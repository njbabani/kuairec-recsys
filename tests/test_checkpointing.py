import json

import polars as pl
import pytest

from recsys.checkpointing import (
    ProgressLog,
    atomic_write,
    fingerprint,
    frame_fingerprint,
    remove_if_empty,
    write_json_atomic,
)


def test_progress_log_saves_after_every_entry_and_resumes_only_matching_work(tmp_path):
    path = tmp_path / "progress.json"
    log = ProgressLog.open(path, fingerprint="abc")

    log.record("config_1", {"score": 0.5})

    assert ProgressLog.open(path, fingerprint="abc").entries == {"config_1": {"score": 0.5}}
    assert ProgressLog.open(path, fingerprint="other").entries == {}


def test_write_json_atomic_leaves_only_the_final_file(tmp_path):
    write_json_atomic({"a": 1}, tmp_path / "nested" / "out.json")

    assert [p.name for p in (tmp_path / "nested").iterdir()] == ["out.json"]
    assert json.loads((tmp_path / "nested" / "out.json").read_text()) == {"a": 1}


def test_fingerprint_changes_with_settings_or_data():
    frame = pl.DataFrame({"x": [1, 2]})

    assert fingerprint({"lr": 0.1}, [frame]) == fingerprint({"lr": 0.1}, [frame.clone()])
    assert fingerprint({"lr": 0.1}, [frame]) != fingerprint({"lr": 0.2}, [frame])
    assert fingerprint({"lr": 0.1}, [frame]) != fingerprint({"lr": 0.1}, [frame.reverse()])
    assert frame_fingerprint(frame) != frame_fingerprint(frame.reverse())


def test_remove_if_empty_only_removes_empty_folders(tmp_path):
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "keep.txt").write_text("x")
    (tmp_path / "empty").mkdir()

    remove_if_empty(tmp_path / "full")
    remove_if_empty(tmp_path / "empty")
    remove_if_empty(None)

    assert (tmp_path / "full" / "keep.txt").exists()
    assert not (tmp_path / "empty").exists()


def test_progress_in_an_unknown_format_is_discarded_not_fatal(tmp_path):
    path = tmp_path / "progress.json"
    path.write_text(json.dumps({"fingerprint": "abc", "scored": {"old": 1}}))

    assert ProgressLog.open(path, fingerprint="abc").entries == {}


def test_atomic_write_removes_the_partial_file_when_writing_fails(tmp_path):
    target = tmp_path / "model.bin"

    def failing_write(partial):
        partial.write_text("half")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        atomic_write(target, failing_write)

    assert list(tmp_path.iterdir()) == []

import hashlib
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest

from recsys.data.download import ChecksumMismatchError, download_file, md5_of

URL = "https://example.org/KuaiRec.zip"
PAYLOAD = b"kuairec-test-archive" * 1000
PAYLOAD_MD5 = hashlib.md5(PAYLOAD, usedforsecurity=False).hexdigest()

Handler = Callable[[httpx.Request], httpx.Response]


def client_from(handler: Handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def client_returning(body: bytes, status_code: int = 200) -> httpx.Client:
    return client_from(lambda request: httpx.Response(status_code, content=body))


def client_that_must_not_be_called() -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request to {request.url}")

    return client_from(handler)


def stream_that_breaks_midway() -> Iterator[bytes]:
    yield PAYLOAD[:100]
    raise httpx.ReadError("connection reset by peer")


def leftover_partials(directory: Path) -> list[Path]:
    return list(directory.glob("*.part"))


def test_md5_of_matches_hashlib(tmp_path):
    path = tmp_path / "file.bin"
    path.write_bytes(PAYLOAD)

    assert md5_of(path, chunk_size=7) == PAYLOAD_MD5


def test_download_writes_file_when_checksum_matches(tmp_path):
    dest = tmp_path / "raw" / "KuaiRec.zip"

    result = download_file(URL, dest, PAYLOAD_MD5, client=client_returning(PAYLOAD))

    assert result == dest
    assert dest.read_bytes() == PAYLOAD
    assert leftover_partials(dest.parent) == []


def test_download_raises_and_keeps_nothing_on_checksum_mismatch(tmp_path):
    dest = tmp_path / "KuaiRec.zip"

    with pytest.raises(ChecksumMismatchError, match=r"expected 0{32}"):
        download_file(URL, dest, "0" * 32, client=client_returning(PAYLOAD))

    assert not dest.exists()
    assert leftover_partials(tmp_path) == []


def test_download_skips_network_when_verified_copy_exists(tmp_path):
    dest = tmp_path / "KuaiRec.zip"
    dest.write_bytes(PAYLOAD)

    result = download_file(URL, dest, PAYLOAD_MD5, client=client_that_must_not_be_called())

    assert result == dest


def test_download_accepts_uppercase_expected_checksum(tmp_path):
    dest = tmp_path / "KuaiRec.zip"
    dest.write_bytes(PAYLOAD)

    download_file(URL, dest, PAYLOAD_MD5.upper(), client=client_that_must_not_be_called())


def test_download_replaces_existing_file_with_wrong_checksum(tmp_path):
    dest = tmp_path / "KuaiRec.zip"
    dest.write_bytes(b"corrupted")

    download_file(URL, dest, PAYLOAD_MD5, client=client_returning(PAYLOAD))

    assert dest.read_bytes() == PAYLOAD


def test_download_retries_transient_errors_with_exponential_backoff(tmp_path):
    attempts: list[httpx.Request] = []

    def flaky(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) == 1:
            raise httpx.ConnectError("temporary DNS failure", request=request)
        return httpx.Response(200, content=PAYLOAD)

    sleeps: list[float] = []
    dest = tmp_path / "KuaiRec.zip"

    download_file(URL, dest, PAYLOAD_MD5, client=client_from(flaky), sleep=sleeps.append)

    assert len(attempts) == 2
    assert sleeps == [2.0]
    assert dest.read_bytes() == PAYLOAD


def test_download_gives_up_after_max_attempts_and_cleans_up_partial_file(tmp_path):
    def always_breaks(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=stream_that_breaks_midway())

    sleeps: list[float] = []
    dest = tmp_path / "KuaiRec.zip"

    with pytest.raises(httpx.ReadError):
        download_file(
            URL, dest, PAYLOAD_MD5, client=client_from(always_breaks), sleep=sleeps.append
        )

    assert sleeps == [2.0, 4.0]
    assert not dest.exists()
    assert leftover_partials(tmp_path) == []


def test_download_does_not_retry_client_errors(tmp_path):
    sleeps: list[float] = []

    with pytest.raises(httpx.HTTPStatusError, match="404"):
        download_file(
            URL,
            tmp_path / "KuaiRec.zip",
            PAYLOAD_MD5,
            client=client_returning(b"", status_code=404),
            sleep=sleeps.append,
        )

    assert sleeps == []


def test_download_retries_server_errors(tmp_path):
    sleeps: list[float] = []

    with pytest.raises(httpx.HTTPStatusError, match="503"):
        download_file(
            URL,
            tmp_path / "KuaiRec.zip",
            PAYLOAD_MD5,
            client=client_returning(b"", status_code=503),
            sleep=sleeps.append,
        )

    assert sleeps == [2.0, 4.0]


def test_failed_redownload_leaves_existing_file_untouched(tmp_path):
    dest = tmp_path / "KuaiRec.zip"
    dest.write_bytes(b"stale-but-present")

    with pytest.raises(httpx.HTTPStatusError):
        download_file(URL, dest, PAYLOAD_MD5, client=client_returning(b"", status_code=404))

    assert dest.read_bytes() == b"stale-but-present"
    assert leftover_partials(tmp_path) == []

"""Guard the reproducibility story: a DVC stage must re-run whenever code it executes changes."""

import ast
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
STAGE_COMMAND_PREFIX = "python -m "


def module_file(module: str) -> Path:
    return SRC / Path(*module.split(".")).with_suffix(".py")


def direct_recsys_imports(path: Path) -> set[str]:
    """Modules imported by ``path`` (``from recsys.viz import style`` yields both candidates)."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("recsys"):
            found.add(node.module)
            found |= {f"{node.module}.{alias.name}" for alias in node.names}
        elif isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names if alias.name.startswith("recsys")}
    return {module for module in found if module_file(module).is_file()}


def executed_modules(entry_point: str) -> set[str]:
    seen: set[str] = set()
    pending = [entry_point]
    while pending:
        module = pending.pop()
        if module not in seen:
            seen.add(module)
            pending.extend(direct_recsys_imports(module_file(module)))
    return seen


def load_stages() -> dict:
    return yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]


@pytest.mark.parametrize("stage_name", sorted(load_stages()))
def test_dvc_stage_lists_every_module_it_executes_as_a_dependency(stage_name):
    stage = load_stages()[stage_name]
    assert stage["cmd"].startswith(STAGE_COMMAND_PREFIX)
    entry_point = stage["cmd"].removeprefix(STAGE_COMMAND_PREFIX).strip()

    required = {
        module_file(module).relative_to(REPO_ROOT).as_posix()
        for module in executed_modules(entry_point)
    }

    missing = required - set(stage["deps"])
    assert not missing, f"stage '{stage_name}' does not depend on {sorted(missing)}"

"""Where generated files are written.

The figures used to be saved to bare file names like ``training_results.png``, which
matplotlib resolves against the current working directory, so running a script from
anywhere but the repository root scattered the outputs. These tests pin the two halves
of the fix: the helper resolves against the repository root, and no call site hands
matplotlib a bare name again.
"""

from __future__ import annotations

import ast
from pathlib import Path

from src import paths

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_output_directories_are_anchored_to_the_repository(tmp_path, monkeypatch):
    """The answer must not depend on where the process was started from."""
    monkeypatch.chdir(tmp_path)

    assert paths.outputs_dir() == REPO_ROOT / "outputs"
    assert paths.artifacts_dir() == REPO_ROOT / "artifacts"
    assert paths.output_path("figure.png") == REPO_ROOT / "outputs" / "figure.png"


def test_output_directories_are_created(tmp_path, monkeypatch):
    """A fresh clone has no outputs/ or artifacts/, and the helper has to make them."""
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path / "clone")

    assert paths.outputs_dir().is_dir()
    assert paths.artifacts_dir().is_dir()


def test_no_figure_is_saved_to_a_bare_file_name():
    """No ``savefig`` call may take a string literal, which would be working-directory relative.

    Checked by parsing rather than by matching text, so it survives reformatting and
    catches a new call site added later, not just the three that used to be wrong.
    """
    offenders = []
    sources = sorted((REPO_ROOT / "src").rglob("*.py")) + sorted((REPO_ROOT / "scripts").rglob("*.py"))

    for file in sources:
        tree = ast.parse(file.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            called = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if called != "savefig":
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                offenders.append(f"{file.relative_to(REPO_ROOT)}:{node.lineno} saves to {first.value!r}")

    assert offenders == [], (
        "these figures would land wherever the process was started from: " + "; ".join(offenders)
    )

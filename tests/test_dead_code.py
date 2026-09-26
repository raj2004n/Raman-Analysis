"""Guards that the retired code stays retired.

Two earlier models, a batch-time data generator, a comparison plot and a
rolling-window loader were removed from ``src/``. Nothing imported them, but a
name can come back by accident (a copy-paste, a merge), and a dead function that
is silently reachable again is how the same code gets maintained twice. These
checks read the parsed tree rather than the text, so a mention inside a docstring
or a comment does not count as a use.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

RETIRED = {
    "CNN_Model1": "src/cnn/model.py",
    "CNN_Model2": "src/cnn/model.py",
    "DataGenerator": "src/cnn/cnn_preprocessing.py",
    "plot_standardised_comparison": "src/cnn/cnn_preprocessing.py",
    "get_area_under_hsi_cube": "src/data/loader.py",
}


def _live_sources():
    """Every module that ships: the package and the command line entry points."""
    return sorted((REPO_ROOT / "src").rglob("*.py")) + sorted((REPO_ROOT / "scripts").rglob("*.py"))


@pytest.mark.parametrize("name", sorted(RETIRED))
def test_a_retired_name_is_not_defined_under_src(name):
    """The name may still be mentioned in a comment, but it must not be defined here."""
    definitions = []
    for file in _live_sources():
        tree = ast.parse(file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
                definitions.append(f"{file.relative_to(REPO_ROOT)}:{node.lineno}")

    assert definitions == [], f"{name} is defined again at " + ", ".join(definitions)


@pytest.mark.parametrize("name", sorted(RETIRED))
def test_a_retired_name_is_not_called_under_src(name):
    """No live module may reference the name, which is what would make it live again.

    Checked against ``Name`` nodes, so a rename of the definition alone does not
    hide a caller that would then fail at run time.
    """
    callers = []
    for file in _live_sources():
        tree = ast.parse(file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == name:
                callers.append(f"{file.relative_to(REPO_ROOT)}:{node.lineno}")

    assert callers == [], f"{name} is referenced again at " + ", ".join(callers)


def test_the_retired_modules_keep_the_code_that_was_moved():
    """The disposition was to keep the code readable on disk, not to lose it.

    ``legacy/`` is not part of the package and is not imported anywhere, so this
    only checks that each moved definition is still there to be read. The directory is
    not tracked by the repository, so a checkout that does not have one skips: it is a
    fact about that machine rather than about the code, and there is nothing there to
    check either way.
    """
    legacy = REPO_ROOT / "legacy"
    if not legacy.is_dir():
        pytest.skip("this checkout has no legacy/ directory")
    for name, source in sorted(RETIRED.items()):
        module = legacy / source.split("/")[-1]
        assert module.is_file(), f"{source.split('/')[-1]} is not in legacy/"
        tree = ast.parse(module.read_text())
        found = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name
        ]
        assert found == [name], f"{name} is not recorded in legacy/{module.name}"

"""Guards that the retired code stays retired.

Two earlier models, a batch-time data generator, a comparison plot and a
rolling-window loader were removed from ``src/``, and the interactive command
line and two of the three viewers it called were retired once the browser
covered the same analyses. Nothing imported the removed code, but a name can
come back by accident (a copy-paste, a merge), and a dead function that is
silently reachable again is how the same code gets maintained twice. These
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
    # The retired command line's prompt layer. Only the prompts are listed: the
    # phase-decomposition viewer it called is still live under
    # ``src/visualisation/``, and a browser test checks the unmixing against it.
    "prompt_args": "scripts/plot_raman.py",
    "prompt_mode": "scripts/plot_raman.py",
    "prompt_heatmap_args": "scripts/plot_raman.py",
    "prompt_phase_decomp_args": "scripts/plot_raman.py",
    "prompt_run_dir": "scripts/plot_raman.py",
    "prompt_predict_args": "scripts/plot_raman.py",
    # The two viewers the browser covers itself. Their figures went with the command
    # line; the windowing the heatmap figure was drawn from, ``query_rolling_auc``, is
    # still live and is what the Heatmap screen asks for its band.
    "show_hsi_viewer": "src/visualisation/view_heatmap.py",
    "pixel_at": "src/visualisation/view_heatmap.py",
    "_apply_intensity_mask": "src/visualisation/view_heatmap.py",
    "show_prediction_map": "src/visualisation/view_predict.py",
    "_hover_annotation": "src/visualisation/view_predict.py",
}


def _live_sources():
    """Every module that ships: the package, and the scripts under ``scripts/``."""
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

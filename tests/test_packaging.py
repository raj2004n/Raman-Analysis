"""The declared dependencies, the recorded ones, and what the code actually imports.

``pyproject.toml`` is the list a person reads and ``pip`` resolves; ``runs.
RECORDED_PACKAGES`` is the list a trained run writes into its own record. The two mean
the same thing, so they are asserted equal here rather than left to drift apart, and a
third check reads the package's syntax tree to confirm nothing is imported that neither
of them declares. A dependency that is used but undeclared is the failure this guards:
it works on the machine it was written on and fails on a fresh install.
"""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path

import pytest

from src.cnn import runs

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"

#: Import names that are not their distribution name. Only the exceptions belong here;
#: everything else is looked up under the name it is imported as.
IMPORT_TO_DISTRIBUTION = {"sklearn": "scikit-learn"}

#: Directories whose imports have to be declared. The notebooks are deliberately not in
#: this list: they are an exploratory record rather than part of the framework, and the
#: one that needs torch states so through the notebooks extra.
SHIPPED = ("src", "scripts")


def _declared_dependencies():
    """The runtime dependency names ``pyproject.toml`` declares, without their bounds."""
    project = tomllib.loads(PYPROJECT.read_text())["project"]

    names = []
    for requirement in project["dependencies"]:
        # "numpy>=2.0,<3" and "ramanspy[extra]>=0.2" both name their distribution first
        name = requirement.split(";")[0].split("[")[0].strip()
        for separator in ("==", ">=", "<=", "~=", "!=", ">", "<"):
            name = name.split(separator)[0]
        names.append(name.strip())
    return names


def _optional_dependencies():
    """The extra name to the dependency names it holds, from ``pyproject.toml``."""
    project = tomllib.loads(PYPROJECT.read_text())["project"]
    return {
        extra: [requirement.split(";")[0].split("[")[0].split("=")[0].split("<")[0]
                .split(">")[0].strip() for requirement in requirements]
        for extra, requirements in project.get("optional-dependencies", {}).items()
    }


def _imported_distributions(root):
    """Every third-party distribution imported under ``root``, as distribution names."""
    imported = set()
    for file in sorted((root / "src").rglob("*.py")) + sorted((root / "scripts").rglob("*.py")):
        tree = ast.parse(file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            else:
                continue
            for name in names:
                if name in sys.stdlib_module_names or name in {"src", "scripts"}:
                    continue
                imported.add(IMPORT_TO_DISTRIBUTION.get(name, name))
    return imported


def test_the_declared_dependencies_are_the_recorded_ones():
    """One list, two readers. A name added to either alone is a run record that lies."""
    assert sorted(_declared_dependencies()) == sorted(runs.RECORDED_PACKAGES)


def test_every_distribution_the_code_imports_is_declared():
    """The check F13 exists for: a 90-line freeze cannot say what is actually used."""
    undeclared = sorted(_imported_distributions(REPO_ROOT) - set(runs.RECORDED_PACKAGES))

    assert undeclared == [], (
        "these are imported under src/ or scripts/ but are not declared as dependencies: "
        + ", ".join(undeclared)
    )


def test_the_import_check_would_notice_a_missing_dependency(monkeypatch, tmp_path):
    """The guard is only worth having if it fails on the thing it forbids."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "stray.py").write_text("import a_package_nobody_declared\n")

    undeclared = sorted(_imported_distributions(tmp_path) - set(runs.RECORDED_PACKAGES))

    assert undeclared == ["a_package_nobody_declared"]


def test_the_declared_dependencies_are_what_an_import_reaches():
    """The other direction: a declared name that nothing imports has to be justified.

    ``tensorflow`` is the Keras backend and ``pydot`` is what ``keras.utils.plot_model``
    imports internally, so both are required without appearing in an import statement.
    Every other declared name is imported directly by the package or its entry points,
    which is what stops a name being declared out of habit.
    """
    required_without_being_imported = {"tensorflow", "pydot"}
    imported = _imported_distributions(REPO_ROOT)

    assert set(_declared_dependencies()) - imported == required_without_being_imported


def test_torch_is_not_a_core_dependency():
    """One notebook imports torch; the framework does not. It belongs in an extra."""
    assert "torch" not in _declared_dependencies()
    assert "torch" in _optional_dependencies()["notebooks"]

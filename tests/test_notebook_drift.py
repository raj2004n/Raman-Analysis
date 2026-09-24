"""Records the state of the notebook scripts under ``notebooks/`` as it is today.

These scripts carry a second, older copy of the data layer in
``notebooks/raman_helper.py``. That duplication is knowingly being left in place:
the notebooks are exploratory records rather than part of the framework, and the
decision was to change nothing about them. This file exists so that decision is
recorded rather than silent.

It matters because the framework's data layer is being changed underneath them.
The notebooks import neither ``src/data/grid.py`` nor ``src/data/loader.py``, so
they cannot move with it. These tests record today's state, which means that if a
notebook genuinely breaks, a test says so instead of a user.

Nothing here asserts that the notebooks work. Most of them do not: they are a
duplicate of a layer that has since moved on, and one of them cannot run at all.
That is the recorded position, not an oversight.
"""

from __future__ import annotations

import ast
import builtins
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = REPO_ROOT / "notebooks"
HELPER = NOTEBOOKS / "raman_helper.py"

# The notebooks that import the helper. t.py does not use it.
HELPER_CONSUMERS = (
    "cluster.py",
    "examples.py",
    "plot_pixel.py",
    "test.py",
    "test_nn.py",
    "unmix.py",
)


def test_the_duplicate_data_layer_still_exists(golden):
    golden.check("notebooks.helper_exists", HELPER.exists())
    golden.check("notebooks.helper_line_count", len(HELPER.read_text().splitlines()))

    assert HELPER.exists(), (
        "notebooks/raman_helper.py is gone. It was deliberately left in place, so "
        "deleting it reverses that decision and the duplication should be re-reviewed."
    )


def test_the_import_styles_are_still_split(golden):
    """One notebook imports the helper by bare name, the other five by package path.

    The split is why the scripts cannot all be run the same way: a bare
    ``from raman_helper import *`` only resolves when the working directory is
    ``notebooks/`` itself.
    """
    styles = {}
    for name in HELPER_CONSUMERS:
        lines = [line.strip() for line in (NOTEBOOKS / name).read_text().splitlines()]
        styles[name] = next(
            (line for line in lines if line.startswith("from ") and "raman_helper" in line),
            None,
        )

    golden.check("notebooks.import_styles", styles)

    assert styles["examples.py"] == "from raman_helper import *"
    assert sum(1 for style in styles.values() if style == "from notebooks.raman_helper import *") == 5


def test_the_broken_call_is_still_broken(golden):
    """``raman_helper.py:260`` calls ``convolve(preprocessed_data, kernel, ...)``.

    The local holding the kernel is named ``auc_kernel``, defined at ``:236``, so
    ``kernel`` is never bound and any notebook reaching that line raises ``NameError``.
    It is recorded rather than fixed, and the value of the record is that the breakage
    stays visible instead of being rediscovered later.

    The check is done by parsing rather than by matching text, so it keeps working if
    the surrounding lines are reformatted.
    """
    tree = ast.parse(HELPER.read_text())

    assigned: set[str] = set()
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                assigned.add(node.id)
            else:
                used.add(node.id)

    undefined = sorted(used - assigned - set(dir(builtins)))
    golden.check("notebooks.names_used_but_never_assigned", undefined)

    assert "auc_kernel" in assigned, "the kernel local was renamed, so line 260 needs re-checking"
    assert "kernel" not in assigned, "a `kernel` binding now exists, so the NameError may be fixed"
    assert "kernel" in used, "the call site no longer references `kernel`"

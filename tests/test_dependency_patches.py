"""The recorded dependency patches, and the tool that puts them on the dependencies.

The patches exist because two of the dependencies are unmaintained and broken against a
current numpy and scipy, in ways that stop the tool running at all. So the tests here are
about the patch set as an artefact:

* That a patch is read as the diff it is written as: the file it names, and the lines it
  changes, context included.
* That applying one to the file it was written for gives back the installed file
  exactly, which is what says the recorded patch is the whole of the change.
* That a patch is not applied to a file it was not written for, and not half applied.
* That this environment carries them, since the suite cannot run without them. That test
  is the one that fails on a fresh environment, and its message says what to run.
"""

from __future__ import annotations

import shutil

from scripts import apply_dependency_patches as applier

NORMALISE_PATCH = "ramanspy-preprocessing-normalise.patch"
NORMALISE_FILE = "ramanspy/preprocessing/normalise.py"


def _patch(name=NORMALISE_PATCH):
    return applier.parse_patch(applier.PATCH_DIR / name)


def test_a_patch_names_its_file_and_the_lines_it_changes():
    """The distribution and the path come from the diff, and the hunks hold the edit."""
    patch = _patch()

    assert patch.distribution == "ramanspy"
    assert patch.relative == NORMALISE_FILE
    assert len(patch.hunks) == 1, "this patch is one line of one function"

    hunk = patch.hunks[0]
    assert b"np.trapz(intensity_data" in hunk.removed
    assert b"np.trapezoid(intensity_data" in hunk.added
    assert b"np.trapz(intensity_data" not in hunk.added


def test_a_hunk_lands_on_its_lines_and_nowhere_else():
    """A file holding exactly the hunk's lines becomes the file holding its replacement."""
    hunk = _patch().hunks[0]

    patched, unplaced = applier.apply_hunks(hunk.removed, (hunk,))

    assert unplaced is None
    assert patched == hunk.added


def test_a_file_that_already_holds_the_change_is_left_alone():
    """What makes the tool safe to run again, which the README tells the reader to do."""
    hunk = _patch().hunks[0]

    assert applier.already_applied(hunk.added, (hunk,))
    assert not applier.already_applied(hunk.removed, (hunk,))


def test_a_file_the_patch_was_not_written_for_is_reported_rather_than_changed():
    """The lines are not there, so the file is not the version the patch was written for.

    Nothing is written in that case: a patch that guessed at where its lines go would
    corrupt a dependency that had moved on, and the report is the point.
    """
    hunk = _patch().hunks[0]
    other = b"def _auc_norm(intensity_data, spectral_axis, *, pixelwise: bool):\n    ...\n"

    patched, unplaced = applier.apply_hunks(other, (hunk,))

    assert unplaced is hunk
    assert patched == other


def test_a_recorded_patch_rebuilds_the_installed_file(tmp_path, monkeypatch):
    """The applier end to end, on a file built by reversing the patch it is handed.

    Reversing the recorded edits gives the file as it stood before them, so applying the
    patch to that has to give back the installed file byte for byte. Anything the patch
    failed to record would show up here as a difference, which is what makes this the
    check on the patch rather than only on the tool.
    """
    installed = applier.installed_path(NORMALISE_FILE)
    assert installed is not None
    patched_bytes = installed.read_bytes()

    patch = _patch()
    unpatched = patched_bytes
    for hunk in patch.hunks:
        assert unpatched.count(hunk.added) == 1, "reversing needs the change to be there"
        unpatched = unpatched.replace(hunk.added, hunk.removed, 1)

    target = tmp_path / NORMALISE_FILE.split("/")[-1]
    target.write_bytes(unpatched)

    alone = tmp_path / "patches"
    alone.mkdir()
    shutil.copy(applier.PATCH_DIR / NORMALISE_PATCH, alone)
    monkeypatch.setattr(applier, "installed_path", lambda relative: target)

    outcomes = applier.apply_patches(patch_dir=alone)

    assert [outcome.status for outcome in outcomes] == [applier.APPLIED]
    assert target.read_bytes() == patched_bytes


def test_a_patch_that_does_not_match_is_not_written(tmp_path, monkeypatch):
    """The same run against a file that is not the one the patch was written for."""
    target = tmp_path / "normalise.py"
    target.write_bytes(b"def _auc_norm():\n    pass\n")

    alone = tmp_path / "patches"
    alone.mkdir()
    shutil.copy(applier.PATCH_DIR / NORMALISE_PATCH, alone)
    monkeypatch.setattr(applier, "installed_path", lambda relative: target)

    outcomes = applier.apply_patches(patch_dir=alone)

    assert [outcome.status for outcome in outcomes] == [applier.NO_MATCH]
    assert target.read_bytes() == b"def _auc_norm():\n    pass\n"


def test_every_recorded_patch_is_in_place():
    """The environment carries them, which is what the dependencies need to work.

    This is the test a fresh environment fails, and the failure is the instruction: the
    patches are applied to an installed copy, so a checkout alone is not enough.
    """
    outcomes = applier.apply_patches(dry_run=True)
    unfinished = [outcome for outcome in outcomes if outcome.status != applier.ALREADY_APPLIED]

    assert not unfinished, (
        "the installed dependencies do not carry every recorded patch: "
        + "; ".join(f"{outcome.patch.path.name}: {outcome.status}" for outcome in unfinished)
        + "\n    Apply them with: python -m scripts.apply_dependency_patches"
    )

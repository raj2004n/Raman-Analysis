"""Puts the patches recorded in ``patches/`` onto the installed dependencies.

Two of the packages this project depends on are no longer maintained and are broken
against a current numpy and scipy, in ways that stop the tool running at all, and
neither has a release to move to. The lines that have to change are kept as unified
diffs in ``patches/``, which is the record of what the environment needs and why, and
this puts them onto the installed copy::

    python -m scripts.apply_dependency_patches

Running it again is safe. A patch that is already in place is reported and not
reapplied, and a patch whose lines are not in the installed file is reported and left
alone rather than forced, since that means the file is not the version the patch was
written for. ``--check`` reports what it would do and writes nothing.

The files are read and written as bytes, so the line endings of a file are left exactly
as they are. That matters here: pysptools' files end their lines the Windows way, and
reading one as text and writing it back would rewrite every line of the file as a side
effect of changing three of them.

The report is printed rather than logged, for the same reason the command line's
prompts are printed: it is the output of the command, and nothing else reads it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from src.paths import REPO_ROOT

#: Where the recorded patches live.
PATCH_DIR = REPO_ROOT / "patches"

#: What can have become of a patch. These are shown to the user, so they read as
#: sentences rather than as codes.
ALREADY_APPLIED = "already applied"
APPLIED = "applied"
WOULD_APPLY = "would apply, and --check was given so nothing was written"
NO_FILE = "the installed file is not there"
NO_MATCH = "the recorded lines are not in the installed file"


@dataclass(frozen=True)
class Hunk:
    """One edit: the lines a patch expects to find, and the lines that replace them.

    Both sides carry the hunk's context lines as well as the lines it changes, because
    the context is what makes the match unambiguous within the file.

    Attributes:
        removed: The context and removed lines, as they stand in the unpatched file.
        added: The context and added lines, as they stand in the patched file.
    """

    removed: bytes
    added: bytes


@dataclass(frozen=True)
class Patch:
    """A recorded patch, and the file it is written for.

    Attributes:
        path: The patch file itself.
        distribution: The installed distribution the patch is aimed at, which is the
            first component of the patched file's path for both packages patched here.
        relative: The patched file's path inside the environment's site-packages.
        hunks: The edits, in the order the patch lists them.
    """

    path: Path
    distribution: str
    relative: str
    hunks: tuple[Hunk, ...]


@dataclass(frozen=True)
class Outcome:
    """What became of one patch, and where it was aimed.

    Attributes:
        patch: The patch, as it was read.
        target: The installed file it was aimed at, or None when the distribution is
            not installed.
        status: One of the module's status values.
    """

    patch: Patch
    target: Path | None
    status: str


def parse_patch(path: Path) -> Patch:
    """Reads one recorded patch.

    The line numbers in the hunk headers are not used. A hunk is placed by its content,
    so a patch aimed at a file it was not written for is reported instead of landing at
    roughly the right place.

    Args:
        path (Path): The patch file.

    Returns:
        Patch: The file it names, and its hunks.

    Raises:
        ValueError: If the patch is not a unified diff of exactly one file, or if it
            ends a file without a newline, which matching cannot express.
    """
    relative = None
    hunks: list[Hunk] = []
    removed: list[bytes] = []
    added: list[bytes] = []
    in_hunk = False

    for line in path.read_bytes().splitlines(keepends=True):
        if line.startswith(b"+++ "):
            if relative is not None:
                raise ValueError(f"{path}: one patch may only name one file")
            named = line[4:].strip().decode()
            if not named.startswith("b/"):
                raise ValueError(f"{path}: expected a b/ path, found {named!r}")
            relative = named[2:]
        elif line.startswith(b"--- "):
            continue
        elif line.startswith(b"@@"):
            if in_hunk:
                hunks.append(Hunk(b"".join(removed), b"".join(added)))
            removed, added, in_hunk = [], [], True
        elif line.startswith(b"\\"):
            raise ValueError(f"{path}: a file without a trailing newline is not supported")
        elif in_hunk:
            # The first byte of a diff line is its role, not part of the file: a space
            # means the line is context and kept, a minus and a plus mean it is on one
            # side of the change only.
            if line[:1] in (b" ", b"-"):
                removed.append(line[1:])
            if line[:1] in (b" ", b"+"):
                added.append(line[1:])

    if in_hunk:
        hunks.append(Hunk(b"".join(removed), b"".join(added)))
    if relative is None:
        raise ValueError(f"{path}: nothing names the file to patch")
    if not hunks:
        raise ValueError(f"{path}: there are no hunks to apply")
    return Patch(path, relative.split("/")[0], relative, tuple(hunks))


def installed_path(relative: str) -> Path | None:
    """Finds where a file of an installed distribution is in this environment.

    Args:
        relative (str): The distribution and the file's path inside site-packages, for
            example ``"ramanspy/preprocessing/normalise.py"``.

    Returns:
        Path or None: The file's path, or None when the distribution is not installed.
    """
    try:
        distribution = metadata.distribution(relative.split("/")[0])
    except metadata.PackageNotFoundError:
        return None
    return Path(distribution.locate_file(relative))


def already_applied(content: bytes, hunks: tuple[Hunk, ...]) -> bool:
    """Whether a file already holds what the patch would put there.

    Args:
        content (bytes): The file as it stands.
        hunks (tuple[Hunk, ...]): The patch's edits.

    Returns:
        bool: True when every hunk's replacement is present exactly once and none of the
        lines it removes is still there.
    """
    return all(content.count(hunk.added) == 1 for hunk in hunks) and not any(
        content.count(hunk.removed) for hunk in hunks
    )


def apply_hunks(content: bytes, hunks: tuple[Hunk, ...]) -> tuple[bytes, Hunk | None]:
    """Applies every hunk to a file's content.

    Args:
        content (bytes): The file as it stands.
        hunks (tuple[Hunk, ...]): The patch's edits.

    Returns:
        tuple: The content with the hunks applied, and the first hunk that could not be
        placed, which is None when all of them were. The content is returned unchanged
        when one could not be placed, so a half-applied file is never written.
    """
    original = content
    for hunk in hunks:
        if content.count(hunk.removed) != 1:
            return original, hunk
        content = content.replace(hunk.removed, hunk.added, 1)
    return content, None


def apply_patches(dry_run: bool = False, patch_dir: Path = PATCH_DIR) -> list[Outcome]:
    """Puts every recorded patch onto the installed dependencies.

    Args:
        dry_run (bool): Report what would happen and write nothing.
        patch_dir (Path): The directory of recorded patches. Defaults to ``patches/``.

    Returns:
        list[Outcome]: One outcome per patch, in the order the files are named.
    """
    outcomes = []
    for path in sorted(patch_dir.glob("*.patch")):
        patch = parse_patch(path)
        target = installed_path(patch.relative)
        if target is None or not target.is_file():
            outcomes.append(Outcome(patch, target, NO_FILE))
            continue

        content = target.read_bytes()
        if already_applied(content, patch.hunks):
            outcomes.append(Outcome(patch, target, ALREADY_APPLIED))
            continue

        patched, unplaced = apply_hunks(content, patch.hunks)
        if unplaced is not None:
            outcomes.append(Outcome(patch, target, NO_MATCH))
            continue

        if not dry_run:
            target.write_bytes(patched)
        outcomes.append(Outcome(patch, target, WOULD_APPLY if dry_run else APPLIED))
    return outcomes


def main(argv=None) -> int:
    """Applies the recorded patches, and reports what became of each one.

    Args:
        argv (list, optional): Command line arguments. Defaults to None, which reads
            them from the process.

    Returns:
        int: 0 when every patch is in place or has just been applied, 1 when one could
        not be accounted for.
    """
    parser = argparse.ArgumentParser(
        prog="python -m scripts.apply_dependency_patches",
        description="Puts the patches in patches/ onto the installed dependencies.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report what would change, and write nothing",
    )
    args = parser.parse_args(argv)

    outcomes = apply_patches(dry_run=args.check)
    for outcome in outcomes:
        print(f"{outcome.status}: {outcome.patch.path.name}")

    failed = [outcome for outcome in outcomes if outcome.status in (NO_FILE, NO_MATCH)]
    if failed:
        for outcome in failed:
            print(f"  {outcome.patch.path.name} is written for {outcome.patch.relative},"
                  f" installed at {outcome.target or 'nowhere in this environment'}")
        print("The installed dependencies are not the versions these patches were"
              " written for, so the tool is unlikely to run correctly.")
        return 1

    print(f"{len(outcomes)} patch(es) accounted for.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

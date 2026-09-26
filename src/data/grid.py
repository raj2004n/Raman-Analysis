import re
from collections import Counter
from pathlib import Path


class RamanGrid:
    """The numbered ``.txt`` files in one directory, treated as a rectangular map.

    A map directory holds files named ``FullGrid[n](-x, -y, -z).txt``, where ``n`` is
    the pixel number; ``traverse`` turns that number into a grid position. Files
    without a number in the name are not part of the map.

    Constructing the grid does not read any file, so building one is cheap. Call
    ``get_sorted_files`` to check the directory and get the files in pixel order.
    """

    #: Matches the pixel number in a ``FullGrid[n](...)`` file name.
    _NUMBER = re.compile(r"\[(\d+)\]")

    def __init__(self, path, x, y):
        """Notes the directory and the grid size, and reads nothing.

        Args:
            path (str or Path): The directory holding the map's files. ``~`` is
                expanded, so a path from a shell prompt can be passed through.
            x, y (int): The rows and columns the files are laid out on.
        """
        self.path = Path(path).expanduser()
        self.x = x
        self.y = y

    def _get_file_number(self, path):
        """Gets the file number of the corresponding file.
            eg. ".../FullGrid[1](-18000, -320, -14675)" -> 1

        Args:
            path (Path): Path to the file.

        Returns:
            int : File number of the corresponding file.
                Returns None if no file number found.
        """
        match = self._NUMBER.search(path.name)
        return int(match.group(1)) if match else None

    def _discovery_error(self, ignored, found):
        """Describes why the directory does not hold the grid it was asked for.

        Args:
            ignored (list[str]): Names of the files skipped for having no number.
            found (list[int]): The pixel numbers that were found, sorted.

        Returns:
            str: A message naming the files that were ignored and the numbers that are
                missing or repeated, so the cause is clear from the message alone.
        """
        expected = self.x * self.y
        lines = [
            f"{self.path} does not hold the {expected} numbered .txt files that a "
            f"{self.x}x{self.y} grid needs (found {len(found)})."
        ]

        duplicates = sorted(n for n, count in Counter(found).items() if count > 1)
        missing = sorted(set(range(1, expected + 1)) - set(found))
        if missing:
            lines.append(f"Pixel numbers with no file: {missing}.")
        if duplicates:
            lines.append(f"Pixel numbers used by more than one file: {duplicates}.")
        if ignored:
            lines.append(
                "Files ignored because their name has no [n] pixel number: "
                + ", ".join(sorted(ignored))
                + "."
            )
        return " ".join(lines)

    def get_sorted_files(self):
        """The map's files, in pixel order.

        Returns:
            list[Path]: Exactly ``x * y`` files, sorted by the number in their name.

        Raises:
            FileNotFoundError: The directory holds no numbered file. A directory with
                only unnumbered ``.txt`` files raises this too, and the message names
                them.
            ValueError: The numbered files are not exactly the pixel numbers
                ``1..x * y``. The message names the numbers that are missing or
                repeated, and any unnumbered files that were skipped. A directory with
                a gap or a duplicate cannot be laid out on the grid, so this is an
                error rather than a warning: carrying on would place every pixel after
                the gap at the wrong position without saying so.
        """
        entries = []  # (pixel number, path) for each numbered file
        ignored = []  # names of files with no pixel number
        for file in sorted(self.path.glob("*.txt")):
            number = self._get_file_number(file)
            if number is None:
                ignored.append(file.name)
            else:
                entries.append((number, file))

        if not entries:
            raise FileNotFoundError(
                f"No numbered .txt files found in {self.path}"
                + (f". Ignored: {', '.join(sorted(ignored))}." if ignored else ".")
            )

        entries.sort(key=lambda entry: entry[0])
        found = [number for number, _ in entries]
        if found != list(range(1, self.x * self.y + 1)):
            raise ValueError(self._discovery_error(ignored, found))

        return [file for _, file in entries]

    def _step(self, cur_x, cur_y, step):
        """The position after one move along a row, turning round at the end of it.

        The walk goes along a row in one direction, steps up to the next row when it
        reaches the boundary, and comes back along that row in the other direction, so
        every position is visited exactly once.

        Args:
            cur_x (int): The row the walk is on.
            cur_y (int): The column it is at within that row.
            step (int): 1 while the walk is heading towards increasing columns, -1 while
                it is heading back.

        Returns:
            tuple: The next row, column and direction, in that order.
        """
        if cur_y == self.y - 1 and step != -1: # on right boundary
            cur_x -= 1 # step up
            step *= -1 # flip step direction
        elif cur_y == 0 and step != 1: # on left boundary
            cur_x -= 1 # step up
            step *= -1 # flip step direction
        else: # not on boundary
            cur_y += step # step
        return cur_x, cur_y, step

    def traverse(self):
        """Walks the map's files in the order the grid was laid out in.

        The order is boustrophedon: from the last row index down to the first, along a
        row one way and back along the next, which is the order the pixel numbers were
        assigned in.

        Yields:
            tuple: The file, and the row and column it belongs at, in that order.
        """
        files = self.get_sorted_files()
        # initial position and step direction
        cur_x, cur_y, step = self.x - 1, 0, 1
        for file in files:
            yield file, cur_x, cur_y
            cur_x, cur_y, step = self._step(cur_x, cur_y, step)

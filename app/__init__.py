"""The browser interface over the framework.

Importing this package chooses matplotlib's backend for the whole process, before
pyplot can be imported by anything under it. A server has no window to draw in and
nobody to close one: matplotlib picks an interactive backend when it finds a display,
so a figure drawn while serving a request would open a window on whichever machine
happened to be running the server. Agg writes files and opens nothing, which is what
every figure this package draws is for.
"""

import matplotlib

matplotlib.use("Agg")

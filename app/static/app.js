/* The four screens, drawn into one shell.
 *
 * The shell is served for every screen URL, so this file reads
 * ``location.pathname`` to decide which screen to draw. The rail, the top bar and
 * the session they share are built here; each screen owns only its own body.
 *
 * The screens are plain DOM. There is no framework and no build step, so a change
 * to a screen is a change to this file and a reload.
 */

// ---------------------------------------------------------------- the server

/** Calls a route and returns its JSON, raising the route's own message on failure.
 *
 * FastAPI puts the reason a request was refused in `detail`, so that is what is
 * raised: the grid's own wording about missing pixel numbers, or the job runner's
 * about one already running, arrives at the screen unedited.
 */
async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "content-type": "application/json" },
    ...options,
  });

  let payload = null;
  try {
    payload = await response.json();
  } catch (error) {
    payload = null;
  }

  if (!response.ok) {
    const detail = payload && payload.detail ? payload.detail : `The server answered ${response.status}.`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return payload;
}

const get = (path) => api(path);
const post = (path, body) => api(path, { method: "POST", body: JSON.stringify(body ?? {}) });

/** Builds an element.
 *
 * Keeping this to one line each is what makes the screens below read as the
 * layout they describe rather than as a wall of createElement calls.
 */
function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "onclick") node.addEventListener("click", value);
    else if (value !== null && value !== undefined) node.setAttribute(key, value);
  }
  for (const child of children) if (child) node.append(child);
  return node;
}

/** A labelled form control. */
function field(label, control) {
  return el("label", { class: "field" }, [
    el("span", { class: "field-label", text: label }),
    control,
  ]);
}

/** Follows a job until it stops, calling back as it moves.
 *
 * Polling rather than a stream: the job writes its progress to a file and this asks
 * for it, so a page that is reloaded mid-job rejoins by polling the same identifier
 * rather than by holding a connection open.
 */
function watchJob(job, { onStep, onDone, onError, onCancelled }) {
  const tick = async () => {
    let current;
    try {
      current = await get(`/api/job/${job.id}`);
    } catch (error) {
      onError(error.message);
      return;
    }

    if (current.state === "running") {
      onStep(current);
      setTimeout(tick, 300);
      return;
    }
    if (current.state === "done") onDone(current);
    else if (current.state === "cancelled") onCancelled(current);
    else onError(current.error || "The job stopped without saying why.");
  };
  tick();
}

// ------------------------------------------------------------------ the shell

const SCREENS = {
  data: {
    title: "Data",
    sub: "Point at a map directory and say how many pixels it has. Nothing is read until Load map.",
    render: renderData,
  },
  heatmap: {
    title: "Heatmap",
    sub: "One band of the map, and the spectra under the pointer.",
    render: renderHeatmap,
  },
  unmix: {
    title: "Unmixing",
    sub: "How many phases the map holds, and where each of them sits.",
    render: renderUnmix,
  },
  predict: {
    title: "Predict",
    sub: "A trained run applied to this map, pixel by pixel.",
    render: renderPending,
  },
};

/** The path this page was served at, which is the screen to draw. */
function currentScreen() {
  const name = location.pathname.replace(/^\/|\/$/g, "");
  return name in SCREENS ? name : "data";
}

/** Marks the rail link for the screen being shown, and dims the rest. */
function markRail(screen) {
  for (const link of document.querySelectorAll(".rail-item")) {
    if (link.dataset.screen === screen) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
}

/** Dims the screens that need a map while none is loaded.
 *
 * The three screens after Data all work on the loaded map, and each of them would
 * otherwise draw an empty shell and a sentence. Dimming says the same thing before the
 * click rather than after it, and leaves the link in place so the rail still reads as
 * the four screens it is.
 */
function markRailAvailability(loaded) {
  for (const link of document.querySelectorAll(".rail-item")) {
    if (link.dataset.screen === "data") continue;
    if (loaded) link.removeAttribute("aria-disabled");
    else link.setAttribute("aria-disabled", "true");
  }
}

/** The receipt for what is loaded, drawn above every screen.
 *
 * A screen that needs a map says so through this rather than by drawing its own
 * version of the same facts, so one map loaded once reads the same everywhere.
 */
async function drawTopbar() {
  const bar = document.getElementById("topbar");
  bar.replaceChildren();

  let session;
  try {
    session = await get("/api/session");
  } catch (error) {
    bar.append(el("span", { class: "chip", text: "the server is not answering" }));
    return;
  }

  if (!session.loaded) {
    markRailAvailability(false);
    bar.append(el("span", { class: "chip dim", text: "no map loaded" }));
    return;
  }

  markRailAvailability(true);

  bar.append(
    el("span", { class: "chip strong", text: session.label }),
    el("span", { class: "chip", text: session.directory }),
    el("span", { class: "chip", text: `${session.rows} × ${session.cols}` }),
    el("span", { class: "chip", text: `${session.files} files` }),
    el("span", { class: "chip", text: `${session.bands} bands` }),
    el("span", {
      class: "chip",
      text: `${session.axis_start.toFixed(1)} – ${session.axis_end.toFixed(1)} cm⁻¹`,
    }),
  );
}

/** Draws the shell around the screen: heading, then the screen's own body. */
function drawScreen(screen) {
  const spec = SCREENS[screen];
  markRail(screen);

  const root = document.getElementById("screen");
  root.replaceChildren(
    el("h1", { text: spec.title }),
    el("p", { class: "sub", text: spec.sub }),
  );
  root.append(el("div", { class: "screens" }));
  spec.render(root.querySelector(".screens"));
}

/** A screen that has not been built yet, so the shell is honest about it. */
function renderPending(root) {
  root.append(el("div", { class: "card" }, [
    el("p", { class: "note", text: "This screen is not built yet." }),
  ]));
}

// ------------------------------------------------------------------- the data

/** The Data screen: the map to open, what the grid will hold, and the read itself. */
function renderData(root) {
  const directory = el("input", {
    type: "text",
    class: "wide",
    id: "data-directory",
    placeholder: "~/Code/Data_SH/SB008",
    spellcheck: "false",
  });
  const browse = el("button", { type: "button", id: "data-browse", text: "Browse…" });
  const rows = el("input", { type: "number", id: "data-rows", min: "1", value: "10" });
  const cols = el("input", { type: "number", id: "data-cols", min: "1", value: "13" });
  const load = el("button", {
    type: "button",
    class: "primary",
    id: "data-load",
    text: "Load map",
  });

  const strip = el("div", { class: "strip", id: "data-strip", hidden: "" });
  const picker = el("div", { class: "picker", id: "data-picker", hidden: "" });
  const preview = el("div", { class: "preview", id: "data-preview" });
  const progress = el("div", { class: "progress", id: "data-progress", hidden: "" });

  const form = el("div", { class: "card" }, [
    field("MAP DIRECTORY", el("div", { class: "row" }, [directory, browse])),
    picker,
    el("div", { class: "row" }, [
      field("GRID", el("div", { class: "row" }, [
        rows,
        el("span", { class: "note", text: "×" }),
        cols,
      ])),
    ]),
    strip,
    el("div", { class: "row" }, [load]),
    progress,
  ]);

  const why = el("div", { class: "card why" }, [
    el("div", { class: "field-label", text: "WHY THIS IS HERE" }),
    el("p", {
      class: "note",
      text:
        "The count has to be exactly rows × cols, and it has to be the count the loaders " +
        "would read, or every pixel after a gap is placed in the wrong position without " +
        "anything saying so. The check runs on every keystroke and reads nothing: it is " +
        "the same file discovery the load itself uses.",
    }),
    el("p", {
      class: "note",
      text:
        "Pipeline and spectral range are deliberately not here. They belong to the " +
        "screens that use them, so one map serves all of them.",
    }),
  ]);

  root.append(el("div", { class: "grid-2" }, [form, el("div", { class: "card" }, [
    el("div", { class: "field-label", text: "GRID PREVIEW" }),
    preview,
    el("p", { class: "note", id: "data-preview-note", text: "No directory yet." }),
  ])]), why);

  // -- the live check ---------------------------------------------------------

  let checkTimer = null;

  const say = (message, state) => {
    strip.hidden = !message;
    strip.textContent = message || "";
    if (state) strip.dataset.state = state;
    else delete strip.dataset.state;
  };

  const drawPreview = (report) => {
    const r = Number(rows.value);
    const c = Number(cols.value);
    preview.replaceChildren();
    if (!(r > 0 && c > 0)) return;

    // One dot per pixel, in the order the grid is laid out in: the same boustrophedon
    // walk the loader makes, so the preview shows the file order and not just the shape.
    for (let i = 0; i < r * c; i += 1) {
      preview.append(el("span", { class: report.ok ? "dot filled" : "dot" }));
    }
    preview.style.gridTemplateColumns = `repeat(${c}, 1fr)`;
  };

  const complete = () => directory.value.trim() !== "" && Number(rows.value) > 0 && Number(cols.value) > 0;

  const check = async () => {
    const r = Number(rows.value);
    const c = Number(cols.value);
    const path = directory.value.trim();

    if (!path || !(r > 0) || !(c > 0)) {
      load.disabled = true;
      say("", null);
      drawPreview({ ok: false });
      document.getElementById("data-preview-note").textContent = "No directory yet.";
      return;
    }

    let report;
    try {
      report = await get(
        `/api/grid?path=${encodeURIComponent(path)}&rows=${r}&cols=${c}`
      );
    } catch (error) {
      load.disabled = true;
      say(error.message, "error");
      return;
    }

    load.disabled = !report.ok;
    say(report.message, report.ok ? null : "error");
    drawPreview(report);
    document.getElementById("data-preview-note").textContent = report.ok
      ? `${report.count} numbered .txt files, laid out ${r} × ${c}.`
      : `${report.expected} needed, ${report.count === null ? "none found" : report.count + " found"}.`;
  };

  const checkSoon = () => {
    clearTimeout(checkTimer);
    checkTimer = setTimeout(check, 150);
  };

  for (const control of [directory, rows, cols]) {
    control.addEventListener("input", checkSoon);
    control.addEventListener("change", checkSoon);
  }

  // -- Browse -----------------------------------------------------------------
  //
  // The browser cannot name a path on the machine the server runs on, so the listing
  // comes from the server. It is a directory walk, not a file picker: the map is a
  // directory, and only the directories under it are offered.

  browse.addEventListener("click", async () => {
    if (!picker.hidden) {
      picker.hidden = true;
      return;
    }
    const start = directory.value.trim() || "~";
    await showPicker(picker, start, (chosen) => {
      directory.value = chosen;
      picker.hidden = true;
      check();
    });
  });

  // -- Load -------------------------------------------------------------------

  load.addEventListener("click", async () => {
    load.disabled = true;
    say("", null);

    let started;
    try {
      started = await post("/api/map", {
        path: directory.value.trim(),
        rows: Number(rows.value),
        cols: Number(cols.value),
      });
    } catch (error) {
      say(error.message, "error");
      load.disabled = false;
      return;
    }

    const label = el("span", { class: "note" });
    const bar = el("div", { class: "bar" }, [el("div", { class: "bar-fill" })]);
    const cancel = el("button", { type: "button", text: "Cancel" });
    progress.replaceChildren(el("div", { class: "row" }, [label, cancel]), bar);
    progress.hidden = false;

    // The read reports which step it is on rather than how far through it is: the
    // loader walks the grid in one call with nothing to count against, so a filling
    // bar would be a bar that lies. It moves, and it says what it is doing.
    const fill = bar.querySelector(".bar-fill");
    fill.classList.add("indeterminate");

    cancel.addEventListener("click", async () => {
      cancel.disabled = true;
      try {
        await post(`/api/job/${started.job.id}/cancel`);
      } catch (error) {
        label.textContent = error.message;
      }
    });

    const finish = () => {
      progress.hidden = true;
      progress.replaceChildren();
      load.disabled = false;
    };

    watchJob(started.job, {
      onStep: (job) => {
        label.textContent = `${describeStep(job.step)}${job.detail ? `: ${job.detail}` : ""}`;
      },
      onDone: async () => {
        label.textContent = "Read.";
        finish();
        await drawTopbar();
        await check();
      },
      onCancelled: () => {
        finish();
        say("The read was cancelled. The map that was loaded is still loaded.", null);
      },
      onError: (message) => {
        finish();
        say(message, "error");
      },
    });
  });

  // The form starts from the map already in hand, and is checked once on arrival, so a
  // directory typed earlier is reported before anything is clicked. A blank field beside
  // a session chip claiming a map is loaded would read as two different answers to the
  // same question.
  (async () => {
    let existing = { loaded: false };
    try {
      existing = await get("/api/session");
    } catch (error) {
      existing = { loaded: false };
    }
    if (existing.loaded) {
      directory.value = existing.directory;
      rows.value = existing.rows;
      cols.value = existing.cols;
    }
    check();
  })();
}

/** The step a job is on, in the words a screen says it in. */
function describeStep(step) {
  const steps = {
    reading: "Reading the map",
    saving: "Writing the session",
    estimating: "Estimating the phase count",
    decomposing: "Decomposing",
    drawing: "Drawing the panels",
  };
  return steps[step] || step || "Working";
}

/** Draws a directory listing and calls back with what was chosen.
 *
 * One level at a time, with a way back up. It offers directories only: the map is a
 * directory, and a file cannot be loaded as one.
 */
async function showPicker(container, start, onChoose) {
  container.hidden = false;
  container.replaceChildren(el("p", { class: "note", text: "Reading the directory…" }));

  let listing;
  try {
    listing = await get(`/api/directories?path=${encodeURIComponent(start)}`);
  } catch (error) {
    container.replaceChildren(el("p", { class: "note", text: error.message }));
    return;
  }

  const children = [];
  if (listing.parent) {
    children.push(el("button", {
      type: "button",
      class: "picker-item",
      text: "↑ ..",
      onclick: () => showPicker(container, listing.parent, onChoose),
    }));
  }

  for (const name of listing.directories) {
    children.push(el("button", {
      type: "button",
      class: "picker-item",
      text: name,
      onclick: () => showPicker(container, `${listing.path}/${name}`, onChoose),
    }));
  }

  if (!listing.directories.length && !listing.parent) {
    children.push(el("p", { class: "note", text: "No directories here." }));
  }

  container.replaceChildren(
    el("div", { class: "picker-head" }, [
      el("span", { class: "mono", text: listing.path }),
      el("button", { type: "button", text: "Use this", onclick: () => onChoose(listing.path) }),
    ]),
    el("div", { class: "picker-list" }, children),
  );
}

// ---------------------------------------------------------------- the heatmap

/* The colormap the maps are drawn in, sampled from matplotlib's magma so a map in
 * the browser and a map in an exported figure are the same colours. Enough stops
 * that the interpolation between them is not visible on a map of a hundred pixels. */
const MAGMA = [
  "#000003", "#0A0722", "#1C1046", "#350F6A", "#50127B", "#691C80", "#822581",
  "#9C2E7F", "#B63679", "#D0416F", "#E65162", "#F56A5B", "#FB8861", "#FEA671",
  "#FEC488", "#FDE1A3", "#FBFCBF",
];

/** How big one pixel is on the canvas bitmap, before the element is scaled to fit. */
const CELL = 24;


/** The colour at a fraction of the way along the colormap. */function magma(fraction) {
  const span = MAGMA.length - 1;
  const position = Math.min(Math.max(fraction, 0), 1) * span;
  const low = Math.floor(position);
  const high = Math.min(low + 1, span);
  const mix = position - low;

  const channel = (hex, offset) => parseInt(hex.slice(1 + offset * 2, 3 + offset * 2), 16);
  const parts = [0, 1, 2].map((offset) =>
    Math.round(channel(MAGMA[low], offset) * (1 - mix) + channel(MAGMA[high], offset) * mix)
  );
  return `rgb(${parts[0]}, ${parts[1]}, ${parts[2]})`;
}

/** The magma colormap as a CSS gradient, for the colour bar under a map. */
function magmaGradient(direction) {
  return `linear-gradient(${direction}, ${MAGMA.join(", ")})`;
}

/** An SVG element in the namespace the document needs.
 *
 * `createElement` makes an HTML element with the right tag name and none of the SVG
 * behaviour, so every element of a drawing has to be made this way instead.
 */
function svgEl(tag, props = {}, children = []) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "text") node.textContent = value;
    else if (value !== null && value !== undefined) node.setAttribute(key, value);
  }
  for (const child of children) if (child) node.append(child);
  return node;
}

/** The spectrum panel: one trace, the window's two ends, and an axis.
 *
 * Drawn as SVG rather than on a canvas because the two dashed lines, the ticks and the
 * labels are marks in a coordinate system, which is what SVG is for, and because they
 * stay crisp when the window is resized or the page is printed.
 */
function spectrumPanel(axis, values, windowStart, windowEnd, xLabel) {
  const width = 660;
  const height = 180;
  const pad = { left: 54, right: 12, top: 10, bottom: 30 };

  const first = axis[0];
  const last = axis[axis.length - 1];
  let low = Math.min(...values);
  let high = Math.max(...values);
  if (high - low < 1e-12) {
    high = low + 1;
  }

  const x = (value) => pad.left + ((value - first) / (last - first)) * (width - pad.left - pad.right);
  const y = (value) => height - pad.bottom - ((value - low) / (high - low)) * (height - pad.top - pad.bottom);

  const line = axis.map((shift, index) => `${index ? "L" : "M"}${x(shift).toFixed(2)},${y(values[index]).toFixed(2)}`).join(" ");

  const marks = [];
  for (const edge of [windowStart, windowEnd]) {
    if (edge >= first && edge <= last) {
      marks.push(svgEl("line", {
        x1: x(edge), y1: pad.top, x2: x(edge), y2: height - pad.bottom,
        stroke: "#E07B54", "stroke-width": 1, "stroke-dasharray": "4 3", opacity: 0.8,
      }));
    }
  }

  const ticks = [0, 0.25, 0.5, 0.75, 1].map((fraction) => {
    const value = first + fraction * (last - first);
    return svgEl("text", {
      x: x(value), y: height - 10, fill: "#8B949E", "font-size": 10,
      "text-anchor": "middle", text: value.toFixed(0),
    });
  });

  const vertical = [
    svgEl("text", { x: 12, y: y(high) + 4, fill: "#8B949E", "font-size": 10, text: high.toPrecision(3) }),
    svgEl("text", { x: 12, y: y(low) + 4, fill: "#8B949E", "font-size": 10, text: low.toPrecision(3) }),
  ];

  return svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
  }, [
    svgEl("rect", {
      x: pad.left, y: pad.top,
      width: width - pad.left - pad.right, height: height - pad.top - pad.bottom,
      fill: "#0E1117", stroke: "#2F3742",
    }),
    ...marks,
    svgEl("path", { d: line, fill: "none", stroke: "#4FB3FF", "stroke-width": 1.3 }),
    ...ticks,
    ...vertical,
    svgEl("text", {
      x: width - pad.right, y: 12, fill: "#8B949E", "font-size": 10,
      "text-anchor": "end", text: xLabel,
    }),
  ]);
}

/** The Heatmap screen: one band, the spectrum of a pixel, and the window both share. */
async function renderHeatmap(root) {
  let session;
  try {
    session = await get("/api/session");
  } catch (error) {
    session = { loaded: false };
  }

  if (!session.loaded) {
    root.append(el("div", { class: "card" }, [
      el("p", { class: "note", text: "No map is loaded. Load one on the Data screen first." }),
      el("p", {}, [el("a", { class: "link", href: "/data", text: "Go to Data" })]),
    ]));
    return;
  }

  // What the screen is showing. Everything drawn is a function of this, so a control
  // changes one field and calls the redraw that field affects.
  const state = {
    center: 0,
    widthCm: 10,
    pixel: Math.min(47, session.rows * session.cols),
    logScale: false,
    intensity: [0, 1],
    frame: null,
  };

  // -- the panels -------------------------------------------------------------

  const canvas = el("canvas", { class: "map", width: 640, height: 420 });
  // The pixel the pointer is over, and the marks that say which cell that is. Both are
  // readouts of the same thing: the badge names the pixel, the crosshair points at it.
  const badge = el("span", { class: "map-badge", id: "map-badge" });
  const crosshair = el("div", { class: "crosshair", hidden: "" }, [
    svgEl("svg", { viewBox: "0 0 24 24", width: 24, height: 24 }, [
      svgEl("circle", {
        cx: 12, cy: 12, r: 5, fill: "none",
        stroke: "#E6EDF3", "stroke-opacity": 0.75, "stroke-width": 1.5,
      }),
      svgEl("path", {
        d: "M12 1 V6 M12 18 V23 M1 12 H6 M18 12 H23",
        stroke: "#E6EDF3", "stroke-opacity": 0.75, "stroke-width": 1.5,
      }),
    ]),
  ]);
  const mapNote = el("p", { class: "note", id: "map-note" });
  const colourBar = el("div", { class: "colourbar" }, [
    el("span", { class: "note", id: "hm-imax-label" }),
    el("div", { class: "colourbar-track" }),
    el("span", { class: "note", id: "hm-imin-label" }),
  ]);

  const spectrumBox = el("div", { class: "panel-plot" });
  const lnBox = el("div", { class: "panel-plot" });

  const mapCard = el("div", { class: "card" }, [
    el("div", { class: "panel-head" }, [
      el("span", { class: "field-label", text: "SPATIAL MAP" }),
      el("span", { class: "note", id: "band-note" }),
    ]),
    el("div", { class: "map-row" }, [
      el("div", { class: "map-wrap" }, [canvas, badge, crosshair]),
      colourBar,
    ]),
    mapNote,
  ]);

  const spectrumCard = el("div", { class: "card" }, [
    el("div", { class: "panel-head" }, [
      el("span", { class: "field-label", text: "INTENSITY SPECTRA" }),
    ]),
    spectrumBox,
  ]);

  const lnCard = el("div", { class: "card" }, [
    el("div", { class: "panel-head" }, [
      el("span", { class: "field-label", text: "INTENSITY SPECTRA (LN SCALE)" }),
    ]),
    lnBox,
  ]);

  // -- the top bar's own controls ---------------------------------------------

  const pipeline = el("select", { id: "hm-pipeline" }, [0, 1, 2, 3].map((id) =>
    el("option", { value: id, text: id === 0 ? "No pipeline (P0)" : `P${id}` })
  ));
  pipeline.value = String(session.pipeline_id ?? 0);

  const rangeStart = el("input", {
    type: "number", class: "narrow", id: "hm-start",
    placeholder: session.axis_start.toFixed(0),
  });
  const rangeEnd = el("input", {
    type: "number", class: "narrow", id: "hm-end",
    placeholder: session.axis_end.toFixed(0),
  });
  const exportButton = el("button", { type: "button", id: "hm-export", text: "Export PNG" });
  const deriveNote = el("span", { class: "note", id: "hm-derive" });

  const toolbar = el("div", { class: "card row toolbar" }, [
    el("label", { class: "inline-field" }, [
      el("span", { class: "field-label", text: "PIPELINE" }), pipeline,
    ]),
    el("label", { class: "inline-field" }, [
      el("span", { class: "field-label", text: "RANGE (CM⁻¹)" }),
      el("span", { class: "row" }, [
        rangeStart, el("span", { class: "note", text: "–" }), rangeEnd,
        // Said here rather than discovered: either control means reading and integrating
        // every file again, so the wait is expected rather than a surprise.
        el("span", { class: "note warn", text: "changing either reloads the cube" }),
      ]),
    ]),
    el("span", { class: "spacer" }),
    deriveNote,
    exportButton,
  ]);

  // -- the controls -----------------------------------------------------------

  const shift = el("input", { type: "range", id: "hm-shift", min: 0, max: 1, step: 1, value: 0 });
  const shiftValue = el("span", { class: "ctrl-value" });
  const width = el("input", { type: "range", id: "hm-width", min: 1, max: 10, step: 0.5, value: 10 });
  const widthValue = el("span", { class: "ctrl-value" });
  const intensityLow = el("input", { type: "range", class: "vertical", id: "hm-imin" });
  const intensityHigh = el("input", { type: "range", class: "vertical", id: "hm-imax" });
  const scaleButton = el("button", { type: "button", id: "hm-scale", text: "Switch to ln scale" });
  const pixel = el("input", { type: "number", class: "narrow", id: "hm-pixel", min: 1, step: 1 });
  const pixelValue = el("span", { class: "note" });

  // Position and width are separate controls because they are separate questions: the
  // readout beside each is the only place the exact numbers appear.
  const controls = el("div", { class: "card controls" }, [
    el("label", { class: "ctrl" }, [
      el("span", { class: "ctrl-head" }, [
        el("span", { class: "field-label", text: "RAMAN SHIFT" }),
        shiftValue,
      ]),
      shift,
    ]),
    el("label", { class: "ctrl" }, [
      el("span", { class: "ctrl-head" }, [
        el("span", { class: "field-label", text: "WINDOW WIDTH (CM⁻¹)" }),
        widthValue,
      ]),
      width,
    ]),
  ]);

  // The intensity range is a range rather than a value, and it is what the colour bar
  // is scaled to, so it stands beside the image rather than under it.
  const intensityCard = el("div", { class: "card" }, [
    el("div", { class: "field-label", text: "INTENSITY RANGE" }),
    el("div", { class: "vertical-pair" }, [
      el("span", { class: "note", id: "hm-imax-value" }),
      intensityHigh,
      intensityLow,
      el("span", { class: "note", id: "hm-imin-value" }),
    ]),
  ]);

  const scaleCard = el("div", { class: "card" }, [scaleButton]);

  const pixelCard = el("label", { class: "card" }, [
    el("span", { class: "field-label", text: "PIXEL" }),
    el("span", { class: "ctrl-head" }, [pixel, pixelValue]),
    el("p", { class: "note", text: "Return moves the spectra to it, and so does a double-click on the map." }),
  ]);

  // -- the rules the screen is written around ---------------------------------

  const pointerCard = el("div", { class: "card why" }, [
    el("div", { class: "field-label", text: "POINTER" }),
    el("p", { class: "note mono", text: "hover the map → reads the pixel number" }),
    el("p", { class: "note mono", text: "double-click → pins that pixel's spectrum" }),
    el("p", { class: "note mono", text: "single click → does nothing, on purpose" }),
    el("p", { class: "note mono", text: "drag a slider → redraws without reloading" }),
    el("p", { class: "note", text: "Panning sends clicks too, which is why a single click must not move the spectrum." }),
  ]);

  const windowCard = el("div", { class: "card why" }, [
    el("div", { class: "field-label", text: "WINDOW" }),
    el("p", { class: "note mono", id: "hm-window-eq", text: "" }),
    el("p", { class: "note mono", text: "min 2 × band step, max half the range" }),
    el("p", { class: "note mono", text: "snaps to the band step" }),
    el("p", { class: "note mono", text: "areas outside the range draw grey" }),
    el("p", { class: "note", text: "Narrower windows separate peaks that overlap in the full range, and cost nothing to compute." }),
  ]);

  root.append(
    toolbar,
    el("div", { class: "split" }, [
      el("div", { class: "split-left" }, [mapCard, spectrumCard, lnCard, controls]),
      el("div", { class: "split-right" }, [
        intensityCard, scaleCard, pixelCard, pointerCard, windowCard,
      ]),
    ]),
  );

  // -- drawing ----------------------------------------------------------------

  /** Sizes the canvas on screen: one square per pixel, and no taller than the window.
   *
   * The whole element is scaled rather than stretched, so the grid's shape survives at
   * every window size. That matters beyond looks: a point on the image is turned back
   * into a pixel number by where it falls in the element, which is only the pixel it is
   * drawn on while the squares stay square.
   */
  const fitMap = () => {
    const frame = state.frame;
    if (!frame) return;

    const available = canvas.parentElement.clientWidth;
    if (!available) return;

    const scale = Math.min(
      available / canvas.width,
      (window.innerHeight * 0.34) / canvas.height,
    );
    canvas.style.width = `${Math.round(canvas.width * scale)}px`;
    canvas.style.height = `${Math.round(canvas.height * scale)}px`;

    // The map is centred in its column, so the badge is placed against the map rather
    // than against the column, which would leave it stranded in the margin.
    badge.style.left = `${canvas.offsetLeft + 12}px`;
    badge.style.top = `${canvas.offsetTop + 12}px`;
  };

  /** The intensity range the picture is drawn in, in plain units.
   *
   * The sliders carry ln values while the ln scale is on, so their ends are turned
   * back into areas by undoing the log and the shift that was added before it. The
   * image is in plain units either way, which is what keeps the colour bar and the
   * picture saying the same thing.
   */
  const drawRange = () => {
    const low = Number(intensityLow.value);
    const high = Number(intensityHigh.value);
    if (!state.logScale) return [low, high];
    return [Math.exp(low) - state.frame.limits.correction, Math.exp(high) - state.frame.limits.correction];
  };

  const drawBand = () => {
    const frame = state.frame;
    if (!frame) return;

    const [iMin, iMax] = drawRange();
    const span = iMax - iMin || 1;

    // One pixel is one square on the bitmap, whatever the grid's shape, so no pixel is
    // stretched into a different shape from the one it was measured as. The element is
    // then scaled as a whole, which keeps the squares square on screen.
    if (canvas.width !== frame.cols * CELL || canvas.height !== frame.rows * CELL) {
      canvas.width = frame.cols * CELL;
      canvas.height = frame.rows * CELL;
    }

    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height);

    for (let row = 0; row < frame.rows; row += 1) {
      for (let col = 0; col < frame.cols; col += 1) {
        const value = frame.band[row][col];
        // Outside the range is grey rather than a colour at the end of the map, so an
        // out-of-range pixel never reads as the brightest one in the picture.
        context.fillStyle = value < iMin || value > iMax ? "#26262E" : magma((value - iMin) / span);
        context.fillRect(col * CELL, row * CELL, CELL, CELL);
      }
    }

    mapNote.textContent =
      `${frame.cols} × ${frame.rows} pixels, ${frame.bands} window positions, ` +
      `window ${frame.window_start.toFixed(0)} – ${frame.window_end.toFixed(0)} cm⁻¹.`;
    document.getElementById("band-note").textContent =
      `position ${frame.index + 1} of ${frame.bands}`;

    const labels = colourBar.querySelectorAll(".note");
    labels[0].textContent = iMax.toPrecision(3);
    labels[1].textContent = iMin.toPrecision(3);
    colourBar.querySelector(".colourbar-track").style.background = magmaGradient("to top");

    fitMap();
  };

  const drawSpectra = () => {
    const frame = state.frame;
    if (!frame) return;

    const correction = frame.limits.correction;
    const logged = frame.spectrum.map((value) => Math.log(Math.max(value + correction + 1e-10, 1e-10)));

    spectrumBox.replaceChildren(
      spectrumPanel(frame.axis, frame.spectrum, frame.window_start, frame.window_end, `pixel ${frame.pixel}`)
    );
    lnBox.replaceChildren(
      spectrumPanel(frame.axis, logged, frame.window_start, frame.window_end, "ln")
    );
  };

  /** Redraws the colour bar and the readouts, which the frame does not change. */
  const drawReadouts = () => {
    const frame = state.frame;
    if (!frame) return;

    const low = Number(intensityLow.value);
    const high = Number(intensityHigh.value);
    const format = (value) => (state.logScale ? value.toFixed(2) : value.toPrecision(3));

    document.getElementById("hm-imin-value").textContent = format(low);
    document.getElementById("hm-imax-value").textContent = format(high);

    shiftValue.textContent = `${frame.window_start.toFixed(0)} – ${frame.window_end.toFixed(0)} cm⁻¹`;
    widthValue.textContent = `${frame.width_cm.toFixed(0)} cm⁻¹`;
    document.getElementById("hm-window-eq").textContent =
      `${frame.width_cm.toFixed(0)} cm⁻¹ = ${frame.window_bands} bands`;
    pixelValue.textContent = `of ${frame.rows * frame.cols}`;
  };

  /** Points the intensity sliders at a range, and draws.
   *
   * Called whenever the frame arrives, because a new width means a new range to work
   * in: the areas under a wider window are larger than under a narrow one, so the old
   * ends would leave the whole picture outside the range.
   */
  const setIntensityBounds = () => {
    const limits = state.frame.limits;
    const low = state.logScale ? limits.log_min : limits.i_min;
    const high = state.logScale ? limits.log_max : limits.i_max;

    for (const slider of [intensityLow, intensityHigh]) {
      slider.min = low;
      slider.max = high;
      slider.step = (high - low) / 500 || 0.001;
    }
    intensityLow.value = low;
    intensityHigh.value = high;
    state.intensity = [low, high];
    drawReadouts();
  };

  // -- talking to the server --------------------------------------------------

  let pending = null;

  /** Asks for the frame the current state describes, and draws it when it arrives.
   *
   * One request at a time, and only the newest one matters: dragging a slider sends
   * more positions than a person can look at, so a request that is overtaken is
   * dropped rather than queued.
   */
  const refresh = async () => {
    if (pending) pending.abort();
    pending = new AbortController();

    const query = `center=${state.center}&width_cm=${state.widthCm}&pixel=${state.pixel}`;
    let frame;
    try {
      frame = await api(`/api/frame?${query}`, { signal: pending.signal });
    } catch (error) {
      if (error.name === "AbortError") return;
      // The window does not fit the range, or the map has gone. Either way the view
      // stays as it was and the message says why, rather than the screen emptying.
      deriveNote.textContent = error.message;
      deriveNote.classList.add("bad");
      return;
    }

    state.frame = frame;
    state.center = frame.index;

    shift.max = Math.max(frame.bands - 1, 0);
    shift.value = frame.index;
    width.min = frame.window_min;
    width.max = frame.window_max;
    width.step = frame.mean_step;
    width.value = frame.width_cm;
    pixel.value = frame.pixel;

    setIntensityBounds();
    drawBand();
    drawSpectra();
    drawReadouts();
    namePixel(frame.pixel);
  };

  /** Re-reads the map with a pipeline and a range, and shows the job while it runs. */
  const derive = async () => {
    deriveNote.classList.remove("bad");
    let started;
    try {
      started = await post("/api/derive", {
        pipeline_id: Number(pipeline.value),
        start: rangeStart.value === "" ? null : Number(rangeStart.value),
        end: rangeEnd.value === "" ? null : Number(rangeEnd.value),
      });
    } catch (error) {
      deriveNote.textContent = error.message;
      deriveNote.classList.add("bad");
      return;
    }

    deriveNote.textContent = "reading the map again…";
    for (const control of [pipeline, rangeStart, rangeEnd]) control.disabled = true;

    watchJob(started.job, {
      onStep: (job) => {
        deriveNote.textContent = `${describeStep(job.step)}${job.detail ? `: ${job.detail}` : ""}`;
      },
      onDone: async () => {
        for (const control of [pipeline, rangeStart, rangeEnd]) control.disabled = false;
        deriveNote.textContent = "";
        state.center = 0;
        await drawTopbar();
        await refresh();
      },
      onCancelled: () => {
        for (const control of [pipeline, rangeStart, rangeEnd]) control.disabled = false;
        deriveNote.textContent = "cancelled; the cube is as it was";
      },
      onError: (message) => {
        for (const control of [pipeline, rangeStart, rangeEnd]) control.disabled = false;
        deriveNote.textContent = message;
        deriveNote.classList.add("bad");
      },
    });
  };

  // -- controls ---------------------------------------------------------------

  window.addEventListener("resize", fitMap);

  shift.addEventListener("input", () => {
    state.center = Number(shift.value);
    refresh();
  });

  width.addEventListener("input", () => {
    state.widthCm = Number(width.value);
    refresh();
  });

  const onIntensity = () => {
    // The two ends cannot cross. Which one moved is not known here, so the pair is
    // put back in order rather than guessing.
    if (Number(intensityLow.value) > Number(intensityHigh.value)) {
      if (document.activeElement === intensityLow) intensityHigh.value = intensityLow.value;
      else intensityLow.value = intensityHigh.value;
    }
    state.intensity = [Number(intensityLow.value), Number(intensityHigh.value)];
    drawReadouts();
    drawBand();
  };
  intensityLow.addEventListener("input", onIntensity);
  intensityHigh.addEventListener("input", onIntensity);

  scaleButton.addEventListener("click", () => {
    const [low, high] = drawRange();
    state.logScale = !state.logScale;
    const limits = state.frame.limits;

    if (state.logScale) {
      intensityLow.min = intensityHigh.min = limits.log_min;
      intensityLow.max = intensityHigh.max = limits.log_max;
      intensityLow.value = Math.log(Math.max(low + limits.correction + 1e-10, 1e-10));
      intensityHigh.value = Math.log(Math.max(high + limits.correction + 1e-10, 1e-10));
      scaleButton.textContent = "Switch to standard scale";
    } else {
      intensityLow.min = intensityHigh.min = limits.i_min;
      intensityLow.max = intensityHigh.max = limits.i_max;
      intensityLow.value = Math.exp(low) - limits.correction;
      intensityHigh.value = Math.exp(high) - limits.correction;
      scaleButton.textContent = "Switch to ln scale";
    }

    const span = Number(intensityHigh.max) - Number(intensityLow.min);
    for (const slider of [intensityLow, intensityHigh]) slider.step = span / 500 || 0.001;

    drawReadouts();
    drawBand();
  });

  const setPixel = (value) => {
    const frame = state.frame;
    if (!frame) return;
    state.pixel = Math.min(Math.max(Math.round(value), 1), frame.rows * frame.cols);
    namePixel(state.pixel);
    refresh();
  };

  pixel.addEventListener("change", () => setPixel(Number(pixel.value)));

  pipeline.addEventListener("change", derive);
  for (const control of [rangeStart, rangeEnd]) {
    control.addEventListener("change", derive);
  }

  exportButton.addEventListener("click", async () => {
    const [iMin, iMax] = drawRange();

    exportButton.disabled = true;
    try {
      const saved = await post("/api/save", {
        kind: "heatmap",
        center: state.center,
        width_cm: state.widthCm,
        pixel: state.pixel,
        // In plain units whatever the scale showing, so the file is drawn in the range
        // the picture on screen is drawn in rather than the whole range of the data.
        i_min: iMin,
        i_max: iMax,
      });
      deriveNote.textContent = `written to ${saved.path}`;
      deriveNote.classList.remove("bad");
    } catch (error) {
      deriveNote.textContent = error.message;
      deriveNote.classList.add("bad");
    }
    exportButton.disabled = false;
  });

  // -- the pointer ------------------------------------------------------------
  //
  // A hover names the pixel and nothing else moves. A double-click pins it. A single
  // click is deliberately not handled: panning and zooming send clicks too, and
  // following those would move the spectrum off the pixel being looked at.

  /** Which cell of the grid a point in the element is over.
   *
   * Rounded and then clamped, which is what the matplotlib viewer does with a click, so
   * the same point on the same map is the same pixel in both.
   */
  const pointerCell = (event) => {
    const frame = state.frame;
    const box = canvas.getBoundingClientRect();
    const clamp = (value, limit) => Math.min(Math.max(Math.round(value), 0), limit - 1);
    return {
      row: clamp(((event.clientY - box.top) / box.height) * frame.rows - 0.5, frame.rows),
      col: clamp(((event.clientX - box.left) / box.width) * frame.cols - 0.5, frame.cols),
    };
  };

  const namePixel = (pixel) => {
    badge.textContent = `Pixel ${pixel}`;
  };

  canvas.addEventListener("mousemove", (event) => {
    if (!state.frame) return;
    const { row, col } = pointerCell(event);
    const box = canvas.getBoundingClientRect();

    namePixel(state.frame.pixel_map[row][col]);
    crosshair.hidden = false;
    // Placed against the wrapper the canvas is centred in rather than against the
    // element itself, since the canvas is narrower than its column whenever the map is
    // drawn shorter than the column is tall.
    crosshair.style.left = `${canvas.offsetLeft + (col + 0.5) * (box.width / state.frame.cols)}px`;
    crosshair.style.top = `${canvas.offsetTop + (row + 0.5) * (box.height / state.frame.rows)}px`;
  });

  canvas.addEventListener("mouseleave", () => {
    crosshair.hidden = true;
    namePixel(state.pixel);
  });

  canvas.addEventListener("dblclick", (event) => {
    const { row, col } = pointerCell(event);
    setPixel(state.frame.pixel_map[row][col]);
  });

  await refresh();
}

// --------------------------------------------------------------- the unmixing

/** One row of the estimator's report: what a method found, or why it found nothing.
 *
 * A rejected method is listed rather than dropped, because the reason is what tells a
 * reader whether the count is worth trusting. It is drawn in the colour the figures use
 * for what went wrong, since a row that decided nothing should not read as one that did.
 */
function reportRow(vote) {
  const props = vote.counted ? {} : { class: "rejected" };
  return el("tr", props, [
    el("td", { class: "mono", text: vote.name }),
    el("td", { class: "mono", text: vote.counted ? String(vote.value) : "none" }),
    el("td", { text: vote.rejected || "" }),
  ]);
}

/** What the confidence in a count means, in the terms of the rule that decided it.
 *
 * The rule counts agreement among the estimates that survived, so what the word means
 * is how many of them landed on the same number: every one of three or more for a high
 * confidence, some of them or fewer than three for a moderate one, and none at all for a
 * low one. Saying which of those happened is more use than repeating the word.
 */
function confidenceNote(report) {
  const counted = report.votes.filter((vote) => vote.counted);
  const values = new Set(counted.map((vote) => vote.value));

  if (report.confidence === "high") return `all ${counted.length} estimates agree`;
  if (report.confidence === "moderate") {
    return values.size === 1
      ? `${counted.length} estimates agree, which is fewer than the three that make a count certain`
      : "some of the estimates agree, but not all of them";
  }
  return "no two of the estimates agree";
}

/** The line under the table: the count, how firm it is, and what that rests on. */
function verdictLine(report) {
  if (report.n === null) {
    return el("div", { class: "verdict" }, [
      el("span", { class: "verdict-count bad", text: "Count: none" }),
      el("span", { class: "pill bad", text: report.confidence }),
      el("span", {
        class: "note",
        text:
          `${report.counted} of ${report.methods} methods gave an estimate, fewer than the ` +
          `${report.min_votes} needed to decide, so nothing is unmixed.`,
      }),
    ]);
  }

  return el("div", { class: "verdict" }, [
    el("span", { class: "verdict-count", text: `Count ${report.n}` }),
    el("span", { class: "pill", text: report.confidence }),
    el("span", { class: "note", text: confidenceNote(report) }),
  ]);
}

/** The Unmixing screen: how many phases the map holds, and where each of them sits. */
async function renderUnmix(root) {
  let session;
  try {
    session = await get("/api/session");
  } catch (error) {
    session = { loaded: false };
  }

  if (!session.loaded) {
    root.append(el("div", { class: "card" }, [
      el("p", { class: "note", text: "No map is loaded. Load one on the Data screen first." }),
      el("p", {}, [el("a", { class: "link", href: "/data", text: "Go to Data" })]),
    ]));
    return;
  }

  // What a decomposition runs with, served by the server rather than written out here so
  // that the card cannot say something the job does not do.
  let spec;
  try {
    spec = await get("/api/unmix/settings");
  } catch (error) {
    root.append(el("div", { class: "card" }, [
      el("p", { class: "note bad", text: error.message }),
    ]));
    return;
  }

  const state = { estimate: true, count: 3, jobId: null, result: null, running: false };

  // -- the top bar -----------------------------------------------------------

  const rangeStart = el("input", {
    type: "number", class: "narrow", id: "un-start",
    placeholder: session.axis_start.toFixed(0),
  });
  const rangeEnd = el("input", {
    type: "number", class: "narrow", id: "un-end",
    placeholder: session.axis_end.toFixed(0),
  });
  const settingsToggle = el("button", {
    type: "button", id: "un-settings-toggle", "aria-expanded": "true", text: "▾ NMF settings",
  });
  const status = el("span", { class: "note", id: "un-status" });
  const run = el("button", { type: "button", class: "primary", id: "un-run", text: "Unmix" });

  const toolbar = el("div", { class: "card row toolbar" }, [
    el("label", { class: "inline-field" }, [
      el("span", { class: "field-label", text: "RANGE (CM⁻¹)" }),
      el("span", { class: "row" }, [
        rangeStart, el("span", { class: "note", text: "–" }), rangeEnd,
        // Said here rather than discovered: this crops the cube the decomposition is taken
        // over, and does not touch the map the other screens are holding.
        el("span", { class: "note", text: "crops before decomposing, leaving the loaded map alone" }),
      ]),
    ]),
    settingsToggle,
    el("span", { class: "spacer" }),
    status,
    run,
  ]);

  const progress = el("div", { class: "progress", id: "un-progress", hidden: "" });

  // -- the phase count -------------------------------------------------------

  const estimateRadio = el("input", {
    type: "radio", name: "un-count-mode", id: "un-estimate", checked: "",
  });
  const fixedRadio = el("input", { type: "radio", name: "un-count-mode", id: "un-fixed" });
  const minus = el("button", { type: "button", id: "un-minus", text: "−" });
  const countInput = el("input", {
    type: "number", id: "un-count", min: spec.count_min, step: 1, value: state.count,
  });
  const plus = el("button", { type: "button", id: "un-plus", text: "+" });

  const countCard = el("div", { class: "card" }, [
    el("div", { class: "field-label", text: "PHASE COUNT" }),
    el("label", { class: "choice" }, [
      estimateRadio,
      el("span", {}, [
        el("span", { class: "choice-title", text: "Estimate from the data" }),
        el("span", {
          class: "note",
          text: "the default, and what −1 means on the command line",
        }),
      ]),
    ]),
    el("label", { class: "choice" }, [
      fixedRadio,
      el("span", { class: "row" }, [
        el("span", { class: "choice-title", text: "Fix the count" }),
        minus, countInput, plus,
      ]),
    ]),
    el("p", {
      class: "note",
      text: "A fixed count is used as given, with nothing checking that the data supports it.",
    }),
  ]);

  // -- the estimator's report ------------------------------------------------

  const reportBody = el("div", { id: "un-report" });
  const reportCard = el("div", { class: "card" }, [
    el("div", { class: "panel-head" }, [
      el("span", { class: "field-label", text: "ESTIMATOR REPORT" }),
    ]),
    reportBody,
  ]);

  // -- the settings ----------------------------------------------------------

  const settingsCard = el("div", { class: "card" }, [
    el("div", { class: "field-label", text: "NMF SETTINGS" }),
    el("div", { class: "chips" }, Object.entries(spec.settings).map(([name, value]) =>
      el("span", { class: "chip", text: `${name} ${value === null ? "none" : value}` })
    )),
    el("p", {
      class: "note",
      text:
        `The unmixing pipeline is fixed: ${spec.pipeline.join(", ")}. It is not the ` +
        "heatmap's pipeline.",
    }),
    el("p", {
      class: "note",
      text:
        "A decomposition this large can take a while, which is why it runs on a button " +
        "rather than on load.",
    }),
  ]);

  // -- what the unmixing produced --------------------------------------------

  const spectraBox = el("div", { class: "panel-plot", id: "un-spectra" });
  const spectraNote = el("span", { class: "note" });
  const mapGrid = el("div", { class: "map-grid", id: "un-maps" });

  const spectraCard = el("div", { class: "card" }, [
    el("div", { class: "panel-head" }, [
      el("span", { class: "field-label", text: "EXTRACTED SPECTRA" }),
      spectraNote,
    ]),
    spectraBox,
  ]);

  const mapsCard = el("div", { class: "card" }, [
    el("div", { class: "panel-head" }, [
      el("span", { class: "field-label", text: "ABUNDANCE MAPS" }),
    ]),
    mapGrid,
    el("p", {
      class: "note",
      text:
        "Up to ten components are laid out in a square grid; beyond that, one panel per " +
        "row. Each map is scaled to its own component, so two panels are not comparable " +
        "by colour alone.",
    }),
  ]);

  root.append(toolbar, progress, el("div", { class: "split" }, [
    el("div", { class: "split-left" }, [countCard, reportCard, settingsCard]),
    el("div", { class: "split-right" }, [spectraCard, mapsCard]),
  ]));

  // -- drawing ---------------------------------------------------------------

  /** A panel standing in for a result that is not there yet, or will not be. */
  const placeholder = (message) => el("div", { class: "panel-empty note", text: message });

  /** The estimator's report, or what to say in its place.
   *
   * A fixed count has no report to show and never will: the estimator is not run at all,
   * which is a different thing from an estimate that came back empty.
   */
  const drawReport = () => {
    reportBody.replaceChildren();

    if (state.result && state.result.estimate) {
      const report = state.result.estimate;
      reportBody.append(
        el("table", { class: "report" }, [
          el("thead", {}, [el("tr", {}, [
            el("th", { text: "METHOD" }),
            el("th", { text: "VOTE" }),
            el("th", { text: "NOTE" }),
          ])]),
          el("tbody", {}, report.votes.map(reportRow)),
        ]),
        verdictLine(report),
      );
      return;
    }

    if (!state.estimate) {
      reportBody.append(placeholder(
        "The count is fixed, so nothing is estimated, and what the methods would have " +
        "found is not looked at."
      ));
      return;
    }

    reportBody.append(placeholder(
      state.running
        ? "Running the estimator…"
        : "Nothing has been estimated yet. Unmix estimates the count first, and what every " +
          "method reported appears here."
    ));
  };

  /** The two panels, or what to say when there is nothing to draw. */
  const drawResult = () => {
    spectraBox.replaceChildren();
    mapGrid.replaceChildren();
    spectraNote.textContent = "";

    const result = state.result;

    if (!result) {
      spectraBox.append(placeholder(state.running ? "Unmixing…" : "Nothing has been unmixed yet."));
      mapGrid.append(placeholder(state.running ? "Unmixing…" : "One panel per component."));
      return;
    }

    if (!result.unmixed) {
      const message =
        "The estimator named no count, so nothing was unmixed. Its report above is the " +
        "whole result.";
      spectraBox.append(placeholder(message));
      mapGrid.append(placeholder(message));
      return;
    }

    spectraNote.textContent =
      `${result.count} components over ${result.axis_start.toFixed(0)} – ` +
      `${result.axis_end.toFixed(0)} cm⁻¹, ${result.bands} bands`;

    spectraBox.append(el("img", {
      class: "panel-image",
      src: figureUrl(result.spectra),
      alt: "The end-member spectra of each component, drawn stacked",
    }));

    // The viewer's rule, kept: a square grid up to ten components, and one per row past
    // that, where a square grid stops being readable.
    const count = result.maps.length;
    mapGrid.style.gridTemplateColumns =
      `repeat(${count <= 10 ? Math.ceil(Math.sqrt(count)) : 1}, minmax(0, 1fr))`;

    for (const entry of result.maps) {
      mapGrid.append(el("figure", {}, [
        el("img", {
          class: "panel-image",
          src: figureUrl(entry.figure),
          alt: `The abundance of component ${entry.component} across the map`,
        }),
        el("figcaption", { class: "note mono", text: `Spectra ${entry.component} map` }),
      ]));
    }
  };

  /** Where one of the running job's own figures is served from. */
  const figureUrl = (name) => `/api/job/${state.jobId}/figure/${encodeURIComponent(name)}`;

  // -- the controls ----------------------------------------------------------

  const setCountMode = (estimating) => {
    state.estimate = estimating;
    estimateRadio.checked = estimating;
    fixedRadio.checked = !estimating;
    for (const control of [minus, countInput, plus]) control.disabled = estimating;
    drawReport();
  };

  const stepCount = (by) => {
    state.count = Math.max(spec.count_min, state.count + by);
    countInput.value = state.count;
  };

  estimateRadio.addEventListener("change", () => setCountMode(true));
  fixedRadio.addEventListener("change", () => setCountMode(false));
  minus.addEventListener("click", () => stepCount(-1));
  plus.addEventListener("click", () => stepCount(1));
  countInput.addEventListener("change", () => {
    state.count = Math.max(spec.count_min, Number(countInput.value) || spec.count_min);
    countInput.value = state.count;
  });

  settingsToggle.addEventListener("click", () => {
    const open = settingsCard.hidden;
    settingsCard.hidden = !open;
    settingsToggle.textContent = `${open ? "▾" : "▸"} NMF settings`;
    settingsToggle.setAttribute("aria-expanded", String(open));
  });

  // -- the run ---------------------------------------------------------------

  const setRunning = (running) => {
    state.running = running;
    run.disabled = running;
    settingsToggle.disabled = running;
    for (const control of [rangeStart, rangeEnd, estimateRadio, fixedRadio]) control.disabled = running;
    for (const control of [minus, countInput, plus]) control.disabled = running || state.estimate;
  };

  run.addEventListener("click", async () => {
    // The panels of the run before are cleared rather than left under this one's progress:
    // one screen shows one decomposition, and a stale panel beside a running job reads as
    // this job's own result.
    state.result = null;
    state.jobId = null;
    status.classList.remove("bad");
    status.textContent = "starting…";
    drawReport();
    drawResult();
    setRunning(true);

    let started;
    try {
      started = await post("/api/unmix", {
        estimate: state.estimate,
        count: state.estimate ? null : state.count,
        start: rangeStart.value === "" ? null : Number(rangeStart.value),
        end: rangeEnd.value === "" ? null : Number(rangeEnd.value),
      });
    } catch (error) {
      status.textContent = error.message;
      status.classList.add("bad");
      setRunning(false);
      return;
    }

    state.jobId = started.job.id;

    const label = el("span", { class: "note", text: "Running the estimator…" });
    const cancel = el("button", { type: "button", text: "Cancel" });
    progress.replaceChildren(
      el("div", { class: "row" }, [label, cancel]),
      el("div", { class: "bar" }, [el("div", { class: "bar-fill indeterminate" })]),
    );
    progress.hidden = false;

    cancel.addEventListener("click", async () => {
      cancel.disabled = true;
      try {
        await post(`/api/job/${state.jobId}/cancel`);
      } catch (error) {
        label.textContent = error.message;
      }
    });

    const finish = () => {
      progress.hidden = true;
      progress.replaceChildren();
      setRunning(false);
    };

    watchJob(started.job, {
      onStep: (job) => {
        // A child reports nothing until it has read its request and got to its first
        // step, so an empty step leaves the line as it is rather than replacing a
        // sentence that names the work with a word that names none of it.
        if (!job.step) return;
        const step = `${describeStep(job.step)}${job.detail ? `: ${job.detail}` : ""}`;
        label.textContent = step;
        status.textContent = step;
      },
      onDone: (job) => {
        state.result = job.result;
        finish();
        status.textContent = "";
        drawReport();
        drawResult();
      },
      onCancelled: () => {
        finish();
        status.textContent = "The unmixing was cancelled. Nothing was produced.";
        drawReport();
        drawResult();
      },
      onError: (message) => {
        finish();
        status.textContent = message;
        status.classList.add("bad");
        drawReport();
        drawResult();
      },
    });
  });

  setCountMode(true);
  drawReport();
  drawResult();
}

drawTopbar();
drawScreen(currentScreen());

#!/usr/bin/env node

import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const outputDir = resolve(repoRoot, "docs", "assets");
const maximumCiEndpointDifference = 5.82e-4;

const color = {
  ink: "#17212B",
  muted: "#596673",
  faint: "#85919C",
  rule: "#DCE3E8",
  grid: "#E9EEF1",
  paper: "#FFFFFF",
  panel: "#F6F9FA",
  rust: "#0072B2",
  r: "#68717A",
  stata: "#D55E00",
  stataText: "#B34700",
  green: "#009E73",
};

const singleFits = [
  {
    label: "Sharp · local linear · NN",
    r: 20.733333333,
    stata: 46.3,
    rust: 1.051301,
  },
  {
    label: "Sharp · local quadratic · NN",
    r: 21.4,
    stata: 45.366666667,
    rust: 1.246694,
  },
  {
    label: "Fuzzy · local linear · NN",
    r: 24.4,
    stata: 47.966666667,
    rust: 1.193516,
  },
  {
    label: "Covariates + weights · HC1",
    r: 3.666666667,
    stata: 32.266666667,
    rust: 0.622738,
  },
  {
    label: "Cluster-robust · CR1",
    r: 24.866666667,
    stata: 34.166666667,
    rust: 1.099593,
  },
];

const batches = [
  {
    fits: 16,
    rustSerial: 16.871429,
    rustBatch: 2.755825,
    r: 333,
    stata: 722,
  },
  {
    fits: 64,
    rustSerial: 67.559633,
    rustBatch: 9.23865,
    r: 1306,
    stata: 2930,
  },
  {
    fits: 256,
    rustSerial: 271.534167,
    rustBatch: 35.241667,
    r: 5313,
    stata: 11775,
  },
];

const parity = [
  {
    label: "Sharp · local linear · NN",
    stataEstimate: 1.98495955700304,
    rustEstimate: 1.9849595498008785,
    stataSe: 0.0927801547061703,
    rustSe: 0.09297169140662266,
  },
  {
    label: "Sharp · local quadratic · NN",
    stataEstimate: 1.9757156108214,
    rustEstimate: 1.9757152602202965,
    stataSe: 0.121844106381872,
    rustSe: 0.12196520703820368,
  },
  {
    label: "Fuzzy · local linear · NN",
    stataEstimate: 1.47463564821699,
    rustEstimate: 1.4746418601240507,
    stataSe: 0.15543688674157,
    rustSe: 0.1557307560649487,
  },
  {
    label: "Covariates + weights · HC1",
    stataEstimate: 1.88836965938914,
    rustEstimate: 1.8883694135701505,
    stataSe: 0.241996149773146,
    rustSe: 0.24199599864428978,
  },
  {
    label: "Cluster-robust · CR1",
    stataEstimate: 1.98437589981556,
    rustEstimate: 1.984375899965675,
    stataSe: 0.0896476090869795,
    rustSe: 0.08971761877070794,
  },
].map((row) => ({
  ...row,
  estimateDifference: Math.abs(row.rustEstimate - row.stataEstimate),
  relativeSeDifference:
    (100 * Math.abs(row.rustSe - row.stataSe)) / row.stataSe,
}));

for (const row of singleFits) {
  assert(row.rust > 0 && row.r > 0 && row.stata > 0);
}
for (const row of batches) {
  assert(row.rustBatch < row.rustSerial);
  assert(row.rustBatch < row.r && row.rustBatch < row.stata);
}
assert(Math.max(...parity.map((row) => row.estimateDifference)) < 6.3e-6);
assert(Math.max(...parity.map((row) => row.relativeSeDifference)) < 0.21);

function escapeXml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function text(x, y, value, className, anchor = "start", attributes = "") {
  return `<text x="${x}" y="${y}" class="${className}" text-anchor="${anchor}" ${attributes}>${escapeXml(value)}</text>`;
}

function line(x1, y1, x2, y2, className, attributes = "") {
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="${className}" ${attributes}/>`;
}

function rect(x, y, width, height, attributes = "") {
  return `<rect x="${x}" y="${y}" width="${width}" height="${height}" ${attributes}/>`;
}

function marker(x, y, kind, markerColor, size = 7) {
  if (kind === "diamond") {
    return `<path d="M ${x} ${y - size} L ${x + size} ${y} L ${x} ${y + size} L ${x - size} ${y} Z" fill="${markerColor}"/>`;
  }
  if (kind === "square") {
    return rect(
      x - size,
      y - size,
      size * 2,
      size * 2,
      `rx="2" fill="${markerColor}"`,
    );
  }
  if (kind === "ring") {
    return `<circle cx="${x}" cy="${y}" r="${size}" fill="${color.paper}" stroke="${markerColor}" stroke-width="3"/>`;
  }
  return `<circle cx="${x}" cy="${y}" r="${size}" fill="${markerColor}"/>`;
}

function legendItem(x, y, label, kind, markerColor) {
  return [
    marker(x, y - 4, kind, markerColor, 6),
    text(x + 15, y, label, "legend-label"),
  ].join("");
}

function card(x, y, width, label, value, note, accent) {
  return [
    rect(x, y, width, 96, `rx="16" fill="${color.panel}"`),
    rect(x, y, 5, 96, `rx="2.5" fill="${accent}"`),
    text(x + 24, y + 27, label, "card-label"),
    text(x + 24, y + 61, value, "card-value"),
    text(x + 24, y + 82, note, "card-note"),
  ].join("");
}

function pill(x, y, value, label) {
  return [
    rect(x, y, 100, 35, `rx="17.5" fill="#E8F2F8"`),
    text(x + 50, y + 16, value, "pill-value", "middle"),
    text(x + 50, y + 29, label, "pill-label", "middle"),
  ].join("");
}

function logPosition(value, minimum, maximum, start, end) {
  const proportion =
    (Math.log(value) - Math.log(minimum)) /
    (Math.log(maximum) - Math.log(minimum));
  return start + proportion * (end - start);
}

function linearPosition(value, minimum, maximum, start, end) {
  return start + ((value - minimum) / (maximum - minimum)) * (end - start);
}

function formatMilliseconds(value) {
  if (value >= 1000) {
    return `${(value / 1000).toFixed(2)} s`;
  }
  if (value < 10) {
    return `${value.toFixed(2)} ms`;
  }
  if (value < 100) {
    return `${value.toFixed(1)} ms`;
  }
  return `${Math.round(value)} ms`;
}

function formatAxisTime(value) {
  if (value >= 1000) {
    return `${Math.round(value / 1000)} s`;
  }
  return `${value} ms`;
}

function formatSpeedup(value) {
  return value < 10 ? `${value.toFixed(1)}×` : `${Math.round(value)}×`;
}

function formatScientific(value) {
  const [coefficient, exponent] = value.toExponential(2).split("e");
  return `${coefficient}e${Number(exponent) < 0 ? "−" : "+"}${Math.abs(Number(exponent))}`;
}

function superscriptInteger(value) {
  return String(value)
    .replace("-", "⁻")
    .replaceAll("0", "⁰")
    .replaceAll("1", "¹")
    .replaceAll("2", "²")
    .replaceAll("3", "³")
    .replaceAll("4", "⁴")
    .replaceAll("5", "⁵")
    .replaceAll("6", "⁶")
    .replaceAll("7", "⁷")
    .replaceAll("8", "⁸")
    .replaceAll("9", "⁹");
}

function formatPower(value) {
  const [coefficient, exponent] = value.toExponential(2).split("e");
  return `${coefficient} × 10${superscriptInteger(Number(exponent))}`;
}

function formatPercent(value) {
  if (value < 0.001) {
    return `${value.toFixed(6)}%`;
  }
  if (value < 0.1) {
    return `${value.toFixed(4)}%`;
  }
  return `${value.toFixed(3)}%`;
}

function svgShell({ width, height, title: titleValue, description, body }) {
  return `<!-- Generated by tools/generate_readme_figures.mjs. -->
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="figure-title figure-description">
  <title id="figure-title">${escapeXml(titleValue)}</title>
  <desc id="figure-description">${escapeXml(description)}</desc>
  <style>
    text {
      font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
      font-variant-numeric: tabular-nums;
    }
    .title { fill: ${color.ink}; font-size: 35px; font-weight: 750; letter-spacing: -0.6px; }
    .subtitle { fill: ${color.muted}; font-size: 16px; }
    .panel-title { fill: ${color.ink}; font-size: 21px; font-weight: 700; }
    .panel-subtitle { fill: ${color.muted}; font-size: 14px; }
    .row-label { fill: ${color.ink}; font-size: 14px; font-weight: 600; }
    .axis-label { fill: ${color.muted}; font-size: 12px; }
    .value-label { font-size: 12px; font-weight: 700; }
    .legend-label { fill: ${color.muted}; font-size: 13px; font-weight: 600; }
    .card-label { fill: ${color.muted}; font-size: 11px; font-weight: 750; letter-spacing: 1.2px; }
    .card-value { fill: ${color.ink}; font-size: 29px; font-weight: 760; letter-spacing: -0.4px; }
    .card-note { fill: ${color.muted}; font-size: 12px; }
    .pill-value { fill: ${color.rust}; font-size: 15px; font-weight: 800; }
    .pill-label { fill: ${color.muted}; font-size: 9px; font-weight: 700; letter-spacing: 0.4px; }
    .footnote { fill: ${color.muted}; font-size: 12px; }
    .grid { stroke: ${color.grid}; stroke-width: 1; }
    .row-guide { stroke: ${color.rule}; stroke-width: 1.5; stroke-linecap: round; }
    .axis { stroke: ${color.faint}; stroke-width: 1; }
  </style>
  ${rect(0, 0, width, height, `fill="${color.paper}"`)}
  ${body}
</svg>
`;
}

function performanceFigure() {
  const width = 1200;
  const height = 1180;
  const plotStart = 310;
  const plotEnd = 990;
  const speedupX = 1050;
  const content = [];

  content.push(
    text(
      64,
      62,
      "Rust cuts wait time across every tested RD specification",
      "title",
    ),
  );
  content.push(
    text(
      64,
      94,
      "Same deterministic 2,000-observation workload · lower is faster · axes use log scales",
      "subtitle",
    ),
  );

  const rSpeedups = singleFits.map((row) => row.r / row.rust);
  const stataSpeedups = singleFits.map((row) => row.stata / row.rust);
  const parallelGains = batches.map((row) => row.rustSerial / row.rustBatch);
  content.push(
    card(
      64,
      126,
      340,
      "VERSUS R",
      `${Math.min(...rSpeedups).toFixed(1)}–${Math.round(Math.max(...rSpeedups))}×`,
      "faster for one fit",
      color.r,
    ),
    card(
      430,
      126,
      340,
      "VERSUS STATA",
      `${Math.round(Math.min(...stataSpeedups))}–${Math.round(Math.max(...stataSpeedups))}×`,
      "faster for one fit",
      color.stata,
    ),
    card(
      796,
      126,
      340,
      "RAYON BATCH GAIN",
      `${Math.min(...parallelGains).toFixed(1)}–${Math.max(...parallelGains).toFixed(1)}×`,
      "versus one-at-a-time Rust",
      color.rust,
    ),
  );

  content.push(
    text(64, 272, "A. One fit under five specifications", "panel-title"),
    text(
      64,
      298,
      "Wall-clock wait time through each implementation",
      "panel-subtitle",
    ),
    legendItem(310, 330, "Rust", "circle", color.rust),
    legendItem(400, 330, "R", "diamond", color.r),
    legendItem(465, 330, "Stata", "square", color.stata),
    text(1100, 330, "Rust vs Stata", "axis-label", "middle"),
  );

  const singleTicks = [0.5, 1, 2, 4, 8, 16, 32, 64];
  for (const tick of singleTicks) {
    const x = logPosition(tick, 0.5, 64, plotStart, plotEnd);
    content.push(
      line(x, 350, x, 635, "grid"),
      text(x, 660, String(tick), "axis-label", "middle"),
    );
  }
  content.push(
    line(plotStart, 635, plotEnd, 635, "axis"),
    text(
      (plotStart + plotEnd) / 2,
      690,
      "Wait time (milliseconds, log scale)",
      "axis-label",
      "middle",
    ),
  );

  singleFits.forEach((row, index) => {
    const y = 382 + index * 57;
    const rLabelY = index === singleFits.length - 1 ? y + 18 : y + 26;
    const rustX = logPosition(row.rust, 0.5, 64, plotStart, plotEnd);
    const rX = logPosition(row.r, 0.5, 64, plotStart, plotEnd);
    const stataX = logPosition(row.stata, 0.5, 64, plotStart, plotEnd);
    content.push(
      text(64, y + 5, row.label, "row-label"),
      line(rustX, y, stataX, y, "row-guide"),
      marker(rustX, y, "circle", color.rust),
      marker(rX, y, "diamond", color.r),
      marker(stataX, y, "square", color.stata),
      text(
        rustX,
        y - 14,
        formatMilliseconds(row.rust),
        "value-label",
        "middle",
        `fill="${color.rust}"`,
      ),
      text(
        rX,
        rLabelY,
        formatMilliseconds(row.r),
        "value-label",
        "middle",
        `fill="${color.r}"`,
      ),
      text(
        stataX,
        y - 14,
        formatMilliseconds(row.stata),
        "value-label",
        "middle",
        `fill="${color.stataText}"`,
      ),
      pill(speedupX, y - 17, formatSpeedup(row.stata / row.rust), "FASTER"),
    );
  });

  content.push(
    line(64, 728, 1136, 728, "grid"),
    text(64, 774, "B. Repeated fits in one workload", "panel-title"),
    text(
      64,
      800,
      "Total wait time for 16, 64, or 256 repetitions of the sharp local-linear fit",
      "panel-subtitle",
    ),
    legendItem(310, 832, "Rust batch", "circle", color.rust),
    legendItem(430, 832, "Rust, one at a time", "ring", color.rust),
    legendItem(605, 832, "R loop", "diamond", color.r),
    legendItem(700, 832, "Stata loop", "square", color.stata),
    text(1100, 832, "Batch gain", "axis-label", "middle"),
  );

  const batchTicks = [2, 8, 32, 128, 512, 2048, 8192];
  for (const tick of batchTicks) {
    const x = logPosition(tick, 2, 16384, plotStart, plotEnd);
    content.push(
      line(x, 852, x, 1054, "grid"),
      text(x, 1082, formatAxisTime(tick), "axis-label", "middle"),
    );
  }
  content.push(
    line(plotStart, 1054, plotEnd, 1054, "axis"),
    text(
      (plotStart + plotEnd) / 2,
      1118,
      "Total wait time (log scale)",
      "axis-label",
      "middle",
    ),
  );

  batches.forEach((row, index) => {
    const y = 882 + index * 78;
    const rustBatchX = logPosition(row.rustBatch, 2, 16384, plotStart, plotEnd);
    const rustSerialX = logPosition(
      row.rustSerial,
      2,
      16384,
      plotStart,
      plotEnd,
    );
    const rX = logPosition(row.r, 2, 16384, plotStart, plotEnd);
    const stataX = logPosition(row.stata, 2, 16384, plotStart, plotEnd);
    content.push(
      text(64, y + 5, `${row.fits} fits`, "row-label"),
      line(rustBatchX, y, stataX, y, "row-guide"),
      marker(rustBatchX, y, "circle", color.rust),
      marker(rustSerialX, y, "ring", color.rust),
      marker(rX, y, "diamond", color.r),
      marker(stataX, y, "square", color.stata),
      text(
        rustBatchX,
        y - 15,
        formatMilliseconds(row.rustBatch),
        "value-label",
        "middle",
        `fill="${color.rust}"`,
      ),
      text(
        rustSerialX,
        y + 27,
        formatMilliseconds(row.rustSerial),
        "value-label",
        "middle",
        `fill="${color.rust}"`,
      ),
      text(
        rX,
        y - 15,
        formatMilliseconds(row.r),
        "value-label",
        "middle",
        `fill="${color.r}"`,
      ),
      text(
        stataX,
        y + 27,
        formatMilliseconds(row.stata),
        "value-label",
        "middle",
        `fill="${color.stataText}"`,
      ),
      pill(
        speedupX,
        y - 17,
        formatSpeedup(row.rustSerial / row.rustBatch),
        "PARALLEL",
      ),
    );
  });

  content.push(
    text(
      64,
      1156,
      "Measured July 28, 2026 · Apple M2 Pro · release Python 3.12 binding · R 4.5.1 / rdrobust 3.0.0 · Stata 19.5 IC / rdrobust.ado 10.0.0",
      "footnote",
    ),
  );

  return svgShell({
    width,
    height,
    title: "Rust cuts wait time across every tested RD specification",
    description:
      "Two log-scale dot plots compare Rust, R, and Stata. Rust single fits take 0.62 to 1.25 milliseconds, 5.9 to 23 times faster than R and 31 to 52 times faster than Stata. Rust parallel batches are 6.1 to 7.7 times faster than one-at-a-time Rust.",
    body: content.join("\n  "),
  });
}

function parityFigure() {
  const width = 1200;
  const height = 750;
  const content = [];
  const maximumEstimateDifference = Math.max(
    ...parity.map((row) => row.estimateDifference),
  );
  const maximumRelativeSeDifference = Math.max(
    ...parity.map((row) => row.relativeSeDifference),
  );

  content.push(
    text(
      64,
      62,
      "Stata and Rust agree closely across matched specifications",
      "title",
    ),
    text(
      64,
      94,
      "Five deterministic 2,000-observation checks · lower differences mean closer agreement",
      "subtitle",
    ),
    card(
      64,
      126,
      340,
      "LARGEST ESTIMATE GAP",
      formatPower(maximumEstimateDifference),
      "bias-corrected estimate",
      color.rust,
    ),
    card(
      430,
      126,
      340,
      "LARGEST ROBUST-SE GAP",
      `${maximumRelativeSeDifference.toFixed(3)}%`,
      "relative to the Stata SE",
      color.stata,
    ),
    card(
      796,
      126,
      340,
      "LARGEST CI-ENDPOINT GAP",
      formatPower(maximumCiEndpointDifference),
      "robust 95% interval",
      color.green,
    ),
  );

  content.push(
    text(64, 284, "A. Bias-corrected estimate", "panel-title"),
    text(
      64,
      310,
      "Absolute Stata–Rust difference · log scale",
      "panel-subtitle",
    ),
    text(650, 284, "B. Robust standard error", "panel-title"),
    text(
      650,
      310,
      "Absolute difference as a percentage of the Stata SE",
      "panel-subtitle",
    ),
    line(610, 276, 610, 666, "grid"),
  );

  const estimateRows = [...parity].sort(
    (left, right) => right.estimateDifference - left.estimateDifference,
  );
  const estimateTicks = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5];
  const estimateStart = 286;
  const estimateEnd = 565;
  for (const tick of estimateTicks) {
    const x = logPosition(tick, 1e-10, 1e-5, estimateStart, estimateEnd);
    const exponent = Math.round(Math.log10(tick));
    content.push(
      line(x, 338, x, 620, "grid"),
      text(x, 648, `10${superscriptInteger(exponent)}`, "axis-label", "middle"),
    );
  }
  content.push(line(estimateStart, 620, estimateEnd, 620, "axis"));

  estimateRows.forEach((row, index) => {
    const y = 364 + index * 52;
    const x = logPosition(
      row.estimateDifference,
      1e-10,
      1e-5,
      estimateStart,
      estimateEnd,
    );
    const isMaximum = row.estimateDifference === maximumEstimateDifference;
    const markerColor = isMaximum ? color.stata : color.rust;
    const labelColor = isMaximum ? color.stataText : color.rust;
    const anchor = x > 500 ? "end" : "start";
    const labelX = x > 500 ? x - 11 : x + 11;
    content.push(
      text(64, y + 5, row.label, "row-label"),
      line(estimateStart, y, estimateEnd, y, "grid"),
      marker(x, y, "circle", markerColor, isMaximum ? 8 : 6),
      text(
        labelX,
        y - 10,
        formatScientific(row.estimateDifference),
        "value-label",
        anchor,
        `fill="${labelColor}"`,
      ),
    );
  });

  const seRows = [...parity].sort(
    (left, right) => right.relativeSeDifference - left.relativeSeDifference,
  );
  const seStart = 882;
  const seEnd = 1120;
  const seMaximum = 0.25;
  const seTicks = [0, 0.05, 0.1, 0.15, 0.2, 0.25];
  for (const tick of seTicks) {
    const x = linearPosition(tick, 0, seMaximum, seStart, seEnd);
    content.push(
      line(x, 338, x, 620, "grid"),
      text(x, 648, `${tick.toFixed(2)}%`, "axis-label", "middle"),
    );
  }
  content.push(line(seStart, 620, seEnd, 620, "axis"));

  seRows.forEach((row, index) => {
    const y = 364 + index * 52;
    const x = linearPosition(
      row.relativeSeDifference,
      0,
      seMaximum,
      seStart,
      seEnd,
    );
    const isMaximum = row.relativeSeDifference === maximumRelativeSeDifference;
    const markerColor = isMaximum ? color.stata : color.rust;
    const labelColor = isMaximum ? color.stataText : color.rust;
    const barEnd = Math.max(x, seStart + 2);
    const labelX = Math.min(barEnd + 10, 1142);
    const anchor = labelX >= 1142 ? "end" : "start";
    content.push(
      text(650, y + 5, row.label, "row-label"),
      rect(
        seStart,
        y - 5,
        Math.max(barEnd - seStart, 2),
        10,
        `rx="5" fill="${markerColor}" opacity="${isMaximum ? 1 : 0.8}"`,
      ),
      marker(barEnd, y, "circle", markerColor, isMaximum ? 7 : 5),
      text(
        labelX,
        y - 10,
        formatPercent(row.relativeSeDifference),
        "value-label",
        anchor,
        `fill="${labelColor}"`,
      ),
    );
  });

  content.push(
    text(
      64,
      708,
      "Stata 19.5 IC / rdrobust.ado 10.0.0 versus the release Rust binding · matched cutoff, kernel, polynomial, variance, covariate, weight, cluster, and bandwidth options",
      "footnote",
    ),
  );

  return svgShell({
    width,
    height,
    title: "Stata and Rust agree closely across matched specifications",
    description:
      "Two panels show numerical differences across five matched specifications. The largest bias-corrected estimate difference is 6.21 times 10 to the minus 6. The largest robust standard-error difference is 0.206 percent of the Stata value.",
    body: content.join("\n  "),
  });
}

mkdirSync(outputDir, { recursive: true });

const outputs = [
  ["readme-performance.svg", performanceFigure()],
  ["readme-parity.svg", parityFigure()],
];

for (const [name, svg] of outputs) {
  const path = resolve(outputDir, name);
  writeFileSync(path, svg, "utf8");
  process.stdout.write(`wrote ${path}\n`);
}

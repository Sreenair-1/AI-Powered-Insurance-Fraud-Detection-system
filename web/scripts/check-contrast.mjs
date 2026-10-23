// Verifies WCAG AA contrast for the text/background token pairs the UI uses,
// in both themes, straight from src/styles/tokens.css. Exit code 1 on failure.
import { readFileSync } from "node:fs";

const css = readFileSync(new URL("../src/styles/tokens.css", import.meta.url), "utf8");

function block(selector) {
  const start = css.indexOf(`${selector} {`);
  const end = css.indexOf("\n}", start);
  const vars = {};
  for (const m of css.slice(start, end).matchAll(/--([\w-]+):\s*([^;]+);/g)) vars[m[1]] = m[2].trim();
  return vars;
}

const light = block(":root");
const dark = { ...light, ...block(".dark") };

function resolve(vars, name) {
  let v = vars[name];
  for (let i = 0; i < 10 && v?.startsWith("var("); i++) v = vars[v.slice(6, -1)];
  return v.split(/\s+/).map(Number);
}

function luminance([r, g, b]) {
  const c = [r, g, b].map((x) => {
    const s = x / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
}

function ratio(a, b) {
  const [l1, l2] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (l1 + 0.05) / (l2 + 0.05);
}

// [foreground, background, minimum]
const pairs = [
  ["text", "surface", 4.5], ["text", "background", 4.5], ["text-muted", "surface", 4.5],
  ["text-muted", "background", 4.5], ["text-subtle", "surface", 4.5], ["text-muted", "surface-sunken", 4.5],
  ["on-primary", "primary", 4.5], ["on-accent", "accent", 4.5], ["primary", "surface", 4.5],
  ["accent", "surface", 4.5], ["primary", "primary-soft", 4.5],
  ["risk-high", "risk-high-soft", 4.5], ["risk-medium", "risk-medium-soft", 4.5], ["risk-low", "risk-low-soft", 4.5],
  ["risk-high", "surface", 4.5], ["risk-medium", "surface", 4.5], ["risk-low", "surface", 4.5],
  ["risk-high-solid", "surface", 3], ["risk-medium-solid", "surface", 3], ["risk-low-solid", "surface", 3],
  ["node-claim", "surface", 3], ["node-provider", "surface", 3], ["node-physician", "surface", 3],
  ["node-beneficiary", "surface", 3], ["node-garage", "surface", 3], ["node-bank", "surface", 3],
  ["ring", "surface", 3], ["border-strong", "surface", 1.3],
];

let failed = 0;
for (const [theme, vars] of [["light", light], ["dark", dark]]) {
  for (const [fg, bg, min] of pairs) {
    const r = ratio(resolve(vars, fg), resolve(vars, bg));
    const ok = r >= min;
    if (!ok) failed++;
    console.log(`${ok ? "ok  " : "FAIL"} ${theme.padEnd(5)} ${fg} on ${bg}: ${r.toFixed(2)} (min ${min})`);
  }
}
process.exit(failed ? 1 : 0);

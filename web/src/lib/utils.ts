import { clsx, type ClassValue } from "clsx";
import { useEffect, useState } from "react";
import { twMerge } from "tailwind-merge";
import { useSettings } from "./settings-store";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const TOKEN_NAMES = [
  "primary",
  "accent",
  "text",
  "text-muted",
  "text-subtle",
  "surface",
  "border",
  "chart-1",
  "chart-2",
  "chart-grid",
  "chart-axis",
  "risk-high-solid",
  "risk-medium-solid",
  "risk-low-solid",
  "node-claim",
  "node-provider",
  "node-physician",
  "node-beneficiary",
  "node-garage",
  "node-bank",
] as const;

export type TokenName = (typeof TOKEN_NAMES)[number];
export type TokenColors = Record<TokenName, string>;

function readTokens(): TokenColors {
  const style = getComputedStyle(document.documentElement);
  const out = {} as TokenColors;
  for (const name of TOKEN_NAMES) {
    out[name] = `rgb(${style.getPropertyValue(`--${name}`).trim().split(/\s+/).join(", ")})`;
  }
  return out;
}

/**
 * Resolved token colours for canvas/SVG libraries (Recharts, force-graph)
 * that cannot read CSS variables themselves. Re-reads on theme change.
 */
export function useTokenColors(): TokenColors {
  const { theme } = useSettings();
  const [colors, setColors] = useState<TokenColors>(readTokens);
  useEffect(() => {
    setColors(readTokens());
  }, [theme]);
  return colors;
}

export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(
    () => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onChange = () => setReduced(mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return reduced;
}

export function useDebounced<T>(value: T, ms = 250): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setV(value), ms);
    return () => clearTimeout(id);
  }, [value, ms]);
  return v;
}

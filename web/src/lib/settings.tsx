import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { DEFAULTS, SAMPLE_LIMITS, SettingsContext, type SampleParams, type SettingsValue, type Theme } from "./settings-store";

function read<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? { ...fallback, ...JSON.parse(raw) } : fallback;
  } catch {
    return fallback;
  }
}

function clampSample(p: SampleParams): SampleParams {
  const out = { ...p };
  for (const key of Object.keys(SAMPLE_LIMITS) as (keyof SampleParams)[]) {
    const { min, max } = SAMPLE_LIMITS[key];
    const v = Number(out[key]);
    out[key] = Number.isFinite(v) ? Math.min(max, Math.max(min, Math.round(v))) : DEFAULTS[key];
  }
  return out;
}

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [sample, setSampleState] = useState<SampleParams>(() => clampSample(read("insuresight.sample", DEFAULTS)));
  const [theme, setTheme] = useState<Theme>(() => {
    try {
      const stored = localStorage.getItem("insuresight.theme");
      if (stored === "dark" || stored === "light") return stored;
    } catch {
      /* storage unavailable */
    }
    return "light";
  });

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    try {
      localStorage.setItem("insuresight.theme", theme);
    } catch {
      /* storage unavailable */
    }
  }, [theme]);

  const setSample = useCallback((next: SampleParams) => {
    const clean = clampSample(next);
    setSampleState(clean);
    try {
      localStorage.setItem("insuresight.sample", JSON.stringify(clean));
    } catch {
      /* storage unavailable */
    }
  }, []);

  const value = useMemo<SettingsValue>(
    () => ({
      sample,
      setSample,
      resetSample: () => setSample(DEFAULTS),
      theme,
      toggleTheme: () => setTheme((t) => (t === "light" ? "dark" : "light")),
    }),
    [sample, setSample, theme],
  );
  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>;
}

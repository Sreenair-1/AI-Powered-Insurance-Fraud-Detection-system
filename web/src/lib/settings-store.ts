import { createContext, useContext } from "react";

// Sample-size bounds and defaults are the Streamlit app's (app/config.py);
// the API enforces the same limits and rejects anything outside them.
export const SAMPLE_LIMITS = {
  motorSampleSize: { min: 100, max: 3000, step: 100, default: 500, label: "Motor claims" },
  inpatientSampleSize: { min: 20, max: 1200, step: 20, default: 150, label: "Health inpatient claims" },
  outpatientSampleSize: { min: 40, max: 2800, step: 40, default: 350, label: "Health outpatient claims" },
} as const;

export interface SampleParams {
  motorSampleSize: number;
  inpatientSampleSize: number;
  outpatientSampleSize: number;
}

export const DEFAULTS: SampleParams = {
  motorSampleSize: SAMPLE_LIMITS.motorSampleSize.default,
  inpatientSampleSize: SAMPLE_LIMITS.inpatientSampleSize.default,
  outpatientSampleSize: SAMPLE_LIMITS.outpatientSampleSize.default,
};

export type Theme = "light" | "dark";

export interface SettingsValue {
  sample: SampleParams;
  setSample: (next: SampleParams) => void;
  resetSample: () => void;
  theme: Theme;
  toggleTheme: () => void;
}

export const SettingsContext = createContext<SettingsValue | null>(null);

export function useSettings(): SettingsValue {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error("useSettings must be used inside SettingsProvider");
  return ctx;
}

export function useSampleParams(): SampleParams {
  return useSettings().sample;
}

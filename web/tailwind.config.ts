import type { Config } from "tailwindcss";

// Every colour/radius/shadow here is a reference to src/styles/tokens.css,
// so components never contain raw hex values.
const token = (name: string) => `rgb(var(--${name}) / <alpha-value>)`;

export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: token("background"),
        surface: { DEFAULT: token("surface"), raised: token("surface-raised"), sunken: token("surface-sunken") },
        border: { DEFAULT: token("border"), strong: token("border-strong") },
        text: { DEFAULT: token("text"), muted: token("text-muted"), subtle: token("text-subtle") },
        primary: { DEFAULT: token("primary"), hover: token("primary-hover"), soft: token("primary-soft"), on: token("on-primary") },
        accent: { DEFAULT: token("accent"), hover: token("accent-hover"), soft: token("accent-soft"), on: token("on-accent") },
        ring: token("ring"),
        risk: {
          high: token("risk-high"),
          "high-soft": token("risk-high-soft"),
          "high-border": token("risk-high-border"),
          "high-solid": token("risk-high-solid"),
          medium: token("risk-medium"),
          "medium-soft": token("risk-medium-soft"),
          "medium-border": token("risk-medium-border"),
          "medium-solid": token("risk-medium-solid"),
          low: token("risk-low"),
          "low-soft": token("risk-low-soft"),
          "low-border": token("risk-low-border"),
          "low-solid": token("risk-low-solid"),
        },
      },
      fontFamily: { sans: ["var(--font-sans)"], mono: ["var(--font-mono)"] },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem" }],
      },
      borderRadius: {
        sm: "var(--radius-sm)",
        md: "var(--radius-md)",
        lg: "var(--radius-lg)",
        xl: "var(--radius-xl)",
      },
      boxShadow: {
        xs: "var(--shadow-xs)",
        sm: "var(--shadow-sm)",
        md: "var(--shadow-md)",
        lg: "var(--shadow-lg)",
      },
      transitionDuration: { fast: "var(--duration-fast)", base: "var(--duration-base)", slow: "var(--duration-slow)" },
      transitionTimingFunction: { out: "var(--ease-out)" },
      keyframes: {
        shimmer: { "100%": { transform: "translateX(100%)" } },
      },
      animation: { shimmer: "shimmer 1.4s infinite" },
    },
  },
  plugins: [],
} satisfies Config;

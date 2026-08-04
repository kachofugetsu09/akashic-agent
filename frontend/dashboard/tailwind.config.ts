import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import type { Config } from "tailwindcss";

// Content globs are resolved against this config's own location so scanning
// works regardless of the process cwd (npm runs from the repo root).
const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");

// Industrial / precision-instrument design tokens. Solid colors use RGB-triplet
// vars (see src/styles.css) so opacity modifiers like bg-danger/20 work.
// Plugin panels use the public preset or their own CSS. They are not part of
// the host bundle's Tailwind content contract.
export default {
  content: [
    resolve(here, "index.html"),
    resolve(here, "src/**/*.{ts,tsx}"),
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      colors: {
        bg: "rgb(var(--ak-color-bg-canvas-rgb) / <alpha-value>)",
        surface: "rgb(var(--ak-color-bg-surface-rgb) / <alpha-value>)",
        "surface-2": "rgb(var(--ak-color-bg-surface-low-rgb) / <alpha-value>)",
        "surface-3": "rgb(var(--ak-color-bg-surface-high-rgb) / <alpha-value>)",
        border: "var(--ak-color-border-default)",
        "border-strong": "var(--ak-color-border-strong)",
        fg: "rgb(var(--ak-color-text-primary-rgb) / <alpha-value>)",
        muted: "rgb(var(--ak-color-text-secondary-rgb) / <alpha-value>)",
        subtle: "rgb(var(--ak-color-text-muted-rgb) / <alpha-value>)",
        accent: "rgb(var(--ak-color-action-primary-rgb) / <alpha-value>)",
        "accent-ink": "rgb(var(--ak-color-on-action-primary-rgb) / <alpha-value>)",
        "accent-soft": "var(--ak-color-action-soft)",
        "accent-deep": "rgb(var(--ak-color-action-hover-rgb) / <alpha-value>)",
        danger: "rgb(var(--ak-color-status-error-rgb) / <alpha-value>)",
        warning: "rgb(var(--ak-color-status-warning-rgb) / <alpha-value>)",
        success: "rgb(var(--ak-color-status-success-rgb) / <alpha-value>)",
      },
      borderRadius: {
        xs: "1px",
        sm: "2px",
        DEFAULT: "4px",
        md: "6px",
        lg: "10px",
        xl: "12px",
        "2xl": "14px",
      },
      letterSpacing: {
        tightest: "-0.04em",
        tighter: "-0.025em",
        tight: "-0.015em",
      },
      boxShadow: {
        "inset-hairline":
          "inset 0 1px 0 0 rgba(255,255,255,0.04), inset 0 0 0 1px rgba(255,255,255,0.02)",
        "inset-deep": "inset 0 2px 4px 0 rgba(0,0,0,0.6), inset 0 1px 0 0 rgba(255,255,255,0.03)",
        "lift-sm":
          "0 1px 0 0 rgba(255,255,255,0.05) inset, 0 1px 2px 0 rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.04)",
        "lift-md":
          "0 1px 0 0 rgba(255,255,255,0.06) inset, 0 2px 8px -1px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.05)",
        "glow-accent": "0 0 0 1px rgba(72,90,226,0.35), 0 0 24px -4px rgba(72,90,226,0.42)",
      },
      keyframes: {
        "pulse-dot": {
          "0%, 100%": { opacity: "0.4", transform: "scale(0.95)" },
          "50%": { opacity: "1", transform: "scale(1.05)" },
        },
        scan: {
          "0%": { transform: "translateX(-100%)" },
          "100%": { transform: "translateX(100%)" },
        },
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "pulse-dot": "pulse-dot 1.6s ease-in-out infinite",
        scan: "scan 3s ease-in-out infinite",
        "fade-up": "fade-up 0.4s cubic-bezier(0.22, 1, 0.36, 1) both",
      },
    },
  },
  plugins: [],
} satisfies Config;

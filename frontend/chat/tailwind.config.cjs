const path = require("node:path");

module.exports = {
  darkMode: ["class"],
  content: [
    path.join(__dirname, "index.html"),
    path.join(__dirname, "src/**/*.{ts,tsx}"),
  ],
  theme: {
    extend: {
      colors: {
        border: "rgb(var(--ak-color-border-default-rgb) / <alpha-value>)",
        input: "rgb(var(--ak-color-border-default-rgb) / <alpha-value>)",
        ring: "rgb(var(--ak-color-action-primary-rgb) / <alpha-value>)",
        background: "rgb(var(--ak-color-bg-canvas-rgb) / <alpha-value>)",
        foreground: "rgb(var(--ak-color-text-primary-rgb) / <alpha-value>)",
        primary: {
          DEFAULT: "rgb(var(--ak-color-action-primary-rgb) / <alpha-value>)",
          foreground: "rgb(var(--ak-color-on-action-primary-rgb) / <alpha-value>)",
        },
        secondary: {
          DEFAULT: "rgb(var(--ak-color-bg-surface-low-rgb) / <alpha-value>)",
          foreground: "rgb(var(--ak-color-text-primary-rgb) / <alpha-value>)",
        },
        destructive: {
          DEFAULT: "rgb(var(--ak-color-status-error-rgb) / <alpha-value>)",
          foreground: "rgb(var(--ak-color-on-action-primary-rgb) / <alpha-value>)",
        },
        muted: {
          DEFAULT: "rgb(var(--ak-color-bg-surface-low-rgb) / <alpha-value>)",
          foreground: "rgb(var(--ak-color-text-secondary-rgb) / <alpha-value>)",
        },
        accent: {
          DEFAULT: "rgb(var(--ak-color-action-soft-rgb) / <alpha-value>)",
          foreground: "rgb(var(--ak-color-text-primary-rgb) / <alpha-value>)",
        },
        popover: {
          DEFAULT: "rgb(var(--ak-color-bg-surface-rgb) / <alpha-value>)",
          foreground: "rgb(var(--ak-color-text-primary-rgb) / <alpha-value>)",
        },
        card: {
          DEFAULT: "rgb(var(--ak-color-bg-surface-rgb) / <alpha-value>)",
          foreground: "rgb(var(--ak-color-text-primary-rgb) / <alpha-value>)",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
    },
  },
};

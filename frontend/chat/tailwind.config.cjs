const path = require("node:path");

module.exports = {
  darkMode: ["class"],
  content: [
    path.join(__dirname, "harness/**/*.{html,tsx}"),
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
        // shadcn 原语接到品牌圆角刻度：md 控件、lg 浮层、xl 对话框；sm 菜单项与 12 圆角、6 内边距的菜单同心。
        xl: "var(--ak-radius-lg)",
        lg: "var(--ak-radius-md)",
        md: "var(--ak-radius-sm)",
        sm: "calc(var(--ak-radius-sm) - 2px)",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};

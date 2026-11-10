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
        // shadcn 原语接到品牌圆角刻度：md 控件与提示（8）、lg 菜单与浮层（12）、xl 对话框（16）。
        // 菜单项用 md：12 − 4 内边距 − 1 边框 = 7，取最近一级 8。sm（6）只给 8 圆角控件里内缩 2px 的小件（input-group 的 kbd、xs 按钮）。
        xl: "var(--ak-radius-lg)",
        lg: "var(--ak-radius-md)",
        md: "var(--ak-radius-sm)",
        sm: "calc(var(--ak-radius-sm) - 2px)",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};

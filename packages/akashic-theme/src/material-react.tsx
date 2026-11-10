import { createElement, type MouseEvent, type ReactNode } from "react";

// md-* 自定义元素全文档只登记一次，仍归这份模块所有；登记改为异步：外壳运行时发布
// （聊天挂载的前置）不再等 @material/web 的下载与求值。登记前渲染的 md-* 元素在
// define 落地时原地升级，React 不需要重渲染；登记与首个插件模块下载并行，先于插件 UI 上屏。
void import("./material-components");

export type MaterialButtonVariant = "filled" | "tonal" | "outlined" | "text" | "danger";

const BUTTON_TAGS = {
  filled: "md-filled-button",
  tonal: "md-filled-tonal-button",
  outlined: "md-outlined-button",
  text: "md-text-button",
  danger: "md-filled-tonal-button",
} as const;

export function MaterialButton({
  children,
  variant = "filled",
  disabled = false,
  loading = false,
  className,
  type = "button",
  onClick,
}: {
  children: ReactNode;
  variant?: MaterialButtonVariant;
  disabled?: boolean;
  loading?: boolean;
  className?: string;
  type?: "button" | "submit" | "reset";
  onClick?: (event: MouseEvent<HTMLElement>) => void;
}) {
  const tag = BUTTON_TAGS[variant];
  return createElement(
    tag,
    {
      className: ["ak-material-button", `ak-material-button--${variant}`, className].filter(Boolean).join(" "),
      disabled: disabled || loading,
      type,
      onClick,
      "aria-busy": loading || undefined,
    },
    loading
      ? createElement("md-circular-progress", { indeterminate: true, "aria-label": "处理中" })
      : null,
    children,
  );
}

export function MaterialFilterChip({
  children,
  selected = false,
  disabled = false,
  className,
  onClick,
}: {
  children: ReactNode;
  selected?: boolean;
  disabled?: boolean;
  className?: string;
  onClick?: (event: MouseEvent<HTMLElement>) => void;
}) {
  return createElement("md-filter-chip", {
    className: ["ak-material-filter-chip", className].filter(Boolean).join(" "),
    selected,
    disabled,
    onClick,
  }, children);
}

export function MaterialIconButton({
  children,
  variant = "filled",
  disabled = false,
  className,
  label,
  onClick,
}: {
  children: ReactNode;
  variant?: "filled" | "tonal" | "standard" | "danger";
  disabled?: boolean;
  className?: string;
  label: string;
  onClick?: (event: MouseEvent<HTMLElement>) => void;
}) {
  const tag = variant === "filled"
    ? "md-filled-icon-button"
    : variant === "standard"
      ? "md-icon-button"
      : "md-filled-tonal-icon-button";
  return createElement(tag, {
    className: ["ak-material-icon-button", `ak-material-icon-button--${variant}`, className].filter(Boolean).join(" "),
    disabled,
    onClick,
    "aria-label": label,
  }, children);
}

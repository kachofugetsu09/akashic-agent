import type { ButtonHTMLAttributes } from "react";

import { IconSwap } from "./icon-swap";

import "./composer-action.css";

export type ComposerActionMode = "send" | "stop";

// 停止是一枚 10px 圆角小方块：比 20px 实心块安静，外圈的进度弧负责说"还在跑"。
const StopGlyph = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
    <rect x="3" y="3" width="10" height="10" rx="2.5" fill="currentColor" />
  </svg>
);

const SendGlyph = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M12 21V7M5 14l7-7 7 7" />
  </svg>
);

export function ComposerActionButton({
  mode,
  label,
  className,
  type = "button",
  ...props
}: Omit<ButtonHTMLAttributes<HTMLButtonElement>, "aria-label"> & {
  mode: ComposerActionMode;
  label: string;
}) {
  return (
    <button
      {...props}
      type={type}
      className={["composer-action-button", className].filter(Boolean).join(" ")}
      data-mode={mode}
      aria-label={label}
    >
      {/* 生成中外圈一道慢转的进度弧，正在停止时转快（pathLength=1 无缝追逐）。 */}
      <svg className="composer-action-button__chase" viewBox="0 0 32 32" aria-hidden="true">
        <circle
          cx="16"
          cy="16"
          r="15.1"
          pathLength={1}
          strokeDasharray="0.24 0.76"
          strokeLinecap="round"
        />
      </svg>
      {/* 发送与停止叠在同一个格子里换挡：按钮尺寸与热区不变，只有图形换手。 */}
      <IconSwap
        state={mode === "stop"}
        a={<SendGlyph />}
        b={<StopGlyph />}
      />
    </button>
  );
}

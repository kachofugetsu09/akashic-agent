import type { ButtonHTMLAttributes } from "react";
import { Square } from "lucide-react";

import { IconSwap } from "./icon-swap";

import "./composer-action.css";

export type ComposerActionMode = "send" | "stop";

const SendGlyph = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M12 19V5M5 12l7-7 7 7" />
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
      {/* 发送与停止叠在同一个格子里换挡：按钮尺寸与热区不变，只有图形换手。 */}
      <IconSwap
        state={mode === "stop"}
        a={<SendGlyph />}
        b={<Square width={20} height={20} aria-hidden="true" fill="currentColor" />}
      />
    </button>
  );
}

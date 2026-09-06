import type { ButtonHTMLAttributes } from "react";
import { Square, SendHorizontal } from "lucide-react";

import "./composer-action.css";

export type ComposerActionMode = "send" | "stop";

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
      {mode === "send" ? <SendHorizontal aria-hidden="true" /> : <Square aria-hidden="true" fill="currentColor" />}
    </button>
  );
}

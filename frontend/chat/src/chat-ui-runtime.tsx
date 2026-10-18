import * as React from "react";
import * as ReactJSXRuntime from "react/jsx-runtime";
import * as ReactDOMClient from "react-dom/client";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { copyText } from "./copy-text";
import { MarqueeText } from "./marquee-text";

// 顶栏右侧的安静标记：平时只是"图标 + 短名"，异常时变色；点开是一块说明面板。
function HeaderMarker({ icon, label, alert = false, ariaLabel, panelLabel, children }: {
  icon: ReactNode;
  label: string;
  alert?: boolean;
  ariaLabel: string;
  panelLabel: string;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  // 点击面板外或按 Esc 收起，Esc 把焦点还给触发按钮。
  useEffect(() => {
    if (!open) return;
    const onPointer = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setOpen(false);
      triggerRef.current?.focus();
    };
    document.addEventListener("pointerdown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);
  return <div className="header-marker" ref={rootRef}>
    <button ref={triggerRef} type="button" className="header-marker__trigger" data-alert={alert || undefined}
      aria-expanded={open} aria-label={ariaLabel} onClick={() => setOpen((value) => !value)}>
      {icon}
      <span><MarqueeText still>{label}</MarqueeText></span>
    </button>
    {open ? <div className="header-marker__panel" role="group" aria-label={panelLabel} data-escape-layer="">
      {children}
    </div> : null}
  </div>;
}

// 面板底部的一排文字按钮。
function HeaderMarkerActions({ children }: { children: ReactNode }) {
  return <div className="header-marker__actions">{children}</div>;
}

// 插件模块经 import map 取用 /assets/sdk/*.js，这些文件从这里读出宿主的同一份 React 与聊天基础件。
const ChatUi = {
  // 在插槽元素里挂一个 React 根；返回的卸载函数可直接作为 mount() 的返回值。
  render(host: HTMLElement, node: ReactNode): () => void {
    const root = ReactDOMClient.createRoot(host);
    root.render(node);
    // 插槽清理发生在宿主提交期间，推迟一拍卸载，避免嵌套同步卸载。
    return () => { queueMicrotask(() => root.unmount()); };
  },
  // 输入框上方的一行状态提示，与宿主自己的连接、上传提示同一外观。
  Notice({ children }: { children: ReactNode }) {
    return <p role="status" className="reply-unavailable">{children}</p>;
  },
  HeaderMarker,
  HeaderMarkerActions,
  MarqueeText,
  copyText,
};

export type ChatUiRuntime = typeof ChatUi;

// 必须在任何插件模块导入之前发布。
export function exposeChatRuntime(): void {
  Object.assign(window, { __akashicRuntime: { React, ReactJSXRuntime, ReactDOMClient, ChatUi } });
}

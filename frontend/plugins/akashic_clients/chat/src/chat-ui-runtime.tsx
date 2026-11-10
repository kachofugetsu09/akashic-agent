import * as React from "react";
import * as ReactJSXRuntime from "react/jsx-runtime";
import * as ReactDOM from "react-dom";
import * as ReactDOMClient from "react-dom/client";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { ChevronDown } from "lucide-react";
import { copyText } from "./copy-text";
import { MarqueeText } from "./marquee-text";
import { formatToolDuration, PreviewText, ToolDataValue } from "./tool-result-view";
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuRadioGroup,
  DropdownMenuRadioItem, DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

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
      if (event.key !== "Escape" || event.defaultPrevented) return;
      setOpen(false);
      // 焦点已经离开面板（例如在工作台里）时不抢回。
      const active = document.activeElement;
      if (active === document.body || rootRef.current?.contains(active)) triggerRef.current?.focus();
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

// 单选菜单：触发按钮显示当前项，展开后每项带一行说明，底部可附一句提示。
function ChoiceMenu({ ariaLabel, value, choices, note, disabled = false, onChange }: {
  ariaLabel: string;
  value: string;
  choices: readonly { value: string; label: string; description?: string }[];
  note?: string;
  disabled?: boolean;
  onChange(value: string): void;
}) {
  const current = choices.find((choice) => choice.value === value);
  return <DropdownMenu>
    <DropdownMenuTrigger asChild>
      <button type="button" className="choice-menu__trigger" disabled={disabled} aria-label={`${ariaLabel}：${current?.label ?? ""}`}>
        {current?.label} <ChevronDown size={15} aria-hidden="true" />
      </button>
    </DropdownMenuTrigger>
    <DropdownMenuContent className="choice-menu" align="start" sideOffset={8} collisionPadding={16}>
      <DropdownMenuRadioGroup value={value} aria-label={ariaLabel}>
        {choices.map((choice) => <DropdownMenuRadioItem key={choice.value} value={choice.value}
          className="choice-menu__choice" onSelect={() => onChange(choice.value)}>
          <span><strong>{choice.label}</strong>{choice.description ? <small>{choice.description}</small> : null}</span>
        </DropdownMenuRadioItem>)}
      </DropdownMenuRadioGroup>
      {note ? <p className="choice-menu__note">{note}</p> : null}
    </DropdownMenuContent>
  </DropdownMenu>;
}

// 面板底部的一排文字按钮。
function HeaderMarkerActions({ children }: { children: ReactNode }) {
  return <div className="header-marker__actions">{children}</div>;
}

// 插件模块经外壳 import map 取用 /dashboard/assets/sdk/*.js，其中 chat-ui-v1.js 从这里读出聊天基础件。
const ChatUi = {
  // 在插槽元素里挂一个 React 根；返回的卸载函数可直接作为 mount() 的返回值。
  render(host: HTMLElement, node: ReactNode): () => void {
    // 根挂在自己的容器里：宿主清理时会清空插槽元素，推迟的卸载只动这个已摘下的容器。
    const box = document.createElement("div");
    box.style.display = "contents";
    host.append(box);
    const root = ReactDOMClient.createRoot(box);
    root.render(node);
    // 插槽清理发生在宿主提交期间，推迟一拍卸载，避免嵌套同步卸载。
    return () => { box.remove(); queueMicrotask(() => root.unmount()); };
  },
  // 输入框上方的一行状态提示，与宿主自己的连接、上传提示同一外观。
  Notice({ children }: { children: ReactNode }) {
    return <p role="status" className="reply-unavailable">{children}</p>;
  },
  HeaderMarker,
  HeaderMarkerActions,
  MarqueeText,
  ChoiceMenu,
  copyText,
  ToolText: PreviewText,
  ToolData: ToolDataValue,
  formatDuration: formatToolDuration,
};

export type ChatUiRuntime = typeof ChatUi;

// 必须在任何插件模块导入之前发布。同文档挂载时外壳已登记 React 等共享运行时（与本模块用的是同一份），只追加 ChatUi。
export function exposeChatRuntime(): void {
  const host = window as unknown as { __akashicRuntime?: Record<string, unknown> };
  host.__akashicRuntime = { React, ReactJSXRuntime, ReactDOM, ReactDOMClient, ...host.__akashicRuntime, ChatUi };
}

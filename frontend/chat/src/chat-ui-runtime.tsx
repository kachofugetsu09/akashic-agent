import * as React from "react";
import * as ReactJSXRuntime from "react/jsx-runtime";
import * as ReactDOMClient from "react-dom/client";
import type { ReactNode } from "react";
import { copyText } from "./copy-text";
import { MarqueeText } from "./marquee-text";

// 插件模块经 import map 取用 /assets/sdk/*.js，这些文件从这里读出宿主的同一份 React 与聊天基础件。
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
  MarqueeText,
  copyText,
};

export type ChatUiRuntime = typeof ChatUi;

// 必须在任何插件模块导入之前发布。
export function exposeChatRuntime(): void {
  Object.assign(window, { __akashicRuntime: { React, ReactJSXRuntime, ReactDOMClient, ChatUi } });
}

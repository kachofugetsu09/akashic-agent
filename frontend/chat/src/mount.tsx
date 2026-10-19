// 外壳同文档挂载入口：conversation_ui 把聊天挂进自己的页面，经 link 收发宿主事件；React 与运行时都用外壳那一份。
import { connectHost, type ChatHandle, type ChatHostLink } from "./chat-host";
import { renderChat } from "./chat-entry";
import { setPageBuild } from "./use-desktop-chat-controller";

export type { ChatHandle, ChatHostLink } from "./chat-host";

export interface MountedChat extends ChatHandle {
  dispose(): void;
}

// 构建时替换为本次产物的版本（vite.module.config.ts 的 stampChatBuild），用来发现服务端已换了前端。
const MODULE_BUILD = "__AKASHIC_CHAT_BUILD__";

// 样式与本模块同目录发布；同一文档只插一次。
function ensureStylesheet(): void {
  const href = new URL(/* @vite-ignore */ "./chat-app.css", import.meta.url).href;
  if ([...document.querySelectorAll<HTMLLinkElement>("link[rel=stylesheet]")].some((link) => link.href === href)) return;
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = href;
  document.head.append(link);
}

export function mountChat(host: HTMLElement, link: ChatHostLink): MountedChat {
  ensureStylesheet();
  setPageBuild(MODULE_BUILD);
  const title = document.title;
  const frame = document.createElement("div");
  frame.className = "akashic-chat";
  frame.dataset.embeddedShell = "true";
  host.replaceChildren(frame);
  const disconnect = connectHost(link);
  const chat = renderChat(frame, frame, true);
  return {
    ...chat.handle,
    dispose() {
      chat.unmount();
      disconnect();
      frame.remove();
      document.title = title;
    },
  };
}

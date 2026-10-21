// 聊天的唯一入口：conversation_ui 把聊天挂进外壳页面（ADR 0126），经 link 收发宿主事件；React 与运行时都用外壳那一份。
import { Suspense } from "react";
import { createRoot } from "react-dom/client";
import { TooltipProvider } from "@/components/ui/tooltip";
import { connectHost, setKeyScope, type ChatHandle, type ChatHostLink } from "./chat-host";
import { observeChatLayout } from "./chat-layout";
import { exposeChatRuntime } from "./chat-ui-runtime";
import { DesktopChatApp } from "./desktop-chat-app";
import { warmMessageRenderer } from "./message-view";
import { notifyShown } from "./plugin-ui-runtime";
import { setShellRailActions } from "./shell-rail-actions";
import { receiveWorkbenchQuote, setHostShortcuts, setWorkbenchIntents } from "./shell-workbench";
import { setPageBuild } from "./use-desktop-chat-controller";
import { WebUiErrorBoundary } from "./webui-error-boundary";

export type { ChatHandle, ChatHostLink } from "./chat-host";

export interface MountedChat extends ChatHandle {
  dispose(): void;
}

// 发布构建时替换为本次产物的版本（vite.config.ts 的 stampChatBuild），用来发现服务端已换了前端；夹具页面不替换，不做版本核对。
const MODULE_BUILD = "__AKASHIC_CHAT_BUILD__";

// 样式与本模块同目录发布；同一文档只插一次。开发服务器经模块注入样式，没有这份文件。
function ensureStylesheet(): void {
  if (import.meta.env.DEV) return;
  const href = new URL(/* @vite-ignore */ "./chat-app.css", import.meta.url).href;
  if ([...document.querySelectorAll<HTMLLinkElement>("link[rel=stylesheet]")].some((link) => link.href === href)) return;
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = href;
  document.head.append(link);
}

// 宿主到聊天的句柄：外壳与工具区的状态只经这里进入聊天；frame 上的属性给样式读。
function chatHandle(frame: HTMLElement): ChatHandle {
  return {
    setRailActions: setShellRailActions,
    setTools(present, shortcuts) {
      frame.dataset.shellTools = present ? "true" : "false";
      setHostShortcuts(shortcuts);
    },
    setWorkbenchIntents,
    quote(quote) {
      if (typeof quote === "object" && quote !== null) receiveWorkbenchQuote(quote as Record<string, unknown>);
    },
    shown: notifyShown,
  };
}

export function mountChat(host: HTMLElement, link: ChatHostLink): MountedChat {
  // 1. 样式、版本与公开运行时先就位，聊天插件模块随后才会加载。
  ensureStylesheet();
  if (!MODULE_BUILD.startsWith("__")) setPageBuild(MODULE_BUILD);
  exposeChatRuntime();
  // 2. 聊天根节点承载布局、按键范围与弹层。
  const title = document.title;
  const frame = document.createElement("div");
  frame.className = "akashic-chat";
  host.replaceChildren(frame);
  const disconnect = connectHost(link);
  const stopLayout = observeChatLayout(frame);
  const stopKeys = setKeyScope(frame);
  // 3. 宿主投影（底栏动作、工作台意图与快捷键）是模块级状态，上一次挂载可能留下：每次挂载从空开始，卸载时撤回。
  const handle = chatHandle(frame);
  const clearProjection = () => {
    handle.setRailActions([]);
    handle.setTools(false, []);
    handle.setWorkbenchIntents([]);
  };
  clearProjection();
  // 4. 渲染应用。
  const root = createRoot(frame);
  root.render(
    <WebUiErrorBoundary>
      <TooltipProvider>
        <Suspense fallback={<div className="webui-entry-loading">正在载入界面…</div>}>
          <DesktopChatApp />
        </Suspense>
      </TooltipProvider>
    </WebUiErrorBoundary>,
  );
  // 5. 空闲预热富文本渲染分块(见 message-view.tsx 的 warmMessageRenderer),
  // 首个含代码/数学的会话不再在切换关键路径上等这份下载与求值。
  // rIC 加 timeout 兜底：挂载繁忙期一直抢不到空闲时，超时后仍强制预热，
  // 否则挂载完立刻切换会话的用户仍要在关键路径上付整笔加载。
  // 5s 取值压在慢网启动 ready（limited 档实测约 3.8s）之后：更早的强制预热会
  // 在启动关键路径上抢带宽与主线程（3s 档实测 limited warm ready +72ms）。
  // 不支持 rIC 的环境用同一时长的 setTimeout 兜底；调度与取消按同一套 API 配对，
  // 不拿 clearTimeout 去清 idle id（两套 id 命名空间，误清会伤到别的定时器）。
  const WARMUP_TIMEOUT_MS = 5000;
  const useIdle = typeof window.requestIdleCallback === "function" && typeof window.cancelIdleCallback === "function";
  const idle = useIdle
    ? window.requestIdleCallback(warmMessageRenderer, { timeout: WARMUP_TIMEOUT_MS })
    : window.setTimeout(warmMessageRenderer, WARMUP_TIMEOUT_MS);
  const cancelWarmup = () => {
    if (useIdle) window.cancelIdleCallback(idle as number);
    else window.clearTimeout(idle as number);
  };
  return {
    ...handle,
    dispose() {
      cancelWarmup();
      clearProjection();
      root.unmount();
      stopKeys();
      stopLayout();
      disconnect();
      frame.remove();
      document.title = title;
    },
  };
}

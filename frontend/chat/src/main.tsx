import { initializeTheme, startCrossPortThemeSync } from "../../theme/src/theme-runtime";
import { connectHost } from "./chat-host";
import { renderChat } from "./chat-entry";
import { frameLink, listenFrameHost } from "./frame-link";

export type { AgentBlock, ChatMessage, MessageAttachment, ThinkingBlock, ToolBlock } from "./chat-message";

// 独立页面与 iframe 入口；iframe 里经消息适配连上父页面的宿主。
const embeddedShell = new URLSearchParams(window.location.search).get("embedded") === "1";
if (embeddedShell) document.documentElement.dataset.embeddedShell = "true";
initializeTheme();
startCrossPortThemeSync();
if (embeddedShell) connectHost(frameLink);
const chat = renderChat(document.getElementById("root")!, document.documentElement, embeddedShell);
if (embeddedShell) listenFrameHost(chat.handle);

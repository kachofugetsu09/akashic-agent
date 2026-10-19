import "./message-ui.css";
import type { MessageAttachmentFacts, PluginUiContext, PluginUiDefinition } from "@akashic/chat-ui-v1";

const OPEN_INTENT = "site.open.v1";
const SUFFIX = ".akashic-site.json";
const OPENED_KEY = "akashic.sites.opened";
// 刚发布的快照才自动打开：消息时间在这个窗口内（容忍少量时钟差），且本页还没自动打开过它。
const FRESH_MS = 2 * 60_000;
// 只认领 show_site 工具产出的网站文件。
function accepts(attachment: MessageAttachmentFacts): boolean {
  return attachment.origin.kind === "tool_result" && attachment.origin.tool === "show_site"
    && attachment.kind === "file" && !!attachment.filename?.endsWith(SUFFIX);
}

// 自动打开记一次就够：刷新、回到会话、历史卡片都不再自动弹出，只有用户点击才打开。
function firstAutoOpen(artifact: string): boolean {
  try {
    const opened: string[] = JSON.parse(window.sessionStorage.getItem(OPENED_KEY) ?? "[]");
    if (opened.includes(artifact)) return false;
    window.sessionStorage.setItem(OPENED_KEY, JSON.stringify([...opened, artifact].slice(-50)));
    return true;
  } catch {
    return false;
  }
}

// 手机宽度下侧栏会盖住整段对话，不自动打开，只留卡片入口。
function wideEnough(): boolean {
  return (window.top ?? window).innerWidth >= 768;
}

function timeLabel(iso: string): string {
  const time = new Date(iso);
  return Number.isNaN(time.getTime()) ? "" : time.toLocaleString("zh-CN", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function mount(host: HTMLElement, context: PluginUiContext<"message.attachment">): () => void {
  const attachment = context.block;
  const title = attachment.filename!.slice(0, -SUFFIX.length) || "网页";
  const card = document.createElement("div");
  card.className = "site-card";
  const text = document.createElement("div");
  text.className = "site-card-text";
  const name = document.createElement("strong");
  name.textContent = title;
  const meta = document.createElement("span");
  meta.textContent = ["网页快照", `${Math.max(1, Math.round(attachment.size_bytes / 1024))} KB`, timeLabel(attachment.origin.at)].filter(Boolean).join(" · ");
  text.append(name, meta);
  const button = document.createElement("button");
  button.type = "button";
  button.className = "site-card-open";
  button.textContent = "打开预览";
  const open = () => context.workbench.open(OPEN_INTENT, { artifact_id: attachment.artifact_id }).then((status) => {
    meta.dataset.state = status === "handled" ? "" : "failed";
    if (status !== "handled") meta.textContent = status === "unhandled" ? "没有可用的侧边栏，可下载源文件" : "打开失败，请再点一次";
  });
  button.addEventListener("click", () => void open());
  card.append(text, button);
  // 宿主的能力广播可能晚于挂载，自动打开等到确认有接受者时再做。
  let auto = Math.abs(Date.now() - Date.parse(attachment.origin.at)) < FRESH_MS && wideEnough();
  const render = () => {
    const available = context.workbench.accepts(OPEN_INTENT);
    button.hidden = !available;
    host.replaceChildren(card);
    if (auto && available) {
      auto = false;
      if (firstAutoOpen(attachment.artifact_id)) void open();
    }
  };
  render();
  const stop = context.onInvalidate(render);
  return () => {
    stop();
    host.replaceChildren();
  };
}

export default { slots: { "message.attachment": { accepts, mount } } } satisfies PluginUiDefinition;

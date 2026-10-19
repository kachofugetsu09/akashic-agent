import template from "./frame.html?raw";
import type { MessageAttachmentFacts, PluginUiDefinition, PluginUiContext } from "@akashic/chat-ui-v1";

const CHANNEL = "akashic.visual.v1";
const MAX_BYTES = 5_000_000;
const TOKENS = ["type-reading", "color-action-primary", "color-text-muted", "color-text-primary", "color-text-secondary", "color-bg-canvas", "color-bg-surface", "color-border-default", "color-border-strong",
  ...[1, 2, 3, 4, 5, 6].map(index => `chart-${index}`)];
// 展示时注入的宿主桥：只汇报尺寸、就绪和错误，并接收主题；不提供查询或执行能力。
const BRIDGE = `<script>(() => {
  const send = (type, value) => parent.postMessage({channel: '${CHANNEL}', type, value}, '*');
  let queued = false, last = 0;
  const measure = () => {
    queued = false;
    const height = Math.ceil(document.body.getBoundingClientRect().height);
    if (height !== last) { last = height; send('height', height); }
  };
  const resize = () => { if (!queued) { queued = true; requestAnimationFrame(measure); } };
  addEventListener('error', event => send('error', event.message));
  addEventListener('unhandledrejection', event => send('error', String(event.reason)));
  addEventListener('message', event => {
    if (event.source !== parent || event.data?.channel !== '${CHANNEL}' || event.data.type !== 'theme') return;
    document.getElementById('ak-host-theme').textContent = event.data.value;
    dispatchEvent(new CustomEvent('akashic:theme'));
    resize();
  });
  addEventListener('DOMContentLoaded', () => {
    new ResizeObserver(resize).observe(document.body);
    measure();
    // 隐藏或屏外 iframe 的 rAF 可暂停；启动确认不能等待下一帧。
    send('ready', true);
  });
})();</` + `script>`;

function isHtml(attachment: MessageAttachmentFacts): boolean {
  return attachment.media_type === "text/html" || /\.html?$/i.test(attachment.filename ?? "");
}

// 旧版 render_visual 保存的 JSON 文档；新文档就是普通 HTML 文件。
function isLegacy(attachment: MessageAttachmentFacts): boolean {
  return attachment.origin?.kind === "tool_result" && attachment.origin.tool === "render_visual"
    && attachment.media_type === "application/json" && !!attachment.filename?.endsWith(".akashic-visual.json");
}

function accepts(attachment: MessageAttachmentFacts): boolean {
  return attachment.kind === "file" && (isHtml(attachment) || isLegacy(attachment));
}

function themeCss(host: HTMLElement): string {
  const style = getComputedStyle(host);
  return `:root{${TOKENS.map(token => `--ak-${token}:${style.getPropertyValue(`--ak-${token}`)};`).join("")}
    color-scheme:${style.colorScheme};}`;
}

// 桥放在文档最前面以便尽早汇报错误；宿主主题放在 head 末尾，覆盖文档自带的同名变量与配色方案。
function withBridge(html: string, theme: string): string {
  const style = `<style id="ak-host-theme">${theme}</style>`;
  const end = html.search(/<\/head\s*>/i);
  const themed = end < 0 ? html : html.slice(0, end) + style + html.slice(end);
  const inject = end < 0 ? BRIDGE + style : BRIDGE;
  const head = /<head\b[^>]*>/i.exec(themed) ?? /<html\b[^>]*>/i.exec(themed) ?? /<!doctype[^>]*>/i.exec(themed);
  if (!head) return inject + themed;
  const at = head.index + head[0].length;
  return themed.slice(0, at) + inject + themed.slice(at);
}

async function readDocument(attachment: MessageAttachmentFacts, host: HTMLElement, signal: AbortSignal): Promise<{ title: string; html: string }> {
  if (attachment.size_bytes > MAX_BYTES) throw new Error("附件超过大小限制");
  const response = await fetch(`/api/chat/artifacts/${encodeURIComponent(attachment.artifact_id)}`, { signal });
  if (!response.ok) throw new Error(`读取失败 (${response.status})`);
  if (!isLegacy(attachment)) return { title: attachment.filename ?? "网页", html: await response.text() };
  const value = await response.json();
  if (!value || value.schema !== CHANNEL || typeof value.title !== "string" || typeof value.html !== "string") throw new Error("不支持的文档格式");
  return { title: value.title, html: template.replace("/*visual-theme*/", () => themeCss(host)).replace("<!--visual-content-->", () => value.html) };
}

/** 每次挂载只读一次不可变附件；主题改变不重建文档。 */
function mount(host: HTMLElement, context: PluginUiContext<"message.attachment">): () => void {
  // 1. 文档在独立 sandbox 文档里运行；用户自己上传的网页先点击再运行。
  const attachment = context.block;
  const abort = new AbortController();
  const status = document.createElement("p");
  status.setAttribute("role", "status");
  const frame = document.createElement("iframe");
  frame.sandbox.add("allow-scripts");
  frame.referrerPolicy = "no-referrer";
  frame.style.cssText = "display:block;width:100%;height:320px;border:0;visibility:hidden";
  host.append(status);
  let ready = false;
  let timeout = 0;
  const fail = (message: string) => {
    window.clearTimeout(timeout);
    abort.abort();
    frame.remove();
    status.textContent = "";
    context.reportError(message);
  };

  // 2. 只接收当前 opaque frame 的展示消息。
  const receive = (event: MessageEvent) => {
    if (event.source !== frame.contentWindow || event.data?.channel !== CHANNEL) return;
    if (event.data.type === "ready") {
      ready = true;
      window.clearTimeout(timeout);
      frame.style.visibility = "visible";
      status.textContent = "";
    } else if (event.data.type === "height" && Number.isFinite(event.data.value)) {
      frame.style.height = `${Math.max(80, Math.min(1600, event.data.value))}px`;
    } else if (event.data.type === "error" && typeof event.data.value === "string") {
      if (ready) status.textContent = `本次交互计算遇到问题：${event.data.value.slice(0, 300)}。可以调整参数再试。`;
      else fail(event.data.value.slice(0, 300));
    }
  };
  window.addEventListener("message", receive);
  let loads = 0;
  let documentHtml = "";
  frame.addEventListener("load", () => {
    loads += 1;
    if (loads === 1) {
      // 独立 HTTP 文档不继承聊天页禁止内联脚本的 CSP；只传入一次展示内容。
      frame.contentWindow?.postMessage({ channel: CHANNEL, type: "init", document: documentHtml }, "*");
      documentHtml = "";
    } else if (loads > 2) {
      fail("文档发生了不允许的导航");
    }
  });
  const theme = new MutationObserver(() => frame.contentWindow?.postMessage(
    { channel: CHANNEL, type: "theme", value: themeCss(host) }, "*"));
  theme.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "style", "class"] });
  const start = async () => {
    status.textContent = "正在打开交互图…";
    timeout = window.setTimeout(() => { if (!ready) fail("加载超时"); }, 10_000);
    try {
      const value = await readDocument(attachment, host, abort.signal);
      if (abort.signal.aborted) return;
      frame.title = value.title;
      documentHtml = withBridge(value.html, themeCss(host));
      frame.src = "/visuals/frame.html";
      host.append(frame);
    } catch (error) {
      if (!abort.signal.aborted) fail(error instanceof Error ? error.message : String(error));
    }
  };
  if (attachment.origin?.author === "user") {
    const run = document.createElement("button");
    run.type = "button";
    run.textContent = `运行网页 ${attachment.filename ?? ""}`.trim();
    run.addEventListener("click", () => { run.remove(); void start(); }, { once: true });
    host.append(run);
  } else {
    void start();
  }

  // 3. 插件卸载释放请求、监听器和整个脚本上下文；不改附件。
  return () => {
    abort.abort();
    window.clearTimeout(timeout);
    theme.disconnect();
    window.removeEventListener("message", receive);
    frame.remove();
    status.remove();
  };
}

export default { slots: { "message.attachment": { accepts, mount } } } satisfies PluginUiDefinition;

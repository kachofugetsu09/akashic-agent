import template from "./frame.html?raw";
import type { PluginUiDefinition, PluginUiContext } from "../../../chat/src/plugin-ui-runtime";
import type { TimelineAttachmentOrigin } from "../../../chat/src/message-view";
import type { TimelineAttachment } from "../../../chat/src/message-timeline";

const CHANNEL = "akashic.visual.v1";
type VisualAttachment = TimelineAttachment & { origin: TimelineAttachmentOrigin };
const TOKENS = ["type-reading", "color-action-primary", "color-text-muted", "color-text-primary", "color-text-secondary", "color-bg-canvas", "color-bg-surface", "color-border-default",
  ...[1, 2, 3, 4, 5, 6].map(index => `chart-${index}`)];

function accepts(block: unknown): block is VisualAttachment {
  if (!block || typeof block !== "object") return false;
  const attachment = block as VisualAttachment;
  return attachment.origin?.kind === "tool_result" && attachment.origin.tool === "render_visual"
    && attachment.kind === "file" && attachment.media_type === "application/json"
    && !!attachment.filename?.endsWith(".akashic-visual.json");
}

function themeCss(host: HTMLElement): string {
  const style = getComputedStyle(host);
  return `:root{${TOKENS.map(token => `--ak-${token}:${style.getPropertyValue(`--ak-${token}`)};`).join("")}
    color-scheme:${style.colorScheme};}`;
}

/** 每次挂载只读一次不可变附件；主题改变不重建交互文档。 */
function mount(host: HTMLElement, context: PluginUiContext): () => void {
  // 1. 附件由宿主边界解析，文档内容在本插件边界校验。
  const attachment = context.block as TimelineAttachment;
  const abort = new AbortController();
  const status = document.createElement("p");
  status.setAttribute("role", "status");
  status.textContent = "正在打开交互图…";
  const frame = document.createElement("iframe");
  frame.sandbox.add("allow-scripts");
  frame.referrerPolicy = "no-referrer";
  frame.style.cssText = "display:block;width:100%;height:320px;border:0;visibility:hidden";
  host.append(status);
  let ready = false;
  const timeout = window.setTimeout(() => {
    if (!ready) fail("加载超时");
  }, 10_000);
  const fail = (message: string) => {
    window.clearTimeout(timeout);
    abort.abort();
    frame.remove();
    status.textContent = "";
    context.reportError(message);
  };

  // 2. 只接收当前 opaque frame 的展示消息，不提供查询或执行桥。
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
  let loaded = false;
  frame.addEventListener("load", () => {
    if (loaded) fail("文档发生了不允许的导航");
    loaded = true;
  });
  const theme = new MutationObserver(() => frame.contentWindow?.postMessage(
    { channel: CHANNEL, type: "theme", value: themeCss(host) }, "*"));
  theme.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "style", "class"] });
  void (async () => {
    try {
      if (attachment.size_bytes > 1_500_000) throw new Error("附件超过大小限制");
      const response = await fetch(`/api/chat/artifacts/${encodeURIComponent(attachment.artifact_id)}`, { signal: abort.signal });
      if (!response.ok) throw new Error(`读取失败 (${response.status})`);
      const value = await response.json();
      if (!value || value.schema !== CHANNEL || typeof value.title !== "string" || typeof value.summary !== "string"
        || typeof value.html !== "string" || Array.from(value.html).length > 200_000) throw new Error("不支持的文档格式");
      if (abort.signal.aborted) return;
      frame.title = value.title;
      const runtime = context.scripts["frame_runtime.js"];
      if (!runtime) throw new Error("插件缺少固定绘图脚本");
      const runtimeUrl = new URL(runtime, location.href);
      const escape = (value: string) => value.replaceAll("&", "&amp;").replaceAll('"', "&quot;");
      frame.srcdoc = template.replace("/*visual-theme*/", () => themeCss(host))
        .replace("__VISUAL_RUNTIME_CSP__", () => escape(runtimeUrl.origin + runtimeUrl.pathname))
        .replace("__VISUAL_RUNTIME_URL__", () => escape(runtimeUrl.href))
        .replace("<!--visual-content-->", () => value.html);
      host.append(frame);
    } catch (error) {
      if (!abort.signal.aborted) fail(error instanceof Error ? error.message : String(error));
    }
  })();

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

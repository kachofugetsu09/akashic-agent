import { defineComponent, h, onBeforeUnmount, onMounted, ref, watch } from "vue";
import type { Editor } from "@open-pencil/core";
import { buildOpenPencilClipboardHTML } from "@open-pencil/core/clipboard";
import { browserHTMLToSceneGraph } from "@open-pencil/dom-css/browser";
import { ensureFallback, loadDocumentFonts, type Documents } from "./documents";

// 网页在自带 CSP sandbox 的不透明源页面里运行（preview.html），拿不到 akashic 的 Cookie、存储与本编辑器。
const FRAME = "/design/assets/preview.html";
const CHANNEL = "akashic.site.v1";

// 预览：每次换内容都重建 iframe，旧页面里的脚本与计时器随之销毁。
export const PreviewFrame = defineComponent({
  props: { html: { type: String, required: true } },
  setup(props) {
    const host = ref<HTMLElement | null>(null);
    let frame: HTMLIFrameElement | null = null;
    const onMessage = (event: MessageEvent<{ channel?: string; type?: string }>) => {
      if (!frame || event.source !== frame.contentWindow || event.data?.channel !== CHANNEL || event.data.type !== "ready") return;
      frame.contentWindow?.postMessage({ channel: CHANNEL, type: "init", document: props.html }, "*");
    };
    const mount = () => {
      frame?.remove();
      frame = document.createElement("iframe");
      frame.className = "design-preview-frame";
      frame.title = "网页预览";
      frame.sandbox.add("allow-scripts", "allow-forms", "allow-modals", "allow-popups");
      frame.src = FRAME;
      host.value?.append(frame);
    };
    window.addEventListener("message", onMessage);
    onMounted(mount);
    watch(() => props.html, mount);
    onBeforeUnmount(() => { window.removeEventListener("message", onMessage); frame?.remove(); });
    return () => h("div", { ref: host, class: "design-preview" });
  },
});

const BODY = "akashic-body";

// 快照把本地样式表内联成 data URI 的 <link>；外链样式表不加载。
function stylesheets(parsed: Document): string {
  return [...parsed.querySelectorAll<HTMLStyleElement | HTMLLinkElement>("style, link[rel~='stylesheet'][href^='data:text/css']")].map((node) => {
    if (node instanceof HTMLStyleElement) return node.textContent ?? "";
    const href = node.getAttribute("href") ?? "";
    const comma = href.indexOf(",");
    const body = href.slice(comma + 1);
    if (!href.slice(0, comma).endsWith(";base64")) return decodeURIComponent(body);
    return new TextDecoder().decode(Uint8Array.from(atob(body), (char) => char.charCodeAt(0)));
  }).join("\n");
}

// 转换只取 body 的子元素当顶层图层；把它们包进一个容器，原来写给 html/body 的规则改写到容器上，背景、内边距和排版才保留。
function rebody(rules: CSSRuleList): void {
  for (const rule of rules) {
    if (rule instanceof CSSStyleRule) rule.selectorText = rule.selectorText.replace(/(^|[\s,>+~(])(html|body|:root)(?=$|[\s,>+~.:#[)])/g, `$1.${BODY}`);
    if (rule instanceof CSSGroupingRule) rebody(rule.cssRules);
  }
}

// 网页导入成图层：用浏览器自己的 CSS 引擎算出样式与布局，整个页面成为一个顶层框架；整次导入是一步“粘贴”，可以撤销。
// 当前页是空的就放进当前页，否则放进新的一页。
export async function importHTML(editor: Editor, page: string, name: string): Promise<number> {
  const parsed = new DOMParser().parseFromString(page, "text/html");
  const sheet = new CSSStyleSheet();
  sheet.replaceSync(stylesheets(parsed));
  rebody(sheet.cssRules);
  const cssText = [...sheet.cssRules].map((rule) => rule.cssText).join("\n");
  const html = `<div class="${BODY}">${parsed.body.innerHTML}</div>`;
  const imported = await browserHTMLToSceneGraph(html, { cssText, sandbox: "iframe", pageName: name });
  const roots = (imported.getPages()[0]?.childIds ?? []).map((id) => imported.getNode(id)).filter((node) => node !== undefined);
  if (!roots.length) throw new Error("页面里没有可以导入的内容");
  for (const root of roots) imported.updateNode(root.id, { name });
  await ensureFallback([...imported.nodes.values()].map((node) => node.type === "TEXT" ? node.text : node.name).join(""));
  if (editor.graph.getNode(editor.state.currentPageId)?.childIds.length) await editor.switchPage(editor.addPage(name));
  await editor.pasteFromHTML(buildOpenPencilClipboardHTML(roots, imported), { x: 0, y: 0 });
  editor.zoomToFit();
  await loadDocumentFonts(editor);
  return countNodes(imported, roots.map((node) => node.id));
}

function countNodes(graph: { getNode(id: string): { childIds: string[] } | undefined }, ids: string[]): number {
  return ids.reduce((total, id) => total + 1 + countNodes(graph, graph.getNode(id)?.childIds ?? []), 0);
}

// 用户点“导入为图层”：排进文档队列，轮到时会话或文档已经换了就不导入；导入期间冻结画布输入。
export function importPreview(editor: Editor, documents: Documents): Promise<number> {
  const { session, current } = documents.state;
  return documents.serial(() => {
    const now = documents.state;
    const page = now.page;
    if (!current || !page || now.session !== session || now.current?.id !== current.id)
      return Promise.reject(new Error("文档已经换了，这次导入没有执行"));
    return documents.hold(async () => {
      const count = await importHTML(editor, page, current.name);
      now.view = "canvas";
      documents.touch();
      return count;
    });
  });
}

import { reactive } from "vue";
import { CORE_TOOLS, FigmaAPI, createDefaultEditorState, createEditor, exportFigFile, getCanvasKit, parseFigFile, type ToolDef } from "@open-pencil/core";
import { executeAtomicTool } from "@open-pencil/core/editor";
import { isAtomicTool, parseToolArgs } from "@open-pencil/core/tools";
import { SceneGraph, type SceneNode } from "@open-pencil/scene-graph";
import canvasKitWasm from "canvaskit-wasm/bin/canvaskit.wasm";
import { ensureFallback, loadDocumentFonts } from "./documents";
import { configureFonts } from "./fonts";
import { importHTML } from "./preview";

// 服务端的无头画布：后端在无头浏览器里打开这一页，Agent 的设计工具在这里执行，文档的读写由后端经 window.akashicDesign 完成。
// 用户不必开着网页；网页里的画布只跟进存储里的新修订。

export type WorkerReply = { ok: true; result: unknown; mutated: boolean } | { ok: false; error: string };

export interface DesignWorkerApi {
  open(data: string | null): Promise<void>;
  call(tool: string, args: Record<string, unknown>): Promise<WorkerReply>;
  importHtml(page: string, name: string): Promise<number>;
  save(): Promise<string>;
}

const TOOLS = new Map<string, ToolDef>(CORE_TOOLS.map((tool) => [tool.name, tool]));
const ID = /^\d+:\d+$/u;
// 内容字段原样保留：文字里的“12:30”“16:9”不是图层 ID
const CONTENT = new Set(["text", "characters", "name", "jsx", "html", "content", "diff", "message", "error"]);

configureFonts((path) => fetch(path));
await getCanvasKit({ locateFile: () => canvasKitWasm });
const graph = new SceneGraph();
const first = graph.getPages()[0];
if (!first) throw new Error("open-pencil 没有初始页面");
const editor = createEditor({ graph, state: reactive(createDefaultEditorState(first.id)), getViewportSize: () => ({ width: 1440, height: 900 }) });

// 图层在 .fig 里的 GUID 每次读入都不变，读入后的图层 ID 却按顺序重新编号：交给 Agent 的一律是 GUID，
// 别处保存后重新读入、浏览器回收后重开，Agent 手里的 ID 仍然指向同一个图层。
function guid(node: SceneNode): string {
  const saved = node.source.id;
  return saved && ID.test(saved) ? saved : node.id;
}

// 只换整串就是图层 ID 的值（参数里的 id、ids、parent，结果里的 id），不在任意文字里查找替换
function translate(value: unknown, map: Map<string, string>): unknown {
  if (typeof value === "string") return ID.test(value) ? map.get(value) ?? value : value;
  if (Array.isArray(value)) return value.map((item) => translate(item, map));
  if (value === null || typeof value !== "object") return value;
  return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, CONTENT.has(key) ? item : translate(item, map)]));
}

function idMaps(): { inward: Map<string, string>; outward: Map<string, string> } {
  const inward = new Map<string, string>();
  const outward = new Map<string, string>();
  for (const node of editor.graph.nodes.values()) {
    const id = guid(node);
    inward.set(id, node.id);
    outward.set(node.id, id);
  }
  return { inward, outward };
}

// 结果要能交回后端：节点代理等对象只留可序列化的字段。
function plain(value: unknown): unknown {
  return value === undefined ? null : JSON.parse(JSON.stringify(value)) as unknown;
}

// 1. 校验参数并把 GUID 换成图层 ID；2. 只读工具直接跑，改文档的工具跑完把排版与字体补齐；3. 结果里的 ID 换回 GUID
async function call(tool: string, raw: Record<string, unknown>): Promise<WorkerReply> {
  try {
    const def = TOOLS.get(tool);
    if (!def) throw new Error(`未知的设计工具：${tool}`);
    const args = parseToolArgs(def.name, def.input, translate(raw, idMaps().inward)) as Record<string, unknown>;
    const figma = new FigmaAPI(editor.graph);
    figma.currentPage = figma.wrapNode(editor.state.currentPageId);
    if (def.mutates) await ensureFallback(JSON.stringify(raw));
    const result = !def.mutates ? await def.execute(figma, args)
      : isAtomicTool(def) ? executeAtomicTool(editor, figma, def, args, { label: "Agent" })
      : await editor.runMutationWithLayout(() => def.execute(figma, args), editor.state.currentPageId, () => loadDocumentFonts(editor));
    if (result && typeof result === "object" && "error" in result) throw new Error(String(result.error));
    if (def.mutates) await loadDocumentFonts(editor);
    return { ok: true, result: translate(plain(result), idMaps().outward), mutated: def.mutates };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) };
  }
}

function decode(data: string): ArrayBuffer {
  return Uint8Array.from(atob(data), (char) => char.charCodeAt(0)).buffer;
}

function encode(bytes: Uint8Array): string {
  let text = "";
  for (let start = 0; start < bytes.length; start += 0x8000) text += String.fromCharCode(...bytes.subarray(start, start + 0x8000));
  return btoa(text);
}

const api: DesignWorkerApi = {
  async open(data) {
    editor.replaceGraph(data ? await parseFigFile(decode(data)) : new SceneGraph());
    editor.undo.clear();
    await loadDocumentFonts(editor);
  },
  call,
  importHtml: (page, name) => importHTML(editor, page, name),
  async save() {
    return encode(await exportFigFile(editor.graph));
  },
};
(window as unknown as { akashicDesign: DesignWorkerApi }).akashicDesign = api;

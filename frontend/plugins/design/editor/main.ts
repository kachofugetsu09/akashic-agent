import { createApp, defineComponent, h, onBeforeUnmount, onMounted, reactive, watchEffect, type PropType } from "vue";
import { createDefaultEditorState, createEditor, fontManager, getCanvasKit, type Editor, type Tool } from "@open-pencil/core";
import { SceneGraph } from "@open-pencil/scene-graph";
import { CanvasRoot, CanvasSurface, EDITOR_KEY, toolCursor, useCanvasContext, useCanvasInput, useEditorCommands, useTextEdit } from "@open-pencil/vue";
import canvasKitWasm from "canvaskit-wasm/bin/canvaskit.wasm";
import interBold from "@open-pencil/core/assets/Inter-Bold.ttf";
import interExtraBold from "@open-pencil/core/assets/Inter-ExtraBold.ttf";
import interMedium from "@open-pencil/core/assets/Inter-Medium.ttf";
import interRegular from "@open-pencil/core/assets/Inter-Regular.ttf";
import interSemiBold from "@open-pencil/core/assets/Inter-SemiBold.ttf";
import notoNaskhArabic from "@open-pencil/core/assets/NotoNaskhArabic-Regular.ttf";
import { DESIGN_CHANNEL, type DesignFromFrame, type DesignToFrame, type RelayedResponse } from "./contract";
import { Documents, type DocumentHost } from "./documents";
import { iconSvg, type IconName } from "./icons";
import { bindKeyboard } from "./keyboard";
import { LayersPanel, PropertiesPanel } from "./panels";
import "./style.css";

const FONTS: Record<string, string> = {
  "Inter-Bold.ttf": interBold, "Inter-ExtraBold.ttf": interExtraBold, "Inter-Medium.ttf": interMedium,
  "Inter-Regular.ttf": interRegular, "Inter-SemiBold.ttf": interSemiBold, "NotoNaskhArabic-Regular.ttf": notoNaskhArabic,
};
const TOOLS: { tool: Tool; label: string; icon: IconName; key: string }[] = [
  { tool: "SELECT", label: "移动", icon: "select", key: "V" },
  { tool: "FRAME", label: "画框", icon: "frame", key: "F" },
  { tool: "RECTANGLE", label: "矩形", icon: "rectangle", key: "R" },
  { tool: "ELLIPSE", label: "椭圆", icon: "ellipse", key: "O" },
  { tool: "LINE", label: "直线", icon: "line", key: "L" },
  { tool: "TEXT", label: "文字", icon: "text", key: "T" },
  { tool: "PEN", label: "钢笔", icon: "pen", key: "P" },
  { tool: "HAND", label: "抓手", icon: "hand", key: "H" },
];
// 面板默认按 iframe 宽度决定：工作台够宽时两侧都开，窄时只留画布，用文档条上的按钮展开。
const panels = reactive({ layers: window.innerWidth >= 720, properties: window.innerWidth >= 960 });
const SAVE_LABELS = { saved: "已保存", dirty: "有修改", saving: "正在保存…", conflict: "文档已在别处更新", error: "保存失败" } as const;

fontManager.setBundledFontLocator((file) => FONTS[file.replace(/^\//u, "")] ?? file);

function IconButton(props: { label: string; icon: IconName; pressed?: boolean; disabled?: boolean; onClick: () => void }) {
  return h("button", {
    type: "button", class: "design-icon", title: props.label, "aria-label": props.label, disabled: props.disabled,
    "aria-pressed": props.pressed === undefined ? undefined : String(props.pressed),
    innerHTML: iconSvg(props.icon), onClick: props.onClick,
  });
}

// 文档条：当前会话的文档切换、新建与保存状态；冲突时只给“重新加载”，不静默覆盖别处的修改。
const DocumentBar = defineComponent({
  props: {
    documents: { type: Object as PropType<Documents>, required: true },
  },
  setup(props) {
    const commands = useEditorCommands();
    return () => {
      const { state } = props.documents;
      const busy = state.busy > 0;
      const current = state.current;
      const picker = state.list.length > 1 && current
        ? h("select", { class: "design-doc-select", "aria-label": "设计文档", value: current.id, disabled: busy,
            onChange: (event: Event) => void props.documents.open((event.target as HTMLSelectElement).value) },
          state.list.map((item) => h("option", { value: item.id }, item.name)))
        : h("strong", { class: "design-doc-name" }, current?.name ?? "设计");
      const status = state.save === "conflict"
        ? h("button", { type: "button", class: "design-status is-alert", disabled: busy, onClick: () => void props.documents.reload() },
            `${SAVE_LABELS.conflict} · 重新加载`)
        : current && h("span", { class: `design-status is-${state.save}`, role: "status" }, SAVE_LABELS[state.save]);
      return h("header", { class: "design-bar" }, [
        picker, status,
        h("span", { class: "design-bar-gap" }),
        IconButton({ label: "图层", icon: "layers", pressed: panels.layers, disabled: !current, onClick: () => { panels.layers = !panels.layers; } }),
        IconButton({ label: "属性", icon: "properties", pressed: panels.properties, disabled: !current,
          onClick: () => { panels.properties = !panels.properties; } }),
        IconButton({ label: "撤销", icon: "undo", disabled: !current || busy, onClick: () => commands.runCommand("edit.undo") }),
        IconButton({ label: "重做", icon: "redo", disabled: !current || busy, onClick: () => commands.runCommand("edit.redo") }),
        IconButton({ label: "缩放到全部内容", icon: "fit", disabled: !current, onClick: () => commands.runCommand("view.zoomFit") }),
        IconButton({ label: "新建设计", icon: "plus", disabled: !state.session || busy, onClick: () => void props.documents.create() }),
      ]);
    };
  },
});

// 浮在画布底部的工具条：与 open-pencil 应用一样按单键切换，按钮上标出键位。
const ToolDock = defineComponent({
  props: { editor: { type: Object as PropType<Editor>, required: true } },
  setup(props) {
    return () => h("nav", { class: "design-dock", "aria-label": "绘制工具" }, TOOLS.map((item) => IconButton({
      label: `${item.label} (${item.key})`, icon: item.icon, pressed: props.editor.state.activeTool === item.tool,
      onClick: () => props.editor.setTool(item.tool),
    })));
  },
});

// CanvasRoot 只负责绘制：指针、文字编辑与光标要像 open-pencil 应用的 EditorCanvas 一样另外接上。
const CanvasInput = defineComponent({
  props: {
    editor: { type: Object as PropType<Editor>, required: true },
    documents: { type: Object as PropType<Documents>, required: true },
  },
  setup(props) {
    const canvas = useCanvasContext();
    const enabled = () => props.documents.state.busy === 0;
    const input = useCanvasInput(canvas.canvasRef, props.editor, canvas.hitTestSectionTitle,
      canvas.hitTestComponentLabel, canvas.hitTestFrameTitle, undefined, undefined, enabled);
    useTextEdit(canvas.canvasRef, props.editor, { isEnabled: enabled });
    watchEffect(() => {
      const element = canvas.canvasRef.value;
      if (element) element.style.cursor = toolCursor(props.editor.state.activeTool, input.cursorOverride.value);
    });
    onBeforeUnmount(() => input.cleanupInteractions());
    return () => null;
  },
});

const Shell = defineComponent({
  props: {
    editor: { type: Object as PropType<Editor>, required: true },
    documents: { type: Object as PropType<Documents>, required: true },
  },
  setup(props) {
    const commands = useEditorCommands();
    let unbind = () => {};
    onMounted(() => { unbind = bindKeyboard(window, props.editor, commands.runCommand, () => props.documents.state.busy > 0); });
    onBeforeUnmount(() => unbind());
    return () => {
      const { state } = props.documents;
      const empty = !state.current && !state.loading;
      return h("div", { class: "design-editor" }, [
        h(DocumentBar, { documents: props.documents }),
        state.error && h("p", { class: "design-error", role: "alert" }, state.error),
        h("div", { class: "design-body", inert: state.busy > 0 }, [
          !empty && panels.layers && h(LayersPanel),
          h("div", { class: "design-stage" }, [
            h(CanvasRoot, null, () => [h(CanvasSurface, { class: "design-canvas" }), h(CanvasInput, { editor: props.editor, documents: props.documents })]),
            !empty && h(ToolDock, { editor: props.editor }),
            empty && h("div", { class: "design-empty" }, [
              h("p", state.session ? "这个会话还没有设计文档。Agent 做设计时会在这里新建，你也可以自己开一份。" : "发出第一条消息后，这个会话就可以有自己的设计文档。"),
              state.session && h("button", { type: "button", class: "design-primary", onClick: () => void props.documents.create() }, "新建设计"),
            ]),
          ]),
          !empty && panels.properties && h(PropertiesPanel, { editor: props.editor }),
        ]),
      ]);
    };
  },
});

// 1. 跑在设计标签的同源 iframe 里：标题与保存指示回报给父页面
function report(message: DesignFromFrame): void {
  parent.postMessage({ channel: DESIGN_CHANNEL, ...message }, location.origin);
}

// 2. 文档请求经父页面代发：父页面带着 Web 模块身份调用本插件接口，再把响应整块转回来
const pending = new Map<number, (reply: RelayedResponse) => void>();
let nextRequest = 0;
async function relayed(path: string, init: RequestInit = {}): Promise<Response> {
  const id = ++nextRequest;
  const body = init.body === undefined || init.body === null ? null
    : typeof init.body === "string" ? new TextEncoder().encode(init.body).buffer
    : init.body instanceof Uint8Array ? new Uint8Array(init.body).buffer : null;
  const reply = await new Promise<RelayedResponse>((resolve) => {
    pending.set(id, resolve);
    report({ type: "request", id, path, method: init.method ?? "GET",
      headers: [...new Headers(init.headers).entries()], body });
  });
  if ("error" in reply) throw new Error(reply.error);
  return new Response(reply.status === 204 ? null : reply.body, { status: reply.status, headers: reply.headers });
}

function boot(documents: Documents, editor: Editor): void {
  const theme = (dark: boolean) => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    editor.requestRender();
  };
  // 3. 只接受父页面同源的消息；会话切换与隐藏时先把未保存的修改写回
  window.addEventListener("message", (event: MessageEvent<DesignToFrame & { channel?: string }>) => {
    if (event.source !== parent || event.origin !== location.origin || event.data?.channel !== DESIGN_CHANNEL) return;
    const data = event.data;
    if (data.type === "init") {
      theme(data.dark);
      void documents.setSession(data.session);
    } else if (data.type === "session") void documents.setSession(data.session);
    else if (data.type === "active" && !data.active) void documents.flush();
    else if (data.type === "theme") theme(data.dark);
    else if (data.type === "changed") void documents.changed(data.document, data.revision);
    else if (data.type === "resync") void documents.resync();
    else if (data.type === "response") {
      pending.get(data.id)?.(data);
      pending.delete(data.id);
    }
  });
  window.addEventListener("pagehide", () => void documents.flush());
  report({ type: "ready" });
}

// 4. CanvasKit 先按我们的资产地址加载完再挂载画布：0.16 的 getCanvasKit 只在加载完成后才缓存实例，
//    并发的第二次调用会按默认地址（站点根目录）再加载一份
await getCanvasKit({ locateFile: () => canvasKitWasm });
const graph = new SceneGraph();
const page = graph.getPages()[0];
if (!page) throw new Error("open-pencil 没有初始页面");
// 5. 编辑器状态是响应式的，面板和工具条跟着选择、工具与页面变化
// 视口是画布区域本身：两侧面板开合时“缩放到全部内容”按剩下的画布宽度算
const viewport = () => {
  const stage = document.querySelector(".design-stage");
  return { width: stage?.clientWidth ?? window.innerWidth, height: stage?.clientHeight ?? window.innerHeight };
};
const editor = createEditor({ graph, state: reactive(createDefaultEditorState(page.id)), getViewportSize: viewport });
const host: DocumentHost = {
  request: relayed,
  setTitle: (title) => report({ type: "title", title }),
  setIndicator: (indicator) => report({ type: "indicator", indicator }),
};
const documents = new Documents(editor, host);
const root = document.getElementById("design-root");
if (!root) throw new Error("设计编辑器页面缺少挂载点");
// 编辑器挂在应用级：外壳自己的 setup 里也能用 open-pencil 的组合式函数。
createApp(Shell, { editor, documents }).provide(EDITOR_KEY, editor).mount(root);
boot(documents, editor);

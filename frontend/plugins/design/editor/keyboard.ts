import { TOOL_SHORTCUTS, type Editor, type Tool } from "@open-pencil/core";
import { EDITOR_COMMAND_METADATA, isCanvasTextInput, type EditorCommandId } from "@open-pencil/vue";
import { ZOOM_STEP, zoomBy } from "./navigation";

const MAC = /Mac|iP(hone|ad|od)/u.test(navigator.platform);

// 键位写法与 open-pencil 的命令表一致（tinykeys："$mod+Shift+KeyZ"）；$mod 在 macOS 是 ⌘，其余是 Ctrl。
function matches(binding: string, event: KeyboardEvent): boolean {
  const parts = binding.split("+");
  const code = parts.pop();
  const wanted = new Set(parts.map((part) => (part === "$mod" ? (MAC ? "Meta" : "Control") : part)));
  return code === event.code
    && wanted.has("Meta") === event.metaKey && wanted.has("Control") === event.ctrlKey
    && wanted.has("Alt") === event.altKey && wanted.has("Shift") === event.shiftKey;
}

function editing(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return isCanvasTextInput(target) || target.isContentEditable
    || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
}

// 编辑器在自己的 iframe 里，快捷键挂在 iframe 的 window 上：焦点在画布、面板或页面上都生效，碰不到聊天输入框。
export function bindKeyboard(root: Window, editor: Editor, run: (id: EditorCommandId) => void, blocked: () => boolean): () => void {
  const commands = Object.entries(EDITOR_COMMAND_METADATA) as [EditorCommandId, { keybinding?: string | string[] }][];
  // 按住空格临时换成抓手，松开换回原来的工具（Figma 的做法）；拖动中松开空格，这次拖动照样做完。
  // 已经按着鼠标时不换：钢笔拖动中按空格是 open-pencil 用来挪动锚点的
  let held: Tool | null = null;
  let pressed = false;
  const onPointer = (event: PointerEvent) => { pressed = event.buttons !== 0; };
  const release = () => {
    if (held !== null && editor.state.activeTool === "HAND") editor.setTool(held);
    held = null;
  };
  const onKey = (event: KeyboardEvent) => {
    if (blocked() || editing(event.target)) return;
    if (event.code === "Space") {
      event.preventDefault();
      if (held === null && !event.repeat && !pressed && editor.state.activeTool !== "HAND") {
        held = editor.state.activeTool;
        editor.setTool("HAND");
      }
      return;
    }
    if (event.repeat && !event.code.startsWith("Arrow") && event.code !== "Equal" && event.code !== "Minus") return;
    // 0. 放大缩小：+/-，带不带 Ctrl/⌘ 都行，同时拦下浏览器自己的页面缩放
    if ((event.code === "Equal" || event.code === "Minus" || event.code === "NumpadAdd" || event.code === "NumpadSubtract") && !event.altKey) {
      event.preventDefault();
      zoomBy(editor, event.code === "Equal" || event.code === "NumpadAdd" ? ZOOM_STEP : 1 / ZOOM_STEP);
      return;
    }
    // 1. 编辑器命令（撤销、编组、复制……）
    for (const [id, meta] of commands) {
      const keys = meta.keybinding === undefined ? [] : [meta.keybinding].flat();
      if (keys.some((binding) => matches(binding, event))) {
        event.preventDefault();
        run(id);
        return;
      }
    }
    // 2. 删除选中的图层：命令表里没有给它键位
    if ((event.code === "Delete" || event.code === "Backspace") && !event.metaKey && !event.ctrlKey) {
      event.preventDefault();
      editor.deleteSelected();
      return;
    }
    // 3. 单键切换工具
    const tool = (TOOL_SHORTCUTS as Record<string, Tool | undefined>)[event.shiftKey ? `Shift+${event.code}` : event.code];
    if (tool && !event.metaKey && !event.ctrlKey && !event.altKey) {
      event.preventDefault();
      editor.setTool(tool);
    }
  };
  const onKeyUp = (event: KeyboardEvent) => { if (event.code === "Space") release(); };
  root.addEventListener("keydown", onKey);
  root.addEventListener("keyup", onKeyUp);
  root.addEventListener("blur", release);
  root.addEventListener("pointerdown", onPointer, true);
  root.addEventListener("pointerup", onPointer, true);
  return () => {
    root.removeEventListener("pointerdown", onPointer, true);
    root.removeEventListener("pointerup", onPointer, true);
    root.removeEventListener("keydown", onKey);
    root.removeEventListener("keyup", onKeyUp);
    root.removeEventListener("blur", release);
  };
}

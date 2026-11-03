import { TOOL_SHORTCUTS, type Editor, type Tool } from "@open-pencil/core";
import { EDITOR_COMMAND_METADATA, isCanvasTextInput, type EditorCommandId } from "@open-pencil/vue";

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
export function bindKeyboard(root: Window, editor: Editor, run: (id: EditorCommandId) => void): () => void {
  const commands = Object.entries(EDITOR_COMMAND_METADATA) as [EditorCommandId, { keybinding?: string | string[] }][];
  const onKey = (event: KeyboardEvent) => {
    if (event.repeat && !event.code.startsWith("Arrow") || editing(event.target)) return;
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
  root.addEventListener("keydown", onKey);
  return () => root.removeEventListener("keydown", onKey);
}

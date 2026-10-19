import { micromark } from "micromark";
import { useEffect, useRef, useState } from "react";
import { copyText } from "./copy-text";
import { chatLayerOpen, keyForChat } from "./chat-host";

type CodeBlock = { code: string; language: string };
const NOTICE_MS = 1800;

// 用 Markdown 解析器取出回复里显示成代码块的部分（含引用、列表里的围栏与缩进代码），与页面上看到的同一组块；
// 自己按行认围栏不懂容器前缀，会跳过引用里的最后一块、静默复制更早的代码。micromark 输出已转义，模板不挂到页面。
export function codeBlocks(markdown: string): CodeBlock[] {
  const template = document.createElement("template");
  template.innerHTML = micromark(markdown);
  return Array.from(template.content.querySelectorAll("pre > code"), (code) => ({
    code: (code.textContent ?? "").replace(/\n$/u, ""),
    language: Array.from(code.classList).find((name) => name.startsWith("language-"))?.slice(9) ?? "",
  }));
}

// 从最新的回复往前找，第一条带代码块的回复里的最后一块。
function lastCodeBlock(replies: readonly string[]): CodeBlock | null {
  for (let index = replies.length - 1; index >= 0; index -= 1) {
    const blocks = codeBlocks(replies[index]);
    if (blocks.length) return blocks[blocks.length - 1];
  }
  return null;
}

function noticeFor(block: CodeBlock | null): string {
  if (!block) return "对话里还没有代码块";
  const lines = block.code.split("\n").length;
  return `已复制最后一个代码块${block.language ? `（${block.language}，${lines} 行）` : `（${lines} 行）`}`;
}

/** Ctrl/⌘ + Shift + ; 复制对话里最后一个代码块（写代码时最常见的"把它拿走"，ChatGPT 同一组合）。
 *  对话框、菜单、选择器打开或输入法组合中不接；结果在输入卡上方用一行状态提示，读屏同样播报。 */
export function useCopyLastCode(replies: readonly string[], onError: (error: unknown) => void): string {
  const [notice, setNotice] = useState("");
  const latest = useRef(replies);
  latest.current = replies;
  useEffect(() => {
    let timer = 0;
    const onKeyDown = (event: KeyboardEvent) => {
      if (!(event.metaKey || event.ctrlKey) || !event.shiftKey || event.altKey || event.isComposing) return;
      if (event.code !== "Semicolon" || !keyForChat(event)) return;
      if (chatLayerOpen()) return;
      event.preventDefault();
      // 1. 找代码块；没有也给提示，免得按了没反应
      const block = lastCodeBlock(latest.current);
      const show = (text: string) => {
        setNotice(text);
        window.clearTimeout(timer);
        timer = window.setTimeout(() => setNotice(""), NOTICE_MS);
      };
      if (!block) { show(noticeFor(null)); return; }
      // 2. 复制成功才报"已复制"，失败交给统一错误行
      void copyText(block.code).then(() => show(noticeFor(block))).catch(onError);
    };
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      window.clearTimeout(timer);
    };
  }, [onError]);
  return notice;
}

/** 状态行常驻（空时不占位、不显示），文字变化才会被读屏播报。 */
export function CopyNotice({ text }: { text: string }) {
  return <p className="copy-notice" role="status">{text}</p>;
}

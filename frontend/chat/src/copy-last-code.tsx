import { useEffect, useRef, useState } from "react";
import { copyText } from "./copy-text";

type CodeBlock = { code: string; language: string };
type OpenFence = { indent: number; fence: string; language: string; lines: string[] };

const FENCE_OPEN = /^( {0,3})(`{3,}|~{3,})(.*)$/u;
const FENCE_CLOSE = /^ {0,3}(`{3,}|~{3,})[ \t]*$/u;
const NOTICE_MS = 1800;

// 按 CommonMark 围栏规则取出 Markdown 里的代码块：闭合须同字符、不短于开栏且其后只有空白，没闭合的延续到文末；
// 开栏缩进几格，正文就去掉至多几格。反引号围栏的信息串里不能再有反引号。
export function fencedCodeBlocks(markdown: string): CodeBlock[] {
  const blocks: CodeBlock[] = [];
  let open: OpenFence | null = null;
  for (const line of markdown.split("\n")) {
    if (open) {
      const close = FENCE_CLOSE.exec(line);
      if (close && close[1][0] === open.fence[0] && close[1].length >= open.fence.length) {
        blocks.push({ code: open.lines.join("\n"), language: open.language });
        open = null;
      } else open.lines.push(line.replace(new RegExp(`^ {0,${open.indent}}`, "u"), ""));
      continue;
    }
    const start = FENCE_OPEN.exec(line);
    if (!start || (start[2][0] === "`" && start[3].includes("`"))) continue;
    open = { indent: start[1].length, fence: start[2], language: start[3].trim().split(/\s+/u)[0] ?? "", lines: [] };
  }
  if (open) blocks.push({ code: open.lines.join("\n"), language: open.language });
  return blocks;
}

// 从最新的回复往前找，第一条带代码块的回复里的最后一块。
function lastCodeBlock(replies: readonly string[]): CodeBlock | null {
  for (let index = replies.length - 1; index >= 0; index -= 1) {
    const blocks = fencedCodeBlocks(replies[index]);
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
      if (event.code !== "Semicolon") return;
      if (document.querySelector("[role='dialog'], [role='menu'], [role='listbox'], [data-radix-popper-content-wrapper]")) return;
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

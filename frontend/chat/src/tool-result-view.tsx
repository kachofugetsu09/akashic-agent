import { Check, X } from "lucide-react";
import { type ReactNode, type RefObject, useLayoutEffect, useMemo, useRef, useState } from "react";
import { readToolData } from "@/message-rendering-policy";

const PREVIEW_LINES = 12;
// 只多出几行时直接全显，避免“显示全部”只换来两三行。
const PREVIEW_SLACK = 3;
const MOTION_EASE = "cubic-bezier(0.2, 0, 0, 1)";
// 吸顶收起后淡入下方内容时最多处理的元素数，长会话里不逐条动画整页。
const FOLLOWING_FADE_LIMIT = 24;

export function formatToolDuration(durationMs: number): string {
  if (durationMs < 1_000) return `${Math.max(1, Math.round(durationMs))}ms`;
  return `${(durationMs / 1_000).toFixed(durationMs < 10_000 ? 1 : 0).replace(/\.0$/, "")}s`;
}

// 脚本动画统一入口；系统要求减少动态时不播放，CSS 过渡由媒体查询另行关闭。
function motion(node: Element | null, frames: Keyframe[], duration: number): void {
  if (!node || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  node.animate(frames, { duration, easing: MOTION_EASE });
}

// 最近的纵向滚动祖先；没有时退回文档滚动。
function scrollParent(node: Element): Element | null {
  for (let current = node.parentElement; current; current = current.parentElement) {
    const { overflowY } = window.getComputedStyle(current);
    if ((overflowY === "auto" || overflowY === "scroll") && current.scrollHeight > current.clientHeight) return current;
  }
  return document.scrollingElement;
}

// 内容高度骤变后把 node 滚回原来的视口位置，视线不跳。
function keepAt(node: Element | null, before: number): void {
  if (!node) return;
  const delta = node.getBoundingClientRect().top - before;
  if (Math.abs(delta) < 1) return;
  const scroller = scrollParent(node);
  if (scroller) scroller.scrollTop += delta;
}

// 从行向外逐层收集其后的兄弟元素，直到消息行为止。
function followingElements(row: Element): Element[] {
  const result: Element[] = [];
  for (let node: Element | null = row.parentElement; node && result.length < FOLLOWING_FADE_LIMIT; node = node.parentElement) {
    for (let next = node.nextElementSibling; next && result.length < FOLLOWING_FADE_LIMIT; next = next.nextElementSibling) result.push(next);
    if (node.classList.contains("message-row")) break;
  }
  return result;
}

export interface StickyDisclosure {
  open: boolean;
  instant: boolean;
  rowRef: RefObject<HTMLDivElement | null>;
  toggle: () => void;
}

// 展开项的标题行吸顶，读到哪里都能收起；吸顶时的收起瞬间完成并把行放回吸顶位置，再淡入下方内容。
export function useStickyDisclosure(initial = false): StickyDisclosure {
  const [open, setOpen] = useState(initial);
  const [instant, setInstant] = useState(false);
  const rowRef = useRef<HTMLDivElement>(null);
  const pendingTop = useRef<number | null>(null);
  const toggle = () => {
    const row = rowRef.current;
    // 1. 行离开了所在容器的顶部，说明正吸在视口上；这时高度过渡会把行带走，改为瞬间收起
    const stuck = open && row !== null && row.parentElement !== null
      && row.getBoundingClientRect().top - row.parentElement.getBoundingClientRect().top > 1;
    pendingTop.current = stuck ? row.getBoundingClientRect().top : null;
    setInstant(stuck);
    setOpen(!open);
  };
  useLayoutEffect(() => {
    const before = pendingTop.current;
    const row = rowRef.current;
    if (before === null || row === null) return;
    pendingTop.current = null;
    // 2. 收起已提交到 DOM：行回到吸顶位置，下一帧恢复过渡，下方内容从 8px 下方淡入
    keepAt(row, before);
    const frame = window.requestAnimationFrame(() => setInstant(false));
    for (const node of followingElements(row)) {
      motion(node, [{ opacity: 0, translate: "0 8px" }, { opacity: 1, translate: "0 0" }], 240);
    }
    return () => window.cancelAnimationFrame(frame);
  }, [open]);
  return { open, instant, rowRef, toggle };
}

// 展开区：grid 行 0fr → 1fr 过渡高度，内层淡入；内层只用 overflow: clip，嵌套行的 sticky 仍对滚动容器生效。
export function Disclosure({ open, instant = false, children }: {
  open: boolean;
  instant?: boolean;
  children: ReactNode;
}) {
  return (
    <div
      className={`tool-step-disclosure${open ? " open" : ""}${instant ? " is-instant" : ""}`}
      aria-hidden={!open}
      inert={!open ? true : undefined}
    >
      <div className="tool-step-disclosure-inner">{children}</div>
    </div>
  );
}

// 长文本先给预览：命令输出看末尾，其它看开头；显示全部原地铺开，流里不出现内部滚动。
function PreviewText({ text, tail, className }: { text: string; tail: boolean; className: string }) {
  const lines = useMemo(() => text.split("\n"), [text]);
  const [full, setFull] = useState(false);
  const preRef = useRef<HTMLPreElement>(null);
  const moreRef = useRef<HTMLButtonElement>(null);
  const pending = useRef<{ height: number; top: number } | null>(null);
  useLayoutEffect(() => {
    const before = pending.current;
    if (before === null) return;
    pending.current = null;
    // 1. 显示全部：输出区从预览高度长到全文高度；2. 收回预览：按钮留在原位，预览轻微淡入
    if (full) motion(preRef.current, [{ height: `${before.height}px` }, { height: `${preRef.current?.offsetHeight ?? 0}px` }], 360);
    else {
      keepAt(moreRef.current, before.top);
      motion(preRef.current, [{ opacity: 0.4 }, { opacity: 1 }], 200);
    }
  }, [full]);
  if (lines.length <= PREVIEW_LINES + PREVIEW_SLACK) return <pre className={className}>{text}</pre>;
  const shown = full ? text : (tail ? lines.slice(-PREVIEW_LINES) : lines.slice(0, PREVIEW_LINES)).join("\n");
  const clipped = full ? "" : tail ? " is-clipped-head" : " is-clipped-tail";
  const moreFirst = tail && !full;
  const more = <button ref={moreRef} type="button" className="tool-result-more" onClick={() => {
    pending.current = { height: preRef.current?.offsetHeight ?? 0, top: moreRef.current?.getBoundingClientRect().top ?? 0 };
    setFull(!full);
  }}>
    {full ? `收起到${tail ? "末尾" : "前"} ${PREVIEW_LINES} 行` : `显示全部 · ${lines.length} 行`}
  </button>;
  return <div className="tool-result-preview">
    {moreFirst ? more : null}
    <pre ref={preRef} className={`${className}${clipped}`}>{shown}</pre>
    {moreFirst ? null : more}
  </div>;
}

interface ShellResult {
  command: string;
  output: string;
  exitCode?: number;
  wallTimeMs?: number;
  status?: string;
  rest: Record<string, unknown>;
}

// 识别命令执行结果：至少带命令与输出字符串；其余字段收进原始数据。
function readShellResult(value: unknown): ShellResult | null {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return null;
  const { command, output, exit_code: exitCode, wall_time_ms: wallTimeMs, process_status: status, ...rest } = value as Record<string, unknown>;
  if (typeof command !== "string" || typeof output !== "string") return null;
  return {
    command,
    output,
    exitCode: typeof exitCode === "number" ? exitCode : undefined,
    wallTimeMs: typeof wallTimeMs === "number" ? wallTimeMs : undefined,
    status: typeof status === "string" ? status : undefined,
    rest: { ...rest, ...(typeof status === "string" ? { process_status: status } : {}) },
  };
}

// 命令块：$ 命令 → 输出（预览看末尾）→ 底栏结果与耗时。
function ShellResultView({ result }: { result: ShellResult }) {
  const failed = result.exitCode !== undefined && result.exitCode !== 0;
  return <>
    <div className="tool-shell">
      <div className="tool-shell-command"><span aria-hidden="true">$</span><span>{result.command}</span></div>
      {result.output.trim()
        ? <PreviewText text={result.output} tail className="tool-result tool-shell-output" />
        : <p className="tool-shell-empty">无输出</p>}
      <div className="tool-shell-foot">
        {result.exitCode === undefined ? <span>{result.status ?? "运行中"}</span>
          : <span className={`tool-shell-exit${failed ? " is-failed" : ""}`}>
            {failed ? <X size={13} aria-hidden="true" /> : <Check size={13} aria-hidden="true" />}
            {failed ? `退出码 ${result.exitCode}` : "成功"}
          </span>}
        {result.wallTimeMs !== undefined ? <span>{formatToolDuration(result.wallTimeMs)}</span> : null}
      </div>
    </div>
    {Object.keys(result.rest).length ? <details className="tool-result-raw">
      <summary>原始数据</summary>
      <ToolDataValue value={result.rest} />
    </details> : null}
  </>;
}

/** 所有工具共用字面文本和数据展示；复制仍使用原始值。 */
export function ToolResultContent({ value, error = false }: { value: unknown; error?: boolean }) {
  const parsed = useMemo(() => readToolData(value), [value]);
  const shell = useMemo(() => readShellResult(parsed), [parsed]);
  return <div className={`tool-result-data${error ? " error" : ""}`}>
    {shell ? <ShellResultView result={shell} /> : <ToolDataValue value={parsed} />}
  </div>;
}

/** 结构可以展开；标量始终保留原始文字，不解释 Markdown 或 HTML。 */
function ToolDataValue({ value }: { value: unknown }) {
  if (value !== null && typeof value === "object") {
    const entries = Object.entries(value);
    if (!entries.length) return <pre className="tool-result">{Array.isArray(value) ? "[]" : "{}"}</pre>;
    return <dl className="tool-result-fields">{entries.map(([name, item]) => <div key={name}>
      <dt>{name}</dt>
      <dd>{Array.isArray(value) ? <ToolResultContent value={item} /> : item !== null && typeof item === "object"
        ? <details><summary>展开数据</summary><ToolDataValue value={item} /></details>
        : <PreviewText text={item === null ? "null" : String(item)} tail={false} className="tool-result" />}</dd>
    </div>)}</dl>;
  }
  return <PreviewText text={value === null ? "null" : String(value)} tail={false} className="tool-result" />;
}

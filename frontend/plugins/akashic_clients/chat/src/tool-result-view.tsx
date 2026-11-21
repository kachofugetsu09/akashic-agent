import { type ReactNode, type RefObject, type TransitionEvent, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";
import { readToolData } from "@/message-rendering-policy";
import { PluginUiSlot, pluginSlotAccepts, usePluginUiCatalogVersion } from "./plugin-ui-runtime";
import { flipDisclosure, settle, watchHeight } from "./motion";

const PREVIEW_LINES = 12;
// 只多出几行时直接全显，避免“显示全部”只换来两三行。
const PREVIEW_SLACK = 3;
const COLLAPSE_FALLBACK_MS = 220;

export function formatToolDuration(durationMs: number): string {
  if (durationMs < 1_000) return `${Math.max(1, Math.round(durationMs))}ms`;
  return `${(durationMs / 1_000).toFixed(durationMs < 10_000 ? 1 : 0).replace(/\.0$/, "")}s`;
}

export function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

// 最近的纵向滚动祖先；没有时退回文档滚动。
export function scrollParent(node: Element): Element | null {
  for (let current = node.parentElement; current; current = current.parentElement) {
    const { overflowY } = window.getComputedStyle(current);
    if ((overflowY === "auto" || overflowY === "scroll") && current.scrollHeight > current.clientHeight) return current;
  }
  return document.scrollingElement;
}

// 内容高度变化后把 node 留在原来的视口位置，视线不跳。
export function keepAt(node: Element | null, before: number): void {
  if (!node) return;
  const delta = node.getBoundingClientRect().top - before;
  if (Math.abs(delta) < 0.1) return;
  const scroller = scrollParent(node);
  if (scroller) scroller.scrollTop += delta;
}

export interface StickyDisclosure {
  open: boolean;
  sticky: boolean;
  rowRef: RefObject<HTMLDivElement | null>;
  toggle: () => void;
  setOpen: (open: boolean) => void;
  motion: DisclosureMotion;
}

// 收起的两个时点：onCollapse 在布局收起后立刻调用（滚动修正，量终点时也会调用）；onSettled 在收起动画结束、落位之后。
export interface DisclosureMotion {
  onCollapse: () => void;
  onSettled: () => void;
}

// 展开项的标题行吸顶，读到哪里都能收起；收起动画期间保持吸顶，落位时把吸住的标题行留在原来的视口位置。
export function useStickyDisclosure(initial = false): StickyDisclosure {
  const [open, setOpenState] = useState(initial);
  const [closing, setClosing] = useState(false);
  const openRef = useRef(initial);
  const rowRef = useRef<HTMLDivElement>(null);
  const collapseAnchorRef = useRef<number | null>(null);

  useLayoutEffect(() => {
    openRef.current = open;
  }, [open]);

  const setOpen = useCallback((next: boolean) => {
    if (openRef.current === next) return;
    openRef.current = next;
    if (next) {
      collapseAnchorRef.current = null;
      setClosing(false);
      setOpenState(true);
      return;
    }
    const row = rowRef.current;
    const stuck = row !== null && row.parentElement !== null
      && row.getBoundingClientRect().top - row.parentElement.getBoundingClientRect().top > 1;
    collapseAnchorRef.current = stuck ? row.getBoundingClientRect().top : null;
    setClosing(true);
    setOpenState(false);
  }, []);

  const toggle = useCallback(() => {
    setOpen(!openRef.current);
  }, [setOpen]);

  const motion = useMemo<DisclosureMotion>(() => ({
    onCollapse: () => {
      if (collapseAnchorRef.current !== null) keepAt(rowRef.current, collapseAnchorRef.current);
    },
    onSettled: () => {
      collapseAnchorRef.current = null;
      setClosing(false);
    },
  }), []);

  return { open, sticky: open || closing, rowRef, toggle, setOpen, motion };
}

// 消息区里会吸顶的元素：展开区开合时不能把 transform 加在吸住元素的祖先上。
const PINNED = ".tool-step-sticky.is-open, .process-trigger-sticky, .code-block-header, .reading-code-stream-label";

// 展开区：常驻 DOM，开合只切 open（grid 行 0fr / 1fr，不过渡）。class 由这里在布局效果里切，
// 切之前才量得到起点；位移与透明度走 motion.ts 的 FLIP，收起先播动画、结束时再切 class。
export function Disclosure({ open, children, motion }: {
  open: boolean;
  children: ReactNode;
  motion?: DisclosureMotion;
}) {
  const [initial] = useState(open);
  const hostRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const shownRef = useRef(open);
  const animationsRef = useRef<Animation[]>([]);
  const motionRef = useRef(motion);
  useLayoutEffect(() => {
    motionRef.current = motion;
  }, [motion]);

  useLayoutEffect(() => {
    const host = hostRef.current;
    const content = contentRef.current;
    if (!host || !content || shownRef.current === open) return;
    shownRef.current = open;
    const apply = (next: boolean) => {
      host.classList.toggle("open", next);
      if (!next) motionRef.current?.onCollapse();
    };
    const finish = () => {
      apply(false);
      settle(animationsRef.current);
      animationsRef.current = [];
      motionRef.current?.onSettled();
    };
    const boundary = host.closest<HTMLElement>(".conversation-scroll") ?? scrollParent(host);
    const animations = boundary instanceof HTMLElement
      ? flipDisclosure({ host, content, boundary, pinned: PINNED, open, apply }, animationsRef.current)
      : (settle(animationsRef.current), apply(open), []);
    animationsRef.current = animations;
    const last = animations.at(-1);
    if (open) {
      // 展开播完就放掉这批动画的引用。
      void last?.finished.then(() => { if (animationsRef.current === animations) animationsRef.current = []; }, () => undefined);
      return;
    }
    if (!last) return finish();
    void last.finished.then(() => { if (animationsRef.current === animations) finish(); }, () => undefined);
    // 收起途中内容又变了高（流式追加、结果区重排）：终点已经不准，直接落位，不留跳位的末帧。
    watchHeight(host, last, () => { if (animationsRef.current === animations) finish(); });
  }, [open]);
  useEffect(() => () => settle(animationsRef.current), []);

  return (
    <div
      ref={hostRef}
      className={`tool-step-disclosure${initial ? " open" : ""}`}
      aria-hidden={!open}
      inert={!open ? true : undefined}
    >
      <div ref={contentRef} className="tool-step-disclosure-inner">{children}</div>
    </div>
  );
}

// 长文本先给预览：按显示高度截（几行很长、折成几十行的也算），命令输出看末尾，其它看开头；
// 显示全部与收起都在原位平滑过渡，流里不出现内部滚动。
export function PreviewText({ text, tail, className }: { text: string; tail: boolean; className: string }) {
  const lines = useMemo(() => text.split("\n"), [text]);
  const [full, setFull] = useState(false);
  const [closing, setClosing] = useState(false);
  // 行数不多但折行后超高：渲染后按行高量一次，宽度变化（折行跟着变）时重量。
  const [tall, setTall] = useState(false);
  const many = lines.length > PREVIEW_LINES + PREVIEW_SLACK;
  const preRef = useRef<HTMLPreElement>(null);
  const moreRef = useRef<HTMLButtonElement>(null);
  const pendingRef = useRef<{ height: number; expanding: boolean } | null>(null);
  const anchorTopRef = useRef<number | null>(null);
  const collapsedHeightRef = useRef<number | null>(null);

  useLayoutEffect(() => {
    const pre = preRef.current;
    if (!pre || many) return;
    const measure = () => {
      const line = parseFloat(getComputedStyle(pre).lineHeight) || 18;
      setTall(pre.scrollHeight > line * (PREVIEW_LINES + PREVIEW_SLACK));
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(pre);
    return () => observer.disconnect();
  }, [many, text]);

  useLayoutEffect(() => {
    const pre = preRef.current;
    if (!pre) return;
    if (!full && !closing) {
      pre.style.maxHeight = "";
      pre.style.transition = "";
      // 看末尾的预览：截掉的是开头，折行后超出的部分也从上方让出。
      if (tail) pre.scrollTop = pre.scrollHeight;
    }
    const before = pendingRef.current;
    if (!before) return;
    pendingRef.current = null;
    if (prefersReducedMotion()) {
      pre.style.maxHeight = "";
      pre.style.transition = "";
      if (!before.expanding) {
        setClosing(false);
        if (anchorTopRef.current !== null) keepAt(moreRef.current, anchorTopRef.current);
        anchorTopRef.current = null;
      }
      return;
    }
    const targetHeight = before.expanding
      ? pre.scrollHeight
      : (collapsedHeightRef.current ?? Math.min(before.height, Math.ceil((parseFloat(getComputedStyle(pre).lineHeight) || 18) * PREVIEW_LINES)));
    pre.style.transition = "none";
    pre.style.maxHeight = `${before.height}px`;
    void pre.offsetHeight;
    // motion-exempt: 工具结果预览的高度过渡：合成写法需要盖板与不透明底色，ADR 0166 已知限制
    pre.style.transition = `max-height ${before.expanding ? "var(--ak-sys-duration-expand)" : "var(--ak-sys-duration-snap)"} var(--ak-sys-motion-standard)`;
    pre.style.maxHeight = `${targetHeight}px`;
  }, [full, closing, tail, tall]);

  // 看末尾的预览在阅读列变宽变窄后重新折行，要跟着贴回末尾；只改滚动位置，不会再触发尺寸变化。
  useEffect(() => {
    const pre = preRef.current;
    if (!pre || !tail || full || closing) return;
    const observer = new ResizeObserver(() => { pre.scrollTop = pre.scrollHeight; });
    observer.observe(pre);
    return () => observer.disconnect();
  }, [tail, full, closing]);

  useEffect(() => {
    if (!closing) return;
    const pre = preRef.current;
    const observer = !tail && pre ? new ResizeObserver(() => {
      if (anchorTopRef.current !== null) keepAt(moreRef.current, anchorTopRef.current);
    }) : null;
    if (pre && observer) observer.observe(pre);
    const timer = window.setTimeout(() => setClosing(false), COLLAPSE_FALLBACK_MS);
    return () => {
      observer?.disconnect();
      window.clearTimeout(timer);
      anchorTopRef.current = null;
    };
  }, [closing, tail]);

  const diff = useMemo(() => looksLikeDiff(lines), [lines]);
  // diff 按行对照着读，折行会把一行增删拆散：整块不折行，在结果区里横向滚动。
  const kind = diff ? `${className} is-diff` : className;
  const clippable = many || tall;
  const renderFull = full || closing || !many;
  const visible = renderFull ? lines : tail ? lines.slice(-PREVIEW_LINES) : lines.slice(0, PREVIEW_LINES);
  const shown = diff ? <DiffLines lines={visible} /> : visible.join("\n");
  const clipped = full || !clippable ? "" : tail ? " is-clipped-head" : " is-clipped-tail";
  const moreFirst = tail && !full;
  const onToggleMore = () => {
    const pre = preRef.current;
    const currentHeight = pre?.getBoundingClientRect().height ?? 0;
    if (!full && !closing && pre) collapsedHeightRef.current = currentHeight;
    pendingRef.current = { height: currentHeight, expanding: !full };
    anchorTopRef.current = moreRef.current?.getBoundingClientRect().top ?? null;
    if (full) {
      setClosing(!prefersReducedMotion());
      setFull(false);
    } else {
      setClosing(false);
      setFull(true);
    }
  };
  const onPreTransitionEnd = (event: TransitionEvent<HTMLPreElement>) => {
    if (event.target !== event.currentTarget || event.propertyName !== "max-height") return;
    if (!full) setClosing(false);
    else if (preRef.current) {
      preRef.current.style.maxHeight = "";
      preRef.current.style.transition = "";
    }
  };
  const more = clippable ? <button ref={moreRef} type="button" className="tool-result-more" aria-expanded={full} onClick={onToggleMore}>
    {full ? "收起" : `显示全部 · ${lines.length} 行`}
  </button> : null;
  return <div className="tool-result-preview">
    {moreFirst ? more : null}
    <pre ref={preRef} className={`${kind}${clipped}`} onTransitionEnd={onPreTransitionEnd}>{shown}</pre>
    {moreFirst ? null : more}
  </div>;
}

// 编辑类工具常把统一 diff 原样回给模型（有时还包在 ```diff 围栏里）。只按行首给增删着色，
// 不解释 Markdown；文字与复制内容都保持原样。
function looksLikeDiff(lines: string[]): boolean {
  return lines.some((line) => line.startsWith("@@ ")) && lines.some((line) => line.startsWith("+++ ") || line.startsWith("--- "));
}

export function DiffLines({ lines }: { lines: string[] }) {
  return lines.map((line, index) => <span key={index} className={`tool-diff-line${diffLineKind(line)}`}>
    {line}{index < lines.length - 1 ? "\n" : null}
  </span>);
}

function diffLineKind(line: string): string {
  if (line.startsWith("+++ ") || line.startsWith("--- ") || line.startsWith("```")) return " is-meta";
  if (line.startsWith("@@")) return " is-hunk";
  if (line.startsWith("+")) return " is-add";
  if (line.startsWith("-")) return " is-del";
  return "";
}

/** 一次调用的身份：结果只交给注册该工具的插件（owner）渲染。 */
export interface ToolIdentity {
  owner?: string;
  name: string;
  arguments: unknown;
}

/** 所有工具共用字面文本和数据展示；工具 owner 的插件认领时改由它渲染，复制仍使用原始值。 */
export function ToolResultContent({ value, error = false, tool }: { value: unknown; error?: boolean; tool?: ToolIdentity }) {
  const parsed = useMemo(() => readToolData(value), [value]);
  usePluginUiCatalogVersion();
  const generic = <ToolDataValue value={parsed} />;
  const facts = tool?.owner ? { name: tool.name, arguments: tool.arguments, value: parsed, error } : null;
  return <div className={`tool-result-data${error ? " error" : ""}`}>
    {facts && pluginSlotAccepts("tool.result", facts, tool?.owner)
      ? <PluginUiSlot name="tool.result" owner={tool?.owner} block={facts} quiet fallback={generic} />
      : generic}
  </div>;
}

/** 结构可以展开；标量始终保留原始文字，不解释 Markdown 或 HTML。 */
export function ToolDataValue({ value }: { value: unknown }) {
  if (value !== null && typeof value === "object") {
    const entries = Object.entries(value);
    if (!entries.length) return <pre className="tool-result">{Array.isArray(value) ? "[]" : "{}"}</pre>;
    return <dl className="tool-result-fields">{entries.map(([name, item]) => <div key={name}>
      <dt>{name}</dt>
      <dd>{Array.isArray(value) ? <ToolResultContent value={item} /> : item !== null && typeof item === "object"
        ? <NestedDataDisclosure value={item} />
        : <PreviewText text={item === null ? "null" : String(item)} tail={false} className="tool-result" />}</dd>
    </div>)}</dl>;
  }
  return <PreviewText text={value === null ? "null" : String(value)} tail={false} className="tool-result" />;
}

function NestedDataDisclosure({ value }: { value: unknown }) {
  const [open, setOpen] = useState(false);
  return <div className="tool-result-nested" data-disclosure-root="">
    <button type="button" className="tool-result-nested__trigger" data-disclosure-trigger="" aria-expanded={open} onClick={() => setOpen((current) => !current)}>
      <ChevronDown className={`tool-result-nested__chevron${open ? " open" : ""}`} size={12} aria-hidden="true" />
      <span>展开数据</span>
    </button>
    <Disclosure open={open}><ToolDataValue value={value} /></Disclosure>
  </div>;
}

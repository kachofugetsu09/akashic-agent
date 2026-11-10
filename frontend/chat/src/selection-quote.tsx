import { Quote } from "lucide-react";
import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { chatPortalRoot } from "./chat-layout";

const POINTER_FINE = "(hover: hover) and (pointer: fine)";
const MESSAGE = ".web-message-anchor";
const BUTTON_GAP = 8;
const BUTTON_HEIGHT = 28;
const EDITABLE = "input, textarea, [contenteditable]";

type Anchor = { left: number; top: number; below: boolean };

// 在一条消息里选中一段文字后，选区上方浮出"引用"：把选中内容作为 Markdown 引用块接到草稿后面，针对一句追问。
// 只在鼠标等精细指针下出现；触屏有系统自带的选择菜单，不再叠一层。整条引用仍走消息操作行的"引用此消息"。
export function SelectionQuote({ onQuote }: { onQuote: (markdown: string) => void }) {
  const [anchor, setAnchor] = useState<Anchor | null>(null);

  // 必要 effect：选区与滚动都是 document 级事件，只能在这里订阅。
  useEffect(() => {
    if (!window.matchMedia(POINTER_FINE).matches) return;
    const measure = () => {
      const lines = selectionLines();
      setAnchor(lines ? placeNear(lines) : null);
    };
    const hide = () => setAnchor(null);
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") hide();
      else if (event.shiftKey) measure();
    };
    // 高亮代码在 shadow DOM 里，document 侧的选区会被折叠成宿主外的一点：按文字是否为空判断，而不是 isCollapsed。
    const onSelectionChange = () => {
      if (!window.getSelection()?.toString().trim()) hide();
    };
    document.addEventListener("mouseup", measure);
    document.addEventListener("keyup", onKey);
    document.addEventListener("selectionchange", onSelectionChange);
    document.addEventListener("scroll", hide, { capture: true, passive: true });
    window.addEventListener("resize", hide);
    return () => {
      document.removeEventListener("mouseup", measure);
      document.removeEventListener("keyup", onKey);
      document.removeEventListener("selectionchange", onSelectionChange);
      document.removeEventListener("scroll", hide, { capture: true });
      window.removeEventListener("resize", hide);
    };
  }, []);

  if (!anchor) return null;
  return createPortal(<button type="button" className={`selection-quote${anchor.below ? " is-below" : ""}`}
    style={{ left: anchor.left, top: anchor.top }}
    // 按下时不能让按钮抢走焦点、清掉选区，否则点击时已经读不到选中的文字。
    onMouseDown={(event) => event.preventDefault()}
    onClick={() => {
      const text = window.getSelection()?.toString().trim() ?? "";
      setAnchor(null);
      if (!text) return;
      window.getSelection()?.removeAllRanges();
      onQuote(`${text.split(/\r?\n/u).map((line) => `> ${line}`.trimEnd()).join("\n")}\n\n`);
    }}>
    <Quote size={14} aria-hidden="true" />
    <span>引用</span>
  </button>, chatPortalRoot());
}

type Endpoints = Pick<StaticRange, "startContainer" | "startOffset" | "endContainer" | "endOffset">;

// 取当前选区的行矩形：有文字、起止在同一条消息里、不在可编辑区域。高亮代码在 shadow DOM 里，要带上 shadow root 才取得到真实端点。
function selectionLines(): DOMRect[] | null {
  const selection = window.getSelection();
  if (!selection?.rangeCount || !selection.toString().trim()) return null;
  // 1. 普通 DOM 直接用；端点被折叠到宿主外时，改用 composed range 的真实端点（可能跨 shadow 边界）。
  const range = selection.getRangeAt(0);
  const ends: Endpoints | null = range.collapsed ? composedRange(selection) : range;
  if (!ends) return null;
  // 2. 起止都要落在同一条消息里（越过 shadow 边界向上找），且不在可编辑区域。
  const message = closestComposed(ends.startContainer, MESSAGE);
  if (!message || message !== closestComposed(ends.endContainer, MESSAGE)) return null;
  if (closestComposed(ends.startContainer, EDITABLE) || closestComposed(ends.endContainer, EDITABLE)) return null;
  // 3. 取逐行矩形；跨树（代码块连着后面的正文）普通 Range 存不下，按树切段分别测量。
  return treeSegments(ends).flatMap((segment) => [...segment.getClientRects()]);
}

// 把可能跨 shadow 边界的选区切成各树内的 Range：起点所在 shadow 的剩余部分、两个宿主之间的文档部分、终点所在 shadow 的开头部分。
// 只处理一层 shadow（高亮代码块的宿主都在文档里）；长选区两端都在视口外时，中间可见的行照样量得到。
function treeSegments(ends: Endpoints): Range[] {
  const startRoot = ends.startContainer.getRootNode();
  const endRoot = ends.endContainer.getRootNode();
  const segment = () => document.createRange();
  if (startRoot === endRoot) {
    const whole = segment();
    whole.setStart(ends.startContainer, ends.startOffset);
    whole.setEnd(ends.endContainer, ends.endOffset);
    return [whole];
  }
  const startHost = startRoot instanceof ShadowRoot ? startRoot.host : null;
  const endHost = endRoot instanceof ShadowRoot ? endRoot.host : null;
  const segments: Range[] = [];
  if (startRoot instanceof ShadowRoot) {
    const head = segment();
    head.setStart(ends.startContainer, ends.startOffset);
    head.setEnd(startRoot, startRoot.childNodes.length);
    segments.push(head);
  }
  const middle = segment();
  if (startHost) middle.setStartAfter(startHost); else middle.setStart(ends.startContainer, ends.startOffset);
  if (endHost) middle.setEndBefore(endHost); else middle.setEnd(ends.endContainer, ends.endOffset);
  segments.push(middle);
  if (endRoot instanceof ShadowRoot) {
    const tail = segment();
    tail.setStart(endRoot, 0);
    tail.setEnd(ends.endContainer, ends.endOffset);
    segments.push(tail);
  }
  return segments;
}

// 现行签名传 { shadowRoots }；旧签名（MDN 记载的早期实现）以剩余参数传 shadow root，传对象会抛 TypeError。
function composedRange(selection: Selection): StaticRange | null {
  if (typeof selection.getComposedRanges !== "function") return null;
  const shadowRoots = [...document.querySelectorAll("diffs-container")].flatMap((host) => host.shadowRoot ? [host.shadowRoot] : []);
  try {
    return selection.getComposedRanges({ shadowRoots })[0] ?? null;
  } catch (error) {
    if (!(error instanceof TypeError)) throw error;
  }
  const legacy = selection.getComposedRanges as unknown as (this: Selection, ...roots: ShadowRoot[]) => StaticRange[];
  try {
    return legacy.apply(selection, shadowRoots)[0] ?? null;
  } catch (error) {
    if (error instanceof TypeError) return null;
    throw error;
  }
}

// 按钮落在选区可见部分（标题条之下、输入卡之上）：优先首个可见行的上方；放不下就放到末个可见行下方；仍放不下时压回可见区。
function placeNear(rects: DOMRect[]): Anchor | null {
  const { top, bottom } = readingBounds();
  const lines = rects.filter((rect) => rect.height > 0 && rect.bottom > top && rect.top < bottom);
  const first = lines[0];
  const last = lines.at(-1);
  if (!first || !last) return null;
  const clampX = (rect: DOMRect) => Math.min(Math.max(rect.left + rect.width / 2, 48), window.innerWidth - 48);
  if (first.top - BUTTON_GAP - BUTTON_HEIGHT >= top) {
    return { left: clampX(first), top: first.top - BUTTON_GAP, below: false };
  }
  const below = Math.min(last.bottom + BUTTON_GAP, bottom - BUTTON_HEIGHT);
  return { left: clampX(last), top: Math.max(below, top), below: true };
}

// 可见阅读区：浮动标题条与输入卡都会盖住正文，按它们的实际几何扣掉（输入卡展开、矮视口文档流都随之变化）。
function readingBounds(): { top: number; bottom: number } {
  const heading = document.querySelector(".conversation-heading")?.getBoundingClientRect();
  const composer = document.querySelector(".chat-main .composer-wrap")?.getBoundingClientRect();
  const top = Math.max(heading?.bottom ?? 0, 0) + BUTTON_GAP;
  const bottom = Math.min(composer && composer.height > 0 ? composer.top : window.innerHeight, window.innerHeight) - BUTTON_GAP;
  return { top, bottom };
}

function closestComposed(node: Node, selector: string): Element | null {
  let element: Element | null = node instanceof Element ? node : node.parentElement;
  while (element) {
    const hit = element.closest(selector);
    if (hit) return hit;
    const root = element.getRootNode();
    element = root instanceof ShadowRoot ? root.host : null;
  }
  return null;
}

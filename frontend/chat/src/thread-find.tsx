import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore, type RefObject } from "react";
import { ChevronDown, ChevronUp, Search, X } from "lucide-react";
import { useStickToBottomContext } from "use-stick-to-bottom";
import { keyForChat } from "./chat-host";

// 会话内查找（Ctrl/⌘F）：窗口右上角一张卡片，只匹配看得见的文字，原文用 CSS Highlight 高亮，不改 DOM。
// 收起时内容不在 DOM 里的折叠（过程摘要、更早步骤、工具详情、思考）登记成区域，
// 区域用数据文本计数并排在原位；跳到区域里的命中时先展开它，离开后再收回。

const MATCH_LIMIT = 1000;
const QUERY_DELAY_MS = 150;
const HIT_HIGHLIGHT = "thread-find";
const ACTIVE_HIGHLIGHT = "thread-find-active";

interface FindRegion {
  container: HTMLElement;
  text: () => string;
  reveal: () => void;
  conceal: () => void;
}

type FindHit =
  | { kind: "text"; range: Range }
  | { kind: "region"; region: FindRegion; ordinal: number };

export interface FindSnapshot {
  open: boolean;
  draft: string;
  query: string;
  count: number;
  active: number;
  capped: boolean;
}

// 同一文档只有一个聊天：区域与查询放在模块里，折叠组件不必穿过整棵消息树拿上下文。
const regions = new Map<HTMLElement, FindRegion>();
let appliedQuery = "";
const queryListeners = new Set<() => void>();

function publishQuery(query: string): void {
  if (query === appliedQuery) return;
  appliedQuery = query;
  for (const listener of queryListeners) listener();
}

function subscribeQuery(listener: () => void): () => void {
  queryListeners.add(listener);
  return () => queryListeners.delete(listener);
}

function countIn(text: string, needle: string, limit = MATCH_LIMIT): number {
  if (!needle) return 0;
  const haystack = text.toLowerCase();
  let count = 0;
  for (let at = haystack.indexOf(needle); at >= 0 && count < limit; at = haystack.indexOf(needle, at + needle.length)) count += 1;
  return count;
}

// 折叠组件登记自己：收起期间用数据文本计数，返回的命中数显示在摘要行上。
export function useFindRegion(rowRef: RefObject<HTMLElement | null>, open: boolean, text: () => string,
  setOpen: (open: boolean) => void): number {
  const query = useSyncExternalStore(subscribeQuery, () => appliedQuery, () => "");
  const latest = useRef({ text, setOpen });
  latest.current = { text, setOpen };
  useEffect(() => {
    const container = rowRef.current?.parentElement;
    if (open || !container) return;
    regions.set(container, {
      container,
      text: () => latest.current.text(),
      reveal: () => latest.current.setOpen(true),
      conceal: () => latest.current.setOpen(false),
    });
    return () => { regions.delete(container); };
  }, [open, rowRef]);
  return useMemo(() => open || !query ? 0 : countIn(text(), query), [open, query, text]);
}

export function FindRegionCount({ count }: { count: number }) {
  return count ? <span className="find-region-count" data-find-skip="">· {count} 处匹配</span> : null;
}

// 不属于消息正文的节点：界面装饰、收起的详情、输入控件。
function skipped(element: Element): boolean {
  return element.hasAttribute("data-find-skip") || element.hasAttribute("inert")
    || element.getAttribute("aria-hidden") === "true"
    || element.matches("svg, style, script, textarea, input, .shared-message-meta");
}

// 按文档顺序收集命中：文字节点逐个匹配；进入收起的区域时先匹配它看得见的摘要行，离开时补上区域里的命中。
function collectHits(root: HTMLElement, needle: string): { hits: FindHit[]; capped: boolean } {
  const hits: FindHit[] = [];
  const open: FindRegion[] = [];
  const flush = (node: Node | null) => {
    while (open.length && (node === null || !open.at(-1)!.container.contains(node))) {
      const region = open.pop()!;
      const count = countIn(region.text(), needle, MATCH_LIMIT - hits.length);
      for (let ordinal = 0; ordinal < count; ordinal += 1) hits.push({ kind: "region", region, ordinal });
    }
  };
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT, {
    acceptNode: (node) => node instanceof Element && skipped(node) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT,
  });
  for (let node = walker.nextNode(); node && hits.length < MATCH_LIMIT; node = walker.nextNode()) {
    flush(node);
    const region = node instanceof HTMLElement ? regions.get(node) : undefined;
    if (region) { open.push(region); continue; }
    if (node.nodeType !== Node.TEXT_NODE) continue;
    const text = (node as Text).data.toLowerCase();
    for (let at = text.indexOf(needle); at >= 0 && hits.length < MATCH_LIMIT; at = text.indexOf(needle, at + needle.length)) {
      const range = document.createRange();
      range.setStart(node, at);
      range.setEnd(node, at + needle.length);
      hits.push({ kind: "text", range });
    }
  }
  flush(null);
  return { hits, capped: hits.length >= MATCH_LIMIT };
}

function sameHit(left: FindHit, right: FindHit): boolean {
  if (left.kind === "text" && right.kind === "text") {
    return left.range.startContainer === right.range.startContainer && left.range.startOffset === right.range.startOffset;
  }
  return left.kind === "region" && right.kind === "region" && left.region.container === right.region.container
    && left.ordinal === right.ordinal;
}

function paint(hits: FindHit[], active: FindHit | undefined): void {
  if (typeof CSS === "undefined" || !("highlights" in CSS)) return;
  const ranges = hits.flatMap((hit) => hit.kind === "text" && hit !== active ? [hit.range] : []);
  CSS.highlights.set(HIT_HIGHLIGHT, new Highlight(...ranges));
  if (active?.kind === "text") CSS.highlights.set(ACTIVE_HIGHLIGHT, new Highlight(active.range));
  else CSS.highlights.delete(ACTIVE_HIGHLIGHT);
}

function clearPaint(): void {
  if (typeof CSS === "undefined" || !("highlights" in CSS)) return;
  CSS.highlights.delete(HIT_HIGHLIGHT);
  CSS.highlights.delete(ACTIVE_HIGHLIGHT);
}

const nextFrame = () => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));

// 把命中放到「卡片以下、输入区和软键盘以上」那一段的中间。
function scrollToRange(scroller: HTMLElement, range: Range, card: HTMLElement | null): void {
  const owner = range.startContainer.parentElement;
  owner?.closest(".user-text-clamp[data-state=\"clamped\"]")?.querySelector<HTMLButtonElement>(".user-text-toggle")?.click();
  owner?.scrollIntoView({ block: "center", behavior: "instant" });
  const rect = range.getBoundingClientRect();
  const box = scroller.getBoundingClientRect();
  const viewport = window.visualViewport;
  const composer = document.querySelector(".composer-wrap")?.getBoundingClientRect().top ?? box.bottom;
  const top = Math.max(box.top, (card?.getBoundingClientRect().bottom ?? box.top) + 8);
  const bottom = Math.min(box.bottom, composer, viewport ? viewport.offsetTop + viewport.height : box.bottom);
  scroller.scrollTop += rect.top + rect.height / 2 - (top + Math.max(bottom, top + rect.height)) / 2;
}

class FindSession {
  snapshot: FindSnapshot = { open: false, draft: "", query: "", count: 0, active: 0, capped: false };
  private hits: FindHit[] = [];
  private revealed = new Set<FindRegion>();
  private listeners = new Set<() => void>();
  root: HTMLElement | null = null;
  scroller: HTMLElement | null = null;
  card: HTMLElement | null = null;
  stopScroll: () => void = () => {};

  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  };

  private emit(next: Partial<FindSnapshot>): void {
    this.snapshot = { ...this.snapshot, ...next };
    for (const listener of this.listeners) listener();
  }

  setOpen(open: boolean): void {
    if (open === this.snapshot.open) return;
    if (!open) {
      for (const region of this.revealed) region.conceal();
      this.revealed.clear();
      this.hits = [];
      publishQuery("");
      clearPaint();
    }
    this.emit(open ? { open } : { open, draft: "", query: "", count: 0, active: 0 });
  }

  setDraft(draft: string): void { this.emit({ draft }); }

  // 输入还没落定就按 Enter：先按当前输入查一次，从第一处开始。
  advance(delta: number): void {
    if (this.snapshot.draft === this.snapshot.query) { this.step(delta); return; }
    this.refresh(this.snapshot.draft, false);
    void this.go(0);
  }

  // 重新收集命中；保留当前那一处，流式新内容不改变位置。
  refresh(query: string = this.snapshot.query, keep = true): void {
    const needle = query.trim().toLowerCase();
    publishQuery(needle);
    const previous = keep ? this.hits[this.snapshot.active] : undefined;
    const { hits, capped } = this.root && needle ? collectHits(this.root, needle) : { hits: [], capped: false };
    const kept = previous ? hits.findIndex((hit) => sameHit(hit, previous)) : -1;
    const active = kept >= 0 ? kept : Math.min(keep ? this.snapshot.active : 0, Math.max(0, hits.length - 1));
    this.hits = hits;
    paint(hits, hits[active]);
    this.emit({ query, count: hits.length, active, capped });
  }

  // 1. 区域里的命中先展开区域，等渲染后按同一序号重新定位（嵌套折叠最多展开几层）
  // 2. 命中落到文字上：高亮、滚到可见段中间，收回不再包含它的、由查找展开的区域
  async go(index: number, depth = 0): Promise<void> {
    const count = this.hits.length;
    if (!count) return;
    const active = (index + count) % count;
    const hit = this.hits[active];
    this.emit({ active });
    if (hit.kind === "region") {
      if (depth > 4) return;
      hit.region.reveal();
      this.revealed.add(hit.region);
      await nextFrame();
      this.refresh(this.snapshot.query, false);
      return this.go(active, depth + 1);
    }
    paint(this.hits, hit);
    for (const region of this.revealed) {
      if (region.container.contains(hit.range.startContainer)) continue;
      region.conceal();
      this.revealed.delete(region);
    }
    if (this.scroller) {
      this.stopScroll();
      scrollToRange(this.scroller, hit.range, this.card);
    }
  }

  step(delta: number): void { void this.go(this.snapshot.active + delta); }
}

// 选中的一小段单行文字作为初始查询，与浏览器查找一致。
function selectedQuery(): string {
  const text = window.getSelection()?.toString().trim() ?? "";
  return text && text.length <= 120 && !text.includes("\n") ? text : "";
}

export function useThreadFind(): { session: FindSession; snapshot: FindSnapshot; openFind: () => void } {
  const [session] = useState(() => new FindSession());
  const snapshot = useSyncExternalStore(session.subscribe, () => session.snapshot, () => session.snapshot);
  const openFind = useCallback(() => {
    const preset = selectedQuery();
    session.setOpen(true);
    if (preset) session.setDraft(preset);
    requestAnimationFrame(() => {
      const input = session.card?.querySelector("input");
      input?.focus();
      input?.select();
    });
  }, [session]);
  return { session, snapshot, openFind };
}

export function ThreadFind({ session, snapshot, openFind }: ReturnType<typeof useThreadFind>) {
  const { scrollRef, contentRef, stopScroll } = useStickToBottomContext();
  const { draft } = snapshot;
  const pending = draft !== snapshot.query;
  session.root = contentRef.current;
  session.scroller = scrollRef.current;
  session.stopScroll = stopScroll;

  // 1. Ctrl/⌘F 打开（已打开时重新选中输入框），⌘G / ⇧⌘G 逐个跳
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!(event.metaKey || event.ctrlKey) || event.altKey || event.isComposing || !keyForChat(event)) return;
      const key = event.key.toLowerCase();
      if (key === "f" && !event.shiftKey) { event.preventDefault(); openFind(); }
      else if (key === "g" && session.snapshot.open) { event.preventDefault(); session.step(event.shiftKey ? -1 : 1); }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [openFind, session]);

  // 2. 输入停 150ms 再查；新查询从第一处开始并跳过去
  useEffect(() => {
    if (!snapshot.open || !pending) return;
    const timer = window.setTimeout(() => {
      session.refresh(draft, false);
      if (session.snapshot.count) void session.go(0);
    }, QUERY_DELAY_MS);
    return () => window.clearTimeout(timer);
  }, [draft, pending, session, snapshot.open]);

  // 3. 正文变化（流式、展开、翻页）后重新计数，当前位置不动也不滚动
  useEffect(() => {
    const root = contentRef.current;
    if (!snapshot.open || !snapshot.query.trim() || !root) return;
    let timer = 0;
    const observer = new MutationObserver(() => {
      window.clearTimeout(timer);
      timer = window.setTimeout(() => session.refresh(), 200);
    });
    observer.observe(root, { childList: true, subtree: true, characterData: true });
    return () => { observer.disconnect(); window.clearTimeout(timer); };
  }, [contentRef, session, snapshot.open, snapshot.query]);

  useEffect(() => () => session.setOpen(false), [session]);

  if (!snapshot.open) return null;
  const hasQuery = draft.trim() !== "";
  const label = !snapshot.count ? "0 个结果"
    : `${snapshot.active + 1} / ${snapshot.count}${snapshot.capped ? "+" : ""} 个结果`;
  const close = () => session.setOpen(false);
  return (
    <div className="thread-find" role="search" data-has-query={hasQuery} ref={(element) => { session.card = element; }}>
      <div className="thread-find__row">
        {pending && hasQuery ? <span className="thread-find__spin" aria-hidden="true" />
          : <Search className="thread-find__glass" size={16} aria-hidden="true" />}
        <input value={draft} placeholder="在对话中查找…" aria-label="在对话中查找" enterKeyHint="search"
          autoComplete="off" spellCheck={false} onChange={(event) => session.setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.nativeEvent.isComposing) { event.preventDefault(); session.advance(event.shiftKey ? -1 : 1); }
            else if (event.key === "Escape") { event.preventDefault(); event.stopPropagation(); close(); }
          }} />
      </div>
      <div className="thread-find__close">
        <span className="thread-find__sep" aria-hidden="true" />
        <button type="button" aria-label="关闭查找" data-tip="关闭（Esc）" onClick={close}><X size={16} aria-hidden="true" /></button>
      </div>
      <div className="thread-find__meta" aria-hidden={!hasQuery}>
        <button type="button" aria-label="上一个结果" disabled={!snapshot.count} onClick={() => session.step(-1)}>
          <ChevronUp size={14} aria-hidden="true" />
        </button>
        <button type="button" aria-label="下一个结果" disabled={!snapshot.count} onClick={() => session.step(1)}>
          <ChevronDown size={14} aria-hidden="true" />
        </button>
        <span className="thread-find__count" aria-live="polite">{hasQuery && !pending ? label : ""}</span>
      </div>
    </div>
  );
}

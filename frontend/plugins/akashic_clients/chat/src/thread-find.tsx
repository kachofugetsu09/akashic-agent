import React, { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore, type RefObject } from "react";
import { ChevronDown, ChevronUp, Search, X } from "lucide-react";
import { useStickToBottomContext } from "use-stick-to-bottom";
import { keyForChat } from "./chat-host";
import { chatSearchResult, fetchChatJson, type ChatSearchHit } from "./web-chat-data";

// 会话内查找（Ctrl/⌘F）：窗口右上角一张卡片。服务端按完整历史匹配正文，卡片列出命中消息的预览，
// 选中一条就把它读进来（近处连续补齐，远处只读前后一段）并跳过去；这条消息里的每一处用 CSS Highlight 标出，不改 DOM。
// 收起时内容不在 DOM 里的折叠（过程摘要、更早步骤、工具详情、思考）登记成区域，
// 区域用数据文本计数并排在原位；跳到区域里的命中时先展开它，离开后再收回。

const MATCH_LIMIT = 1000;
const QUERY_DELAY_MS = 120;
const RESULT_LIMIT = 200;
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
  hits: ChatSearchHit[];
  total: number;
  capped: boolean;
  /** 当前消息在 hits 里的位置；-1 表示还没选。 */
  active: number;
  /** 当前消息里第几处、共几处（按页面上实际渲染的文字计）。 */
  local: number;
  localCount: number;
  error: string;
}

// 查找跳转需要的宿主能力：确保某条消息已渲染，以及把并进别的轮次的消息展开定位。
export interface FindHost {
  sessionId: string;
  reveal: (seq: number) => Promise<void>;
}

// 同一文档只有一个聊天：区域、查询与时间线定位放在模块里，折叠组件不必穿过整棵消息树拿上下文。
const regions = new Map<HTMLElement, FindRegion>();
let navigateTimeline: ((messageId: string) => void) | null = null;

// 时间线登记自己的定位：并进别的轮次、收在过程里的消息由持有它的那一轮展开定位。
export function registerFindNavigation(navigate: ((messageId: string) => void) | null): void {
  navigateTimeline = navigate;
}
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
function scrollToRange(scroller: HTMLElement, range: Range | HTMLElement, card: HTMLElement | null): void {
  const owner = range instanceof HTMLElement ? range : range.startContainer.parentElement;
  owner?.closest(".user-text-clamp[data-state=\"clamped\"]")?.querySelector<HTMLButtonElement>(".user-text-toggle")?.click();
  owner?.scrollIntoView({ block: "center", behavior: "instant" });
  const rect = range.getBoundingClientRect();
  const box = scroller.getBoundingClientRect();
  const viewport = window.visualViewport;
  const composer = document.querySelector(".composer-wrap")?.getBoundingClientRect().top ?? box.bottom;
  const top = Math.max(box.top, (card?.getBoundingClientRect().bottom ?? box.top) + 8);
  const bottom = Math.min(box.bottom, composer, viewport ? viewport.offsetTop + viewport.height : box.bottom);
  const height = Math.min(rect.height, bottom - top);
  scroller.scrollTop += rect.top + height / 2 - (top + Math.max(bottom, top + height)) / 2;
}

// 页面上这条消息的元素：自己的行，或并进别的轮次后在过程里的那一段。
function messageElement(root: HTMLElement, messageId: string): HTMLElement | null {
  const id = CSS.escape(messageId);
  return root.querySelector<HTMLElement>(`.web-message-anchor[data-message-id="${id}"]`)
    ?? root.querySelector<HTMLElement>(`[data-process-message-id="${id}"]`);
}

class FindSession {
  snapshot: FindSnapshot = { open: false, draft: "", query: "", hits: [], total: 0, capped: false,
    active: -1, local: 0, localCount: 0, error: "" };
  private local: FindHit[] = [];
  private scope: HTMLElement | null = null;
  private revealed = new Set<FindRegion>();
  private listeners = new Set<() => void>();
  private request: AbortController | null = null;
  private jump = 0;
  host: FindHost | null = null;
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

  private clearLocal(): void {
    for (const region of this.revealed) region.conceal();
    this.revealed.clear();
    this.local = [];
    this.scope = null;
    clearPaint();
  }

  setOpen(open: boolean): void {
    if (open === this.snapshot.open) return;
    if (!open) {
      this.request?.abort();
      this.jump += 1;
      this.clearLocal();
      publishQuery("");
    } else void this.fetch("").catch(() => undefined);
    this.emit(open ? { open } : { open, draft: "", query: "", hits: [], total: 0, capped: false, active: -1, local: 0, localCount: 0, error: "" });
  }

  setDraft(draft: string): void { this.emit({ draft }); }

  // 空查询只让服务端建好索引：长会话首建约半秒，打开查找框时发出，输入完成前就绪。
  private async fetch(query: string): Promise<ReturnType<typeof chatSearchResult> | null> {
    const sessionId = this.host?.sessionId;
    if (!sessionId) return null;
    this.request?.abort();
    const controller = new AbortController();
    this.request = controller;
    const endpoint = `/api/chat/sessions/${encodeURIComponent(sessionId)}/search`;
    const result = chatSearchResult(await fetchChatJson<unknown>(
      `${endpoint}?q=${encodeURIComponent(query)}&limit=${RESULT_LIMIT}`, { signal: controller.signal }), endpoint);
    return controller.signal.aborted || this.host?.sessionId !== sessionId ? null : result;
  }

  // 1. 新查询：取结果，选中最新一条并跳过去
  // 2. 刷新（新消息到达）：保留当前选中的消息，不滚动
  async search(query: string, jump: boolean): Promise<void> {
    const needle = query.trim();
    publishQuery(needle.toLowerCase());
    if (!needle) { this.clearLocal(); this.emit({ query, hits: [], total: 0, capped: false, active: -1, local: 0, localCount: 0, error: "" }); return; }
    let result;
    try { result = await this.fetch(needle); }
    catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      this.emit({ query, error: error instanceof Error ? error.message : String(error) });
      return;
    }
    if (!result || this.snapshot.draft.trim() !== needle && jump) return;
    const previous = this.snapshot.hits[this.snapshot.active]?.messageId;
    const kept = jump ? -1 : result.hits.findIndex((hit) => hit.messageId === previous);
    const active = kept >= 0 ? kept : result.hits.length - 1;
    this.emit({ query, hits: result.hits, total: result.total, capped: result.capped, active, error: "",
      ...(kept >= 0 ? {} : { local: 0, localCount: 0 }) });
    if (jump && active >= 0) await this.select(active);
    else if (kept < 0) this.clearLocal();
  }

  // 输入还没落定就按 Enter：先按当前输入查一次。
  advance(delta: number): void {
    if (this.snapshot.draft.trim() === this.snapshot.query.trim() && this.snapshot.hits.length) this.step(delta);
    else void this.search(this.snapshot.draft, true);
  }

  // 1. 让这条消息出现在页面上（可能要读历史），等渲染后找到它的元素
  // 2. 收在别的轮次过程里时交给时间线展开，再找一次
  // 3. 在这条消息里收集命中并定位到第 at 处（-1 为最后一处）；页面文字与正文不完全一致时只把消息放到中间
  async select(index: number, at = 0): Promise<void> {
    const hit = this.snapshot.hits[index];
    if (!hit || !this.host) return;
    const ticket = ++this.jump;
    this.clearLocal();
    this.emit({ active: index, local: 0, localCount: 0 });
    try { await this.host.reveal(hit.seq); }
    catch (error) { this.emit({ error: error instanceof Error ? error.message : String(error) }); return; }
    await nextFrame();
    if (ticket !== this.jump || !this.root) return;
    let element = messageElement(this.root, hit.messageId);
    if (!element || element.closest(".turn-process:not(.is-open)")) {
      navigateTimeline?.(hit.messageId);
      await nextFrame();
      if (ticket !== this.jump) return;
      element = messageElement(this.root, hit.messageId) ?? element;
    }
    if (!element) { this.emit({ error: "这条消息在当前视图里不显示" }); return; }
    this.scope = element;
    this.stopScroll();
    this.collect();
    if (!this.local.length) { if (this.scroller) scrollToRange(this.scroller, element, this.card); return; }
    await this.go(at < 0 ? this.local.length - 1 : at);
  }

  // 页面重新渲染（窗口两端补读、流式输出、展开收起）后，旧的文字区间会失效：
  // 在当前消息里重新收集并重画当前这一处，不滚动；消息元素被替换时按 ID 重新找。
  repaint(): void {
    const hit = this.snapshot.hits[this.snapshot.active];
    if (!hit || !this.root || (!this.scope && !this.local.length)) return;
    if (!this.scope?.isConnected) this.scope = messageElement(this.root, hit.messageId);
    if (!this.scope) return;
    this.collect();
    const local = Math.min(this.snapshot.local, Math.max(0, this.local.length - 1));
    if (local !== this.snapshot.local) this.emit({ local });
    paint(this.local, this.local[local]);
  }

  private collect(): void {
    const needle = this.snapshot.query.trim().toLowerCase();
    this.local = this.scope && needle ? collectHits(this.scope, needle).hits : [];
    this.emit({ localCount: this.local.length });
  }

  // 1. 区域里的命中先展开区域，等渲染后按同一序号重新定位（嵌套折叠最多展开几层）
  // 2. 命中落到文字上：高亮、滚到可见段中间，收回不再包含它的、由查找展开的区域
  private async go(index: number, depth = 0): Promise<void> {
    const hit = this.local[index];
    if (!hit) return;
    this.emit({ local: index });
    if (hit.kind === "region") {
      if (depth > 4) return;
      hit.region.reveal();
      this.revealed.add(hit.region);
      await nextFrame();
      this.collect();
      return this.go(index, depth + 1);
    }
    paint(this.local, hit);
    for (const region of this.revealed) {
      if (region.container.contains(hit.range.startContainer)) continue;
      region.conceal();
      this.revealed.delete(region);
    }
    if (this.scroller) scrollToRange(this.scroller, hit.range, this.card);
  }

  // 先在当前消息里逐处走，走完再换到相邻的命中消息。
  step(delta: number): void {
    const { hits, active, local } = this.snapshot;
    if (!hits.length) return;
    const next = local + delta;
    if (active >= 0 && next >= 0 && next < this.local.length) { void this.go(next); return; }
    const target = active < 0 ? hits.length - 1 : (active + delta + hits.length) % hits.length;
    void this.select(target, delta < 0 ? -1 : 0);
  }
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

function hitTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  const now = new Date();
  const time = `${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
  if (date.toDateString() === now.toDateString()) return time;
  const day = `${date.getMonth() + 1}月${date.getDate()}日`;
  return date.getFullYear() === now.getFullYear() ? `${day} ${time}` : `${date.getFullYear()}年${day}`;
}

// 摘要里其余几处同样标出（不区分大小写，与服务端匹配一致）。
function marked(text: string, query: string): React.ReactNode[] {
  const needle = query.trim().toLowerCase();
  const lower = text.toLowerCase();
  const nodes: React.ReactNode[] = [];
  let from = 0;
  for (let at = needle && lower.length === text.length ? lower.indexOf(needle) : -1; at >= 0; at = lower.indexOf(needle, from)) {
    nodes.push(text.slice(from, at), <mark key={at}>{text.slice(at, at + needle.length)}</mark>);
    from = at + needle.length;
  }
  nodes.push(text.slice(from));
  return nodes;
}

type ThreadFindProps = ReturnType<typeof useThreadFind> & { host: FindHost; revision: string };

export function ThreadFind({ session, snapshot, openFind, host, revision }: ThreadFindProps) {
  const { scrollRef, contentRef, stopScroll } = useStickToBottomContext();
  const { draft } = snapshot;
  const pending = draft.trim() !== snapshot.query.trim();
  const listRef = useRef<HTMLOListElement>(null);
  session.root = contentRef.current;
  session.scroller = scrollRef.current;
  session.stopScroll = stopScroll;
  session.host = host;

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

  // 2. 输入停 120ms 再查；新查询从最新一条开始并跳过去
  useEffect(() => {
    if (!snapshot.open || !pending) return;
    const timer = window.setTimeout(() => { void session.search(draft, true); }, QUERY_DELAY_MS);
    return () => window.clearTimeout(timer);
  }, [draft, pending, session, snapshot.open]);

  // 3. 会话有新消息时按原查询重查：保留当前选中的消息，不滚动
  const query = snapshot.query;
  useEffect(() => {
    if (!snapshot.open || !query.trim()) return;
    const timer = window.setTimeout(() => { void session.search(query, false); }, 300);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- 只随会话内容变化重查，查询变化由第 2 步负责
  }, [revision]);

  // 4. 正文变化后重画当前消息里的高亮（区间随 DOM 失效），不移动位置
  useEffect(() => {
    const root = contentRef.current;
    if (!snapshot.open || !root) return;
    let timer = 0;
    const observer = new MutationObserver(() => {
      window.clearTimeout(timer);
      timer = window.setTimeout(() => session.repaint(), 150);
    });
    observer.observe(root, { childList: true, subtree: true, characterData: true });
    return () => { observer.disconnect(); window.clearTimeout(timer); };
  }, [contentRef, session, snapshot.open]);

  // 5. 换会话或关闭时收起
  useEffect(() => () => session.setOpen(false), [session, host.sessionId]);

  // 6. 当前项保持在列表可见范围内
  useEffect(() => {
    listRef.current?.querySelector<HTMLElement>("[aria-selected=\"true\"]")?.scrollIntoView({ block: "nearest" });
  }, [snapshot.active]);

  if (!snapshot.open) return null;
  const hasQuery = draft.trim() !== "";
  const { hits, total, active } = snapshot;
  const position = active >= 0 ? total - hits.length + active + 1 : 0;
  const label = snapshot.error ? snapshot.error : !hits.length ? "0 条消息"
    : `${position} / ${total}${snapshot.capped ? "+" : ""} 条消息${snapshot.localCount > 1 ? ` · 第 ${snapshot.local + 1}/${snapshot.localCount} 处` : ""}`;
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
      <div className="thread-find__meta-wrap" aria-hidden={!hasQuery}>
        <div className="thread-find__meta">
          <button type="button" aria-label="上一个结果" disabled={!hits.length} onClick={() => session.step(-1)}>
            <ChevronUp size={14} aria-hidden="true" />
          </button>
          <button type="button" aria-label="下一个结果" disabled={!hits.length} onClick={() => session.step(1)}>
            <ChevronDown size={14} aria-hidden="true" />
          </button>
          <span className="thread-find__count" aria-live="polite">{hasQuery && !pending ? label : ""}</span>
        </div>
      </div>
      {hasQuery && hits.length ? <ol className="thread-find__results" ref={listRef} role="listbox" aria-label="命中的消息">
        {total > hits.length ? <li className="thread-find__more" role="presentation">只列出最近 {hits.length} 条，更早的请换个更具体的词</li> : null}
        {hits.map((hit, index) => <li key={hit.messageId} role="option" aria-selected={index === active}>
          <button type="button" onClick={() => void session.select(index)}>
            <span className="thread-find__who">{hit.kind === "input" ? "你" : "回复"}<time dateTime={hit.timestamp}>{hitTime(hit.timestamp)}</time>
              {hit.count > 1 ? <span>{hit.count} 处</span> : null}</span>
            <span className="thread-find__snippet">{marked(hit.before, query)}<mark>{hit.match}</mark>{marked(hit.after, query)}</span>
          </button>
        </li>)}
      </ol> : null}
    </div>
  );
}

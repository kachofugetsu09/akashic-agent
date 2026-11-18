import { useEffect, useLayoutEffect, useRef, useState, useSyncExternalStore, type PointerEvent as ReactPointerEvent } from "react";
import { createPortal } from "react-dom";
import { ArrowDown, ArrowUp } from "lucide-react";
import { useStickToBottomContext } from "use-stick-to-bottom";
import type { MessageSelection, MessageSelectionRequest, MessageSelectionStage } from "@akashic/chat-ui-v1";
import "./message-selection.css";

// 消息选择模式：插件经 selectMessages 请求，宿主接管顶栏与输入区，用户框一段连续消息。
// 范围按 DOM 里的消息锚点计：实时行与时间线行都带 data-message-id，不必穿过两套消息模型。

const ANCHOR = ".web-message-anchor[data-message-id]";
const STAGE_OMIT = ".shared-message-actions, .shared-message-delivery, .plugin-ui-host, .reading-code-copy, .code-header-actions, [data-static-code-copy]";
// 过程：时间线的折叠摘要与思考/工具步骤，实时行的思考触发行与工具步骤；正文同在 .process-trace 里，不能整块删。
const STAGE_PROCESS = ".turn-process, .flow-thinking, .flow-tool, .flow-notice, .tool-group, .tool-step, .process-trigger-sticky, .process-content";
const EDGE_SCROLL = 64;

interface Pending {
  id: number;
  request: MessageSelectionRequest;
  resolve(value: MessageSelection | null): void;
}

interface Range {
  from: string;
  to: string;
}

type Geometry = { top: number; height: number; left: number; width: number } | null;

// 同一文档只有一个聊天：请求放在模块里，插件挂载与消息行都不必穿过组件树拿上下文。
let pending: Pending | null = null;
let layers = 0;
let requests = 0;
const listeners = new Set<() => void>();

function publish(next: Pending | null): void {
  pending = next;
  for (const listener of [...listeners]) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

function settle(value: MessageSelection | null): void {
  const current = pending;
  publish(null);
  current?.resolve(value);
}

// 插件入口：没有会话视图时直接得到 null；新请求顶替旧请求，旧请求得到 null。
export function selectMessages(request: MessageSelectionRequest): Promise<MessageSelection | null> {
  if (layers === 0) return Promise.resolve(null);
  settle(null);
  return new Promise((resolve) => publish({ id: ++requests, request, resolve }));
}

// 选择期间视口外的行也要完整渲染，确认后的副本才有富文本。
export function useMessageSelectionActive(): boolean {
  return useSyncExternalStore(subscribe, () => pending !== null);
}

function anchorsIn(scroller: HTMLElement): HTMLElement[] {
  return [...scroller.querySelectorAll<HTMLElement>(ANCHOR)]
    .filter((anchor) => anchor.offsetHeight > 0 && !anchor.classList.contains("is-compaction-boundary"));
}

const idOf = (anchor: HTMLElement): string => anchor.dataset.messageId ?? "";
const isInput = (anchor: HTMLElement): boolean => anchor.dataset.messageKind === "input" || anchor.classList.contains("user");

function bounds(anchors: HTMLElement[], range: Range | null): [number, number] | null {
  if (!range) return null;
  const from = anchors.findIndex((anchor) => idOf(anchor) === range.from);
  const to = anchors.findIndex((anchor) => idOf(anchor) === range.to);
  return from < 0 || to < 0 ? null : [Math.min(from, to), Math.max(from, to)];
}

// around 取它所在的一轮：往前找到最近一次提问；没有提问就只取它自己。
function initialRange(anchors: HTMLElement[], initial: MessageSelectionRequest["initial"]): Range | null {
  if (anchors.length === 0) return null;
  if (initial && "from" in initial) {
    const restored = bounds(anchors, initial);
    if (restored) return { from: idOf(anchors[restored[0]]), to: idOf(anchors[restored[1]]) };
  }
  const around = initial && "around" in initial ? initial.around : "";
  let end = anchors.findIndex((anchor) => idOf(anchor) === around);
  if (end < 0) end = anchors.length - 1;
  let start = end;
  while (start > 0 && !isInput(anchors[start])) start -= 1;
  if (!isInput(anchors[start])) start = end;
  return { from: idOf(anchors[start]), to: idOf(anchors[end]) };
}

// 点范围外：扩到它；点端点：退掉这一条；点范围内：离得近的一端收到它。
function tapRange(anchors: HTMLElement[], range: Range | null, index: number): Range | null {
  const current = bounds(anchors, range);
  const pick = (a: number, b: number): Range => ({ from: idOf(anchors[a]), to: idOf(anchors[b]) });
  if (!current) return pick(index, index);
  const [a, b] = current;
  if (index < a) return pick(index, b);
  if (index > b) return pick(a, index);
  if (a === b) return null;
  if (index === a) return pick(a + 1, b);
  if (index === b) return pick(a, b - 1);
  return index - a <= b - index ? pick(index, b) : pick(a, index);
}

// 只读副本：沿用聊天样式挂在 chat-main 里的页外位置；去掉操作、插件挂载与延迟渲染标记。
// cloneNode 不带影子树：代码块等自定义元素的内容在 open shadow root 里，按同序遍历逐个补上。
function copyShadowRoots(source: Element, copy: Element): void {
  const from = [source, ...source.querySelectorAll("*")];
  const to = [copy, ...copy.querySelectorAll("*")];
  from.forEach((element, index) => {
    const root = element.shadowRoot;
    const target = to[index];
    if (!root || !target) return;
    const shadow = target.shadowRoot ?? target.attachShadow({ mode: "open" });
    const children = [...root.children];
    shadow.replaceChildren(...[...root.childNodes].map((node) => node.cloneNode(true)));
    shadow.adoptedStyleSheets = [...root.adoptedStyleSheets];
    [...shadow.children].forEach((child, at) => { if (children[at]) copyShadowRoots(children[at], child); });
  });
}

function stageMessages(scroller: HTMLElement, ids: readonly string[], options: { width: number; process: boolean; time: boolean }): MessageSelectionStage {
  // 1. 沿用消息列表容器的类名：列表的纵向排布与间距由它的组件类给出，写死会让问答叠在一起
  const sources = ids.flatMap((id) => scroller.querySelector<HTMLElement>(`.web-message-anchor[data-message-id="${CSS.escape(id)}"]`) ?? []);
  if (sources.length !== ids.length) throw new Error("选中的消息已不在当前会话里，请重新选择");
  const stage = document.createElement("div");
  stage.className = `${sources[0]?.parentElement?.className ?? "conversation-content"} message-selection-stage`;
  stage.style.width = `${options.width}px`;
  // 副本只供排版拍照：不进 Tab 序与读屏，正文链接等也点不到。
  stage.inert = true;
  stage.setAttribute("aria-hidden", "true");
  // 2. 逐条复制并去掉交互件，按选项去掉过程与时间
  const omit = [STAGE_OMIT, options.process ? "" : STAGE_PROCESS, options.time ? "" : ".shared-message-meta time"].filter(Boolean).join(", ");
  for (const source of sources) {
    const copy = source.cloneNode(true) as HTMLElement;
    copyShadowRoots(source, copy);
    copy.classList.remove("history-isolated");
    copy.removeAttribute("data-selection");
    copy.removeAttribute("tabindex");
    for (const element of copy.querySelectorAll(omit)) element.remove();
    for (const meta of copy.querySelectorAll(".shared-message-meta")) if (!meta.querySelector("time")) meta.remove();
    for (const image of copy.querySelectorAll("img")) image.loading = "eager";
    stage.append(copy);
  }
  (scroller.closest(".chat-main") ?? document.body).append(stage);
  return { element: stage, release: () => stage.remove() };
}

export function MessageSelectionLayer({ sessionId }: { sessionId: string }) {
  const { scrollRef } = useStickToBottomContext();
  const current = useSyncExternalStore(subscribe, () => pending);
  useEffect(() => {
    layers += 1;
    return () => {
      layers -= 1;
      if (layers === 0) settle(null);
    };
  }, []);
  // 切走会话即取消：范围只属于开始选择时的那段对话。
  useEffect(() => () => settle(null), [sessionId]);
  const [scroller, setScroller] = useState<HTMLElement | null>(null);
  useLayoutEffect(() => { setScroller(current ? scrollRef.current : null); }, [current, scrollRef]);
  // 按请求身份重建：顶替的新请求要重新算初始范围，不沿用上一个会话的状态。
  return current && scroller ? <SelectionSession key={current.id} pending={current} scroller={scroller} /> : null;
}

function SelectionSession({ pending: request, scroller }: { pending: Pending; scroller: HTMLElement }) {
  const [range, setRange] = useState<Range | null>(() => initialRange(anchorsIn(scroller), request.request.initial));
  const [geometry, setGeometry] = useState<Geometry>(null);
  const [layout, setLayout] = useState(0);
  const [offscreen, setOffscreen] = useState({ above: false, below: false });
  const drag = useRef<{ edge: "from" | "to"; y: number } | null>(null);
  const [dragging, setDragging] = useState(false);
  const main = scroller.closest<HTMLElement>(".chat-main");
  const headingHeight = main?.querySelector<HTMLElement>(".conversation-heading")?.offsetHeight ?? 53;

  useSelectionChrome(main, scroller, () => setLayout((value) => value + 1));
  useSelectionClicks(scroller, setRange);

  // 1. 范围变了或内容重排后，给锚点写选中标记并重算外框
  useLayoutEffect(() => {
    const anchors = anchorsIn(scroller);
    const current = bounds(anchors, range);
    anchors.forEach((anchor, index) => {
      anchor.dataset.selection = current && index >= current[0] && index <= current[1] ? "in" : "out";
    });
    if (!current) { setGeometry(null); return; }
    const box = scroller.getBoundingClientRect();
    const first = anchors[current[0]].getBoundingClientRect();
    const last = anchors[current[1]].getBoundingClientRect();
    setGeometry({
      top: first.top - box.top + scroller.scrollTop - 6,
      height: last.bottom - first.top + 12,
      left: first.left - box.left - 8,
      width: first.width + 16,
    });
  }, [layout, range, scroller]);

  // 2. 起点或终点滚出视口时给出回跳
  useEffect(() => {
    const update = () => setOffscreen(geometry ? {
      above: geometry.top < scroller.scrollTop - 8,
      below: geometry.top + geometry.height > scroller.scrollTop + scroller.clientHeight + 8,
    } : { above: false, below: false });
    update();
    scroller.addEventListener("scroll", update, { passive: true });
    return () => scroller.removeEventListener("scroll", update);
  }, [geometry, scroller]);

  // 3. Esc 取消；退出时停掉拖动循环（按住把手时被取消，把手已卸载收不到 pointerup）
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      settle(null);
    };
    document.addEventListener("keydown", onKey, true);
    return () => {
      document.removeEventListener("keydown", onKey, true);
      drag.current = null;
    };
  }, []);

  const anchors = anchorsIn(scroller);
  const current = bounds(anchors, range);
  const confirm = () => {
    if (!current) return;
    const ids = anchors.slice(current[0], current[1] + 1).map(idOf);
    settle({ messageIds: ids, stage: (options) => stageMessages(scroller, ids, options) });
  };
  const startDrag = (edge: "from" | "to", y: number) => {
    drag.current = { edge, y };
    setDragging(true);
    dragLoop(scroller, drag, setRange);
  };
  const endDrag = () => {
    drag.current = null;
    setDragging(false);
  };

  return <>
    {geometry ? createPortal(<SelectionFrame geometry={geometry} dragging={dragging}
      onStart={startDrag} onMove={(y) => { if (drag.current) drag.current.y = y; }} onEnd={endDrag} />, scroller) : null}
    {main ? createPortal(<SelectionBars title={request.request.title} confirmLabel={request.request.confirmLabel}
      headingHeight={headingHeight} count={current ? current[1] - current[0] + 1 : 0}
      screens={geometry ? Math.round((geometry.height / Math.max(1, scroller.clientHeight)) * 2) / 2 : 0}
      everything={current?.[0] === 0 && current[1] === anchors.length - 1} offscreen={offscreen}
      onToggleAll={(all) => setRange(all || anchors.length === 0 ? null
        : { from: idOf(anchors[0]), to: idOf(anchors[anchors.length - 1]) })}
      onJump={(edge) => {
        if (!geometry) return;
        const top = edge === "from" ? geometry.top - 24 : geometry.top + geometry.height - scroller.clientHeight + 24;
        scroller.scrollTo({ top, behavior: "smooth" });
      }}
      onConfirm={confirm} />, main) : null}
  </>;
}

// 范围外框挂在滚动区里随内容滚动；上下两个把手按住拖动调整端点。
function SelectionFrame({ geometry, dragging, onStart, onMove, onEnd }: {
  geometry: NonNullable<Geometry>;
  dragging: boolean;
  onStart(edge: "from" | "to", y: number): void;
  onMove(y: number): void;
  onEnd(): void;
}) {
  const down = (edge: "from" | "to") => (event: ReactPointerEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    onStart(edge, event.clientY);
  };
  return <div className="message-selection-frame" data-dragging={dragging ? "" : undefined}
    style={{ top: geometry.top, height: geometry.height, left: geometry.left, width: geometry.width }}>
    {(["from", "to"] as const).map((edge) => <button key={edge} type="button" className={`message-selection-handle is-${edge}`}
      aria-label={edge === "from" ? "拖动调整起点" : "拖动调整终点"} onPointerDown={down(edge)}
      onPointerMove={(event) => onMove(event.clientY)}
      onPointerUp={onEnd} onPointerCancel={onEnd}>
      <span aria-hidden="true" />
    </button>)}
  </div>;
}

// 顶栏换成取消/标题/全选，输入区换成已选条数与确认；起点终点滚出视口时浮出回跳。
function SelectionBars({ title, confirmLabel, headingHeight, count, screens, everything, offscreen, onToggleAll, onJump, onConfirm }: {
  title: string;
  confirmLabel: string;
  headingHeight: number;
  count: number;
  screens: number;
  everything: boolean;
  offscreen: { above: boolean; below: boolean };
  onToggleAll(all: boolean): void;
  onJump(edge: "from" | "to"): void;
  onConfirm(): void;
}) {
  return <>
    <div className="message-selection-bar" style={{ blockSize: headingHeight }}>
      <button type="button" onClick={() => settle(null)}>取消</button>
      <div className="message-selection-bar__title">
        <strong>{title}</strong>
        <span>点消息或拖把手调整范围</span>
      </div>
      <button type="button" onClick={() => onToggleAll(everything)}>{everything ? "清空" : "全选"}</button>
    </div>
    {offscreen.above ? <button type="button" className="message-selection-jump is-above" style={{ insetBlockStart: headingHeight + 12 }}
      onClick={() => onJump("from")}><ArrowUp size={14} aria-hidden="true" />起点在上方</button> : null}
    {offscreen.below ? <button type="button" className="message-selection-jump is-below"
      onClick={() => onJump("to")}><ArrowDown size={14} aria-hidden="true" />终点在下方</button> : null}
    <div className="message-selection-dock" role="group" aria-label={title}>
      <div className="message-selection-dock__count" aria-live="polite">
        <strong>{count ? `已选 ${count} 条` : "还没选"}</strong>
        <span>{!count ? "点一条消息开始" : screens <= 1 ? "不到一屏" : `约 ${screens} 屏`}</span>
      </div>
      <button type="button" className="message-selection-confirm" disabled={!count} onClick={onConfirm}>{confirmLabel}</button>
    </div>
  </>;
}

// 接管顶栏与输入区：标记 chat-main 供样式隐藏原控件；内容重排时通知重算外框；退出时擦掉锚点标记。
function useSelectionChrome(main: HTMLElement | null, scroller: HTMLElement, relayout: () => void): void {
  useEffect(() => {
    main?.setAttribute("data-message-selecting", "");
    // 外框按滚动内容定位；滚动区原本没有定位时临时补上，退出时还原。
    const lifted = getComputedStyle(scroller).position === "static";
    if (lifted) scroller.style.setProperty("position", "relative");
    const content = scroller.firstElementChild;
    const resize = new ResizeObserver(relayout);
    const mutation = new MutationObserver(relayout);
    if (content) {
      resize.observe(content);
      mutation.observe(content, { childList: true });
    }
    return () => {
      resize.disconnect();
      mutation.disconnect();
      main?.removeAttribute("data-message-selecting");
      if (lifted) scroller.style.removeProperty("position");
      for (const anchor of scroller.querySelectorAll<HTMLElement>("[data-selection]")) delete anchor.dataset.selection;
    };
  // relayout 每次渲染都是新函数；只在进入时接管一次。
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [main, scroller]);
}

// 选择期间消息区的点击只用来选范围：链接、折叠、按钮都不响应。
function useSelectionClicks(scroller: HTMLElement, setRange: (update: (range: Range | null) => Range | null) => void): void {
  useEffect(() => {
    const onClick = (event: MouseEvent) => {
      const target = event.target as Element | null;
      if (!target || target.closest(".message-selection-frame")) return;
      event.preventDefault();
      event.stopPropagation();
      const anchor = target.closest<HTMLElement>(ANCHOR);
      if (!anchor) return;
      setRange((range) => {
        const anchors = anchorsIn(scroller);
        const index = anchors.indexOf(anchor);
        return index < 0 ? range : tapRange(anchors, range, index);
      });
    };
    scroller.addEventListener("click", onClick, true);
    return () => scroller.removeEventListener("click", onClick, true);
  }, [scroller, setRange]);
}

// 拖把手：指针所在的消息成为新的端点；贴近可见区上下沿时按距离加速滚动。
// 可见区扣掉选择栏与底栏：窄屏上两者压在滚动区上，按滚动区边沿算时手指够不到加速带。
function dragLoop(scroller: HTMLElement, drag: { current: { edge: "from" | "to"; y: number } | null }, setRange: (update: (range: Range | null) => Range | null) => void): void {
  const main = scroller.closest(".chat-main");
  const step = () => {
    const state = drag.current;
    if (!state) return;
    const box = scroller.getBoundingClientRect();
    const top = Math.max(box.top, main?.querySelector(".message-selection-bar")?.getBoundingClientRect().bottom ?? box.top);
    const bottom = Math.min(box.bottom, main?.querySelector(".message-selection-dock")?.getBoundingClientRect().top ?? box.bottom);
    const speed = (depth: number) => Math.ceil(Math.min(depth, EDGE_SCROLL * 2) / 4);
    if (state.y < top + EDGE_SCROLL) scroller.scrollTop -= speed(top + EDGE_SCROLL - state.y);
    else if (state.y > bottom - EDGE_SCROLL) scroller.scrollTop += speed(state.y - bottom + EDGE_SCROLL);
    setRange((range) => {
      const anchors = anchorsIn(scroller);
      const current = bounds(anchors, range);
      if (!current) return range;
      let index = anchors.findIndex((anchor) => state.y < anchor.getBoundingClientRect().bottom);
      if (index < 0) index = anchors.length - 1;
      const [a, b] = state.edge === "from" ? [Math.min(index, current[1]), current[1]] : [current[0], Math.max(index, current[0])];
      const next = { from: idOf(anchors[a]), to: idOf(anchors[b]) };
      return range && next.from === range.from && next.to === range.to ? range : next;
    });
    requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

import { ArrowDown } from "lucide-react";
import { useCallback, useEffect, useLayoutEffect, useRef, useState, useSyncExternalStore } from "react";
import { useStickToBottomContext } from "use-stick-to-bottom";

import type { ChatMessage } from "./chat-message";
import { MarqueeText } from "./marquee-text";
import type { StreamProjectionStore } from "./stream-projection";
import { isGeneratingChatStatus, type ChatStatus } from "./web-chat-status";
import { chatLayout, subscribeChatLayout } from "./chat-layout";
import type { ReplyActivity, TimelineMessage } from "./message-timeline";

export function DesktopAutoScroll({
  messages,
  status,
  streamStore,
  timelineMessages,
  replyActivities,
}: {
  messages: ChatMessage[];
  status: ChatStatus;
  streamStore: StreamProjectionStore<ChatMessage>;
  timelineMessages: TimelineMessage[];
  replyActivities: ReplyActivity[];
}) {
  const { escapedFromLock, isAtBottom, scrollRef, scrollToBottom, stopScroll } = useStickToBottomContext();
  const lastMessageCountRef = useRef(messages.length);
  const anchoredAnswerRef = useRef("");
  const baselineLastMessage = messages.at(-1);
  const baselineLastMessageId = baselineLastMessage?.id;
  const subscribe = useCallback(
    (listener: () => void) => baselineLastMessageId
      ? streamStore.subscribe(baselineLastMessageId, listener)
      : () => {},
    [baselineLastMessageId, streamStore],
  );
  const getSnapshot = useCallback(
    () => baselineLastMessage
      ? streamStore.read(baselineLastMessage.id, baselineLastMessage)
      : undefined,
    [baselineLastMessage, streamStore],
  );
  const lastMessage = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
  const lastBlock = lastMessage?.blocks.at(-1);
  const scrollKey = [
    messages.length,
    lastMessage?.id ?? "",
    lastMessage?.content.length ?? 0,
    lastMessage?.blocks.length ?? 0,
    lastBlock?.kind === "thinking" ? lastBlock.content.length : "",
    timelineMessages.at(-1)?.id ?? "",
    ...replyActivities.map((item) => `${item.handle}:${item.preview?.message_id}:${item.preview?.text.length}:${item.preview?.thinking.length}`),
  ].join(":");
  const lastMessageRole = lastMessage?.role;

  // 必要 effect：新用户消息/流式输出时 DOM 滚动到底部（响应数据变化执行 DOM 副作用），不可改为渲染期计算
  useEffect(() => {
    const hasNewUserMessage = messages.length > lastMessageCountRef.current && lastMessageRole === "user";
    lastMessageCountRef.current = messages.length;

    if (hasNewUserMessage) {
      void scrollToBottom({ animation: "smooth", ignoreEscapes: true });
      return;
    }

    if (isAtBottom && !escapedFromLock) {
      void scrollToBottom({ animation: "instant", ignoreEscapes: false });
    }
  }, [escapedFromLock, isAtBottom, lastMessageRole, messages.length, scrollKey, status, scrollToBottom]);

  const generating = isGeneratingChatStatus(status);
  const followingRef = useRef(false);
  // 必要 effect：把贴底状态同步给滚动监听；用户已经读过回答开头后自己滚开，这条回答不再接管
  useEffect(() => {
    followingRef.current = isAtBottom && !escapedFromLock;
    const scroller = scrollRef.current;
    const answer = scroller && generating && escapedFromLock ? currentAnswer(scroller) : undefined;
    if (scroller && answer && offsetToReadingTop(scroller, answer.element) <= 0) anchoredAnswerRef.current = answer.key;
  }, [escapedFromLock, generating, isAtBottom, scrollRef]);

  // 必要 effect：长回答在贴底跟随中把停靠点顶到阅读区顶缘时停下，读者从开头读起。
  // 正文由逐字动画长出、不经过 React 状态，只能在跟随产生的滚动里检查。
  useEffect(() => {
    const scroller = scrollRef.current;
    if (!scroller || !generating) return;
    const onScroll = () => {
      const answer = anchorableAnswer(scroller, anchoredAnswerRef.current);
      if (!answer || !followingRef.current) return;
      anchoredAnswerRef.current = answer.key;
      stopScroll();
      scroller.scrollTop += answer.offset;
    };
    scroller.addEventListener("scroll", onScroll, { passive: true });
    return () => scroller.removeEventListener("scroll", onScroll);
  }, [generating, scrollRef, stopScroll]);

  return null;
}

// MediaQueryList 的 matches 是实时的；每个滚动帧重新 matchMedia 会重复解析查询，这里只建一次。
const mediaLists = new Map<string, MediaQueryList>();
function media(query: string): MediaQueryList {
  let list = mediaLists.get(query);
  if (!list) {
    list = window.matchMedia(query);
    mediaLists.set(query, list);
  }
  return list;
}

// 浮在滚动区上方的标题栏会遮住内容顶部；返回被遮挡的高度加 8px 呼吸。
function readingTopInset(scroller: HTMLElement): number {
  const heading = scroller.closest(".chat-main")?.querySelector(".conversation-heading");
  const covered = heading ? heading.getBoundingClientRect().bottom - scroller.getBoundingClientRect().top : 0;
  return Math.max(0, covered) + 8;
}

// 元素顶部需要滚动多少才能停在阅读区顶缘。
function offsetToReadingTop(scroller: HTMLElement, element: HTMLElement): number {
  return element.getBoundingClientRect().top - scroller.getBoundingClientRect().top - readingTopInset(scroller);
}

// 跟随停在哪里：窄屏屏幕小，停在回答开头；宽屏停在本轮提问，问题与回答开头同屏。
function currentAnswer(scroller: HTMLElement) {
  const element = chatLayout().narrow
    ? Array.from(scroller.querySelectorAll<HTMLElement>(".reply-activity, .web-message-anchor.streaming")).at(-1)
    : latestTurnStart(scroller);
  const key = element?.dataset.replyHandle ?? element?.dataset.messageId ?? "";
  return element && key ? { element, key } : undefined;
}

// 正在生成的回答一旦把停靠点顶到阅读区顶缘，就返回它的身份与回退距离。
function anchorableAnswer(scroller: HTMLElement, anchored: string) {
  const answer = currentAnswer(scroller);
  if (!answer || answer.key === anchored) return null;
  const offset = offsetToReadingTop(scroller, answer.element);
  return offset <= 0 ? { key: answer.key, offset } : null;
}

// 每一轮提问的锚点；"最新一轮"按钮与提问目录共用。
const TURN_START = ".web-message-anchor.timeline-input, .web-message-anchor.user";

function latestTurnStart(scroller: HTMLElement): HTMLElement | undefined {
  return Array.from(scroller.querySelectorAll<HTMLElement>(TURN_START)).at(-1);
}

/** 离开底部后的返回入口：生成中提示还在写；翻回历史时先回到最新一轮开头，再到底部。 */
export function ReadingReturnButton({ generating }: { generating: boolean }) {
  const { isAtBottom, scrollRef, scrollToBottom } = useStickToBottomContext();
  const [latestBelow, setLatestBelow] = useState(false);

  // 必要 effect：订阅滚动位置，判断最新一轮是否还在视口下方（DOM 几何只能在滚动时读取）
  useEffect(() => {
    const scroller = scrollRef.current;
    if (!scroller) return;
    let frame = 0;
    const measure = () => {
      frame = 0;
      const start = latestTurnStart(scroller);
      setLatestBelow(Boolean(start && start.getBoundingClientRect().top > scroller.getBoundingClientRect().bottom));
    };
    const onScroll = () => { if (!frame) frame = window.requestAnimationFrame(measure); };
    scroller.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      scroller.removeEventListener("scroll", onScroll);
      window.cancelAnimationFrame(frame);
    };
  }, [scrollRef]);

  if (isAtBottom) return null;
  const toTurnStart = latestBelow && !generating;
  const label = generating ? "还在生成" : toTurnStart ? "最新一轮" : "";
  return <button type="button" className="desktop-scroll-return reading-return" data-label={label ? "true" : undefined}
    aria-label={generating ? "还在生成，滚动到底部" : toTurnStart ? "回到最新一轮开头" : "滚动到底部"}
    onClick={() => {
      const scroller = scrollRef.current;
      const start = scroller ? latestTurnStart(scroller) : undefined;
      if (toTurnStart && scroller && start) {
        scroller.scrollTo({ top: scroller.scrollTop + offsetToReadingTop(scroller, start), behavior: "smooth" });
      } else void scrollToBottom();
    }}>
    {generating ? <i className="reading-return__live" aria-hidden="true" /> : null}
    {label ? <span>{label}</span> : null}
    <ArrowDown size={16} aria-hidden="true" />
  </button>;
}

// 焦点在输入框时 PageUp/PageDown 翻对话（参考 Slack、Discord）：发送后焦点留在输入框，不必先点进消息区才能往回读。
// 草稿本身长到能滚动时仍交给输入框；一页是标题条与输入卡之间的可见高度，留 48px 重叠以便接着读。
export function ComposerPageKeys() {
  const { scrollRef, stopScroll } = useStickToBottomContext();

  // 必要 effect：输入框在 Conversation 之外，只能在 document 上接它的按键。
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.key !== "PageUp" && event.key !== "PageDown") || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      // 输入法组合中翻的是候选页（Safari 确认候选时还会补发 keyCode 229）；别处已处理的按键也不接管。
      if (event.isComposing || event.keyCode === 229 || event.defaultPrevented) return;
      const input = event.target;
      const scroller = scrollRef.current;
      if (!(input instanceof HTMLTextAreaElement) || !input.closest(".composer") || !scroller) return;
      if (input.scrollHeight > input.clientHeight + 1) return;
      event.preventDefault();
      const heading = scroller.closest(".chat-main")?.querySelector(".conversation-heading")?.getBoundingClientRect();
      const composer = input.closest(".composer-wrap")?.getBoundingClientRect();
      const frame = scroller.getBoundingClientRect();
      const top = Math.max(frame.top, heading?.bottom ?? frame.top);
      const bottom = Math.min(frame.bottom, composer && composer.height > 0 ? composer.top : frame.bottom);
      const page = Math.max(bottom - top - 48, 120);
      // 往回翻要先解除贴底跟随，否则生成中下一帧又被拉回底部。
      if (event.key === "PageUp") stopScroll();
      scroller.scrollBy({ top: event.key === "PageUp" ? -page : page, behavior: "smooth" });
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [scrollRef, stopScroll]);
  return null;
}

/** 窄屏阅读态：向下读时收起浮动标题栏，向上翻或回到顶部时放回。 */
export function NarrowReadingChrome() {
  const { scrollRef } = useStickToBottomContext();

  // 必要 effect：订阅滚动方向并切换 chat-main 的阅读态 class（纯 DOM 副作用，不触发渲染）
  useEffect(() => {
    const scroller = scrollRef.current;
    const main = scroller?.closest(".chat-main");
    if (!scroller || !main) return;
    let last = scroller.scrollTop;
    let touchedAt = Number.NEGATIVE_INFINITY;
    // 只认用户手势带来的滚动（含松手后的惯性）；贴底跟随、加载定位等程序滚动不收起标题栏。
    const onInput = () => { touchedAt = performance.now(); };
    const onScroll = () => {
      const top = scroller.scrollTop;
      const delta = top - last;
      last = top;
      if (!chatLayout().narrow || top < 64) main.classList.remove("is-reading");
      else if (performance.now() - touchedAt > 1200) return;
      else if (delta > 8) main.classList.add("is-reading");
      else if (delta < -8) main.classList.remove("is-reading");
    };
    // 轻点没有 touchmove，不会把随后到来的程序滚动误认为用户在下滑。
    const inputs = ["touchmove", "wheel"] as const;
    inputs.forEach((type) => scroller.addEventListener(type, onInput, { passive: true }));
    scroller.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      inputs.forEach((type) => scroller.removeEventListener(type, onInput));
      scroller.removeEventListener("scroll", onScroll);
      main.classList.remove("is-reading");
    };
  }, [scrollRef]);
  return null;
}

// 视口外的消息先按估算高度占位，靠近视口才换成完整内容；一次算好的目标会随之漂移。
// 瞬时跳到目标后逐帧复测并校正，连续两帧稳定或 12 帧后停止。
function settleAtReadingTop(scroller: HTMLElement, anchor: HTMLElement) {
  let frames = 0;
  let stable = 0;
  const step = () => {
    const offset = offsetToReadingTop(scroller, anchor);
    if (Math.abs(offset) > 1) {
      scroller.scrollTop += offset;
      stable = 0;
    } else stable += 1;
    frames += 1;
    if (stable < 2 && frames < 12 && anchor.isConnected) window.requestAnimationFrame(step);
  };
  step();
}

interface TurnSegment {
  id: string;
  preview: string;
  anchor: HTMLElement;
}

// 提问目录只在右侧留白放得下刻度的宽屏指针设备上出现，与样式表同一条件（指针设备 + 聊天宽度至少 1200px）。
const TURN_RAIL_POINTER = "(hover: hover) and (pointer: fine)";

function subscribeTurnRailMedia(onChange: () => void): () => void {
  const list = media(TURN_RAIL_POINTER);
  list.addEventListener("change", onChange);
  const stopLayout = subscribeChatLayout(onChange);
  return () => { list.removeEventListener("change", onChange); stopLayout(); };
}

// 当前已载入的每一轮提问，按文档顺序；预览取气泡文字的第一行。
function turnSegments(scroller: HTMLElement): TurnSegment[] {
  return Array.from(scroller.querySelectorAll<HTMLElement>(TURN_START))
    .map((anchor) => ({
      id: anchor.dataset.messageId ?? "",
      preview: (anchor.querySelector(".user-bubble")?.textContent ?? "").trim().replace(/\s+/g, " ").slice(0, 80),
      anchor,
    }))
    .filter((segment) => segment.id && segment.preview);
}

// 读到哪一轮：贴底时是最后一轮；否则是顶缘已越过阅读区上方三分之一的最后一轮提问。
// 锚点在收集时取好，每个滚动帧只读几何，不再查询 DOM。
function activeTurn(scroller: HTMLElement, segments: TurnSegment[]): string {
  if (scroller.scrollTop + scroller.clientHeight >= scroller.scrollHeight - 2) return segments.at(-1)?.id ?? "";
  const line = scroller.getBoundingClientRect().top + scroller.clientHeight / 3;
  // 锚点按文档顺序排列、顶缘单调递增：二分找最后一个越过阅读线的，每帧只读 log n 次几何。
  let low = 0;
  let high = segments.length - 1;
  let found = 0;
  while (low <= high) {
    const middle = (low + high) >> 1;
    if (segments[middle].anchor.getBoundingClientRect().top <= line) {
      found = middle;
      low = middle + 1;
    } else high = middle - 1;
  }
  return segments[found]?.id ?? "";
}

/** 宽屏右侧的提问目录：每轮一道刻度，悬停或聚焦展开为提问列表，点选停到那一轮开头。 */
export function TurnRail({ revision }: { revision: string }) {
  const { scrollRef, stopScroll } = useStickToBottomContext();
  const [segments, setSegments] = useState<TurnSegment[]>([]);
  const [active, setActive] = useState("");
  const [open, setOpen] = useState(false);
  const closeTimer = useRef(0);
  const wide = useSyncExternalStore(subscribeTurnRailMedia, () => media(TURN_RAIL_POINTER).matches && chatLayout().wide);

  // 必要 effect：消息变化后从 DOM 取提问列表，并随滚动同步当前所在的一轮；窄屏与触屏不挂监听
  useEffect(() => {
    const scroller = scrollRef.current;
    if (!scroller || !wide) return;
    let frame = 0;
    let current: TurnSegment[] = [];
    const sync = () => {
      frame = 0;
      setActive(activeTurn(scroller, current));
    };
    const collect = window.requestAnimationFrame(() => {
      current = turnSegments(scroller);
      setSegments(current);
      sync();
    });
    const onScroll = () => { if (!frame) frame = window.requestAnimationFrame(sync); };
    scroller.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      window.cancelAnimationFrame(collect);
      window.cancelAnimationFrame(frame);
      scroller.removeEventListener("scroll", onScroll);
    };
  }, [revision, scrollRef, wide]);
  useEffect(() => () => window.clearTimeout(closeTimer.current), []);
  const listRef = useRef<HTMLOListElement>(null);
  // 必要 effect：展开时把当前一轮滚进列表视野（长会话的列表超过 60vh）
  useEffect(() => {
    if (open) listRef.current?.querySelector("[aria-current]")?.scrollIntoView({ block: "nearest" });
  }, [open]);

  if (!wide || segments.length < 3) return null;
  const show = () => { window.clearTimeout(closeTimer.current); setOpen(true); };
  const hide = () => { window.clearTimeout(closeTimer.current); closeTimer.current = window.setTimeout(() => setOpen(false), 150); };
  const jump = (segment: TurnSegment) => {
    const scroller = scrollRef.current;
    const anchor = segment.anchor.isConnected ? segment.anchor
      : scroller?.querySelector<HTMLElement>(`.web-message-anchor[data-message-id="${CSS.escape(segment.id)}"]`);
    if (!scroller || !anchor) return;
    // 离开贴底跟随，否则生成中的回答会把视图拉回底部。
    stopScroll();
    settleAtReadingTop(scroller, anchor);
    setActive(segment.id);
    setOpen(false);
  };
  return <nav className="turn-rail" aria-label="按提问跳转" data-open={open || undefined}
    onMouseEnter={show} onMouseLeave={hide} onFocus={show}
    onBlur={(event) => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) hide(); }}
    onKeyDown={(event) => {
      if (event.key !== "Escape") return;
      // 收起后列表不可见，焦点不能留在里面；交还给输入框，与"读回复时直接打字"同一落点。
      setOpen(false);
      const composer = document.querySelector<HTMLTextAreaElement>(".composer-wrap textarea");
      if (composer) composer.focus({ preventScroll: true });
      else (document.activeElement as HTMLElement | null)?.blur();
    }}>
    <div className="turn-rail__ticks" aria-hidden="true">
      {segments.map((segment) => <i key={segment.id} data-active={segment.id === active || undefined} />)}
    </div>
    <ol ref={listRef} className="turn-rail__list">
      {segments.map((segment) => <li key={segment.id}>
        <button type="button" aria-current={segment.id === active ? "true" : undefined}
          tabIndex={open ? 0 : segment.id === active ? 0 : -1} onClick={() => jump(segment)}><MarqueeText>{segment.preview}</MarqueeText></button>
      </li>)}
    </ol>
  </nav>;
}

interface ReadingPosition {
  id: string;
  offset: number;
}

// 本标签页内每个会话读到哪里；贴近底部时不记，回来照常贴底。按最近使用保留 50 个会话，删掉的会话自然淘汰。
const readingPositions = new Map<string, ReadingPosition>();
const READING_POSITIONS_LIMIT = 50;

function rememberReading(sessionId: string, position: ReadingPosition): void {
  readingPositions.delete(sessionId);
  readingPositions.set(sessionId, position);
  if (readingPositions.size > READING_POSITIONS_LIMIT) {
    const oldest = readingPositions.keys().next().value;
    if (oldest !== undefined) readingPositions.delete(oldest);
  }
}

// 阅读区顶缘下的第一条消息：用 elementFromPoint 取，不逐条扫描；落在消息间隙时往下多探两次。
// 横坐标取阅读列中线而不是滚动区中线：超宽屏阅读列偏左，滚动区中线会落在列外空白。
function readingPosition(scroller: HTMLElement): ReadingPosition | null {
  const box = scroller.getBoundingClientRect();
  const column = scroller.querySelector(".conversation-content")?.getBoundingClientRect() ?? box;
  const x = column.left + column.width / 2;
  const top = box.top + readingTopInset(scroller);
  for (const dy of [2, 24, 48]) {
    const hit = document.elementFromPoint(x, top + dy)?.closest<HTMLElement>(".web-message-anchor[data-message-id]");
    if (hit && scroller.contains(hit)) return { id: hit.dataset.messageId ?? "", offset: hit.getBoundingClientRect().top - box.top };
  }
  return null;
}

/** 切走再切回会话时回到上次读到的位置；那条消息不在已载入范围内时照常贴底。 */
export function SessionReadingMemory({ sessionId, ready }: { sessionId: string; ready: string }) {
  const { scrollRef, stopScroll } = useStickToBottomContext();
  const restoringRef = useRef<ReadingPosition | null>(null);
  // 逐帧恢复进行中：压回位置产生的滚动不是用户在读，不能记成阅读位置。
  const applyingRef = useRef(false);

  // 必要 effect：切换会话时取出该会话的阅读位置，等消息就绪后再恢复；恢复前不记录
  useLayoutEffect(() => {
    restoringRef.current = sessionId ? readingPositions.get(sessionId) ?? null : null;
  }, [sessionId]);

  // 必要 effect：消息就绪后把记住的那条消息放回原偏移；虚拟占位会让首次定位漂移，逐帧校正到稳定
  useLayoutEffect(() => {
    const target = restoringRef.current;
    if (!target || !ready) return;
    restoringRef.current = null;
    applyingRef.current = true;
    // 1. 会话之间切换时滚动区是同一个节点；从首页（还没有对话区）切回时它才挂载，ref 与目标消息都可能晚一帧出现。
    //    每帧重新取滚动区与锚点，找不到就等下一帧，不立即放弃。
    const find = () => {
      const scroller = scrollRef.current;
      const anchor = scroller?.isConnected
        ? scroller.querySelector<HTMLElement>(`.web-message-anchor[data-message-id="${CSS.escape(target.id)}"]`) : null;
      return scroller && anchor ? { scroller, anchor } : null;
    };
    // 2. 同一次提交里贴底逻辑还拿着"正在贴底"的旧状态会再调一次 scrollToBottom；定位放到之后的帧里逐帧压回。
    let frames = 0;
    let touched = false;
    let cancelled = false;
    const onInput = () => { touched = true; };
    const inputs = ["wheel", "touchmove", "keydown", "pointerdown"] as const;
    inputs.forEach((type) => window.addEventListener(type, onInput, { passive: true, capture: true }));
    let located = false;
    let frame = 0;
    const finish = () => {
      applyingRef.current = false;
      inputs.forEach((type) => window.removeEventListener(type, onInput, { capture: true }));
    };
    const step = () => {
      frame = 0;
      // 3. 用户一有输入就把滚动交还给用户；最多尝试 20 帧。
      if (cancelled || touched || frames >= 20) return finish();
      frames += 1;
      const found = find();
      if (found) {
        // 4. 第一次找到锚点时离开贴底跟随，之后的内容增长不再把视图拉回底部。
        if (!located) stopScroll();
        located = true;
        const delta = found.anchor.getBoundingClientRect().top - found.scroller.getBoundingClientRect().top - target.offset;
        if (Math.abs(delta) > 1) found.scroller.scrollTop += delta;
      }
      frame = window.requestAnimationFrame(step);
    };
    frame = window.requestAnimationFrame(step);
    return () => {
      cancelled = true;
      window.cancelAnimationFrame(frame);
      finish();
    };
  }, [ready, scrollRef, stopScroll]);

  // 必要 effect：随滚动记录阅读位置（合并到每帧一次）；等待恢复与逐帧恢复期间都不覆盖。
  // 切换期间旧消息被清空时会话 id 可能还没变；只在消息确实属于当前会话（ready）时记录，免得清空引起的滚动抹掉旧位置。
  useEffect(() => {
    const scroller = scrollRef.current;
    if (!scroller || !sessionId || !ready) return;
    let frame = 0;
    const record = () => {
      frame = 0;
      if (restoringRef.current || applyingRef.current) return;
      const gap = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight;
      const position = gap > 80 ? readingPosition(scroller) : null;
      if (position?.id) rememberReading(sessionId, position);
      else if (gap <= 80) readingPositions.delete(sessionId);
    };
    const onScroll = () => { if (!frame) frame = window.requestAnimationFrame(record); };
    scroller.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      window.cancelAnimationFrame(frame);
      scroller.removeEventListener("scroll", onScroll);
    };
  }, [ready, scrollRef, sessionId]);

  return null;
}

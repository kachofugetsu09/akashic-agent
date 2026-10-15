import { ArrowDown } from "lucide-react";
import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import { useStickToBottomContext } from "use-stick-to-bottom";

import type { ChatMessage } from "./chat-message";
import type { StreamProjectionStore } from "./stream-projection";
import { isGeneratingChatStatus, type ChatStatus } from "./web-chat-status";
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

const NARROW_READING = "(max-width: 36rem)";

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
  const element = media(NARROW_READING).matches
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

/** 窄屏阅读态：向下读时收起浮动标题栏，向上翻或回到顶部时放回。 */
export function NarrowReadingChrome() {
  const { scrollRef } = useStickToBottomContext();

  // 必要 effect：订阅滚动方向并切换 chat-main 的阅读态 class（纯 DOM 副作用，不触发渲染）
  useEffect(() => {
    const scroller = scrollRef.current;
    const main = scroller?.closest(".chat-main");
    if (!scroller || !main) return;
    const narrow = media(NARROW_READING);
    let last = scroller.scrollTop;
    let touchedAt = Number.NEGATIVE_INFINITY;
    // 只认用户手势带来的滚动（含松手后的惯性）；贴底跟随、加载定位等程序滚动不收起标题栏。
    const onInput = () => { touchedAt = performance.now(); };
    const onScroll = () => {
      const top = scroller.scrollTop;
      const delta = top - last;
      last = top;
      if (!narrow.matches || top < 64) main.classList.remove("is-reading");
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

// 提问目录只在右侧留白放得下刻度的宽屏指针设备上出现，与样式表的媒体查询同一条件。
const TURN_RAIL_MEDIA = "(hover: hover) and (pointer: fine) and (min-width: 1200px)";

function subscribeTurnRailMedia(onChange: () => void): () => void {
  const list = media(TURN_RAIL_MEDIA);
  list.addEventListener("change", onChange);
  return () => list.removeEventListener("change", onChange);
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
  const wide = useSyncExternalStore(subscribeTurnRailMedia, () => media(TURN_RAIL_MEDIA).matches);

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
          tabIndex={open ? 0 : segment.id === active ? 0 : -1} onClick={() => jump(segment)}>{segment.preview}</button>
      </li>)}
    </ol>
  </nav>;
}

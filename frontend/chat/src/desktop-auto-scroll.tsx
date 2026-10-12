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

  // 必要 effect：窄屏长回答在贴底跟随中顶到阅读区顶缘时停在开头。
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

function currentAnswer(scroller: HTMLElement) {
  const element = Array.from(scroller.querySelectorAll<HTMLElement>(".reply-activity, .web-message-anchor.streaming")).at(-1);
  const key = element?.dataset.replyHandle ?? element?.dataset.messageId ?? "";
  return element && key ? { element, key } : undefined;
}

// 正在生成的回答一旦顶到阅读区顶缘，就返回它的身份与回退距离。
function anchorableAnswer(scroller: HTMLElement, anchored: string) {
  if (!window.matchMedia(NARROW_READING).matches) return null;
  const answer = currentAnswer(scroller);
  if (!answer || answer.key === anchored) return null;
  const offset = offsetToReadingTop(scroller, answer.element);
  return offset <= 0 ? { key: answer.key, offset } : null;
}

function latestTurnStart(scroller: HTMLElement): HTMLElement | undefined {
  return Array.from(scroller.querySelectorAll<HTMLElement>(".web-message-anchor.timeline-input, .web-message-anchor.user")).at(-1);
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
    const narrow = window.matchMedia(NARROW_READING);
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

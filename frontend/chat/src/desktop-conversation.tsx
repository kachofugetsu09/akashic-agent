import { focusMessagePart } from "./message-actions";
import { copyText } from "./copy-text";
import { timelineVisibleMessages, timelineFailureRetries, timelineToolResults, timelineReplyGroups, timelineAnchorIndexes, timelineInputStarts, timelineSourceKey, timelineSourceRefreshTokens, needsBeforeReasoningFallback, type ReplyActivity } from "./message-timeline";
import { timelineReply, timelineText, type TimelineMessage, type TimelineReply } from "./message-timeline";
import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { useStickToBottomContext } from "use-stick-to-bottom";

import { MessageReplyReference, SharedMessageActions } from "./message-actions";
import { PluginUiSlot } from "./plugin-ui-runtime";
import type { ChatMessage } from "./chat-message";
import { ChatMessageView, TimelineMessageView } from "./message-view";
import { StreamProjectionStore } from "./stream-projection";
import type { ChatStatus } from "./web-chat-status";
import { chatLayout } from "./chat-layout";
import { webTurnTrace } from "./web-turn-trace";

const LIVE_TAIL_MESSAGES = 120;

type ProjectedChatMessageViewProps = React.ComponentProps<typeof ChatMessageView> & {
  streamStore: StreamProjectionStore<ChatMessage>;
};

/** Subscribe one desktop row to its independent stream projection. */
function ProjectedChatMessageView({
  message: baselineMessage,
  streamStore,
  ...props
}: ProjectedChatMessageViewProps) {
  const subscribe = useCallback(
    (listener: () => void) => streamStore.subscribe(baselineMessage.id, listener),
    [baselineMessage.id, streamStore],
  );
  const getSnapshot = useCallback(
    () => streamStore.read(baselineMessage.id, baselineMessage),
    [baselineMessage, streamStore],
  );
  const message = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
  useLayoutEffect(() => {
    const kinds = webTurnTrace.markReactCommit(message.id);
    if (kinds.length === 0) return;
    window.requestAnimationFrame(() => webTurnTrace.markNextFrame(message.id, kinds));
  }, [message]);
  return <ChatMessageView {...props} message={message} />;
}

export interface DesktopConversationMessagesProps {
  messages: ChatMessage[];
  status: ChatStatus;
  copiedMessageId: string;
  /** 时间线列表末条消息的日期键；实时消息延续该日不重复分隔。 */
  carryDayKey?: string;
  streamStore: StreamProjectionStore<ChatMessage>;
  messageElementsRef: React.RefObject<Map<string, HTMLDivElement>>;
  onReply?: (message: ChatMessage) => void;
  onCopied: (messageId: string) => void;
  onError: (error: unknown) => void;
}

/** Render desktop history with stable rows and viewport-deferred rich content. */
export function DesktopConversationMessages({
  messages,
  status,
  copiedMessageId,
  carryDayKey,
  streamStore,
  messageElementsRef,
  onReply,
  onCopied,
  onError,
}: DesktopConversationMessagesProps) {
  const { stopScroll } = useStickToBottomContext();
  const messageIds = useMemo(() => new Set(messages.map((message) => message.id)), [messages]);
  const dayBreaks = useMemo(() => {
    const breaks = new Map<string, string>();
    let previous = carryDayKey ?? "";
    for (const message of messages) {
      const key = messageDayKey(message.createdAt);
      if (!key) continue;
      if (key !== previous) breaks.set(message.id, messageDayLabel(message.createdAt));
      previous = key;
    }
    return breaks;
  }, [carryDayKey, messages]);

  return (
    <>
      {messages.map((message, index) => (
        <DesktopMessageRow
          key={message.id}
          message={message}
          dayBreak={dayBreaks.get(message.id)}
          initiallyVisible={index >= messages.length - 8}
          replySourceUnavailable={Boolean(message.reply && !messageIds.has(message.reply.messageId))}
          canReply={Boolean(onReply && message.canonical) && status === "idle"}
          copied={copiedMessageId === message.id}
          enhancementSuspended={status !== "idle"}
          waitingForResponse={status === "streaming" && index === messages.length - 1}
          streamStore={streamStore}
          messageElementsRef={messageElementsRef}
          onReply={onReply}
          onCopied={onCopied}
          onError={onError}
          stopScroll={stopScroll}
        />
      ))}
    </>
  );
}

const DesktopMessageRow = React.memo(function DesktopMessageRow({
  message,
  dayBreak,
  initiallyVisible,
  replySourceUnavailable,
  canReply,
  copied,
  enhancementSuspended,
  waitingForResponse,
  streamStore,
  messageElementsRef,
  onReply,
  onCopied,
  onError,
  stopScroll,
}: {
  message: ChatMessage;
  dayBreak?: string;
  initiallyVisible: boolean;
  replySourceUnavailable: boolean;
  canReply: boolean;
  copied: boolean;
  enhancementSuspended: boolean;
  waitingForResponse: boolean;
  streamStore: StreamProjectionStore<ChatMessage>;
  messageElementsRef: React.RefObject<Map<string, HTMLDivElement>>;
  onReply?: (message: ChatMessage) => void;
  onCopied: (messageId: string) => void;
  onError: (error: unknown) => void;
  stopScroll: () => void;
}) {
  const anchorRef = useRef<HTMLDivElement | null>(null);
  const [nearViewport, setNearViewport] = useState(initiallyVisible);

  useEffect(() => {
    if (nearViewport || enhancementSuspended) return;
    let observer: IntersectionObserver | undefined;
    // 1. Let initial bottom positioning settle without observing its transient scroll path.
    const timer = window.setTimeout(() => {
      const anchor = anchorRef.current;
      if (!anchor) return;
      // 2. Upgrade only rows that are visible or one viewport away.
      observer = new IntersectionObserver((entries) => {
        if (!entries.some((entry) => entry.isIntersecting)) return;
        observer?.disconnect();
        setNearViewport(true);
      }, { rootMargin: "800px 0px" });
      observer.observe(anchor);
    }, 500);
    return () => {
      window.clearTimeout(timer);
      observer?.disconnect();
    };
  }, [enhancementSuspended, nearViewport]);

  const renderFullMessage = nearViewport || message.streaming === true;
  return (
    <>
      {dayBreak ? <DaySeparator label={dayBreak} /> : null}
      <div
        className={`web-message-anchor ${message.role} ${message.streaming === true ? "streaming" : "history-isolated"}`}
        data-message-id={message.id}
        ref={(element) => {
          anchorRef.current = element;
          if (element) messageElementsRef.current.set(message.id, element);
          else messageElementsRef.current.delete(message.id);
        }}
      >
        {renderFullMessage ? (
          <>
            <ProjectedChatMessageView
              message={message}
              streamStore={streamStore}
              deferRichContent
              waitingForResponse={waitingForResponse}
              leadingContent={message.reply ? (
                <MessageReplyReference
                  role={message.reply.role}
                  preview={message.reply.preview}
                  unavailable={replySourceUnavailable}
                  onNavigate={() => {
                    stopScroll();
                    const targetId = message.reply!.messageId;
                    const target = messageElementsRef.current.get(targetId)
                      ?? [...document.querySelectorAll<HTMLElement>("[data-message-id]")]
                        .find((element) => element.dataset.messageId === targetId);
                    target?.scrollIntoView({ behavior: "instant", block: "center" });
                  }}
                />
              ) : undefined}
              onCopyToolDetail={(text) => {
                void copyText(text).catch(onError);
              }}
              onError={onError}
            />
            <WebMessageMeta
              message={message}
              copied={copied}
              canReply={canReply}
              onReply={() => onReply?.(message)}
              onCopy={() => {
                void copyText(message.content).then(() => onCopied(message.id)).catch(onError);
              }}
            />
          </>
        ) : <DesktopMessagePlaceholder message={message} />}
      </div>
    </>
  );
});

function DesktopMessagePlaceholder({ message }: { message: ChatMessage }) {
  return (
    <div className={`message-row desktop-message-placeholder ${message.role === "user" ? "user-row" : "agent-row"}`}>
      <div className={message.role === "user" ? "user-bubble" : "agent-content"}>
        {message.reply ? <p className="plain-message-response">回复 · {message.reply.preview}</p> : null}
        {message.blocks.map((block, index) => (
          <p className="plain-message-response" key={block.kind === "tool" ? block.callId : `thinking-${index}`}>
            {block.kind === "thinking"
              ? block.content
              : `工具 · ${block.name} · ${JSON.stringify(block.input)} · ${JSON.stringify(block.output)} · ${block.errorText ?? ""}`}
          </p>
        ))}
        {message.content ? <p className="plain-message-response">{message.content}</p> : null}
        {message.attachments?.map((attachment) => (
          <p className="plain-message-response" key={attachment.id}>{attachment.filename ?? attachment.url}</p>
        ))}
      </div>
    </div>
  );
}

function WebMessageMeta({
  message,
  copied,
  canReply,
  onReply,
  onCopy,
}: {
  message: ChatMessage;
  copied: boolean;
  canReply: boolean;
  onReply: () => void;
  onCopy: () => void;
}) {
  return (
    <div className={`shared-message-meta ${message.role}`}>
      {message.delivery ? <span className="shared-message-delivery" role="status">
        {message.delivery === "retrying" ? "网络不稳，正在重发…" : "发送中…"}
      </span> : message.createdAt ? <time dateTime={message.createdAt}>{formatMessageTime(message.createdAt)}</time> : null}
      <SharedMessageActions
        canReply={canReply}
        canCopy={Boolean(message.content)}
        copied={copied}
        onReply={onReply}
        onCopy={onCopy}
      />
    </div>
  );
}

const chatMessageTimeFormatter = new Intl.DateTimeFormat("zh-CN", {
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});

function formatMessageTime(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "" : chatMessageTimeFormatter.format(date);
}

const daySeparatorFormatter = new Intl.DateTimeFormat("zh-CN", {
  month: "long",
  day: "numeric",
  weekday: "short",
});

const daySeparatorYearFormatter = new Intl.DateTimeFormat("zh-CN", {
  year: "numeric",
  month: "long",
  day: "numeric",
});

/** 按本地日历日聚合；无效时间返回空串，不产生分隔。 */
export function messageDayKey(value: string | undefined): string {
  if (!value) return "";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "" : `${date.getFullYear()}-${date.getMonth()}-${date.getDate()}`;
}

/** 分页可能恰好切在工具调用与结果之间：窗口开头的结果找不到调用，只能孤零零显示成"调用不在当前记录中"。
 * 还有更早历史时先不画，等上一页载入后它会并回所属调用；已到对话开头才按原样显示。 */
export function withoutLeadingOrphanResults(messages: TimelineMessage[], hasMore: boolean): TimelineMessage[] {
  if (!hasMore) return messages;
  let start = 0;
  while (start < messages.length && messages[start].body.kind === "tool_result") start += 1;
  return start ? messages.slice(start) : messages;
}

/** 今天、昨天直接说；同年写"10月14日周二"，跨年补全年份。"10/14" 这类斜杠日期在中文里读起来像分数。 */
function messageDayLabel(value: string | undefined): string {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  const today = new Date();
  const days = Math.round((startOfDay(today) - startOfDay(date)) / 86_400_000);
  if (days === 0) return "今天";
  if (days === 1) return "昨天";
  return date.getFullYear() === today.getFullYear()
    ? daySeparatorFormatter.format(date)
    : daySeparatorYearFormatter.format(date);
}

function startOfDay(date: Date): number {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
}

function DaySeparator({ label }: { label: string }) {
  return <div className="day-separator" role="separator" aria-label={label}><span>{label}</span></div>;
}

/** 展示完整日志，引用只定位原消息，不给工具结果补造助手身份。 */
export const DesktopTimelineMessages = React.memo(function DesktopTimelineMessages({ messages, activities, replySynced, refresh = 0, status, messageElementsRef, copiedMessageId, onReply, onEditInput, onRetry, onCopied, onError }: {
  messages: TimelineMessage[];
  activities: ReplyActivity[];
  /** 本会话的回复状态已同步（收到 reply.status）：之后没有实时活动的未收尾轮次按已结束显示。 */
  replySynced: boolean;
  refresh?: number;
  status: ChatStatus;
  messageElementsRef: React.RefObject<Map<string, HTMLDivElement>>;
  copiedMessageId: string;
  onReply: (reply: TimelineReply) => void;
  onEditInput: (text: string) => void;
  onRetry: (inputId: string) => Promise<void>;
  onCopied: (id: string) => void;
  onError: (error: unknown) => void;
}) {
  const { stopScroll, scrollRef } = useStickToBottomContext();
  const byId = useMemo(() => new Map(messages.map((message) => [message.id, message])), [messages]);
  const lookupMessage = useCallback((id: string) => byId.get(id), [byId]);
  const swipeReply = useCallback((id: string) => {
    const message = byId.get(id);
    if (message && status === "idle" && (message.body.kind === "input" || message.body.kind === "output")) onReply(timelineReply(message));
  }, [byId, onReply, status]);
  useNarrowMessageGestures(scrollRef, swipeReply);
  const toolResults = useMemo(() => timelineToolResults(messages), [messages]);
  const groups = useMemo(() => timelineReplyGroups(messages, activities, replySynced), [messages, activities, replySynced]);
  const inputStarts = useMemo(() => timelineInputStarts(messages), [messages]);
  const refreshTokens = useMemo(() => timelineSourceRefreshTokens(messages, activities, refresh), [messages, activities, refresh]);
  const visibleMessages = useMemo(() => timelineVisibleMessages(messages, groups), [messages, groups]);
  const failures = useMemo(() => timelineFailureRetries(messages), [messages]);
  const dayBreaks = useMemo(() => {
    const breaks = new Map<string, string>();
    let previous = "";
    for (const message of visibleMessages) {
      const key = messageDayKey(message.timestamp);
      if (!key) continue;
      if (key !== previous) breaks.set(message.id, messageDayLabel(message.timestamp));
      previous = key;
    }
    return breaks;
  }, [visibleMessages]);
  const anchorIndexes = useMemo(() => timelineAnchorIndexes(messages, groups), [messages, groups]);
  // 每条输出所在轮次的开始时间：同来源最近一条用户输入。
  const turnStarts = useMemo(() => {
    const starts = new Map<string, string>();
    const inputs = new Map<string, string>();
    for (const message of messages) {
      const key = timelineSourceKey(message);
      if (message.body.kind === "input" && message.author === "user") inputs.set(key, message.timestamp);
      else if (message.body.kind === "output" && inputs.has(key)) starts.set(message.id, inputs.get(key)!);
    }
    return starts;
  }, [messages]);
  const onNavigate = useCallback((id: string, partIndex?: number) => {
    stopScroll();
    const processOwner = partIndex === undefined ? undefined : [...groups.inline.leading].find(
      ([, members]) => members.some(message => message.id === id))?.[0];
    const index = anchorIndexes.get(processOwner ?? id);
    const activity = index === undefined ? undefined : activities[index - visibleMessages.length];
    const row = messageElementsRef.current.get(visibleMessages[index ?? -1]?.id)
      ?? (activity ? document.querySelector<HTMLElement>(`[data-reply-handle="${CSS.escape(activity.handle)}"]`) : null);
    if (row) focusMessagePart(row, id, partIndex);
  }, [activities, anchorIndexes, groups.inline.leading, messageElementsRef, stopScroll, visibleMessages]);
  // 末尾一段不跳过渲染：新插入或刚切到 content-visibility:auto 的元素首帧按 160px 占位布局，
  // 一轮落定时正式回复会先塌成占位高度、滚动被夹紧，下一帧再弹回，形成整屏闪跳。
  // 窗口取得足够宽，切换只发生在远离视口的上方，由原生滚动锚定吸收。
  // 宽度约两页历史：content-visibility 每次解锁都要一帧布局，用户最常滚动的近段不锁更省；
  // 更早的历史仍然锁住，改窗口宽度、开合侧栏时重排的范围有上限。
  const settledFrom = visibleMessages.length - LIVE_TAIL_MESSAGES;
  return <>{visibleMessages.map((message, index) => <React.Fragment key={message.id}>
    {dayBreaks.has(message.id) ? <DaySeparator label={dayBreaks.get(message.id)!} /> : null}
    <div
    className={`web-message-anchor${index < settledFrom ? " history-isolated" : ""} timeline-${message.body.kind}`}
    tabIndex={-1} data-message-id={message.id} data-message-kind={message.body.kind} data-message-seq={message.seq}
    ref={(element) => {
      if (element) messageElementsRef.current.set(message.id, element);
      else messageElementsRef.current.delete(message.id);
    }}>
    <TimelineMessageView message={message} turnStartedAt={turnStarts.get(message.id)} hideBody={groups.hiddenBodies.has(message.id) || groups.moved.has(message.id)} processMessages={groups.completed.get(message.id)} turnOpen={groups.unsettled.has(message.id)}
      inlineReply={groups.inline.outputs.has(message.id)} leadingProcess={groups.inline.leading.get(message.id)}
      hideProcess={groups.moved.has(message.id)} lookupMessage={lookupMessage} toolResults={toolResults} onNavigate={onNavigate} onError={onError}
      beforeReasoning={(origin) => needsBeforeReasoningFallback(origin, inputStarts) ? <PluginUiSlot name="turn.before_reasoning"
        sessionId={origin.session_id} messageId={origin.id} block={{ source: origin.source }}
        refreshToken={refreshTokens.get(timelineSourceKey(origin))} /> : null}
      afterBody={message.body.kind === "output" && message.body.finish === "complete" ? <PluginUiSlot
        name="turn.after_answer" sessionId={message.session_id} messageId={message.id} />
        : message.body.kind === "control" && message.body.action === "failure" ? <FailureRetry
          inputId={failures.retryable.get(message.id)} earlier={failures.earlier.get(message.id)}
          disabled={status !== "idle"} onRetry={onRetry} /> : undefined} />
    {/* 轮次未结束的输出还会接着长，时间与复制等落定后挂在末条上，和实时视图一致。 */}
    {!groups.hiddenBodies.has(message.id) && !groups.moved.has(message.id) && !groups.joined.has(message.id) && !groups.unsettled.has(message.id) ? <div className={`shared-message-meta timeline-meta ${message.body.kind === "input" ? "user" : "assistant"}`}>
      <time dateTime={message.timestamp}>{formatMessageTime(message.timestamp)}</time>
      <SharedMessageActions
        canReply={status === "idle" && (message.body.kind === "input" || message.body.kind === "output")}
        canCopy={Boolean(timelineText(message))} copied={copiedMessageId === message.id}
        onReply={() => onReply(timelineReply(message))}
        onEdit={message.body.kind === "input" && message.author === "user" && timelineText(message)
          ? () => onEditInput(timelineText(message)) : undefined}
        onCopy={() => { void copyText(timelineText(message)).then(() => onCopied(message.id)).catch(onError); }} />
    </div> : null}
    {message.body.kind === "input" && message.author === "user" ? <div className="message-row agent-row">
      <div className="agent-content"><PluginUiSlot name="turn.before_reasoning"
        sessionId={message.session_id} messageId={message.id} block={{ source: message.source }}
        refreshToken={refreshTokens.get(timelineSourceKey(message))} /></div>
    </div> : null}
  </div>
  </React.Fragment>)}</>;
}, (previous, next) => {
  // 草稿正文只更新活动行；身份/活动阶段变化才让稳定输入槽位重新读取事实。
  return previous.messages === next.messages
    && previous.status === next.status
    && previous.replySynced === next.replySynced
    && previous.refresh === next.refresh
    && previous.messageElementsRef === next.messageElementsRef
    && previous.copiedMessageId === next.copiedMessageId
    && previous.onReply === next.onReply
    && previous.onEditInput === next.onEditInput
    && previous.onRetry === next.onRetry
    && previous.onCopied === next.onCopied
    && previous.onError === next.onError
    && previous.activities.length === next.activities.length
    && previous.activities.every((activity, index) => {
      const other = next.activities[index];
      return activity.handle === other.handle
        && activity.session_id === other.session_id
        && activity.source === other.source
        && activity.active === other.active
        && activity.preview?.message_id === other.preview?.message_id;
    });
});

/** 失败块的收尾：重跑这一轮的文字按钮，以及收起的此前失败。 */
function FailureRetry({ inputId, earlier = [], disabled, onRetry }: {
  inputId?: string;
  earlier?: TimelineMessage[];
  disabled: boolean;
  onRetry: (inputId: string) => Promise<void>;
}) {
  const [pending, setPending] = useState(false);
  if (!inputId && !earlier.length) return null;
  return <>
    {inputId ? <div className="timeline-failure-actions">
      <button type="button" disabled={disabled || pending} aria-busy={pending || undefined}
        onClick={() => { setPending(true); void onRetry(inputId).finally(() => setPending(false)); }}>
        {pending ? "正在重试…" : "重试"}
      </button>
    </div> : null}
    {earlier.length ? <details className="timeline-failure-history">
      <summary>此前失败 {earlier.length} 次</summary>
      <ol>{earlier.map((item) => <li key={item.id}>
        <time dateTime={item.timestamp}>{formatMessageTime(item.timestamp)}</time>
        {item.body.kind === "control" ? item.body.reason?.split("\n")[0] : null}
      </li>)}</ol>
    </details> : null}
  </>;
}

type TouchGesture = { x: number; y: number; anchor: HTMLElement | null; fromEdge: boolean; axis: "x" | "y" | null; dx: number };

// 窄屏触控：点消息显示它的时间与操作；在消息上左滑引用；离开屏幕边缘右滑打开会话抽屉。
function useNarrowMessageGestures(scrollRef: React.RefObject<HTMLElement | null>, onSwipeReply: (id: string) => void) {
  const replyRef = useRef(onSwipeReply);
  useEffect(() => { replyRef.current = onSwipeReply; }, [onSwipeReply]);

  // 必要 effect：在滚动区上委托触摸与点击（DOM 手势只能在原生事件里读取几何）
  useEffect(() => {
    const scroller = scrollRef.current;
    if (!scroller) return;
    let gesture: TouchGesture | null = null;
    // 1. 代码块和表格保留自己的横向滚动，不参与手势。
    const onStart = (event: TouchEvent) => {
      const target = event.target instanceof Element ? event.target : null;
      gesture = null;
      if (!chatLayout().narrow || event.touches.length !== 1 || !target || target.closest("pre, table, .reading-table, input, textarea")) return;
      const touch = event.touches[0];
      gesture = { x: touch.clientX, y: touch.clientY, anchor: target.closest<HTMLElement>(".web-message-anchor"),
        fromEdge: touch.clientX < 24, axis: null, dx: 0 };
    };
    // 2. 先判定方向；横向时消息跟手左移，给出引用反馈。
    const onMove = (event: TouchEvent) => {
      if (!gesture) return;
      const dx = event.touches[0].clientX - gesture.x;
      const dy = event.touches[0].clientY - gesture.y;
      if (!gesture.axis && Math.hypot(dx, dy) > 10) gesture.axis = Math.abs(dx) > Math.abs(dy) * 1.5 ? "x" : "y";
      if (gesture.axis !== "x") return;
      gesture.dx = dx;
      if (gesture.anchor && dx < 0) {
        gesture.anchor.style.transition = "none";
        gesture.anchor.style.translate = `${Math.max(dx, -64)}px 0`;
      }
    };
    // 3. 松手时按距离决定引用或打开抽屉，消息回弹到原位。
    const onEnd = () => {
      if (!gesture) return;
      const { anchor, axis, dx, fromEdge } = gesture;
      gesture = null;
      if (axis !== "x") return;
      if (anchor) {
        anchor.style.transition = "";
        anchor.style.translate = "";
      }
      if (dx < -56 && anchor?.dataset.messageId) replyRef.current(anchor.dataset.messageId);
      else if (dx > 72 && !fromEdge) scroller.closest(".chat-main")?.querySelector<HTMLButtonElement>(".compact-navigation-trigger")?.click();
    };
    // 4. 点在正文上切换该条的时间与操作；可交互元素和选中文字不受影响。
    const onClick = (event: MouseEvent) => {
      const target = event.target instanceof Element ? event.target : null;
      if (!chatLayout().narrow || !target || window.getSelection()?.toString()) return;
      if (target.closest("a, button, summary, input, textarea, [role='button'], .shared-message-meta")) return;
      const anchor = target.closest<HTMLElement>(".web-message-anchor");
      for (const open of scroller.querySelectorAll(".web-message-anchor.is-revealed")) if (open !== anchor) open.classList.remove("is-revealed");
      anchor?.classList.toggle("is-revealed");
    };
    scroller.addEventListener("touchstart", onStart, { passive: true });
    scroller.addEventListener("touchmove", onMove, { passive: true });
    scroller.addEventListener("touchend", onEnd);
    scroller.addEventListener("touchcancel", onEnd);
    scroller.addEventListener("click", onClick);
    return () => {
      scroller.removeEventListener("touchstart", onStart);
      scroller.removeEventListener("touchmove", onMove);
      scroller.removeEventListener("touchend", onEnd);
      scroller.removeEventListener("touchcancel", onEnd);
      scroller.removeEventListener("click", onClick);
    };
  }, [scrollRef]);
}

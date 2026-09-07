import { timelineReply, timelineText, type TimelineMessage, type TimelineReply } from "./message-timeline";
import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { useStickToBottomContext } from "use-stick-to-bottom";

import { MessageReplyReference, SharedMessageActions } from "./message-actions";
import { MobilePluginSlot } from "./mobile-plugin-runtime";
import type { ChatMessage, ChatRole } from "./chat-message";
import { ChatMessageView, TimelineMessageView } from "./message-view";
import { StreamProjectionStore } from "./stream-projection";
import type { ChatStatus } from "./web-chat-status";
import { webTurnTrace } from "./web-turn-trace";

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
  streamStore,
  messageElementsRef,
  onReply,
  onCopied,
  onError,
}: DesktopConversationMessagesProps) {
  const { stopScroll } = useStickToBottomContext();
  const messageIds = useMemo(() => new Set(messages.map((message) => message.id)), [messages]);

  return (
    <>
      {messages.map((message, index) => (
        <DesktopMessageRow
          key={message.id}
          message={message}
          initiallyVisible={index >= messages.length - 8}
          followsSameRole={messages[index - 1]?.role === message.role}
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
  initiallyVisible,
  followsSameRole,
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
  initiallyVisible: boolean;
  followsSameRole: boolean;
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
      {followsSameRole ? <RoleDivider role={message.role} /> : null}
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
                void navigator.clipboard.writeText(text).catch(onError);
              }}
              onError={onError}
            />
            <WebMessageMeta
              message={message}
              copied={copied}
              canReply={canReply}
              onReply={() => onReply?.(message)}
              onCopy={() => {
                void navigator.clipboard.writeText(message.content).then(() => onCopied(message.id)).catch(onError);
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

function RoleDivider({ role }: { role: ChatRole }) {
  return <div aria-hidden="true" className={`role-divider ${role}-divider`} />;
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
      {message.createdAt ? <time dateTime={message.createdAt}>{formatMessageTime(message.createdAt)}</time> : null}
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

/** 展示完整日志，引用只定位原消息，不给工具结果补造助手身份。 */
export function DesktopTimelineMessages({ messages, status, messageElementsRef, copiedMessageId, onReply, onCopied, onError }: {
  messages: TimelineMessage[];
  status: ChatStatus;
  messageElementsRef: React.RefObject<Map<string, HTMLDivElement>>;
  copiedMessageId: string;
  onReply: (reply: TimelineReply) => void;
  onCopied: (id: string) => void;
  onError: (error: unknown) => void;
}) {
  const { stopScroll } = useStickToBottomContext();
  const byId = useMemo(() => new Map(messages.map((message) => [message.id, message])), [messages]);
  const lookupMessage = useCallback((id: string) => byId.get(id), [byId]);
  const onNavigate = useCallback((id: string, partIndex?: number) => {
    stopScroll();
    const row = messageElementsRef.current.get(id);
    const target = partIndex === undefined ? row : row?.querySelector<HTMLElement>(`[data-part-index="${partIndex}"]`);
    target?.focus({ preventScroll: true });
    target?.scrollIntoView({ behavior: "instant", block: "center" });
  }, [messageElementsRef, stopScroll]);
  return <>{messages.map((message) => <div key={message.id}
    className={`web-message-anchor history-isolated timeline-${message.body.kind}`}
    tabIndex={-1} data-message-id={message.id} data-message-kind={message.body.kind} data-message-seq={message.seq}
    ref={(element) => {
      if (element) messageElementsRef.current.set(message.id, element);
      else messageElementsRef.current.delete(message.id);
    }}>
    <TimelineMessageView message={message} lookupMessage={lookupMessage} onNavigate={onNavigate} onError={onError}
      leadingContent={message.body.kind === "output" ? <MobilePluginSlot name="turn.before_reasoning"
        sessionId={message.session_id} messageId={message.id} /> : undefined}
      beforePart={(part, index) => part.kind === "tool_call" && !("display" in part) ? <MobilePluginSlot
        name="turn.before_tool" sessionId={message.session_id} messageId={message.id} block={{ ...part, message_id: message.id, part_index: index }} /> : null}
      afterBody={message.body.kind === "output" && message.body.finish === "complete" ? <MobilePluginSlot
        name="turn.after_answer" sessionId={message.session_id} messageId={message.id} /> : undefined} />
    <div className="shared-message-meta timeline-meta">
      <span>{message.author}</span><span>来源 · {message.source}</span>
      <time dateTime={message.timestamp}>{formatMessageTime(message.timestamp)}</time>
      <details><summary>消息详情</summary><pre>{message.id}{"\n"}序号 {message.seq}</pre></details>
      <SharedMessageActions
        canReply={status === "idle" && (message.body.kind === "input" || message.body.kind === "output")}
        canCopy={Boolean(timelineText(message))} copied={copiedMessageId === message.id}
        onReply={() => onReply(timelineReply(message))}
        onCopy={() => { void navigator.clipboard.writeText(timelineText(message)).then(() => onCopied(message.id)).catch(onError); }} />
    </div>
  </div>)}</>;
}

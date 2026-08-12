import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";

import { MessageReplyReference, SharedMessageActions } from "./message-actions";
import { MobilePluginSlot } from "./mobile-plugin-runtime";
import type { ChatMessage, ChatRole } from "./chat-message";
import { ChatMessageView } from "./message-view";
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
  activeSessionId: string;
  status: ChatStatus;
  copiedMessageId: string;
  streamStore: StreamProjectionStore<ChatMessage>;
  messageElementsRef: React.RefObject<Map<string, HTMLDivElement>>;
  onReply: (message: ChatMessage) => void;
  onCopied: (messageId: string) => void;
  onError: (error: unknown) => void;
}

/** Render desktop history with stable rows and viewport-deferred rich content. */
export function DesktopConversationMessages({
  messages,
  activeSessionId,
  status,
  copiedMessageId,
  streamStore,
  messageElementsRef,
  onReply,
  onCopied,
  onError,
}: DesktopConversationMessagesProps) {
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
          activeSessionId={activeSessionId}
          canReply={Boolean(message.canonical) && status === "idle"}
          copied={copiedMessageId === message.id}
          enhancementSuspended={status !== "idle"}
          streamStore={streamStore}
          messageElementsRef={messageElementsRef}
          onReply={onReply}
          onCopied={onCopied}
          onError={onError}
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
  activeSessionId,
  canReply,
  copied,
  enhancementSuspended,
  streamStore,
  messageElementsRef,
  onReply,
  onCopied,
  onError,
}: {
  message: ChatMessage;
  initiallyVisible: boolean;
  followsSameRole: boolean;
  replySourceUnavailable: boolean;
  activeSessionId: string;
  canReply: boolean;
  copied: boolean;
  enhancementSuspended: boolean;
  streamStore: StreamProjectionStore<ChatMessage>;
  messageElementsRef: React.RefObject<Map<string, HTMLDivElement>>;
  onReply: (message: ChatMessage) => void;
  onCopied: (messageId: string) => void;
  onError: (error: unknown) => void;
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
              leadingContent={message.reply ? (
                <MessageReplyReference
                  role={message.reply.role}
                  preview={message.reply.preview}
                  unavailable={replySourceUnavailable}
                  onNavigate={() => {
                    messageElementsRef.current.get(message.reply!.messageId)?.scrollIntoView({ behavior: "smooth", block: "center" });
                  }}
                />
              ) : undefined}
              processStartContent={message.role === "assistant" ? (
                <MobilePluginSlot
                  name="turn.before_reasoning"
                  sessionId={activeSessionId}
                  messageId={message.streaming ? `assistant:${message.id}` : message.id}
                  turnId={message.streaming ? message.id : undefined}
                />
              ) : undefined}
              beforeProcessBlock={(block) => message.role === "assistant" && block.kind === "tool" ? (
                <MobilePluginSlot
                  name="turn.before_tool"
                  sessionId={activeSessionId}
                  messageId={message.streaming ? `assistant:${message.id}` : message.id}
                  turnId={message.streaming ? message.id : undefined}
                  block={block}
                />
              ) : null}
              answerEndContent={message.role === "assistant" ? (
                <MobilePluginSlot
                  name="turn.after_answer"
                  sessionId={activeSessionId}
                  messageId={message.streaming ? `assistant:${message.id}` : message.id}
                  turnId={message.streaming ? message.id : undefined}
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
              onReply={() => onReply(message)}
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

import { useCallback, useEffect, useRef, useSyncExternalStore } from "react";
import { useStickToBottomContext } from "use-stick-to-bottom";

import type { ChatMessage } from "./chat-message";
import type { StreamProjectionStore } from "./stream-projection";
import type { ChatStatus } from "./web-chat-status";

export function DesktopAutoScroll({
  messages,
  status,
  streamStore,
}: {
  messages: ChatMessage[];
  status: ChatStatus;
  streamStore: StreamProjectionStore<ChatMessage>;
}) {
  const { escapedFromLock, isAtBottom, scrollToBottom } = useStickToBottomContext();
  const lastMessageCountRef = useRef(messages.length);
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
  ].join(":");
  const lastMessageRole = lastMessage?.role;

  useEffect(() => {
    const hasNewUserMessage = messages.length > lastMessageCountRef.current && lastMessageRole === "user";
    lastMessageCountRef.current = messages.length;

    if (hasNewUserMessage) {
      void scrollToBottom({ animation: "smooth", ignoreEscapes: true });
      return;
    }

    if ((status === "streaming" || status === "submitted") && isAtBottom && !escapedFromLock) {
      void scrollToBottom({ animation: "smooth", ignoreEscapes: false });
    }
  }, [escapedFromLock, isAtBottom, lastMessageRole, messages.length, scrollKey, status, scrollToBottom]);

  return null;
}

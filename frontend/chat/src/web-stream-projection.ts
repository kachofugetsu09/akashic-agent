import type { ChatMessage } from "./chat-message.ts";
import { StreamProjectionStore } from "./stream-projection.ts";

/** Active last-row mutations can stay on the per-message projection path. */
export function canProjectWebStreamWithoutRoot(
  previousMessages: readonly ChatMessage[],
  nextMessages: readonly ChatMessage[],
): boolean {
  if (previousMessages.length === 0 || previousMessages.length !== nextMessages.length) return false;
  const lastIndex = nextMessages.length - 1;
  for (let index = 0; index < lastIndex; index += 1) {
    if (previousMessages[index] !== nextMessages[index]) return false;
  }
  const previous = previousMessages[lastIndex];
  const next = nextMessages[lastIndex];
  return previous !== next
    && previous.id === next.id
    && previous.role === "assistant"
    && next.role === "assistant"
    && previous.streaming === true
    && next.streaming === true;
}

/** Publish only active assistant mutations; every patch lands immediately. */
export function publishWebStreamChanges(
  previousMessages: readonly ChatMessage[],
  nextMessages: readonly ChatMessage[],
  streamStore: StreamProjectionStore<ChatMessage>,
): void {
  const previousById = new Map(previousMessages.map((message) => [message.id, message]));

  for (const target of nextMessages) {
    const previous = previousById.get(target.id);
    if (
      target.role !== "assistant"
      || previous === undefined
      || target === previous
      || (previous.streaming !== true && target.streaming !== true)
    ) {
      continue;
    }
    streamStore.publish(previous.id, target);
  }
}

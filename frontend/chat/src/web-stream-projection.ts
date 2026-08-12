import type { ChatMessage } from "./main.tsx";
import { StreamProjectionStore } from "./stream-projection.ts";

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

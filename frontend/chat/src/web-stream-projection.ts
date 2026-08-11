import type { ChatMessage } from "./main.tsx";
import {
  appendCodePoints,
  StreamProjectionStore,
  streamFrameBudget,
} from "./stream-projection.ts";

/** Advance desktop thinking and answer text by exactly one visible code point. */
export function advanceWebStreamPresentation<T extends ChatMessage>(
  current: T,
  target: T,
  elapsedMs: number,
): T {
  if (current.id !== target.id) return target;

  // 1. Use one shared budget without rescanning the full pending text every frame.
  let budget = streamFrameBudget(elapsedMs, 1);

  // 2. Apply tools immediately while progressively revealing textual blocks.
  const projectedBlocks = target.blocks.map((block, index) => {
    if (block.kind !== "thinking") return block;
    const previous = current.blocks[index];
    const previousContent = previous?.kind === "thinking" ? previous.content : "";
    if (!block.content.startsWith(previousContent)) return block;
    if (budget === 0) {
      return previousContent === block.content ? block : { ...block, content: previousContent };
    }
    const advanced = appendCodePoints(previousContent, block.content, budget);
    budget -= advanced.appended;
    return advanced.text === block.content ? block : { ...block, content: advanced.text };
  });
  const blocks = projectedBlocks.every((block, index) => block === target.blocks[index])
    ? target.blocks
    : projectedBlocks;
  const content = appendCodePoints(current.content, target.content, budget).text;
  if (
    content === target.content
    && blocks.every((block, index) => block === target.blocks[index])
  ) {
    return target;
  }
  return { ...target, content, blocks };
}

/** Publish only active assistant mutations; terminal states always bypass smoothing. */
export function publishWebStreamChanges(
  previousMessages: readonly ChatMessage[],
  nextMessages: readonly ChatMessage[],
  streamStore: StreamProjectionStore<ChatMessage>,
  revealImmediately: boolean,
): void {
  const previousById = new Map(previousMessages.map((message) => [message.id, message]));

  // 1. Keep persisted history and user messages on the ordinary React path.
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

    // 2. Stream active text by frame, but reveal the authoritative terminal result now.
    streamStore.publish(
      previous.id,
      previous,
      target,
      revealImmediately || target.streaming !== true,
    );
  }
}

import type { ChatMessage } from "./main.tsx";
import {
  advanceStreamingTexts,
  flushStreamingTexts,
  prepareStreamingTexts,
  StreamProjectionStore,
  type StreamTextIO,
} from "./stream-projection.ts";

function webIO<T extends ChatMessage>(): StreamTextIO<T> {
  return {
    blockCount: (message) => message.blocks.length,
    content: (message) => message.content,
    blockText: (message, index) => (message.blocks[index]?.kind === "thinking" ? message.blocks[index].content : null),
    withContent: (message, content) => ({ ...message, content }),
    withBlockTexts: (message, texts) => ({
      ...message,
      blocks: message.blocks.map((block, blockIndex) => {
        const content = texts.get(blockIndex);
        return content === undefined ? block : { ...block, content };
      }),
    }),
  };
}

export interface WebStreamAdvance {
  <T extends ChatMessage>(current: T, target: T, elapsedMs: number, windowAllowance?: number): T;
  prepare?: <T extends ChatMessage>(current: T, target: T) => T;
  flush?: <T extends ChatMessage>(current: T, target: T) => T;
}

/**
 * 桌面端 thinking 与 answer 文本按 grapheme 逐帧揭示；
 * 工具结构立即生效，token 桶按真实 rAF 时间累积，
 * windowAllowance 由 store 的 rolling 1s ledger 提供。
 */
export const advanceWebStreamPresentation: WebStreamAdvance = Object.assign(
  <T extends ChatMessage>(current: T, target: T, elapsedMs: number, windowAllowance?: number): T =>
    advanceStreamingTexts(current, target, elapsedMs, webIO<T>(), windowAllowance),
  {
    prepare: <T extends ChatMessage>(current: T, target: T): T =>
      prepareStreamingTexts(current, target, webIO<T>()),
    flush: <T extends ChatMessage>(current: T, target: T): T =>
      flushStreamingTexts(current, target),
  },
);

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

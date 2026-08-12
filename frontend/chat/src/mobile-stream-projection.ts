import {
  advanceStreamingTexts,
  flushStreamingTexts,
  prepareStreamingTexts,
  streamFrameBudget,
  type StreamTextIO,
} from "./stream-projection.ts";

export {
  StreamProjectionStore as MobileStreamProjectionStore,
  attachReducedMotionFlush,
} from "./stream-projection.ts";
export type {
  StreamFrameScheduler as MobileStreamFrameScheduler,
} from "./stream-projection.ts";

export interface MobileStreamPresentationBlock {
  id: string;
  kind: "thinking" | "tool";
  detail: string;
}

export interface MobileStreamPresentationMessage {
  id: string;
  content: string;
  blocks: MobileStreamPresentationBlock[];
}

function mobileIO<T extends MobileStreamPresentationMessage>(): StreamTextIO<T> {
  return {
    blockCount: (message) => message.blocks.length,
    content: (message) => message.content,
    blockText: (message, index) => (message.blocks[index]?.kind === "thinking" ? message.blocks[index].detail : null),
    withContent: (message, content) => ({ ...message, content }),
    withBlockTexts: (message, texts) => ({
      ...message,
      blocks: message.blocks.map((block, blockIndex) => {
        const detail = texts.get(blockIndex);
        return detail === undefined ? block : { ...block, detail };
      }),
    }),
  };
}

export interface MobileStreamAdvance {
  <T extends MobileStreamPresentationMessage>(current: T, target: T, elapsedMs: number, windowAllowance?: number): T;
  prepare?: <T extends MobileStreamPresentationMessage>(current: T, target: T) => T;
  flush?: <T extends MobileStreamPresentationMessage>(current: T, target: T) => T;
}

/**
 * 每帧推进：token 桶按真实 rAF 时间累积，thinking 与 answer 公平分配，
 * 结构字段（tool 块）立即生效，文本按 grapheme 逐步揭示；
 * windowAllowance 由 store 的 rolling 1s ledger 提供。
 */
export const advanceMobileStreamPresentation: MobileStreamAdvance = Object.assign(
  <T extends MobileStreamPresentationMessage>(current: T, target: T, elapsedMs: number, windowAllowance?: number): T =>
    advanceStreamingTexts(current, target, elapsedMs, mobileIO<T>(), windowAllowance),
  {
    prepare: <T extends MobileStreamPresentationMessage>(current: T, target: T): T =>
      prepareStreamingTexts(current, target, mobileIO<T>()),
    flush: <T extends MobileStreamPresentationMessage>(current: T, target: T): T =>
      flushStreamingTexts(current, target),
  },
);

export function mobileStreamFrameBudget(elapsedMs: number, backlog: number): number {
  return streamFrameBudget(elapsedMs, backlog);
}

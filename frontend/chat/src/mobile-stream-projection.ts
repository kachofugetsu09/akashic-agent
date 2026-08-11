import {
  appendCodePoints,
  streamFrameBudget,
} from "./stream-projection.ts";

export {
  StreamProjectionStore as MobileStreamProjectionStore,
} from "./stream-projection.ts";
export type {
  StreamAdvance as MobileStreamAdvance,
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

/** Advance text by a frame-sized Unicode slice while applying structural fields immediately. */
export function advanceMobileStreamPresentation<T extends MobileStreamPresentationMessage>(
  current: T,
  target: T,
  elapsedMs: number,
): T {
  if (current.id !== target.id) return target;

  // 1. Keep every visible frame to one code point without rescanning the backlog.
  let budget = mobileStreamFrameBudget(elapsedMs, 1);

  // 2. Apply tool structure immediately, then reveal thinking and answer text in order.
  const projectedBlocks = target.blocks.map((block, index) => {
    if (block.kind !== "thinking") return block;
    const previous = current.blocks[index];
    const previousDetail = previous?.kind === "thinking" && previous.id === block.id
      ? previous.detail
      : "";
    if (!block.detail.startsWith(previousDetail)) return block;
    if (budget === 0) return previousDetail === block.detail ? block : { ...block, detail: previousDetail };
    const advanced = appendCodePoints(previousDetail, block.detail, budget);
    budget -= advanced.appended;
    return advanced.text === block.detail ? block : { ...block, detail: advanced.text };
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

export function mobileStreamFrameBudget(elapsedMs: number, backlog: number): number {
  return streamFrameBudget(elapsedMs, backlog);
}

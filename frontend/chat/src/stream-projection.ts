/** 基准推进时间片：仅用于第一帧的默认经过时间。 */
export const STREAM_FRAME_MS = 1000 / 60;

/** 单帧最大时间贡献：隐藏标签页恢复时不会一次刷出一大片。 */
export const STREAM_MAX_FRAME_MS = 250;

/** 基准速率（grapheme/秒）：无积压时也不低于该速率。 */
export const STREAM_BASE_RATE_CPS = 120;

/** 积压每增加一个 grapheme，速率提升的幅度（grapheme/秒）。 */
export const STREAM_RATE_PER_BACKLOG_CPS = 10;

/** 速率上限：积压再大也不会超过该值，保证追赶有界。 */
export const STREAM_MAX_RATE_CPS = 600;

/** 单帧最大揭示 grapheme 数：每帧变化保持视觉连续，不出现大片跳变。 */
export const STREAM_FRAME_BUDGET_CAP = 12;

/** rolling 窗口长度：揭示节奏以真实 rAF 时间戳记账的半开连续窗口为契约。 */
export const STREAM_WINDOW_MS = 1000;

/** 任意半开连续 1000ms 窗口的揭示 grapheme 硬上限（与 STREAM_MAX_RATE_CPS 一致）。 */
export const STREAM_WINDOW_REVEAL_CAP = 600;

/** 正 token carry 上限：约一个基准帧在 600 g/s 下产生的 credit（≈10），负债不受限。 */
export const STREAM_TOKEN_CARRY_MAX = (STREAM_MAX_RATE_CPS * STREAM_FRAME_MS) / 1000;

export interface StreamFrameScheduler {
  request(callback: (timestamp: number) => void): number;
  cancel(handle: number): void;
}

/** 在 publish 时把新增 delta 一次性分割进投影状态的入口。 */
export type StreamPrepare<T> = (current: T, target: T) => T;

/** 一次性揭示全部积压：把投影精确推进到最新 target，不做任何伪造。 */
export type StreamFlush<T> = (current: T, target: T) => T;

/** 帧推进函数：可选携带 prepare，让 store 把分割提前到 publish 时刻。 */
export interface StreamAdvance<T> {
  (current: T, target: T, elapsedMs: number, windowAllowance?: number): T;
  prepare?: StreamPrepare<T>;
  flush?: StreamFlush<T>;
}

/**
 * 按字段缓存的 grapheme 队列：delta 文本 + 每段结束偏移 + 已消费头指针。
 * tail 是已揭示部分的最后一个 grapheme：队列排空后，新 delta 可能扩展它，
 * 需要它来做跨 delta 的原子修正（权威 target 分割是唯一字素 owner）。
 */
export interface StreamQueuedText {
  text: string;
  bounds: number[];
  head: number;
  tail: string;
}

/** 挂在投影消息上的流式状态：队列、增量 count、分数 token 桶。 */
export interface StreamProjectionState {
  target: unknown;
  content: StreamQueuedText | null;
  blocks: (StreamQueuedText | null)[];
  queued: number;
  token: number;
  fresh: boolean;
}

export const STREAM_STATE: unique symbol = Symbol("akashic.stream-projection-state");

export function streamStateOf<T>(projection: T): StreamProjectionState | undefined {
  if (typeof projection !== "object" || projection === null) return undefined;
  return (projection as { [STREAM_STATE]?: StreamProjectionState })[STREAM_STATE];
}

/** 消息形状适配器：引擎借此在 web / mobile 两种消息结构上通用。 */
export interface StreamTextIO<T> {
  blockCount(message: T): number;
  content(message: T): string;
  blockText(message: T, index: number): string | null;
  withContent(message: T, content: string): T;
  /**
   * 一次批量 immutable block 文本更新：把 texts 里每个 index 的可见文本原子
   * 替换进返回副本。prepare 与 advance 每帧至多调用一次 —— 单次 O(B) map/copy
   * 完成全部块更新；逐块重复全列表复制会让 B 个 thinking 块退化成 O(B²)。
   */
  withBlockTexts(message: T, texts: ReadonlyMap<number, string>): T;
}

/**
 * Intl.Segmenter 是 grapheme 分割的唯一依据，不可用必须 fail-loud：
 * 绝不静默退化成按码点分割（会拆开 ZWJ/组合序列）。
 */
const graphemeSegmenter: Intl.Segmenter = (() => {
  if (typeof Intl === "undefined" || typeof Intl.Segmenter !== "function") {
    throw new Error("stream projection requires Intl.Segmenter for grapheme segmentation");
  }
  return new Intl.Segmenter(undefined, { granularity: "grapheme" });
})();

/** 统计字符串的 grapheme 数（ZWJ 组合与 emoji 序列算一个）。 */
export function graphemeCount(text: string): number {
  return Array.from(graphemeSegmenter.segment(text)).length;
}

/** 计算 current 之后追加的 grapheme 数；非前缀关系返回 0。 */
export function appendedGraphemeCount(current: string, target: string): number {
  if (!target.startsWith(current)) return 0;
  return graphemeCount(target.slice(current.length));
}

/** 有界速率：rate = clamp(120 + 10 * backlog, 120, 600) grapheme/秒。 */
export function streamRate(backlog: number): number {
  return Math.min(
    STREAM_MAX_RATE_CPS,
    Math.max(STREAM_BASE_RATE_CPS, STREAM_BASE_RATE_CPS + STREAM_RATE_PER_BACKLOG_CPS * backlog),
  );
}

/** 纯时间片预算：按当前积压速率积累的 token 数，封顶单帧预算。不含 fresh 保底。 */
export function streamFrameBudget(elapsedMs: number, backlog: number): number {
  if (backlog <= 0) return 0;
  const boundedElapsed = Math.min(STREAM_MAX_FRAME_MS, Math.max(0, elapsedMs));
  const tokens = (streamRate(backlog) * boundedElapsed) / 1000;
  return Math.min(STREAM_FRAME_BUDGET_CAP, Math.floor(tokens));
}

function newQueue(): StreamQueuedText {
  return { text: "", bounds: [], head: 0, tail: "" };
}

/**
 * 把距最近整数 ≤ 1e-9 的浮点残余规范为该整数：
 * rawElapsedMs 在 30s 量级 timestamp 上做减法时，双精度 ulp 噪声会把本应是
 * 整数的 token 推离整数边界（如 5.000000000001455、0.999999999999273），
 * 不归整会让噪声逐帧累积、偶发跨过整数预算，造成一帧多揭/少揭。
 * 真实非近整数分数与负债（如 2.3338、-0.86）原样保留。
 */
function normalizeTokenNoise(token: number): number {
  const nearest = Math.round(token);
  return Math.abs(token - nearest) <= 1e-9 ? nearest : token;
}

function queuedCount(queue: StreamQueuedText | null): number {
  return queue === null ? 0 : queue.bounds.length - queue.head;
}

/** 把 text 按 Intl.Segmenter 分割，把每段结束偏移（相对 base）追加进队列。 */
function pushSegments(queue: StreamQueuedText, text: string, base: number): void {
  for (const segment of graphemeSegmenter.segment(text)) {
    queue.bounds.push(base + segment.index + segment.segment.length);
  }
}

/** 把 text 分割成 grapheme 数组（权威 target 分割的唯一实现来源）。 */
function segmentsOf(text: string): string[] {
  return Array.from(graphemeSegmenter.segment(text), (segment) => segment.segment);
}

/** 取文本最后一个 grapheme：仅用于队列为空但可见文本非空的重入场景。 */
function lastGrapheme(text: string): string {
  let last = "";
  for (const segment of graphemeSegmenter.segment(text)) last = segment.segment;
  return last;
}

/**
 * 把 delta 追加进队列，并把可能被扩展的尾部字素与新 delta 一起重分割：
 * 权威 target 的 Intl.Segmenter 分段是唯一字素 owner，任何跨 delta 扩展
 * （组合音标/肤色修饰/区域指示符/ZWJ 序列）都必须按最新 target 修正边界。
 * 返回修正后的可见文本（无扩展时原样返回）。
 *
 * 阶段 1：全新队列且无可见上下文 —— 整段独立分割，无跨 delta 问题。
 * 阶段 2：还有未揭示项 —— 最后一项就是接缝；把 [最后一项 + delta] 作为
 *   window 重分割并替换其边界。接缝项起点必为权威字素边界（前项都是偶数
 *   个 RI 的完整簇），window 单独分割即全文本分割，无需扫描 backlog。
 * 阶段 3：已全部揭示 —— 可见尾字素可能被 delta 扩展；把 [可见尾 + delta]
 *   重分割。首个簇若越过旧尾边界，可见尾在同一提交内原子替换为完整新簇
 *   （不新增 grapheme、不消耗 pacing/rolling 配额），其余簇入队等待揭示。
 */
function appendQueued(queue: StreamQueuedText, deltaText: string, visible: string): string {
  if (queue.bounds.length === 0 && visible.length === 0) {
    queue.text += deltaText;
    pushSegments(queue, deltaText, 0);
    return visible;
  }
  if (queue.bounds.length > 0 && queue.bounds.length - queue.head > 0) {
    const junctionStart = queue.bounds.length >= 2 ? queue.bounds[queue.bounds.length - 2] : 0;
    const window = queue.text.slice(junctionStart) + deltaText;
    queue.text += deltaText;
    queue.bounds.length = queue.bounds.length - 1;
    pushSegments(queue, window, junctionStart);
    return visible;
  }
  if (queue.tail === "") queue.tail = lastGrapheme(visible);
  const oldTail = queue.tail;
  const clusters = segmentsOf(oldTail + deltaText);
  const merged = clusters[0].length > oldTail.length;
  if (merged) queue.tail = clusters[0];
  queue.text = clusters.slice(1).join("");
  queue.bounds = [];
  queue.head = 0;
  pushSegments(queue, queue.text, 0);
  return merged ? visible.slice(0, visible.length - oldTail.length) + clusters[0] : visible;
}

/** 摊还 O(1) 的队列压缩：只有头部远超存活区时才搬移，绝不每帧扫描。 */
function compactQueue(queue: StreamQueuedText): void {
  if (queue.head < 4096) return;
  if (queue.bounds.length - queue.head > 8192) return;
  const cut = queue.head === 0 ? 0 : queue.bounds[queue.head - 1];
  queue.text = queue.text.slice(cut);
  const remaining = queue.bounds.slice(queue.head);
  // 文本被截掉前 cut 个字符后，边界偏移必须重新以新文本为原点：
  // 否则下一次揭示会用旧绝对偏移切片，把整段积压重复刷到可见文本里。
  for (let index = 0; index < remaining.length; index += 1) remaining[index] -= cut;
  queue.bounds = remaining;
  queue.head = 0;
}

/** 揭示前 take 个 grapheme：推进队列头，记录可见尾字素，返回揭示文本与实际数。 */
function cutQueue(queue: StreamQueuedText, take: number): { text: string; count: number } {
  const start = queue.head === 0 ? 0 : queue.bounds[queue.head - 1];
  const end = queue.bounds[queue.head + take - 1];
  const text = queue.text.slice(start, end);
  if (take > 0) {
    const lastStart = queue.head + take - 1 === 0 ? 0 : queue.bounds[queue.head + take - 2];
    queue.tail = queue.text.slice(lastStart, end);
  }
  queue.head += take;
  compactQueue(queue);
  return { text, count: take };
}

/**
 * 从队列头部消费至多 budget 个 grapheme；队列边界以最新 target 的
 * Intl.Segmenter 分段为准（publish 时已修正），逐项揭示天然保持
 * EGC 序列前缀，无需再向后扩展。maxTake（rolling 1s ledger 余量）
 * 是实际揭示数的硬上限，保证任何 1000ms 窗口严格 ≤ 600。
 */
function revealQueued(queue: StreamQueuedText, budget: number, maxTake?: number): { text: string; count: number } {
  const pending = queue.bounds.length - queue.head;
  const take = Math.min(budget, pending, maxTake === undefined ? pending : maxTake);
  if (take <= 0) return { text: "", count: 0 };
  return cutQueue(queue, take);
}

function attachState<T>(projection: T, state: StreamProjectionState): T {
  const copy = Object.assign({}, projection);
  Object.defineProperty(copy, STREAM_STATE, { value: state, enumerable: false });
  return copy as T;
}

/**
 * 为每个 thinking 块分割增量并维护各自队列：跨 delta 扩展的块尾字素在
 * 本提交内原子修正，块与块之间完全隔离；非前缀纠正的块立即走权威文本。
 * 返回每块修正后的可见文本（null 表示保持当前可见）。
 */
function prepareThinkingBlocks<T>(
  state: StreamProjectionState,
  current: T,
  target: T,
  io: StreamTextIO<T>,
  extendsReference: boolean,
  previousTarget: unknown,
  previousState: StreamProjectionState | undefined,
): (string | null)[] {
  const visibleBlocks: (string | null)[] = [];
  for (let index = 0; index < io.blockCount(target); index += 1) {
    const targetText = io.blockText(target, index);
    if (targetText === null) {
      state.blocks.push(null);
      visibleBlocks.push(null);
      continue;
    }
    const previousText = io.blockText(current, index);
    const referenceText = extendsReference && previousTarget !== undefined
      ? io.blockText(previousTarget as T, index)
      : previousText;
    if (previousText === null) {
      const queue = newQueue();
      if (targetText.length > 0) appendQueued(queue, targetText, "");
      state.blocks.push(queue);
      visibleBlocks.push(null);
      continue;
    }
    if (referenceText !== null && targetText.startsWith(referenceText)) {
      const queue = extendsReference ? (previousState?.blocks[index] ?? newQueue()) : newQueue();
      const delta = targetText.slice(referenceText.length);
      visibleBlocks.push(delta.length > 0 ? appendQueued(queue, delta, previousText) : null);
      state.blocks.push(queue);
      continue;
    }
    if (targetText.startsWith(previousText)) {
      const queue = newQueue();
      const delta = targetText.slice(previousText.length);
      visibleBlocks.push(delta.length > 0 ? appendQueued(queue, delta, previousText) : null);
      state.blocks.push(queue);
      continue;
    }
    // 非前缀 block 纠正：立即展示权威文本。
    state.blocks.push(null);
    visibleBlocks.push(null);
  }
  return visibleBlocks;
}

/**
 * publish 时刻执行：只分割新增 delta（相对上一个已分割 target 的增量），
 * 把 grapheme 队列与增量 count 存入投影状态。已入队但未揭示的部分绝不复扫、
 * 绝不重复入队；跨界扩展的尾部字素与新 delta 一起重分割，可见尾被扩展时
 * 在同一提交内原子替换为完整新簇（不新增 grapheme、不消耗 pacing 配额）。
 * 非前缀纠正直接返回权威 target。
 */
export function prepareStreamingTexts<T>(current: T, target: T, io: StreamTextIO<T>): T {
  const previousState = streamStateOf(current);
  const currentContent = io.content(current);
  const targetContent = io.content(target);
  if (!targetContent.startsWith(currentContent)) return target;

  // 增量参照优先取上一个已分割 target 的文本：visible 可能落后于它，
  // 若按 visible 切 delta 会把已经入队的部分重复入队。
  const previousTarget = previousState?.target;
  const referenceContent = previousTarget === undefined
    ? currentContent
    : io.content(previousTarget as T);
  const extendsReference = targetContent.startsWith(referenceContent);

  const contentQueue = extendsReference ? (previousState?.content ?? newQueue()) : newQueue();
  const contentDelta = extendsReference
    ? targetContent.slice(referenceContent.length)
    : targetContent.slice(currentContent.length);
  const state: StreamProjectionState = {
    target,
    content: contentQueue,
    blocks: [],
    queued: 0,
    token: previousState?.token ?? 0,
    fresh: true,
  };
  const visibleContent = contentDelta.length > 0 ? appendQueued(contentQueue, contentDelta, currentContent) : currentContent;
  const visibleBlocks = prepareThinkingBlocks(state, current, target, io, extendsReference, previousTarget, previousState);

  // 复用队列里可能还有未揭示部分：增量 count 从队列本身重算，不重复计数。
  state.queued = queuedCount(state.content);
  for (const queue of state.blocks) state.queued += queuedCount(queue);

  // 投影保持当前可见文本：target 的权威字段（结构、工具块）立即生效，
  // 文本增量只进队列，等帧推进逐步揭示；所有 thinking 块文本变化在一次
  // 批量 immutable 更新里落地（单次 adapter map/copy，不逐块克隆全列表）。
  let projection: T = Object.assign({}, target);
  projection = io.withContent(projection, visibleContent);
  const blockTexts = new Map<number, string>();
  for (let index = 0; index < state.blocks.length; index += 1) {
    if (state.blocks[index] === null) continue;
    // 新块（previous 为 null）需要显式置空等待揭示；已到权威文本的块不重写。
    const overrideText = visibleBlocks[index] ?? io.blockText(current, index) ?? "";
    if (overrideText !== io.blockText(target, index)) blockTexts.set(index, overrideText);
  }
  if (blockTexts.size > 0) projection = io.withBlockTexts(projection, blockTexts);
  return attachState(projection, state);
}

/**
 * 每帧推进：O(1) 看状态，分数 token 桶按真实 rAF 时间累积，
 * 单帧预算封顶 12，thinking 与 answer 按积压比例公平分配，双向最低份额：
 * 预算 ≥ 2 且两条 lane 都有积压时各自至少 1（不超各自 pending）；
 * budget=1 保持 answer 首帧优先，下一帧最低份额保证另一 lane 也能获得预算。
 * windowAllowance 由 store 的 rolling 1s ledger 给出：本帧实际揭示数严格
 * ≤ 该余量，保证任意半开连续 1000ms 窗口 ≤ 600，hidden 恢复首帧同样受限。
 * 队列边界已在 publish 时按最新 target 修正，逐项揭示不拆 EGC；
 * 不扫描完整 backlog，也不每帧重新分割。
 */
export function advanceStreamingTexts<T>(
  current: T,
  target: T,
  elapsedMs: number,
  io: StreamTextIO<T>,
  windowAllowance?: number,
): T {
  let state = streamStateOf(current);
  if (state === undefined || state.target !== target) {
    const prepared = prepareStreamingTexts(current, target, io);
    if (prepared === target) return target;
    state = streamStateOf(prepared) as StreamProjectionState;
    current = prepared;
  }
  if (state.queued === 0) return attachState(target, state);

  const elapsed = Math.min(STREAM_MAX_FRAME_MS, Math.max(0, elapsedMs));
  // 正余额只许 carry 一个基准帧的 credit：30s 隐藏恢复等长帧不能靠越攒越多，
  // 把下一秒推过 600 g/s 硬上限；负债（负数）原样保留，由后续帧偿还。
  state.token = Math.min(STREAM_TOKEN_CARRY_MAX, state.token + (streamRate(state.queued) * elapsed) / 1000);
  // 预算决策前归整：把 rawElapsed 的 ulp 噪声吸收在整数边界上，
  // 避免本应是 5 的 earned 被 floor 成 4、或残余 + earned 被推成 6。
  state.token = normalizeTokenNoise(state.token);
  let budget = Math.floor(state.token);
  // fresh 保底只在下一次 rAF 且未欠债时生效：揭示扣减造成的负债务
  // 必须先由后续帧偿还（暂停揭示），不能靠每次 publish 重置 fresh 白拿吞吐。
  if (state.fresh && budget < 1 && state.token >= 0) budget = 1;
  if (budget < 1) return current;
  // rolling 1s ledger 的余量是硬上限：窗口已满时整帧按住（包括 fresh 保底），
  // 恢复首帧与 hidden 后的追帧同样受限，不靠 token carry 积累无限 credit。
  if (windowAllowance !== undefined) budget = Math.min(budget, windowAllowance);
  if (budget < 1) return current;
  if (budget > STREAM_FRAME_BUDGET_CAP) budget = STREAM_FRAME_BUDGET_CAP;

  const contentPending = queuedCount(state.content);
  const thinkingIndexes: number[] = [];
  let thinkingPending = 0;
  for (let index = 0; index < state.blocks.length; index += 1) {
    const queue = state.blocks[index];
    if (queue === null) continue;
    thinkingIndexes.push(index);
    thinkingPending += queuedCount(queue);
  }
  const total = contentPending + thinkingPending;

  let contentBudget = 0;
  let thinkingBudget = 0;
  if (budget >= total) {
    contentBudget = contentPending;
    thinkingBudget = thinkingPending;
  } else if (contentPending > 0 && thinkingPending > 0 && budget >= 2) {
    // 双向最低份额：预算 ≥ 2 且两条 lane 都有积压时各自至少 1，且不超过各自 pending。
    // 先按积压比例取 content 目标，夹到 [1, min(contentPending, budget - 1)]，
    // 剩余归 thinking 并夹到 [1, thinkingPending]；夹掉的部分回退给另一条 lane，
    // 预算始终整额用满（不降低总吞吐，也不突破单帧 12 上限）。
    contentBudget = Math.min(contentPending, Math.max(1, Math.round((budget * contentPending) / total)));
    if (contentBudget > budget - 1) contentBudget = budget - 1;
    thinkingBudget = budget - contentBudget;
    if (thinkingBudget > thinkingPending) {
      thinkingBudget = thinkingPending;
      contentBudget = budget - thinkingBudget;
    }
  } else if (contentPending > 0) {
    contentBudget = Math.min(contentPending, budget);
  } else {
    thinkingBudget = budget;
  }

  // thinking 队列间按各自积压比例分配，余数逐个补足，避免前排队列独吞预算。
  const shares = new Array<number>(thinkingIndexes.length).fill(0);
  if (thinkingBudget > 0) {
    if (thinkingBudget >= thinkingPending) {
      for (let i = 0; i < thinkingIndexes.length; i += 1) {
        shares[i] = queuedCount(state.blocks[thinkingIndexes[i]]);
      }
    } else {
      let remainder = thinkingBudget;
      for (let i = 0; i < thinkingIndexes.length; i += 1) {
        const pending = queuedCount(state.blocks[thinkingIndexes[i]]);
        shares[i] = Math.min(pending, Math.floor((thinkingBudget * pending) / thinkingPending));
        remainder -= shares[i];
      }
      for (let i = 0; remainder > 0; i = (i + 1) % thinkingIndexes.length) {
        const pending = queuedCount(state.blocks[thinkingIndexes[i]]);
        if (shares[i] < pending) {
          shares[i] += 1;
          remainder -= 1;
        }
      }
    }
  }

  let next: T = target;
  // 只按实际揭示的 grapheme 数扣 token：队列边界权威，揭示即完整 EGC。
  let revealedCount = 0;
  if (contentBudget > 0 && state.content !== null) {
    const revealed = revealQueued(state.content, contentBudget);
    revealedCount += revealed.count;
    next = io.withContent(next, io.content(current) + revealed.text);
  }
  // 所有 thinking 块文本变化一次性批量落地：一帧至多一次 adapter map/copy；
  // share=0 且已到权威文本的块不重写（零复制），未揭示的可见前缀仍需保留。
  const blockTexts = new Map<number, string>();
  for (let i = 0; i < thinkingIndexes.length; i += 1) {
    const queue = state.blocks[thinkingIndexes[i]];
    if (queue === null) continue;
    let text = io.blockText(current, thinkingIndexes[i]) ?? "";
    if (shares[i] > 0) {
      const revealed = revealQueued(queue, shares[i]);
      revealedCount += revealed.count;
      text += revealed.text;
    }
    if (text !== io.blockText(target, thinkingIndexes[i])) blockTexts.set(thinkingIndexes[i], text);
  }
  if (blockTexts.size > 0) next = io.withBlockTexts(next, blockTexts);

  state.queued = queuedCount(state.content);
  for (const queue of state.blocks) state.queued += queuedCount(queue);
  state.token -= revealedCount;
  // 揭示扣减后归整：距最近整数 ≤ 1e-9 的残余规范为整数，不把噪声留给后续帧累积。
  state.token = normalizeTokenNoise(state.token);
  state.fresh = false;
  return attachState(next, state);
}

/**
 * 一次性揭示全部积压（reduced-motion 切换或显式 flush）：
 * 精确推进到权威 target 的文本与结构字段，不伪造任何内容；
 * 清空队列并归零 token，随后若再有 publish，按 clean 状态重新分割。
 */
export function flushStreamingTexts<T>(current: T, target: T): T {
  void current;
  return attachState(target, {
    target,
    content: null,
    blocks: [],
    queued: 0,
    token: 0,
    fresh: false,
  });
}

interface PendingProjection<T> {
  previousId: string;
  target: T;
}

/**
 * 每条消息的 rolling reveal ledger：以真实 rAF 时间戳记录 (timestamp, count)，
 * 维护最近 ~1000ms 内的揭示总和。advanceFrame 是唯一 owner：逐帧 prune 超龄
 * 事件，据此计算本帧 windowAllowance = 600 - 窗口内已揭示数。
 * 闭左窗口 [t - 1000 - EPS, t] 保证严格覆盖任何枚举得到的半开窗口
 * [t, t + 1000)（含 hidden 恢复首帧），EPS 吸收 1s 边界上的浮点 ulp 噪声。
 * hidden 期间不产生帧事件，恢复首帧照常记账：窗口只清算真实时间内的揭示，
 * 无法积累无限 credit。
 */
interface RevealLedger {
  events: { t: number; count: number }[];
  head: number;
  sum: number;
}

/** Keep stream presentation outside the app root and notify only affected message rows. */
export class StreamProjectionStore<T extends { id: string }> {
  private readonly scheduler: StreamFrameScheduler;
  private readonly advance: StreamAdvance<T>;
  private readonly projections = new Map<string, T>();
  private readonly pending = new Map<string, PendingProjection<T>>();
  private readonly ledgers = new Map<string, RevealLedger>();
  private readonly listeners = new Map<string, Set<() => void>>();
  private frameHandle: number | null = null;
  private lastFrameAt: number | null = null;

  constructor(
    scheduler: StreamFrameScheduler,
    advance: StreamAdvance<T>,
  ) {
    this.scheduler = scheduler;
    this.advance = advance;
  }

  read(messageId: string, fallback: T): T {
    return this.projections.get(messageId) ?? fallback;
  }

  subscribe(messageId: string, listener: () => void): () => void {
    const listeners = this.listeners.get(messageId) ?? new Set();
    listeners.add(listener);
    this.listeners.set(messageId, listeners);
    return () => {
      listeners.delete(listener);
      if (listeners.size === 0) this.listeners.delete(messageId);
    };
  }

  /**
   * 发布不可变 target。publish 时只分割新增 delta 并存入投影状态；
   * token 桶跨发布轮次连续累积，刷新率不再影响排空节奏。
   */
  publish(previousId: string, previous: T, target: T, immediate: boolean): void {
    const current = this.projections.get(previousId) ?? previous;
    if (immediate) {
      this.pending.delete(previousId);
      this.setProjection(previousId, target);
      if (target.id !== previousId) this.setProjection(target.id, target);
      this.cancelFrameWhenIdle();
      return;
    }
    const prepare = this.advance.prepare;
    const prepared = prepare !== undefined ? prepare(current, target) : current;
    this.setProjection(previousId, prepared);
    if (target.id !== previousId) this.setProjection(target.id, prepared);
    const queued = prepare !== undefined ? (streamStateOf(prepared)?.queued ?? 0) : 1;

    this.pending.set(previousId, { previousId, target });
    if (queued === 0) {
      this.pending.delete(previousId);
      this.cancelFrameWhenIdle();
      return;
    }
    if (this.frameHandle === null) {
      this.frameHandle = this.scheduler.request(this.advanceFrame);
    }
  }

  /**
   * 立即把所有 pending 积压精确推进到最新 target：取消已排 rAF，
   * 清理 token/carry/队列状态；只通知实际受影响的消息行（每行一次）。
   * 不改写权威消息（streaming、结构等字段保持 target 原值），不伪造终态。
   */
  flushAll(): void {
    if (this.frameHandle !== null) {
      this.scheduler.cancel(this.frameHandle);
      this.frameHandle = null;
    }
    this.lastFrameAt = null;
    if (this.pending.size === 0) return;
    const flush = this.advance.flush;
    for (const [key, projection] of this.pending) {
      const current = this.projections.get(key);
      if (current === undefined) throw new Error(`stream projection missing current message: ${key}`);
      const flushed = flush !== undefined ? flush(current, projection.target) : projection.target;
      this.setProjection(projection.previousId, flushed);
      if (flushed.id !== projection.previousId) this.setProjection(flushed.id, flushed);
      // flush 即时全量揭示属于合同豁免：不计入窗口，后续展示从干净窗口重新记账。
      this.ledgers.delete(key);
    }
    this.pending.clear();
  }

  /** Drop projections already committed into the React-owned coarse snapshot. */
  reconcileBaseline(messages: readonly T[]): void {
    const baseline = new Map(messages.map((message) => [message.id, message]));
    for (const [key, projection] of this.projections) {
      if (baseline.get(projection.id) === projection) {
        this.projections.delete(key);
        this.ledgers.delete(key);
      } else if ((streamStateOf(projection)?.queued ?? 0) === 0) {
        this.projections.delete(key);
        this.ledgers.delete(key);
      }
    }
  }

  clear(): void {
    if (this.frameHandle !== null) this.scheduler.cancel(this.frameHandle);
    this.frameHandle = null;
    this.lastFrameAt = null;
    this.pending.clear();
    this.projections.clear();
    this.ledgers.clear();
  }

  private readonly advanceFrame = (timestamp: number) => {
    const rawElapsedMs = this.lastFrameAt === null ? STREAM_FRAME_MS : timestamp - this.lastFrameAt;
    this.lastFrameAt = timestamp;
    this.frameHandle = null;

    for (const [key, projection] of this.pending) {
      const current = this.projections.get(key);
      if (current === undefined) throw new Error(`stream projection missing current message: ${key}`);
      let ledger = this.ledgers.get(key);
      if (ledger === undefined) {
        ledger = { events: [], head: 0, sum: 0 };
        this.ledgers.set(key, ledger);
      }
      this.pruneLedger(ledger, timestamp);
      const allowance = ledger.sum >= STREAM_WINDOW_REVEAL_CAP ? 0 : STREAM_WINDOW_REVEAL_CAP - ledger.sum;
      const queuedBefore = streamStateOf(current)?.queued ?? 0;
      const next = this.advance(current, projection.target, rawElapsedMs, allowance);
      this.setProjection(projection.previousId, next);
      if (next.id !== projection.previousId) this.setProjection(next.id, next);
      // 按实际揭示数（含 ZWJ 扩展）记账：ledger 与 token 桶一样只认真实输出。
      const revealed = queuedBefore - (streamStateOf(next)?.queued ?? 0);
      if (revealed > 0) {
        ledger.events.push({ t: timestamp, count: revealed });
        ledger.sum += revealed;
      }
      if ((streamStateOf(next)?.queued ?? 0) === 0) {
        this.pending.delete(key);
        if (ledger.events.length === 0) this.ledgers.delete(key);
      }
    }

    if (this.pending.size > 0) {
      this.frameHandle = this.scheduler.request(this.advanceFrame);
    } else {
      this.lastFrameAt = null;
    }
  };

  /** 移除窗口外的旧事件并维护滚动总和；head 落后时按 compactQueue 同款策略搬移。 */
  private pruneLedger(ledger: RevealLedger, timestamp: number): void {
    const cutoff = timestamp - STREAM_WINDOW_MS - 1e-6;
    while (ledger.head < ledger.events.length && ledger.events[ledger.head].t < cutoff) {
      ledger.sum -= ledger.events[ledger.head].count;
      ledger.head += 1;
    }
    if (ledger.head >= 4096 && ledger.events.length - ledger.head <= 8192) {
      ledger.events = ledger.events.slice(ledger.head);
      ledger.head = 0;
    }
  }

  private setProjection(messageId: string, projection: T): void {
    if (this.projections.get(messageId) === projection) return;
    this.projections.set(messageId, projection);
    const listeners = this.listeners.get(messageId);
    if (!listeners) return;
    for (const listener of listeners) listener();
  }

  private cancelFrameWhenIdle(): void {
    if (this.pending.size > 0 || this.frameHandle === null) return;
    this.scheduler.cancel(this.frameHandle);
    this.frameHandle = null;
    this.lastFrameAt = null;
  }
}

/**
 * 订阅 prefers-reduced-motion 切换：切入 reduce 时立即 flushAll 补齐积压，
 * 即使没有新的 delta 也完整显示 backlog；切回 no-preference 不做任何事，
 * 只影响后续 pacing。返回移除 listener 的清理函数（组件卸载时调用）。
 */
export function attachReducedMotionFlush(
  store: { flushAll(): void },
  media: Pick<MediaQueryList, "addEventListener" | "removeEventListener"> = window.matchMedia("(prefers-reduced-motion: reduce)"),
): () => void {
  const handleChange = (event: MediaQueryListEvent): void => {
    if (event.matches) store.flushAll();
  };
  media.addEventListener("change", handleChange);
  return () => media.removeEventListener("change", handleChange);
}

import assert from "node:assert/strict";
import test from "node:test";

import {
  advanceStreamingTexts,
  attachReducedMotionFlush,
  flushStreamingTexts,
  graphemeCount,
  prepareStreamingTexts,
  streamRate,
  streamStateOf,
} from "./stream-projection.ts";
import {
  advanceMobileStreamPresentation,
  MobileStreamProjectionStore,
  mobileStreamFrameBudget,
} from "./mobile-stream-projection.ts";

class TestFrameScheduler {
  callback = null;
  timestamp = 0;

  request(callback) {
    assert.equal(this.callback, null);
    this.callback = callback;
    return 1;
  }

  cancel() {
    this.callback = null;
  }

  advance(elapsedMs = 16.67) {
    const callback = this.callback;
    assert.notEqual(callback, null);
    this.callback = null;
    this.timestamp += elapsedMs;
    callback(this.timestamp);
  }
}

function message(id, content, detail = "") {
  return {
    id,
    content,
    blocks: detail ? [{ id: "thinking", kind: "thinking", detail }] : [],
  };
}

function percentile(samples, p) {
  const sorted = [...samples].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.ceil(p * (sorted.length - 1)))];
}

function mean(samples) {
  return samples.length === 0 ? 0 : samples.reduce((a, b) => a + b, 0) / samples.length;
}

const testSegmenter = new Intl.Segmenter(undefined, { granularity: "grapheme" });

function graphemesOf(text) {
  return Array.from(testSegmenter.segment(text), (segment) => segment.segment);
}

/**
 * 用 Intl.Segmenter 断言 visible 是最新 target 的 EGC 序列前缀：
 * 不接受仅字符串前缀 —— 组合音标/肤色修饰/区域指示符/ZWJ 序列都会让
 * 字符串前缀不等于 EGC 前缀。
 */
function assertEgcPrefix(targetText, visibleText, label) {
  const targetEgc = graphemesOf(targetText);
  const visibleEgc = graphemesOf(visibleText);
  assert.ok(
    visibleEgc.length <= targetEgc.length
      && targetEgc.slice(0, visibleEgc.length).every((egc, index) => egc === visibleEgc[index]),
    `${label}: visible EGCs ${JSON.stringify(visibleEgc)} must be a prefix of target EGCs ${JSON.stringify(targetEgc)}`,
  );
}

/** 断言 queued 精确等于最新 target 剩余权威 EGC 数（content lane）。 */
function assertQueuedExact(projection, targetText, label) {
  const state = streamStateOf(projection);
  const remaining = graphemeCount(targetText) - graphemeCount(projection.content);
  assert.equal(state.queued, remaining, `${label}: queued ${state.queued} != remaining authoritative EGCs ${remaining}`);
}

/**
 * 用真实 rAF timestamp（单调累积的帧时间）模拟连续流式源：
 * 源按 sourceCps 产出 grapheme，每 chunkGraphemes 个发布一次。
 * 全程断言可见内容始终是权威 source 的精确前缀（无重复、无错序），
 * 排空后按完整字符串断言与权威 source 完全相等。
 * 返回每帧采样的可见滞后（ms）以及最终状态。
 */
function runStreamSimulation(hz, sourceCps, { chunkGraphemes = 10, unit = "流", durationMs = 10_000, exactDrain = true } = {}) {
  const scheduler = new TestFrameScheduler();
  const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
  const frameMs = 1000 / hz;
  const id = "assistant:turn";
  const contentFor = (graphemes) => unit.repeat(graphemes);
  const fallback = message(id, "");
  let published = 0;
  let authoritative = fallback;
  const samples = [];
  let visibleAtEnd = "";

  for (let t = frameMs; t <= durationMs; t += frameMs) {
    const sourceGraphemes = Math.floor((sourceCps * t) / 1000);
    while (published < sourceGraphemes) {
      const next = Math.min(sourceGraphemes, published + chunkGraphemes);
      const target = message(id, contentFor(next));
      store.publish(id, authoritative, target, false);
      authoritative = target;
      published = next;
    }
    if (scheduler.callback !== null) {
      scheduler.advance(frameMs);
    }
    const visible = store.read(id, fallback).content;
    assert.ok(
      authoritative.content.startsWith(visible),
      `visible must stay an exact prefix of the authoritative source at t=${t.toFixed(1)}ms`,
    );
    assertEgcPrefix(authoritative.content, visible, `frame at t=${t.toFixed(1)}ms`);
    const visibleCount = graphemeCount(visible);
    samples.push(((published - visibleCount) * 1000) / sourceCps);
    visibleAtEnd = visible;
  }

  let guard = 0;
  while (scheduler.callback !== null && guard < 2000) {
    scheduler.advance(frameMs);
    guard += 1;
  }
  const drained = store.read(id, fallback).content;
  assert.ok(authoritative.content.startsWith(drained), "drained content must stay an exact source prefix");
  assertEgcPrefix(authoritative.content, drained, "drained content");
  if (exactDrain) {
    assert.equal(drained, authoritative.content, "drained content must exactly equal the authoritative source");
  }
  return { samples, store, id, fallback, published, authoritative, drained, visibleAtEnd };
}

/**
 * 枚举 (timestamp, revealCount) 轨迹上所有可能的半开窗口起点 [t_i, t_i + 1000)，
 * 返回其中最大揭示 grapheme 数。窗口按 t_j - t_i < 1000 判定，与 ledger 记账
 * 的闭左窗口同算术，浮点边界不会产生 601 逃逸。
 */
function rollingWindowMax(trace) {
  const prefix = [0];
  for (const entry of trace) prefix.push(prefix[prefix.length - 1] + entry.count);
  let max = 0;
  for (let start = 0; start < trace.length; start += 1) {
    let end = start;
    while (end + 1 < trace.length && trace[end + 1].t - trace[start].t < 1000) end += 1;
    max = Math.max(max, prefix[end + 1] - prefix[start]);
  }
  return max;
}

/**
 * 800 g/s 连续流式输入（按 chunk 分段发布）10 秒，逐帧记录 (timestamp, revealCount)，
 * 返回滚动窗口最大值与 10 秒总揭示数（用于平均速率）。
 */
function continuous800Rolling(hz, chunkGraphemes, unit = "字") {
  const scheduler = new TestFrameScheduler();
  const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
  const id = "assistant:turn";
  const fallback = message(id, "");
  const frameMs = 1000 / hz;
  let published = 0;
  let authoritative = fallback;
  const trace = [];
  let previous = 0;
  for (let t = frameMs; t <= 10_000; t += frameMs) {
    const sourceGraphemes = Math.floor((800 * t) / 1000);
    while (published < sourceGraphemes) {
      const next = Math.min(sourceGraphemes, published + chunkGraphemes);
      const target = message(id, unit.repeat(next));
      store.publish(id, authoritative, target, false);
      authoritative = target;
      published = next;
    }
    if (scheduler.callback !== null) scheduler.advance(frameMs);
    const visible = graphemeCount(store.read(id, fallback).content);
    trace.push({ t: scheduler.timestamp, count: visible - previous });
    previous = visible;
  }
  return { max: rollingWindowMax(trace), total: previous };
}

/**
 * 30s hidden 恢复场景：800 g/s 输入 5 秒后单帧迟到 30 秒，
 * 从恢复那一帧开始记录 (timestamp, revealCount)（恢复帧必须计入），
 * 返回滚动窗口最大值与恢复帧的揭示数。
 */
function hiddenGapMaxWindow(hz, chunkGraphemes, unit = "字") {
  const scheduler = new TestFrameScheduler();
  const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
  const id = "assistant:turn";
  const fallback = message(id, "");
  const frameMs = 1000 / hz;
  let published = 0;
  let authoritative = fallback;
  for (let t = frameMs; t <= 5_000; t += frameMs) {
    const sourceGraphemes = Math.floor((800 * t) / 1000);
    while (published < sourceGraphemes) {
      const next = Math.min(sourceGraphemes, published + chunkGraphemes);
      const target = message(id, unit.repeat(next));
      store.publish(id, authoritative, target, false);
      authoritative = target;
      published = next;
    }
    if (scheduler.callback !== null) scheduler.advance(frameMs);
  }
  const trace = [];
  let previous = graphemeCount(store.read(id, fallback).content);
  scheduler.advance(30_000);
  const recoveryVisible = graphemeCount(store.read(id, fallback).content);
  trace.push({ t: scheduler.timestamp, count: recoveryVisible - previous });
  previous = recoveryVisible;
  for (let frame = 0; frame < hz * 10 && scheduler.callback !== null; frame += 1) {
    scheduler.advance(frameMs);
    const visible = graphemeCount(store.read(id, fallback).content);
    trace.push({ t: scheduler.timestamp, count: visible - previous });
    previous = visible;
  }
  return { max: rollingWindowMax(trace), recoveryCount: trace[0].count };
}

test("after a 30s hidden gap every sliding one-second window including the recovery frame stays under the 600 g/s cap", () => {
  // 轨迹从恢复那一帧开始记录：恢复帧必须计入，且枚举所有窗口起点。
  for (const hz of [60, 90, 120, 144]) {
    for (const chunkGraphemes of [1, 10, 50]) {
      const { max, recoveryCount } = hiddenGapMaxWindow(hz, chunkGraphemes);
      assert.ok(recoveryCount > 0, `${hz}Hz chunk=${chunkGraphemes} recovery frame revealed nothing`);
      // 600 g/s 硬上限，无容差：恢复首帧计入后，任意半开连续 1000ms 窗口仍 ≤ 600。
      assert.ok(max <= 600, `${hz}Hz chunk=${chunkGraphemes} sliding window revealed ${max} > 600`);
    }
  }
});

test("continuous 800 g/s keeps every rolling one-second window under 600 and averages near the cap", () => {
  for (const hz of [60, 90, 120, 144]) {
    for (const chunkGraphemes of [1, 10, 50]) {
      const { max, total } = continuous800Rolling(hz, chunkGraphemes);
      // 逐滚窗检查，不是对齐秒窗/总平均：每个可能的 [t_i, t_i + 1000) 都 ≤ 600。
      assert.ok(max <= 600, `${hz}Hz chunk=${chunkGraphemes} rolling window revealed ${max} > 600`);
      const average = total / 10;
      // 稳定展示接近 600 g/s：不因窗口记账显著掉速，也不超过 600。
      assert.ok(average >= 570 && average <= 600, `${hz}Hz chunk=${chunkGraphemes} average ${average.toFixed(1)} g/s not near 600`);
    }
  }
});

test("stream projection wakes only the subscribed message row", () => {
  const scheduler = new TestFrameScheduler();
  const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
  const before = message("assistant:turn", "正");
  const target = message("assistant:turn", "正在检查流式链路");
  let activeUpdates = 0;
  let historyUpdates = 0;
  store.subscribe("assistant:turn", () => { activeUpdates += 1; });
  store.subscribe("history", () => { historyUpdates += 1; });

  store.publish(before.id, before, target, false);
  assert.equal(store.read(before.id, before).content, "正");
  scheduler.advance();

  assert.equal(store.read(before.id, before).content, "正在检查");
  assert.equal(activeUpdates, 2);
  assert.equal(historyUpdates, 0);
});

test("the first target starts revealing on the very next frame", () => {
  const scheduler = new TestFrameScheduler();
  const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
  const target = message("assistant:turn", "立即启动");

  store.publish("assistant:turn", message("assistant:turn", ""), target, false);
  scheduler.advance(0.1);

  assert.ok(store.read("assistant:turn", target).content.length > 0);
});

test("terminal projection bypasses smoothing and preserves an id migration alias", () => {
  const scheduler = new TestFrameScheduler();
  const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
  const before = message("assistant:turn", "正在");
  const terminal = message("message:canonical", "正在分析完成");

  store.publish(before.id, before, terminal, true);

  assert.equal(store.read(before.id, before), terminal);
  assert.equal(store.read(terminal.id, before), terminal);
  assert.equal(scheduler.callback, null);
});

test("presentation applies tool structure immediately and smooths thinking text", () => {
  const before = message("assistant:turn", "", "先");
  const target = {
    ...message("assistant:turn", "", "先检查调用链"),
    blocks: [
      { id: "thinking", kind: "thinking", detail: "先检查调用链" },
      { id: "tool", kind: "tool", detail: "读取配置", state: "running" },
    ],
  };

  const next = advanceMobileStreamPresentation(before, target, 16.67);

  assert.equal(next.blocks[0].detail, "先检查");
  assert.equal(next.blocks[1], target.blocks[1]);
});

test("answer-only frames preserve the shared process block list", () => {
  const blocks = [{ id: "tool", kind: "tool", detail: "完成", state: "completed" }];
  const before = { id: "assistant:turn", content: "回", blocks };
  const target = { id: "assistant:turn", content: "回答继续", blocks };

  const next = advanceMobileStreamPresentation(before, target, 16.67);

  assert.equal(next.blocks, blocks);
  assert.equal(next.content, "回答继");
});

test("a non-prefix correction replaces the visible text immediately", () => {
  const corrected = message("assistant:turn", "权威纠正");

  const next = advanceMobileStreamPresentation(message("assistant:turn", "旧前缀"), corrected, 16.67);

  assert.equal(next, corrected);
});

test("resetting for a coarse snapshot does not wake streaming rows twice", () => {
  const scheduler = new TestFrameScheduler();
  const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
  const before = message("assistant:turn", "正");
  const target = message("assistant:turn", "正在检查");
  let updates = 0;
  store.subscribe(before.id, () => { updates += 1; });
  store.publish(before.id, before, target, false);
  scheduler.advance();

  store.clear();

  assert.equal(updates, 2);
  assert.equal(store.read(before.id, before), before);
});

test("a 300 grapheme burst drains in under 800ms without batch jumps", () => {
  const scheduler = new TestFrameScheduler();
  const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
  const content = "流式输出".repeat(75);
  const authoritative = message("assistant:turn", "");
  let lastLength = 0;
  let largestFrame = 0;
  store.subscribe(authoritative.id, () => {
    const visible = store.read(authoritative.id, authoritative);
    const length = graphemeCount(visible.content);
    largestFrame = Math.max(largestFrame, length - lastLength);
    lastLength = length;
  });

  store.publish(authoritative.id, authoritative, message(authoritative.id, content), false);
  let frames = 0;
  for (; frames < 600 && scheduler.callback !== null; frames += 1) {
    scheduler.advance(16.67);
  }

  assert.equal(store.read(authoritative.id, authoritative).content, content);
  const drainMs = frames * 16.67;
  assert.ok(drainMs >= 450 && drainMs < 800, `drain time ${drainMs}ms`);
  assert.ok(largestFrame <= 12, `largest frame revealed ${largestFrame}`);
});

test("drain time is refresh-rate independent at 60/90/120/144Hz", () => {
  const content = "流式输出".repeat(75);
  const drainWallMs = (frameMs) => {
    const scheduler = new TestFrameScheduler();
    const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
    const authoritative = message("assistant:turn", "");
    store.publish(authoritative.id, authoritative, message(authoritative.id, content), false);
    let frames = 0;
    for (; frames < 3000 && scheduler.callback !== null; frames += 1) {
      scheduler.advance(frameMs);
    }
    assert.equal(store.read(authoritative.id, authoritative).content, content);
    return frames * frameMs;
  };

  const t60 = drainWallMs(16.67);
  const t90 = drainWallMs(11.11);
  const t120 = drainWallMs(8.33);
  const t144 = drainWallMs(6.94);
  assert.ok(t60 >= 450 && t60 < 800, `60Hz drain ${t60}ms`);
  for (const [label, t] of [["90", t90], ["120", t120], ["144", t144]]) {
    assert.ok(Math.abs(t - t60) / t60 < 0.2, `${label}Hz ${t}ms vs 60Hz ${t60}ms`);
  }
});

test("resuming after a hidden gap reveals only a bounded frame", () => {
  const scheduler = new TestFrameScheduler();
  const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
  const authoritative = message("assistant:turn", "");
  store.publish(authoritative.id, authoritative, message(authoritative.id, "长文".repeat(100)), false);
  scheduler.advance(16.67);
  const afterFirst = store.read(authoritative.id, authoritative).content;
  scheduler.advance(30_000);
  const afterResume = store.read(authoritative.id, authoritative).content;

  const firstRevealed = graphemeCount(afterFirst);
  const resumeRevealed = graphemeCount(afterResume) - firstRevealed;
  assert.ok(firstRevealed <= 12, `first frame revealed ${firstRevealed}`);
  assert.ok(resumeRevealed > 0 && resumeRevealed <= 12, `resume frame revealed ${resumeRevealed}`);
});

test("the answer's first grapheme appears within two frames after a large thinking backlog", () => {
  const scheduler = new TestFrameScheduler();
  const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
  const id = "assistant:turn";
  const thinking = "先".repeat(1000);
  const thinkingOnly = message(id, "", thinking);
  const full = message(id, "答", thinking);
  store.publish(id, message(id, "", ""), thinkingOnly, false);

  let frames = 0;
  while (store.read(id, thinkingOnly).blocks[0].detail.length === 0 && frames < 100) {
    scheduler.advance(16.67);
    frames += 1;
  }
  assert.ok(frames > 0 && frames < 100, `thinking started after ${frames} frames`);

  store.publish(id, thinkingOnly, full, false);
  let answerFrames = 0;
  let content = store.read(id, full).content;
  while (content === "" && answerFrames < 10) {
    scheduler.advance(16.67);
    content = store.read(id, full).content;
    answerFrames += 1;
  }

  assert.equal(content, "答");
  assert.ok(answerFrames <= 2, `answer visible after ${answerFrames} frames`);
  // 主 lane（thinking）未被清空：answer 首字出现时 thinking 仍有积压未揭示。
  const mid = store.read(id, full);
  assert.ok(mid.blocks[0].detail.length < thinking.length, "thinking must not drain while the answer still streams");
  while (scheduler.callback !== null) scheduler.advance(16.67);
  const final = store.read(id, full);
  assert.equal(final.content, "答");
  assert.equal(final.blocks[0].detail, thinking);
});

test("thinking's first grapheme appears within two frames after a large answer backlog", () => {
  const scheduler = new TestFrameScheduler();
  const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
  const id = "assistant:turn";
  const answer = "答".repeat(1000);
  const emptyThinking = [{ id: "thinking", kind: "thinking", detail: "" }];
  const answerOnly = { ...message(id, answer), blocks: emptyThinking };
  const full = { ...message(id, answer), blocks: [{ id: "thinking", kind: "thinking", detail: "思" }] };
  store.publish(id, message(id, "", ""), answerOnly, false);

  let frames = 0;
  while (store.read(id, answerOnly).content.length === 0 && frames < 100) {
    scheduler.advance(16.67);
    frames += 1;
  }
  assert.ok(frames > 0 && frames < 100, `answer started after ${frames} frames`);

  store.publish(id, answerOnly, full, false);
  let thinkingFrames = 0;
  let thinking = store.read(id, full).blocks[0].detail;
  while (thinking === "" && thinkingFrames < 10) {
    scheduler.advance(16.67);
    thinking = store.read(id, full).blocks[0].detail;
    thinkingFrames += 1;
  }

  assert.equal(thinking, "思");
  assert.ok(thinkingFrames <= 2, `thinking visible after ${thinkingFrames} frames`);
  // 主 lane（answer）未被清空：thinking 首字出现时 answer 仍有积压未揭示。
  const mid = store.read(id, full);
  assert.ok(mid.content.length > 0 && mid.content.length < answer.length, "answer must keep streaming, not drain early");
  while (scheduler.callback !== null) scheduler.advance(16.67);
  const final = store.read(id, full);
  assert.equal(final.content, answer);
  assert.equal(final.blocks[0].detail, "思");
});

test("a ZWJ emoji sequence is never split across reveals", () => {
  const target = message("assistant:turn", "👨‍👩‍👧好");
  const next = advanceMobileStreamPresentation(message("assistant:turn", ""), target, 0);

  assert.equal(next.content, "👨‍👩‍👧");
  assert.equal(Array.from(next.content).length, 5);
});

test("cross-delta ZWJ sequences never show a dangling joiner", () => {
  const scheduler = new TestFrameScheduler();
  const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
  const id = "assistant:turn";
  const fallback = message(id, "");
  const firstTarget = message(id, "👨");
  store.publish(id, fallback, firstTarget, false);
  scheduler.advance(16.67);
  assert.equal(store.read(id, fallback).content, "👨");

  // 后续 delta 从 ZWJ 续接开始：已揭示尾簇被扩展，同一次 publish 原子替换
  // 为完整新簇，任何时刻可见文本都是最新 target 的 EGC 序列前缀。
  const fullTarget = message(id, "👨‍👩‍👧好");
  store.publish(id, firstTarget, fullTarget, false);
  const afterPublish = store.read(id, fallback);
  assert.equal(afterPublish.content, "👨‍👩‍👧", "atomic replace right at publish");
  assertEgcPrefix(fullTarget.content, afterPublish.content, "after atomic publish");
  assert.equal(streamStateOf(afterPublish).queued, 1, "only 好 stays queued");
  let frames = 0;
  let visible = afterPublish;
  while (visible.content !== "👨‍👩‍👧好" && frames < 600) {
    scheduler.advance(16.67);
    frames += 1;
    visible = store.read(id, fallback);
    assertEgcPrefix(fullTarget.content, visible.content, `frame ${frames}`);
  }
  assert.equal(visible.content, "👨‍👩‍👧好");
});

test("a 50k grapheme backlog is never scanned on a single frame", () => {
  const content = "字".repeat(50_000);
  const target = message("assistant:turn", content);
  const first = advanceMobileStreamPresentation(message("assistant:turn", ""), target, 0);
  const state = streamStateOf(first);
  assert.ok(state.queued >= 49_999, `queued ${state.queued}`);

  const queue = state.content;
  let reads = 0;
  queue.bounds = new Proxy(queue.bounds, {
    get(bounds, property) {
      reads += 1;
      const value = Reflect.get(bounds, property);
      return typeof value === "function" ? value.bind(bounds) : value;
    },
  });

  let worst = 0;
  let current = first;
  const readsBefore = reads;
  for (let frame = 0; frame < 200; frame += 1) {
    const before = reads;
    current = advanceMobileStreamPresentation(current, target, 16.67);
    worst = Math.max(worst, reads - before);

    // 必须持续消费同一个队列对象：任何重分割都会把 5 万积压重新全量入队。
    assert.equal(streamStateOf(current)?.content, queue, `re-segmented at frame ${frame}`);
    // 可见文本始终是权威 source 的精确前缀（只按可见长度核对，不扫描积压）。
    assert.equal(current.content, content.slice(0, current.content.length), `prefix violated at frame ${frame}`);
  }

  // 每帧只能接触 O(budget) 个边界，绝不遍历 5 万积压。
  assert.ok(worst <= 60, `worst frame touched ${worst} bounds entries`);
  const consumed = reads - readsBefore;
  assert.ok(consumed <= 4000, `consumed ${consumed} reads across 200 frames`);

  // 按 600 g/s 上限以 ~10 grapheme/帧 有界推进。
  const visible = graphemeCount(current.content);
  assert.ok(visible >= 1500 && visible <= 2500, `visible ${visible} after 200 frames`);
});

test("continuous 400 g/s keeps P95 visible lag under 100ms at 60/90/120/144Hz", () => {
  for (const hz of [60, 90, 120, 144]) {
    const { samples } = runStreamSimulation(hz, 400);
    const p95 = percentile(samples, 0.95);
    assert.ok(p95 <= 100, `${hz}Hz P95 lag ${p95.toFixed(1)}ms`);

    // 滞后不无界增长：后段均值不超过前段的 1.3 倍。
    const early = samples.slice(hz * 2, hz * 5);
    const late = samples.slice(hz * 7, hz * 10);
    const bound = mean(early) * 1.3 + 10;
    assert.ok(mean(late) <= bound, `${hz}Hz lag grew: early ${mean(early).toFixed(1)}ms late ${mean(late).toFixed(1)}ms`);
  }
});

test("a 100 g/s source is never artificially slowed down", () => {
  for (const hz of [60, 144]) {
    const { samples, store, id, fallback, published, authoritative } = runStreamSimulation(hz, 100);
    const p95 = percentile(samples, 0.95);
    assert.ok(p95 <= 150, `${hz}Hz 100g/s P95 lag ${p95.toFixed(1)}ms`);
    assert.ok(Math.max(...samples) <= 250, `${hz}Hz 100g/s max lag ${Math.max(...samples).toFixed(1)}ms`);
    assert.ok(published >= 990, `${hz}Hz published ${published}`);
    // 连续发布后完整字符串必须与权威 source 逐字符相等（无重复、无错序）。
    assert.equal(store.read(id, fallback).content, authoritative.content);
  }
});

test("cross-delta cluster extensions replace the visible tail atomically without spending tokens", () => {
  const id = "assistant:turn";
  const target = message(id, "先👨‍好");
  const empty = message(id, "");

  // 帧 1（elapsed 0）：fresh 保底揭示 1 个 EGC"先"，形成负债务。
  let current = advanceMobileStreamPresentation(empty, target, 0);
  assert.equal(current.content, "先");
  let state = streamStateOf(current);
  assert.ok(state.token < 0, `fresh floor borrowed: token ${state.token}`);

  // 帧 2（1ms）：收益不足以偿还债务，本帧不揭示 —— 债务偿还的节奏。
  current = advanceMobileStreamPresentation(current, target, 1);
  assert.equal(current.content, "先");

  // 逐帧推进直到排空：任何帧都是 EGC 序列前缀，完整簇"👨\u200D"、"好"逐项揭示。
  let frame = 0;
  for (; frame < 200; frame += 1) {
    assertEgcPrefix(target.content, current.content, `frame ${frame}`);
    if (streamStateOf(current).queued === 0) break;
    current = advanceMobileStreamPresentation(current, target, 16.67);
  }
  assert.ok(frame < 200, "target must drain");
  assert.equal(current.content, "先👨‍好");
  assert.equal(streamStateOf(current).queued, 0);

  // 扩展 delta：已揭示尾"好"被组合音标扩展 → 同一次提交原子替换为完整新簇，
  // 不新增 grapheme、不消耗 pacing 配额（token 原样保留）。
  const extended = message(id, "先👨‍好\u0301");
  const tokenBefore = streamStateOf(current).token;
  const next = advanceMobileStreamPresentation(current, extended, 0);
  assert.equal(next.content, "先👨‍好\u0301");
  state = streamStateOf(next);
  assert.equal(state.queued, 0);
  assert.equal(state.token, tokenBefore, `atomic replacement must not spend tokens: ${tokenBefore} -> ${state.token}`);
  assertEgcPrefix(extended.content, next.content, "atomic replacement");
});

test("a complete trailing cluster displays and a later extension merges atomically", () => {
  const scheduler = new TestFrameScheduler();
  const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
  const id = "assistant:turn";
  const fallback = message(id, "");
  const tail = "先".repeat(10) + "👨‍";
  store.publish(id, fallback, message(id, tail), false);

  // 逐帧推进：每一帧可见文本都是 target 的 EGC 序列前缀；
  // "👨\u200D" 是当时 target 下的完整簇，正常显示（不为未来未知 delta 按住）。
  let frames = 0;
  let visible = store.read(id, fallback);
  while (frames < 200 && graphemeCount(visible.content) < 11) {
    scheduler.advance(16.67);
    frames += 1;
    visible = store.read(id, fallback);
    assertEgcPrefix(tail, visible.content, `reveal frame ${frames}`);
  }
  assert.ok(frames < 200, "a complete trailing cluster must reveal, not be held forever");
  assert.equal(visible.content, tail);

  // 后续 delta"好"不与"👨\u200D"合并：正常揭示，最终精确。
  const full = message(id, tail + "好");
  store.publish(id, message(id, tail), full, false);
  let releaseFrames = 0;
  while (scheduler.callback !== null && releaseFrames < 100) {
    scheduler.advance(16.67);
    releaseFrames += 1;
    const now = store.read(id, full);
    assertEgcPrefix(full.content, now.content, `release frame ${releaseFrames}`);
  }
  assert.equal(store.read(id, full).content, tail + "好");
});

test("structure-only publishes keep the queued content and thinking text intact", () => {
  const scheduler = new TestFrameScheduler();
  const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
  const id = "assistant:turn";
  const t1 = {
    ...message(id, "回答内容", "思考过程"),
    blocks: [
      { id: "think", kind: "thinking", detail: "思考过程" },
      { id: "tool", kind: "tool", detail: "工具A", state: "running" },
    ],
  };
  store.publish(id, message(id, ""), t1, false);
  scheduler.advance(16.67);
  const mid = store.read(id, t1);
  assert.ok(mid.content.length < "回答内容".length, "streaming in progress before structure-only publish");

  const t2 = {
    ...t1,
    blocks: [
      ...t1.blocks,
      { id: "tool2", kind: "tool", detail: "工具B", state: "queued" },
    ],
  };
  store.publish(id, t1, t2, false);
  while (scheduler.callback !== null) scheduler.advance(16.67);

  const visible = store.read(id, t2);
  assert.equal(visible.content, "回答内容");
  assert.equal(visible.blocks[0].detail, "思考过程");
  assert.equal(visible.blocks[1], t2.blocks[1]);
  assert.equal(visible.blocks[2], t2.blocks[2]);
});

test("a non-prefix correction through the store replaces the projection immediately", () => {
  const scheduler = new TestFrameScheduler();
  const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
  const before = message("assistant:turn", "正在流式输出");
  const streaming = message("assistant:turn", "正在流式输出更多");
  store.publish(before.id, before, streaming, false);
  scheduler.advance();

  const corrected = message("assistant:turn", "权威纠正");
  store.publish(streaming.id, streaming, corrected, false);

  assert.equal(store.read(before.id, before), corrected);
  assert.equal(scheduler.callback, null);
});

test("multiple thinking queues share the frame budget fairly", () => {
  const scheduler = new TestFrameScheduler();
  const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
  const id = "assistant:turn";
  const thinking = "思".repeat(200);
  const target = {
    ...message(id, "", thinking),
    blocks: [
      { id: "a", kind: "thinking", detail: thinking },
      { id: "b", kind: "thinking", detail: thinking },
    ],
  };
  store.publish(id, message(id, ""), target, false);
  for (let frame = 0; frame < 12; frame += 1) scheduler.advance(16.67);

  const visible = store.read(id, target);
  const a = graphemeCount(visible.blocks[0].detail);
  const b = graphemeCount(visible.blocks[1].detail);
  assert.ok(a > 0 && b > 0, `both queues revealed: a=${a} b=${b}`);
  assert.ok(Math.abs(a - b) <= 4, `fair shares: a=${a} b=${b}`);
});

test("150 and 200 g/s sources show no refresh-rate divergence at 60/90/120/144Hz", () => {
  for (const sourceCps of [150, 200]) {
    const runs = [60, 90, 120, 144].map((hz) => ({ hz, ...runStreamSimulation(hz, sourceCps, { chunkGraphemes: 4 }) }));
    const p95s = runs.map((run) => percentile(run.samples, 0.95));
    for (const [hz, p95] of runs.map((run, index) => [run.hz, p95s[index]])) {
      assert.ok(p95 <= 100, `${sourceCps} g/s at ${hz}Hz P95 lag ${p95.toFixed(1)}ms`);
    }
    const spread = Math.max(...p95s) - Math.min(...p95s);
    assert.ok(spread <= 30, `${sourceCps} g/s P95 lag spread ${spread.toFixed(1)}ms across refresh rates`);
  }
});

test("an 800 g/s source never lets the consumer exceed the 600 g/s cap", () => {
  // 纯字持续输入：ZWJ 正确性由独立测试验证，这里只用无悬空样本做吞吐 oracle，
  // 严格验证 600 g/s 硬上限（无容差）。
  const runs = [60, 90, 120, 144].map((hz) => ({
    hz,
    ...runStreamSimulation(hz, 800, { chunkGraphemes: 2, unit: "字" }),
  }));
  for (const run of runs) {
    const sustained = graphemeCount(run.visibleAtEnd) / 10;
    assert.ok(sustained <= 600, `${run.hz}Hz sustained ${sustained.toFixed(1)} g/s exceeds the 600 cap`);
  }
  const counts = runs.map((run) => graphemeCount(run.visibleAtEnd));
  const max = Math.max(...counts);
  const min = Math.min(...counts);
  assert.ok((max - min) / max < 0.2, `visible spread across refresh rates: ${counts.join(", ")}`);
});

test("trailing ZWJ-ending clusters reveal as complete EGCs and stay exact EGC prefixes", () => {
  const id = "assistant:turn";
  const target = message(id, "字\u200D字\u200D");
  const empty = message(id, "");

  // "字\u200D" 是当时 target 下的完整 EGC：正常揭示，任何时刻都是 EGC 序列前缀
  // （不再为未来未知 delta 按住，也不允许展示半簇"字"）。
  let current = advanceMobileStreamPresentation(empty, target, 0);
  for (let frame = 0; frame < 120; frame += 1) {
    assertEgcPrefix(target.content, current.content, `frame ${frame}`);
    if (current.content === "字\u200D字\u200D") break;
    current = advanceMobileStreamPresentation(current, target, 16.67);
  }
  assert.equal(current.content, "字\u200D字\u200D");

  // 追加安全 grapheme 后按权威 EGC 序列继续揭示，最终精确。
  const full = message(id, "字\u200D字\u200D好");
  for (let frame = 0; frame < 200; frame += 1) {
    current = advanceMobileStreamPresentation(current, full, 16.67);
    assertEgcPrefix(full.content, current.content, `release frame ${frame}`);
  }
  assert.equal(current.content, "字\u200D字\u200D好");
});

test("reconcile drops converged projections so read falls back to the baseline", () => {
  const scheduler = new TestFrameScheduler();
  const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
  const id = "assistant:turn";
  const fallback = message(id, "");
  const target = message(id, "短句");
  store.publish(id, fallback, target, false);
  let frames = 0;
  while (scheduler.callback !== null && frames < 100) {
    scheduler.advance(16.67);
    frames += 1;
  }
  assert.equal(store.read(id, fallback).content, "短句");

  store.reconcileBaseline([target]);
  assert.equal(store.read(id, fallback), fallback);
});

test("frame budget is time-based with a rate that adapts to backlog", () => {
  assert.equal(mobileStreamFrameBudget(16.67, 0), 0);
  assert.equal(mobileStreamFrameBudget(0, 5), 0);
  assert.equal(mobileStreamFrameBudget(16.67, 5), 2);
  assert.equal(mobileStreamFrameBudget(16.67, 48), 10);
  assert.equal(mobileStreamFrameBudget(16.67, 1000), 10);
  assert.equal(mobileStreamFrameBudget(250, 100000), 12);
  assert.equal(streamRate(0), 120);
  assert.equal(streamRate(48), 600);
  assert.equal(streamRate(1000), 600);
});

test("flushStreamingTexts reveals every queued grapheme exactly and cleans the projection state", () => {
  const target = {
    ...message("assistant:turn", "回答内容", "思考过程"),
    blocks: [
      { id: "thinking", kind: "thinking", detail: "思考过程" },
      { id: "tool", kind: "tool", detail: "工具", state: "running" },
    ],
  };

  const flushed = flushStreamingTexts(message("assistant:turn", "回", "思"), target);

  assert.equal(flushed.content, "回答内容");
  assert.equal(flushed.blocks[0].detail, "思考过程");
  assert.equal(flushed.blocks[1], target.blocks[1]);
  const state = streamStateOf(flushed);
  assert.equal(state.queued, 0);
  assert.equal(state.token, 0);
  assert.equal(state.content, null);
});

test("flushAll reveals a standing backlog without further publishes and notifies each row once", () => {
  const scheduler = new TestFrameScheduler();
  const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
  const id = "assistant:turn";
  const fallback = message(id, "");
  const thinking = "思".repeat(300);
  const content = "答".repeat(300);
  const target = {
    ...message(id, content, thinking),
    blocks: [{ id: "thinking", kind: "thinking", detail: thinking }],
  };
  store.publish(id, fallback, target, false);
  scheduler.advance(16.67);
  const before = store.read(id, fallback);
  assert.ok(before.content.length < content.length, "backlog must be pending before flushAll");

  let notifications = 0;
  let idleNotifications = 0;
  store.subscribe(id, () => { notifications += 1; });
  store.subscribe("other:row", () => { idleNotifications += 1; });
  store.flushAll();

  const after = store.read(id, fallback);
  assert.equal(after.content, content, "flushAll reveals the full answer without a new delta");
  assert.equal(after.blocks[0].detail, thinking, "flushAll reveals the full thinking text");
  assert.equal(notifications, 1, "affected row notified exactly once");
  assert.equal(idleNotifications, 0, "unaffected rows stay quiet");
  assert.equal(scheduler.callback, null, "no residual frame after flushAll");
  const state = streamStateOf(after);
  assert.equal(state.queued, 0);
  assert.equal(state.token, 0);

  // 清理后再次 publish 的增量照常入队推进，不残留旧队列状态。
  const more = {
    ...message(id, content + "答".repeat(50), thinking),
    blocks: [{ id: "thinking", kind: "thinking", detail: thinking }],
  };
  store.publish(id, message(id, content), more, false);
  scheduler.advance(16.67);
  const resumed = store.read(id, fallback);
  assert.ok(resumed.content.length > content.length && resumed.content.length < more.content.length, "pacing resumes from the flushed baseline");
  while (scheduler.callback !== null) scheduler.advance(16.67);
  assert.equal(store.read(id, more).content, more.content);
});

test("flushAll with no pending backlog cancels nothing and stays silent", () => {
  const scheduler = new TestFrameScheduler();
  const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
  const id = "assistant:turn";
  store.publish(id, message(id, ""), message(id, "终态"), true);
  let notifications = 0;
  store.subscribe(id, () => { notifications += 1; });

  store.flushAll();

  assert.equal(notifications, 0);
  assert.equal(scheduler.callback, null);
});

function fakeMediaQueryList() {
  const listeners = new Set();
  return {
    listeners,
    addEventListener(type, listener) {
      assert.equal(type, "change");
      listeners.add(listener);
    },
    removeEventListener(type, listener) {
      assert.equal(type, "change");
      listeners.delete(listener);
    },
    fire(matches) {
      for (const listener of [...listeners]) listener({ matches });
    },
  };
}

test("the reduced-motion listener flushes the backlog on switch to reduce and detaches on cleanup", () => {
  const scheduler = new TestFrameScheduler();
  const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
  const id = "assistant:turn";
  const fallback = message(id, "");
  const content = "长文".repeat(50);
  store.publish(id, fallback, message(id, content), false);

  const media = fakeMediaQueryList();
  const detach = attachReducedMotionFlush(store, media);
  assert.equal(media.listeners.size, 1, "change listener must be registered");

  media.fire(false);
  assert.ok(store.read(id, fallback).content.length < content.length, "no-preference switch must not flush");

  media.fire(true);
  assert.equal(store.read(id, fallback).content, content, "reduce switch flushes the standing backlog without a new delta");
  assert.equal(scheduler.callback, null, "no residual frame after the reduce switch");

  detach();
  assert.equal(media.listeners.size, 0, "cleanup must remove the change listener");

  const more = message(id, content + "更多");
  store.publish(id, message(id, content), more, false);
  media.fire(true);
  assert.ok(store.read(id, fallback).content.length < more.content.length, "a detached listener must not flush");
});

test("compactQueue rebases a 13k double-UTF-16 backlog exactly and appends 1k more without loss", () => {
  const scheduler = new TestFrameScheduler();
  const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
  const id = "assistant:turn";
  const fallback = message(id, "");
  const unit = "👨";
  const first = unit.repeat(13_000);
  store.publish(id, fallback, message(id, first), false);

  const queueBefore = streamStateOf(store.read(id, fallback)).content;
  let frames = 0;
  for (; frames < 2000 && scheduler.callback !== null; frames += 1) {
    scheduler.advance(16.67);
    const visible = store.read(id, fallback);
    assert.equal(streamStateOf(visible).content, queueBefore, "queue must never be re-segmented");
    assert.ok(first.startsWith(visible.content), `prefix violated at frame ${frames}`);
    if (graphemeCount(visible.content) >= 4872) break;
  }
  const mid = store.read(id, fallback);
  assert.ok(graphemeCount(mid.content) >= 4872, `head crossed the compaction threshold (${graphemeCount(mid.content)})`);
  assert.equal(mid.content.length % unit.length, 0, "reveals must never split a surrogate pair");
  assert.ok(streamStateOf(mid).content.text.length < first.length, "compaction must have rebased the queued text");

  const full = message(id, first + unit.repeat(1000));
  store.publish(id, message(id, first), full, false);
  let guard = 0;
  while (scheduler.callback !== null && guard < 3000) {
    scheduler.advance(16.67);
    guard += 1;
    const visible = store.read(id, fallback).content;
    assert.ok(full.content.startsWith(visible), `prefix violated during rebase drain at frame ${guard}`);
    assert.equal(visible.length % unit.length, 0, `drain split a surrogate pair at frame ${guard}`);
  }
  const drained = store.read(id, full).content;
  assert.equal(drained, full.content, "final content must equal the authoritative source exactly");
  assert.equal(graphemeCount(drained), 14_000);
});

test("cross-delta EGC extensions in an unrevealed backlog never split clusters", () => {
  // 每步 delta 只追加扩展字符：a→a+组合音标、👍→👍🏽、🇺→🇺🇸、👩→👩+ZWJ💻，
  // 全部在未揭示积压中跨 delta 完成，权威 EGC 数从 1 长到 4。
  const steps = [
    "a",
    "a\u0301",
    "a\u0301👍",
    "a\u0301👍🏽",
    "a\u0301👍🏽🇺",
    "a\u0301👍🏽🇺🇸",
    "a\u0301👍🏽🇺🇸👩",
    "a\u0301👍🏽🇺🇸👩\u200D💻",
  ];
  const scheduler = new TestFrameScheduler();
  const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
  const id = "assistant:turn";
  const fallback = message(id, "");
  let authoritative = fallback;
  for (const text of steps) {
    const target = message(id, text);
    store.publish(id, authoritative, target, false);
    authoritative = target;
    const projection = store.read(id, fallback);
    assertEgcPrefix(authoritative.content, projection.content, `publish ${JSON.stringify(text)}`);
    assertQueuedExact(projection, authoritative.content, `publish ${JSON.stringify(text)}`);
  }
  // 排空期间每帧断言可见 grapheme 数组 = 最新 target grapheme 数组前缀，最终精确。
  let frames = 0;
  while (scheduler.callback !== null && frames < 2000) {
    scheduler.advance(16.67);
    frames += 1;
    const projection = store.read(id, fallback);
    assertEgcPrefix(authoritative.content, projection.content, `drain frame ${frames}`);
    assertQueuedExact(projection, authoritative.content, `drain frame ${frames}`);
  }
  assert.equal(store.read(id, fallback).content, authoritative.content);
  assert.equal(graphemeCount(authoritative.content), 4);
});

test("an already-revealed base EGC is atomically replaced when the extension delta arrives", () => {
  for (const [base, extended] of [
    ["a", "a\u0301"],
    ["👍", "👍🏽"],
    ["🇺", "🇺🇸"],
    ["👩", "👩\u200D💻"],
  ]) {
    const scheduler = new TestFrameScheduler();
    const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
    const id = "assistant:turn";
    const fallback = message(id, "");
    store.publish(id, fallback, message(id, base), false);
    while (scheduler.callback !== null) scheduler.advance(16.67);
    assert.equal(store.read(id, fallback).content, base, `base ${base} must drain`);

    // 基础 EGC 已可见后才收到扩展 delta：下一次 publish 立即原子替换。
    const visibleBefore = graphemeCount(store.read(id, fallback).content);
    store.publish(id, message(id, base), message(id, extended), false);
    const projection = store.read(id, fallback);
    assert.equal(projection.content, extended, `atomic replace ${base} -> ${extended}`);
    assertEgcPrefix(extended, projection.content, `atomic replace ${base}`);
    assert.equal(graphemeCount(projection.content), visibleBefore, "replacement adds no grapheme");
    assert.equal(streamStateOf(projection).queued, 0, "replacement does not over-count queued");
    assert.equal(scheduler.callback, null, "no frame needed after an empty-replacement publish");

    // 原子替换不消耗 pacing/rolling 配额：随后 delta 的揭示照常精确。
    const more = extended + "b";
    store.publish(id, message(id, extended), message(id, more), false);
    while (scheduler.callback !== null) scheduler.advance(16.67);
    assert.equal(store.read(id, fallback).content, more);
  }
});

test("three-segment family emoji and consecutive combining marks stay exact across deltas", () => {
  for (const [steps, final] of [
    [["👩", "👩\u200D👨", "👩\u200D👨\u200D👧"], "👩\u200D👨\u200D👧"],
    [["e", "e\u0301", "e\u0301\u0302", "e\u0301\u0302\u0303"], "e\u0301\u0302\u0303"],
  ]) {
    const scheduler = new TestFrameScheduler();
    const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
    const id = "assistant:turn";
    const fallback = message(id, "");
    let authoritative = fallback;
    for (const text of steps) {
      const target = message(id, text);
      store.publish(id, authoritative, target, false);
      authoritative = target;
      const projection = store.read(id, fallback);
      assertEgcPrefix(authoritative.content, projection.content, `step ${JSON.stringify(text)}`);
      assertQueuedExact(projection, authoritative.content, `step ${JSON.stringify(text)}`);
      // 推进一帧后再断言：揭示与原子替换都不得产生半簇。
      if (scheduler.callback !== null) scheduler.advance(16.67);
      const advanced = store.read(id, fallback);
      assertEgcPrefix(authoritative.content, advanced.content, `step+frame ${JSON.stringify(text)}`);
      assertQueuedExact(advanced, authoritative.content, `step+frame ${JSON.stringify(text)}`);
    }
    while (scheduler.callback !== null) scheduler.advance(16.67);
    assert.equal(store.read(id, fallback).content, final);
  }
});

test("thinking and answer lanes repair cross-delta clusters independently without bleeding", () => {
  const scheduler = new TestFrameScheduler();
  const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
  const id = "assistant:turn";
  const fallback = message(id, "");
  const t1 = message(id, "a", "e");
  store.publish(id, fallback, t1, false);
  scheduler.advance(16.67);
  const t2 = message(id, "a\u0301", "e\u0301");
  store.publish(id, t1, t2, false);
  const projection = store.read(id, fallback);
  assert.equal(projection.content, "a\u0301");
  assert.equal(projection.blocks[0].detail, "e\u0301");
  assertEgcPrefix(t2.content, projection.content, "answer lane");
  assertEgcPrefix(t2.blocks[0].detail, projection.blocks[0].detail, "thinking lane");
  assert.equal(streamStateOf(projection).queued, 0, "both lanes fully replaced");

  // 两个 thinking 块各自隔离：一个块的尾簇扩展不会与另一个块拼接。
  const scheduler2 = new TestFrameScheduler();
  const store2 = new MobileStreamProjectionStore(scheduler2, advanceMobileStreamPresentation);
  const multi = {
    ...message(id, "", "e"),
    blocks: [
      { id: "think1", kind: "thinking", detail: "e" },
      { id: "think2", kind: "thinking", detail: "f" },
    ],
  };
  store2.publish(id, message(id, ""), multi, false);
  while (scheduler2.callback !== null) scheduler2.advance(16.67);
  const multiExt = {
    ...message(id, "", "e\u0301"),
    blocks: [
      { id: "think1", kind: "thinking", detail: "e\u0301" },
      { id: "think2", kind: "thinking", detail: "f" },
    ],
  };
  store2.publish(id, multi, multiExt, false);
  const seen = store2.read(id, fallback);
  assert.equal(seen.blocks[0].detail, "e\u0301", "block 0 tail replaced atomically");
  assert.equal(seen.blocks[1].detail, "f", "unrelated block must not absorb another block's tail");
});

test("after compactQueue rebase a later extension tail repairs exactly without duplication or loss", () => {
  const scheduler = new TestFrameScheduler();
  const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
  const id = "assistant:turn";
  const fallback = message(id, "");
  const base = "字".repeat(6000) + "a";
  store.publish(id, fallback, message(id, base), false);

  // 排空越过 compaction 阈值（4096）后仍留积压，随后发布扩展组合音标。
  let frames = 0;
  while (scheduler.callback !== null && frames < 3000) {
    scheduler.advance(16.67);
    frames += 1;
    if (graphemeCount(store.read(id, fallback).content) >= 5500) break;
  }
  const mid = store.read(id, fallback);
  assert.ok(streamStateOf(mid).content.text.length < base.length, "compaction must have rebased the queued text");
  assert.ok(graphemeCount(mid.content) < graphemeCount(base), "must still be streaming before the extension");

  const extended = message(id, base + "\u0301");
  store.publish(id, message(id, base), extended, false);
  let guard = 0;
  while (scheduler.callback !== null && guard < 3000) {
    scheduler.advance(16.67);
    guard += 1;
    const now = store.read(id, fallback);
    assertEgcPrefix(extended.content, now.content, `drain frame ${guard}`);
    assertQueuedExact(now, extended.content, `drain frame ${guard}`);
  }
  assert.equal(store.read(id, fallback).content, extended.content, "final content must equal the authoritative source exactly");
  assert.equal(graphemeCount(extended.content), 6001);

  // 完全排空后再次扩展：可见尾原子替换，同样不重复不丢失。
  const extended2 = message(id, base + "\u0301\u0302");
  store.publish(id, message(id, extended.content), extended2, false);
  const replaced = store.read(id, fallback);
  assert.equal(replaced.content, base + "\u0301\u0302", "fully-drained tail replaced atomically after compaction");
  assert.equal(streamStateOf(replaced).queued, 0);
});

/**
 * 以复合 cluster 单元（每 unit 多个 EGC）模拟连续流式源：源按 sourceCps
 * 产出 grapheme、按整 unit 分块发布；每帧用 Intl.Segmenter 断言可见
 * grapheme 数组是权威文本的前缀（节奏负载下不拆簇），排空后断言精确相等。
 * 返回每帧采样的可见滞后（ms）。
 */
function runClusterStream(hz, sourceCps, unit, chunkUnits, durationMs = 10_000) {
  const unitCount = graphemeCount(unit);
  const scheduler = new TestFrameScheduler();
  const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
  const frameMs = 1000 / hz;
  const id = "assistant:turn";
  const fallback = message(id, "");
  let published = 0;
  let authoritative = fallback;
  const samples = [];
  for (let t = frameMs; t <= durationMs; t += frameMs) {
    const sourceUnits = Math.floor((sourceCps * t) / 1000 / unitCount);
    while (published < sourceUnits) {
      const next = Math.min(sourceUnits, published + chunkUnits);
      const target = message(id, unit.repeat(next));
      store.publish(id, authoritative, target, false);
      authoritative = target;
      published = next;
    }
    if (scheduler.callback !== null) scheduler.advance(frameMs);
    const visible = store.read(id, fallback).content;
    assertEgcPrefix(authoritative.content, visible, `${hz}Hz cluster frame at t=${t.toFixed(1)}ms`);
    samples.push(((published * unitCount - graphemeCount(visible)) * 1000) / sourceCps);
  }
  let guard = 0;
  while (scheduler.callback !== null && guard < 6000) {
    scheduler.advance(frameMs);
    guard += 1;
    assertEgcPrefix(authoritative.content, store.read(id, fallback).content, `${hz}Hz cluster drain frame ${guard}`);
  }
  assert.equal(store.read(id, fallback).content, authoritative.content, `${hz}Hz cluster drain must be exact`);
  return { samples };
}

/**
 * 按真实 grapheme 数计速的 cluster 连续流：800 g/s 源以整 unit 分块发布
 * （unit 含 4 个 EGC 时 unit.repeat 的份数不能当作 grapheme 数）。
 * 返回滚动窗口最大值与 10 秒总揭示数。
 */
function continuous800RollingCluster(hz, unit, chunkUnits) {
  const unitCount = graphemeCount(unit);
  const scheduler = new TestFrameScheduler();
  const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
  const id = "assistant:turn";
  const fallback = message(id, "");
  const frameMs = 1000 / hz;
  let published = 0;
  let authoritative = fallback;
  const trace = [];
  let previous = 0;
  for (let t = frameMs; t <= 10_000; t += frameMs) {
    const sourceGraphemes = Math.floor((800 * t) / 1000);
    while (published + unitCount <= sourceGraphemes) {
      const units = Math.min(chunkUnits, Math.floor((sourceGraphemes - published) / unitCount));
      const next = published + units * unitCount;
      const target = message(id, unit.repeat(next / unitCount));
      store.publish(id, authoritative, target, false);
      authoritative = target;
      published = next;
    }
    if (scheduler.callback !== null) scheduler.advance(frameMs);
    const visible = graphemeCount(store.read(id, fallback).content);
    trace.push({ t: scheduler.timestamp, count: visible - previous });
    previous = visible;
  }
  return { max: rollingWindowMax(trace), total: previous };
}

/**
 * 按真实 grapheme 数计速的 cluster hidden 恢复：800 g/s 输入 5 秒后单帧
 * 迟到 30 秒，恢复帧必须计入，返回滚动窗口最大值。
 */
function hiddenGapMaxWindowCluster(hz, unit, chunkUnits) {
  const unitCount = graphemeCount(unit);
  const scheduler = new TestFrameScheduler();
  const store = new MobileStreamProjectionStore(scheduler, advanceMobileStreamPresentation);
  const id = "assistant:turn";
  const fallback = message(id, "");
  const frameMs = 1000 / hz;
  let published = 0;
  let authoritative = fallback;
  for (let t = frameMs; t <= 5_000; t += frameMs) {
    const sourceGraphemes = Math.floor((800 * t) / 1000);
    while (published + unitCount <= sourceGraphemes) {
      const units = Math.min(chunkUnits, Math.floor((sourceGraphemes - published) / unitCount));
      const next = published + units * unitCount;
      const target = message(id, unit.repeat(next / unitCount));
      store.publish(id, authoritative, target, false);
      authoritative = target;
      published = next;
    }
    if (scheduler.callback !== null) scheduler.advance(frameMs);
  }
  const trace = [];
  let previous = graphemeCount(store.read(id, fallback).content);
  scheduler.advance(30_000);
  const recoveryVisible = graphemeCount(store.read(id, fallback).content);
  trace.push({ t: scheduler.timestamp, count: recoveryVisible - previous });
  previous = recoveryVisible;
  for (let frame = 0; frame < hz * 10 && scheduler.callback !== null; frame += 1) {
    scheduler.advance(frameMs);
    const visible = graphemeCount(store.read(id, fallback).content);
    trace.push({ t: scheduler.timestamp, count: visible - previous });
    previous = visible;
  }
  return { max: rollingWindowMax(trace), recoveryCount: trace[0].count };
}

test("cluster-heavy streaming keeps exact EGC prefixes and 400 g/s pacing at 60/90/120/144Hz", () => {
  // 每个 unit 是 4 个 EGC 的复合序列（组合音标+肤色+RI+ZWJ），chunk 只取整 unit。
  const unit = "a\u0301👍🏽🇺🇸👩\u200D💻";
  for (const hz of [60, 90, 120, 144]) {
    const { samples } = runClusterStream(hz, 400, unit, 2);
    assert.ok(percentile(samples, 0.95) <= 100, `${hz}Hz cluster P95 lag ${percentile(samples, 0.95).toFixed(1)}ms`);
  }
});

test("cluster-heavy 800 g/s rolling and hidden windows measured by true grapheme count stay under the caps", () => {
  // unit 含 4 个 EGC：unit.repeat 的份数不是 grapheme 数，按真实计数计速。
  const unit = "a\u0301👍🏽🇺🇸👩\u200D💻";
  for (const hz of [60, 90, 120, 144]) {
    const cont = continuous800RollingCluster(hz, unit, 2);
    assert.ok(cont.max <= 600, `${hz}Hz cluster rolling window revealed ${cont.max} > 600`);
    const hidden = hiddenGapMaxWindowCluster(hz, unit, 2);
    assert.ok(hidden.max <= 600, `${hz}Hz cluster hidden-window revealed ${hidden.max} > 600`);
  }
});

test("stream pacing metrics report: 400 g/s lag, 800 g/s rolling, hidden-gap rolling max", () => {
  const lines = ["hz | 400g/s P95/max lag | 800g/s rolling max/avg | hidden rolling max"];
  for (const hz of [60, 90, 120, 144]) {
    const { samples } = runStreamSimulation(hz, 400);
    const p95 = percentile(samples, 0.95);
    const maxLag = Math.max(...samples);
    const cont = continuous800Rolling(hz, 10);
    const hidden = hiddenGapMaxWindow(hz, 10);
    lines.push(
      `${hz}Hz | ${p95.toFixed(1)}ms / ${maxLag.toFixed(1)}ms | ${cont.max} / ${(cont.total / 10).toFixed(1)}g/s | ${hidden.max}`,
    );
  }
  console.log(`\n[mobile-stream-projection metrics]\n${lines.join("\n")}`);
});

/** 计数 adapter：每次批量 block 文本更新都记一次调用（一帧的 map/copy 次数）。 */
function countingMobileIO() {
  const counters = { batch: 0 };
  return {
    counters,
    io: {
      blockCount: (message) => message.blocks.length,
      content: (message) => message.content,
      blockText: (message, index) => (message.blocks[index]?.kind === "thinking" ? message.blocks[index].detail : null),
      withContent: (message, content) => ({ ...message, content }),
      withBlockTexts: (message, texts) => {
        counters.batch += 1;
        return {
          ...message,
          blocks: message.blocks.map((block, blockIndex) => {
            const detail = texts.get(blockIndex);
            return detail === undefined ? block : { ...block, detail };
          }),
        };
      },
    },
  };
}

function mobileThinkingMessage(blockCount, contentGraphemes, blockGraphemes) {
  return {
    id: "assistant:turn",
    content: "回".repeat(contentGraphemes),
    blocks: Array.from({ length: blockCount }, (_, index) => ({
      id: `b${index}`,
      kind: "thinking",
      detail: "思".repeat(blockGraphemes),
    })),
  };
}

test("prepare and each advance frame touch the adapter with at most one batch block update regardless of block count", () => {
  // O(B) 而非 O(B²)：B 个 thinking 块无论多大，prepare 与每一帧都只做一次
  // 批量 adapter map/copy（调用次数不随 B 增长）。
  for (const blockCount of [4, 64]) {
    const { io, counters } = countingMobileIO();
    const target = mobileThinkingMessage(blockCount, 240, 240);
    const empty = { id: "assistant:turn", content: "", blocks: [] };
    let current = prepareStreamingTexts(empty, target, io);
    assert.equal(counters.batch, 1, `B=${blockCount}: prepare must batch all block text into one adapter call`);
    for (let frame = 0; frame < 20; frame += 1) {
      counters.batch = 0;
      current = advanceStreamingTexts(current, target, 16.67, io);
      assert.ok(counters.batch <= 1, `B=${blockCount} frame ${frame}: adapter batch called ${counters.batch} times`);
    }
    assert.ok(streamStateOf(current).queued > 0, "backlog must remain after the fixed frame window");
  }
});

test("no block list is copied once thinking reached its authoritative text or the queue drained", () => {
  const { io, counters } = countingMobileIO();
  const target = mobileThinkingMessage(4, 2000, 20);
  const empty = { id: "assistant:turn", content: "", blocks: [] };
  let current = prepareStreamingTexts(empty, target, io);
  let blocklessContentFrames = 0;
  let guard = 0;
  while (streamStateOf(current).queued > 0 && guard < 1000) {
    counters.batch = 0;
    current = advanceStreamingTexts(current, target, 16.67, io);
    if (counters.batch === 0 && streamStateOf(current).queued > 0) blocklessContentFrames += 1;
    guard += 1;
  }
  assert.ok(blocklessContentFrames > 0, "content-only frames after thinking drained must skip block copies");
  assert.equal(current.content, target.content, "drain must reveal the authoritative content exactly");
  for (let index = 0; index < target.blocks.length; index += 1) {
    assert.equal(current.blocks[index].detail, target.blocks[index].detail);
  }
  counters.batch = 0;
  advanceStreamingTexts(current, target, 16.67, io);
  assert.equal(counters.batch, 0, "drained advance must not touch the adapter");
});

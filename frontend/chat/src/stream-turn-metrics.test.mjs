import assert from "node:assert/strict";
import test from "node:test";

import {
  ClientTurnMetricsTracker,
  deriveTokensPerSecond,
  estimateOutputTokens,
  formatComposerStatsLine,
  formatLatencySeconds,
  formatTokensPerSecond,
} from "./stream-turn-metrics.ts";

test("format helpers match dsh rounding", () => {
  assert.equal(formatLatencySeconds(800), "0.8");
  assert.equal(formatLatencySeconds(12_400), "12");
  assert.equal(formatTokensPerSecond(20.4), "20");
  assert.equal(formatTokensPerSecond(4.25), "4.3");
});

test("TPS requires both decode window and positive tokens", () => {
  assert.equal(deriveTokensPerSecond(200, 5_000), 40);
  assert.equal(deriveTokensPerSecond(0, 5_000), undefined);
  assert.equal(deriveTokensPerSecond(200, 0), undefined);
});

test("tracker folds TTFT and decode throughput like dsh turn-metrics", () => {
  const tracker = new ClientTurnMetricsTracker();
  tracker.onTurnStarted("turn-1", 1_000);
  assert.equal(formatComposerStatsLine(tracker.getSnapshot()), "等待首 token…");

  tracker.onDelta("turn-1", "hello", 1_800);
  tracker.onDelta("turn-1", " world", 2_800);
  const live = tracker.getSnapshot();
  assert.equal(live?.phase, "decoding");
  assert.equal(live?.ttftMs, 800);
  assert.ok((live?.tokensPerSecond ?? 0) > 0);

  tracker.onSettled("turn-1", 40, 6_800);
  const settled = tracker.getSnapshot();
  assert.equal(settled?.phase, "settled");
  assert.equal(settled?.ttftMs, 800);
  assert.equal(settled?.tokensPerSecond, 40 / 5);
  assert.equal(formatComposerStatsLine(settled), "首 token 0.8s · 8 tok/s");
});

test("estimate keeps CJK denser than Latin", () => {
  assert.ok(estimateOutputTokens("你好世界") > estimateOutputTokens("abcd"));
});

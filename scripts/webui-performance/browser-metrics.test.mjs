import assert from "node:assert/strict";
import test from "node:test";

import {
  aggregateBrowserRuns,
  compareBrowserMetrics,
  createBrowserBudgets,
} from "./browser-metrics.mjs";

test("browser aggregation uses median and p75 across five runs", () => {
  const runs = [10, 50, 20, 40, 30].map((durationMs, index) => ({
    scenarios: { desktopHistory: { durationMs, domRows: 100, optional: index === 0 ? null : undefined } },
  }));

  const aggregate = aggregateBrowserRuns(runs);

  assert.deepEqual(aggregate.desktopHistory.durationMs, { median: 30, p75: 40 });
  assert.equal("optional" in aggregate.desktopHistory, false);
});

test("browser budgets keep DOM bounds tight and allow measured timing variance", () => {
  const aggregate = {
    mobileHistory300: {
      durationMs: { median: 100, p75: 120 },
      virtualRows: { median: 12, p75: 13 },
    },
  };
  const budgets = createBrowserBudgets(aggregate);
  const checks = compareBrowserMetrics(aggregate, budgets);

  assert.equal(budgets.mobileHistory300.durationMs, 138);
  assert.equal(budgets.mobileHistory300.virtualRows, 15);
  assert.equal(checks.every((check) => check.passed), true);
});

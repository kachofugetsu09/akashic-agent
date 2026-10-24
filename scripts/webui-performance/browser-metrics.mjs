export function aggregateBrowserRuns(runs) {
  if (runs.length === 0) throw new Error("浏览器性能采样不能为空");
  const scenarioNames = Object.keys(runs[0].scenarios);
  return Object.fromEntries(scenarioNames.map((scenario) => {
    const metricNames = Object.keys(runs[0].scenarios[scenario]);
    const metrics = metricNames.flatMap((metric) => {
      const values = runs
        .map((run) => run.scenarios[scenario][metric])
        .filter((value) => typeof value === "number" && Number.isFinite(value));
      return values.length === 0 ? [] : [[metric, {
        median: percentile(values, 0.5),
        p75: percentile(values, 0.75),
      }]];
    });
    return [scenario, Object.fromEntries(metrics)];
  }));
}

const diagnosticOnlyMetrics = new Set(["frameGapMaxMs", "settledFrameGapMaxMs"]);
// 基线为零的指标给一次测量噪声的余量；布局偏移是小数，向上取整会把零预算卡死在 0。
const zeroBudgets = { longTaskCount: 1, longTaskTotalMs: 60, longTaskMaxMs: 60, layoutShift: 0.01 };
// 堆大小随 GC 时机波动，比时延指标多留余量。
const noisyMetricFactors = { jsHeapBytes: 1.3 };

export function createBrowserBudgets(aggregate) {
  return Object.fromEntries(Object.entries(aggregate).map(([scenario, metrics]) => [scenario, Object.fromEntries(
    Object.entries(metrics).flatMap(([metric, values]) => {
      // 最大帧间隔是离群诊断，不作为有限样本的回归门禁。
      if (diagnosticOnlyMetrics.has(metric)) return [];
      if (values.p75 === 0 && metric in zeroBudgets) return [[metric, zeroBudgets[metric]]];
      return [[metric, metric === "virtualRows" || metric === "domRows"
        ? Math.ceil(values.p75 + 2)
        : Math.ceil(values.p75 * (noisyMetricFactors[metric] ?? 1.15))]];
    }),
  )]));
}

export function compareBrowserMetrics(aggregate, budgets) {
  const checks = [];
  for (const [scenario, scenarioBudgets] of Object.entries(budgets)) {
    if (!aggregate[scenario]) throw new Error(`缺少浏览器性能场景: ${scenario}`);
    for (const [metric, maximum] of Object.entries(scenarioBudgets)) {
      if (diagnosticOnlyMetrics.has(metric)) continue;
      const values = aggregate[scenario][metric];
      if (!values) throw new Error(`缺少浏览器性能指标: ${scenario}.${metric}`);
      checks.push({
        scenario,
        metric,
        actual: values.p75,
        maximum,
        passed: values.p75 <= maximum,
      });
    }
  }
  return checks;
}

function percentile(values, ratio) {
  const sorted = values.toSorted((left, right) => left - right);
  return sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * ratio) - 1)];
}

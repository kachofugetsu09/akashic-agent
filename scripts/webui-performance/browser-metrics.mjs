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

export function createBrowserBudgets(aggregate) {
  return Object.fromEntries(Object.entries(aggregate).map(([scenario, metrics]) => [scenario, Object.fromEntries(
    Object.entries(metrics).map(([metric, values]) => [metric, metric === "virtualRows" || metric === "domRows"
      ? Math.ceil(values.p75 + 2)
      : Math.ceil(values.p75 * 1.15)]),
  )]));
}

export function compareBrowserMetrics(aggregate, budgets) {
  const checks = [];
  for (const [scenario, scenarioBudgets] of Object.entries(budgets)) {
    if (!aggregate[scenario]) throw new Error(`缺少浏览器性能场景: ${scenario}`);
    for (const [metric, maximum] of Object.entries(scenarioBudgets)) {
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

export interface ModelCallStats {
  call_record_id: string;
  model: string;
  state: "started" | "success" | "error";
  first_token_ms: number | null;
  duration_ms: number | null;
  usage: {
    output_tokens: number | null;
    request_count: number;
    covered_request_count: number;
    coverage: "exact" | "partial" | "unavailable";
  } | null;
}

/** 校验 call.latest 返回的调用统计；缺失用量与耗时不补成零。 */
export function readModelCallStats(value: unknown): ModelCallStats {
  const raw = record(value);
  if (!raw || typeof raw.call_record_id !== "string" || !raw.call_record_id || typeof raw.model !== "string" || !raw.model
    || !["started", "success", "error"].includes(String(raw.state))
    || !nullableTime(raw.first_token_ms) || !nullableTime(raw.duration_ms)
    || (typeof raw.first_token_ms === "number" && typeof raw.duration_ms === "number" && raw.duration_ms < raw.first_token_ms)) {
    throw new Error("模型调用统计无效");
  }
  if (raw.usage !== null) {
    const usage = record(raw.usage);
    if (!usage || !(usage.output_tokens === null || count(usage.output_tokens))
      || !count(usage.request_count) || !count(usage.covered_request_count)
      || usage.covered_request_count > usage.request_count
      || !["exact", "partial", "unavailable"].includes(String(usage.coverage))) {
      throw new Error("模型调用用量无效");
    }
  }
  return raw as unknown as ModelCallStats;
}

/** 结算后的摘要：仅完整真实用量参与速度计算；无流式首段时只显示总耗时。 */
export function formatModelCallStats(stats: ModelCallStats): string {
  const parts: string[] = [];
  if (stats.first_token_ms !== null) parts.push(`首 token ${(stats.first_token_ms / 1000).toFixed(1)}s`);
  const usage = stats.usage;
  if (stats.state === "success" && stats.first_token_ms !== null && stats.duration_ms !== null
    && stats.duration_ms > stats.first_token_ms && usage?.coverage === "exact"
    && usage.output_tokens !== null && usage.request_count > 0 && usage.covered_request_count === usage.request_count) {
    parts.push(`${(usage.output_tokens * 1000 / (stats.duration_ms - stats.first_token_ms)).toFixed(1)} tok/s`);
  } else if (stats.duration_ms !== null) parts.push(`耗时 ${formatCallDuration(stats.duration_ms)}`);
  if (stats.state === "error" ? usage === null : stats.state === "started") parts.push("用量未结算");
  if (!parts.length) return "暂无耗时数据";
  return parts.join(" · ");
}

// 一分钟内保留一位小数，超过一分钟改成"1m 52s"，与本轮摘要的时长写法一致。
function formatCallDuration(durationMs: number): string {
  if (durationMs < 60_000) return `${(durationMs / 1000).toFixed(1)}s`;
  const seconds = Math.round(durationMs / 1000);
  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : null;
}
function nullableTime(value: unknown): value is number | null {
  return value === null || (typeof value === "number" && Number.isFinite(value) && value >= 0);
}
function count(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

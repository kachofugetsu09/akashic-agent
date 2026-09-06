import type { ReplyActivity, TimelineMessage } from "./message-timeline.ts";

export interface ModelCallStats {
  call_record_id: string;
  model: string;
  state: "started" | "success" | "unknown";
  first_token_ms: number | null;
  duration_ms: number | null;
  usage: {
    output_tokens: number | null;
    request_count: number;
    covered_request_count: number;
    coverage: "exact" | "partial" | "unavailable";
  } | null;
}

export type LoadModelCallStats = (callId: string, signal: AbortSignal) => Promise<ModelCallStats>;

/** 两个传输边界共用校验；缺失用量与耗时不补成零。 */
export function readModelCallStats(value: unknown, callId: string): ModelCallStats {
  const raw = record(value);
  if (!raw || raw.call_record_id !== callId || typeof raw.model !== "string" || !raw.model
    || !["started", "success", "unknown"].includes(String(raw.state))
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

/** 活动调用优先；空闲时显示最后一条模型输出，绝不混入另一会话。 */
export function selectModelCall(messages: readonly TimelineMessage[], activities: readonly ReplyActivity[]) {
  for (let index = activities.length - 1; index >= 0; index--) {
    const item = activities[index];
    if (item.active) return { callId: item.preview?.call_record_id ?? null, active: true };
  }
  for (let index = messages.length - 1; index >= 0; index--) {
    const body = messages[index].body;
    if (body.kind !== "output") continue;
    for (const part of body.parts) {
      if (part.kind === "model.facts" && "value" in part && typeof part.value !== "string") {
        return { callId: part.value.call_record_id, active: false };
      }
    }
  }
  return { callId: null, active: false };
}

/** 仅完整真实用量参与速度计算；无流式首段时只显示总耗时。 */
export function formatModelCallStats(stats: ModelCallStats, active: boolean): string {
  const parts: string[] = [];
  if (stats.first_token_ms !== null) parts.push(`首 token ${(stats.first_token_ms / 1000).toFixed(1)}s`);
  const usage = stats.usage;
  if (stats.state === "success" && stats.first_token_ms !== null && stats.duration_ms !== null
    && stats.duration_ms > stats.first_token_ms && usage?.coverage === "exact"
    && usage.output_tokens !== null && usage.request_count > 0 && usage.covered_request_count === usage.request_count) {
    parts.push(`${(usage.output_tokens * 1000 / (stats.duration_ms - stats.first_token_ms)).toFixed(1)} tok/s`);
  } else if (stats.duration_ms !== null) parts.push(`耗时 ${(stats.duration_ms / 1000).toFixed(1)}s`);
  if (stats.state === "unknown" || (stats.state === "started" && !active)) parts.push("用量未结算");
  if (!parts.length) return active ? "等待首 token…" : "暂无耗时数据";
  return parts.join(" · ");
}

export const loadWebModelCallStats: LoadModelCallStats = async (callId, signal) => {
  const response = await fetch(`/api/chat/model-settings/calls/${encodeURIComponent(callId)}`, { signal });
  if (!response.ok) throw new Error("统计暂不可用");
  return readModelCallStats(await response.json(), callId);
};

const pending = new Map<string, { receive: (value: unknown) => void }>();

/** 原生只转交只读查询；切页、超时和卸载会移除本地请求。 */
export const loadMobileModelCallStats: LoadModelCallStats = (callId, signal) => new Promise((resolve, reject) => {
  if (signal.aborted) { reject(signal.reason); return; }
  const bridge = window.AkashicNative;
  if (!bridge) { reject(new Error("统计暂不可用")); return; }
  const requestId = crypto.randomUUID();
  const cleanup = () => {
    pending.delete(requestId);
    clearTimeout(timer);
    signal.removeEventListener("abort", aborted);
  };
  const aborted = () => { cleanup(); reject(signal.reason); };
  const timer = setTimeout(() => { cleanup(); reject(new Error("统计查询超时")); }, 15000);
  signal.addEventListener("abort", aborted, { once: true });
  pending.set(requestId, { receive(value) {
    cleanup();
    try { resolve(readModelCallStats(value, callId)); } catch (error) { reject(error); }
  } });
  try { bridge.readModelCallStats(requestId, callId); } catch (error) { cleanup(); reject(error); }
});

export function receiveMobileModelCallStats(requestId: string, value: unknown): void {
  pending.get(requestId)?.receive(value);
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

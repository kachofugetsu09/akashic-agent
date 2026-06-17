/// <reference path="../../types/akashic-dashboard.d.ts" />
import { useEffect, useState, type ReactElement } from "react";
import { MetricTile, BarChart, Chip, api } from "@akashic/dashboard-ui";

interface Overview {
  range: string;
  turns: number;
  errors: number;
  error_rate: number | null;
  input_tokens: number;
  cache_prompt_tokens: number;
  cache_hit_tokens: number;
  cache_hit_rate: number | null;
  avg_iteration: number | null;
  max_iteration: number;
  last_ts: string | null;
}

interface SeriesPoint {
  bucket: string;
  turns: number;
  errors: number;
  input_tokens: number;
  cache_hit_rate: number | null;
  avg_iteration: number | null;
}

interface ErrorRow {
  id: number;
  ts: string;
  session_key: string;
  user_preview: string;
  error: string;
}

interface ErrorGroup {
  signature: string;
  count: number;
  last_ts: string;
}

const RANGES: { key: string; label: string }[] = [
  { key: "24h", label: "24 小时" },
  { key: "7d", label: "7 天" },
  { key: "30d", label: "30 天" },
  { key: "all", label: "全部" },
];

function _compact(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return String(value);
}

function _pct(value: number | null): string {
  return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "—";
}

// Shorten an ISO-bucket label: "2026-06-17T11" -> "11:00", "2026-06-17" -> "6-17".
function _bucketLabel(bucket: string): string {
  if (bucket.includes("T")) {
    const hour = bucket.slice(11, 13);
    return `${hour}:00`;
  }
  const [, m, d] = bucket.split("-");
  return m && d ? `${Number(m)}-${d}` : bucket;
}

function _shortTs(value: string): string {
  const dt = new Date(value);
  if (Number.isNaN(dt.getTime())) return value || "—";
  return `${dt.getMonth() + 1}-${String(dt.getDate()).padStart(2, "0")} ${String(dt.getHours()).padStart(2, "0")}:${String(dt.getMinutes()).padStart(2, "0")}`;
}

// Grafana-style monitoring overview over the agent-loop telemetry in observe.db:
// KPI tiles (turns / errors / KV hit rate / iterations) + trend charts + errors.
function ObserveMain(_props: { dispatch: PluginDispatch }): ReactElement {
  const [range, setRange] = useState<string>("24h");
  const [overview, setOverview] = useState<Overview | null>(null);
  const [points, setPoints] = useState<SeriesPoint[]>([]);
  const [errorRows, setErrorRows] = useState<ErrorRow[]>([]);
  const [groups, setGroups] = useState<ErrorGroup[]>([]);

  useEffect(() => {
    let alive = true;
    void (async () => {
      const [ov, series, errs] = await Promise.all([
        api<Overview>(`/api/dashboard/observe/overview?range=${range}`),
        api<{ points: SeriesPoint[] }>(`/api/dashboard/observe/timeseries?range=${range}`),
        api<{ items: ErrorRow[]; groups: ErrorGroup[] }>(`/api/dashboard/observe/errors?range=${range}&page=1&page_size=30`),
      ]);
      if (!alive) return;
      setOverview(ov);
      setPoints(series.points ?? []);
      setErrorRows(errs.items ?? []);
      setGroups(errs.groups ?? []);
    })();
    return () => {
      alive = false;
    };
  }, [range]);

  if (!overview) {
    return <div className="p-5 text-[13px] text-muted">加载中…</div>;
  }

  const turnSpark = points.map((p) => p.turns);
  const errorSpark = points.map((p) => p.errors);
  const hitSpark = points.map((p) => (p.cache_hit_rate ?? 0) * 100);
  const iterSpark = points.map((p) => p.avg_iteration ?? 0);

  return (
    <div className="p-5">
      {/* header + range switcher */}
      <div className="flex items-end justify-between">
        <div>
          <div className="detail-title">Observe · 监测</div>
          <div className="detail-subtext">Agent 主循环遥测 · Token / 迭代 / 错误</div>
        </div>
        <div className="flex gap-1 rounded-md border border-border bg-surface-2 p-1">
          {RANGES.map((r) => (
            <button
              key={r.key}
              onClick={() => setRange(r.key)}
              className={`rounded-[4px] px-2.5 py-1 font-mono text-[11px] transition-colors ${
                range === r.key ? "bg-accent text-accent-ink" : "text-muted hover:text-fg"
              }`}
            >
              {r.label}
            </button>
          ))}
        </div>
      </div>

      {/* KPI tiles */}
      <div className="mt-5 grid grid-cols-4 gap-3">
        <MetricTile
          label="对话轮数"
          value={_compact(overview.turns)}
          sub={overview.last_ts ? `最近 ${_shortTs(overview.last_ts)}` : "无记录"}
          tone="accent"
          spark={turnSpark}
        />
        <MetricTile
          label="错误"
          value={_compact(overview.errors)}
          unit={overview.error_rate != null ? `· ${_pct(overview.error_rate)}` : undefined}
          sub="error 非空轮次"
          tone="danger"
          spark={errorSpark}
        />
        <MetricTile
          label="KV 缓存命中率"
          value={_pct(overview.cache_hit_rate)}
          sub={`${_compact(overview.cache_hit_tokens)} / ${_compact(overview.cache_prompt_tokens)} tok`}
          tone="success"
          spark={hitSpark}
        />
        <MetricTile
          label="平均迭代"
          value={overview.avg_iteration != null ? overview.avg_iteration.toFixed(1) : "—"}
          unit={`· 峰 ${overview.max_iteration}`}
          sub="每轮 LLM 调用次数"
          tone="warning"
          spark={iterSpark}
        />
      </div>

      {/* trend charts */}
      <div className="mt-4 grid grid-cols-2 gap-3">
        <div className="rounded-lg border border-border bg-surface p-4 shadow-lift-sm">
          <div className="mb-3 font-mono text-[10px] uppercase tracking-[0.14em] text-subtle">输入 Token 趋势</div>
          <BarChart
            points={points.map((p) => ({ label: _bucketLabel(p.bucket), value: p.input_tokens }))}
            tone="accent"
            valueFmt={_compact}
          />
        </div>
        <div className="rounded-lg border border-border bg-surface p-4 shadow-lift-sm">
          <div className="mb-3 font-mono text-[10px] uppercase tracking-[0.14em] text-subtle">错误趋势</div>
          <BarChart
            points={points.map((p) => ({ label: _bucketLabel(p.bucket), value: p.errors }))}
            tone="danger"
            valueFmt={(n) => String(n)}
          />
        </div>
      </div>

      {/* error aggregation + recent errors */}
      <div className="mt-4 grid grid-cols-[280px_1fr] gap-3">
        <div className="rounded-lg border border-border bg-surface p-4 shadow-lift-sm">
          <div className="mb-3 font-mono text-[10px] uppercase tracking-[0.14em] text-subtle">错误聚合 · Top</div>
          {groups.length === 0 ? (
            <div className="text-[12.5px] text-muted">无错误 🎉</div>
          ) : (
            <div className="flex flex-col gap-2">
              {groups.map((g, i) => (
                <div key={i} className="flex items-center justify-between gap-2">
                  <span className="truncate text-[12px] text-fg" title={g.signature}>{g.signature}</span>
                  <Chip tone="danger">{g.count}</Chip>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="overflow-hidden rounded-lg border border-border bg-surface shadow-lift-sm">
          <div className="border-b border-border-strong bg-surface-2 px-3 py-2 font-mono text-[10px] uppercase tracking-[0.14em] text-subtle">
            最近错误
          </div>
          <div className="max-h-[36vh] overflow-auto">
            {errorRows.length === 0 ? (
              <div className="px-3 py-4 text-[12.5px] text-muted">区间内无错误记录。</div>
            ) : (
              errorRows.map((row) => (
                <div key={row.id} className="border-b border-border px-3 py-2 last:border-b-0 hover:bg-surface-2">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono text-[11px] tabular-nums text-muted">{row.session_key}</span>
                    <span className="font-mono text-[10px] tabular-nums text-subtle">{_shortTs(row.ts)}</span>
                  </div>
                  <div className="mt-1 truncate text-[12.5px] text-danger" title={row.error}>{row.error}</div>
                  {row.user_preview && (
                    <div className="mt-0.5 truncate text-[11.5px] text-subtle" title={row.user_preview}>{row.user_preview}</div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

window.AkashicDashboard.registerPlugin({
  id: "observe",
  label: "Observe 监测",
  viewLabel: "observe",
  layout: "workbench",
  pageSize: 30,
  rowKey: "id",

  countTitle(total: number): string {
    return `${total} 轮遥测`;
  },

  columns: [
    { key: "session_key", label: "Session", width: 120, cellClass: "mono cell-session", rawTitle: true },
    { key: "ts", label: "Time", width: 96, fmt: "mono-time", cellClass: "mono cell-time", rawTitle: true },
    { key: "error", label: "Error", flex: true, cellClass: "content-preview" },
  ],

  async getCount(): Promise<number | null> {
    try {
      const ov = await api<Overview>("/api/dashboard/observe/overview?range=all");
      return ov.turns || 0;
    } catch {
      return null;
    }
  },

  async fetchPage({ page, pageSize }: { page: number; pageSize: number }) {
    const data = await api<{ items: Record<string, unknown>[]; total: number }>(
      `/api/dashboard/observe/errors?range=all&page=${page}&page_size=${pageSize}`,
    );
    return { items: data.items || [], total: data.total || 0 };
  },

  Main: ObserveMain,
});

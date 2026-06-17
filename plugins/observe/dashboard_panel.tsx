/// <reference path="../../types/akashic-dashboard.d.ts" />
import { useEffect, useState, type ReactElement, type ReactNode } from "react";
import { MetricTile, TrendChart, Chip, api } from "@akashic/dashboard-ui";

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
  return String(Math.round(value));
}

function _pct(value: number | null): string {
  return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "—";
}

// Shorten an ISO-bucket label: "2026-06-17T11" -> "11:00", "2026-06-17" -> "6-17".
function _bucketLabel(bucket: string): string {
  if (bucket.includes("T")) return `${bucket.slice(11, 13)}:00`;
  const [, m, d] = bucket.split("-");
  return m && d ? `${Number(m)}-${d}` : bucket;
}

function _shortTs(value: string): string {
  const dt = new Date(value);
  if (Number.isNaN(dt.getTime())) return value || "—";
  return `${dt.getMonth() + 1}-${String(dt.getDate()).padStart(2, "0")} ${String(dt.getHours()).padStart(2, "0")}:${String(dt.getMinutes()).padStart(2, "0")}`;
}

// Percentage change of the last bucket vs the previous one, for the tile delta.
function _delta(values: number[]): number | null {
  if (values.length < 2) return null;
  const last = values[values.length - 1];
  const prev = values[values.length - 2];
  if (!prev) return null;
  return ((last - prev) / prev) * 100;
}

// A monitoring widget card with a hairline header — mirrors the superlog widget
// chrome (uppercase mono title, bottom-bordered header, padded body).
function Card({ title, children, bodyClass }: { title: string; children: ReactNode; bodyClass?: string }): ReactElement {
  return (
    <div className="flex flex-col overflow-hidden rounded-lg border border-border bg-surface shadow-lift-sm">
      <div className="flex items-center justify-between border-b border-border px-4 py-2.5">
        <h3 className="font-mono text-[10px] uppercase tracking-[0.2em] text-muted">{title}</h3>
      </div>
      <div className={bodyClass ?? "p-4"}>{children}</div>
    </div>
  );
}

// Grafana-style monitoring overview over observe.db agent-loop telemetry.
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
    return <div className="p-6 text-[13px] text-muted">加载中…</div>;
  }

  const turnSeries = points.map((p) => p.turns);
  const errorSeries = points.map((p) => p.errors);
  const tokenSeries = points.map((p) => p.input_tokens);
  const hitSeries = points.map((p) => (p.cache_hit_rate ?? 0) * 100);
  const iterSeries = points.map((p) => p.avg_iteration ?? 0);

  const labelled = (vals: number[]) => points.map((p, i) => ({ label: _bucketLabel(p.bucket), value: vals[i] }));

  return (
    <div className="flex flex-col gap-4 p-6">
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
      <div className="grid grid-cols-4 gap-4">
        <MetricTile
          label="对话轮数"
          value={_compact(overview.turns)}
          delta={_delta(turnSeries)}
          sub={overview.last_ts ? `最近 ${_shortTs(overview.last_ts)}` : "无记录"}
          tone="accent"
          spark={turnSeries}
        />
        <MetricTile
          label="错误"
          value={_compact(overview.errors)}
          sub={overview.error_rate != null ? `错误率 ${_pct(overview.error_rate)}` : "error 非空轮次"}
          tone="danger"
          spark={errorSeries}
        />
        <MetricTile
          label="KV 缓存命中率"
          value={_pct(overview.cache_hit_rate)}
          sub={`${_compact(overview.cache_hit_tokens)} / ${_compact(overview.cache_prompt_tokens)} tok`}
          tone="success"
          spark={hitSeries}
        />
        <MetricTile
          label="平均迭代"
          value={overview.avg_iteration != null ? overview.avg_iteration.toFixed(1) : "—"}
          unit={`峰 ${overview.max_iteration}`}
          sub="每轮 LLM 调用次数"
          tone="warning"
          spark={iterSeries}
        />
      </div>

      {/* trend charts */}
      <div className="grid grid-cols-2 gap-4">
        <Card title="输入 Token 趋势">
          <TrendChart data={labelled(tokenSeries)} kind="area" tone="accent" valueFmt={_compact} />
        </Card>
        <Card title="平均迭代趋势">
          <TrendChart data={labelled(iterSeries)} kind="area" tone="warning" valueFmt={(n) => n.toFixed(1)} />
        </Card>
        <Card title="KV 缓存命中率趋势">
          <TrendChart data={labelled(hitSeries)} kind="area" tone="success" valueFmt={(n) => `${n.toFixed(0)}%`} />
        </Card>
        <Card title="错误趋势">
          <TrendChart data={labelled(errorSeries)} kind="bar" tone="danger" valueFmt={(n) => String(n)} empty="区间内无错误 🎉" />
        </Card>
      </div>

      {/* error aggregation + recent errors */}
      <div className="grid grid-cols-[300px_1fr] gap-4">
        <Card title="错误聚合 · Top">
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
        </Card>

        <Card title="最近错误" bodyClass="max-h-[34vh] overflow-auto">
          {errorRows.length === 0 ? (
            <div className="px-4 py-4 text-[12.5px] text-muted">区间内无错误记录。</div>
          ) : (
            errorRows.map((row) => (
              <div key={row.id} className="border-b border-border px-4 py-2 last:border-b-0 hover:bg-surface-2">
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
        </Card>
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

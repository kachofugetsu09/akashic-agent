/// <reference path="../../types/akashic-dashboard.d.ts" />
import { useEffect, useState, type ReactElement } from "react";
import { Pie, api } from "@akashic/dashboard-ui";

interface KVCacheSummary {
  tracked_turn_count: number;
  prompt_tokens: number;
  hit_tokens: number;
  miss_tokens: number;
  hit_rate: number | null;
  last_tracked_at: string | null;
  passive: KVCacheSourceSummary;
  proactive: KVCacheSourceSummary;
}

interface KVCacheSourceSummary {
  tracked_turn_count: number;
  prompt_tokens: number;
  hit_tokens: number;
  miss_tokens: number;
  hit_rate: number | null;
}

interface KVCacheTurn {
  id: number;
  ts: string;
  source: string;
  session_key: string;
  user_preview: string;
  prompt_tokens: number;
  hit_tokens: number;
  miss_tokens: number;
  hit_rate: number | null;
}

function _formatNumber(value: unknown): string {
  return new Intl.NumberFormat("zh-CN").format(Number(value || 0));
}

function _formatRate(value: unknown): string {
  if (typeof value !== "number") {
    return "-";
  }
  return `${(value * 100).toFixed(1)}%`;
}

// Color-code a hit rate: high -> green, mid -> amber, low -> red.
function _hitTone(rate: number | null): string {
  if (rate == null) return "text-muted";
  if (rate >= 0.8) return "text-success";
  if (rate >= 0.5) return "text-warning";
  return "text-danger";
}

function _shortTs(value: string): string {
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) {
    return value || "-";
  }
  return `${d.getMonth() + 1}-${String(d.getDate()).padStart(2, "0")} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

const TABLE_GRID = "128px 92px 60px 92px 92px 1fr";

function KvMain(_props: { dispatch: PluginDispatch }): ReactElement {
  const [overview, setOverview] = useState<KVCacheSummary | null>(null);
  const [turns, setTurns] = useState<KVCacheTurn[]>([]);

  useEffect(() => {
    let alive = true;
    void (async () => {
      const [ov, page] = await Promise.all([
        api<KVCacheSummary>("/api/dashboard/status-commands/kvcache/overview"),
        api<{ items: KVCacheTurn[] }>("/api/dashboard/status-commands/kvcache/turns?page=1&page_size=50"),
      ]);
      if (alive) {
        setOverview(ov);
        setTurns(page.items ?? []);
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  if (!overview) {
    return <div className="p-5 text-[13px] text-muted">加载中…</div>;
  }
  const passive = overview.passive ?? overview;
  const recentPassive = turns.filter((turn) => turn.source === "agent").slice(0, 10);
  const recentPassiveHit = recentPassive.reduce((sum, turn) => sum + (turn.hit_tokens || 0), 0);
  const recentPassiveMiss = recentPassive.reduce((sum, turn) => sum + (turn.miss_tokens || 0), 0);
  const recentPassiveRate = (
    recentPassiveHit + recentPassiveMiss > 0
      ? recentPassiveHit / (recentPassiveHit + recentPassiveMiss)
      : null
  );
  const proactive = overview.proactive ?? {
    tracked_turn_count: 0,
    prompt_tokens: 0,
    hit_tokens: 0,
    miss_tokens: 0,
    hit_rate: null,
  };

  return (
    <div className="p-5">
      <div className="detail-title">KV Cache</div>
      <div className="detail-subtext">最近几次 KVCache 调用 · token 复用</div>

      <div className="mt-5 grid gap-3" style={{ gridTemplateColumns: "repeat(3, minmax(0, 1fr))" }}>
        <div className="min-w-0 overflow-hidden rounded-lg border border-border bg-surface p-3 shadow-lift-sm animate-fade-up">
          <Pie title={`最近 10 次被动链路 · ${recentPassive.length} 轮`} rate={recentPassiveRate} hit={recentPassiveHit} miss={recentPassiveMiss} />
        </div>
        <div className="min-w-0 overflow-hidden rounded-lg border border-border bg-surface p-3 shadow-lift-sm animate-fade-up" style={{ animationDelay: "80ms" }}>
          <Pie title={`全局被动链路 · ${passive.tracked_turn_count} 轮`} rate={passive.hit_rate} hit={passive.hit_tokens} miss={passive.miss_tokens} />
        </div>
        <div className="min-w-0 overflow-hidden rounded-lg border border-border bg-surface p-3 shadow-lift-sm animate-fade-up" style={{ animationDelay: "160ms" }}>
          <Pie title={`全局主动链路 · ${proactive.tracked_turn_count} 轮`} rate={proactive.hit_rate} hit={proactive.hit_tokens} miss={proactive.miss_tokens} />
        </div>
      </div>

      <div className="animate-fade-up mt-5 overflow-hidden rounded-lg border border-border" style={{ animationDelay: "220ms" }}>
        <div
          className="grid items-center border-b border-border-strong bg-surface-2 px-3 py-2 font-mono text-[10px] uppercase tracking-[0.14em] text-subtle"
          style={{ gridTemplateColumns: TABLE_GRID, columnGap: "10px" }}
        >
          <div>Session</div>
          <div>Time</div>
          <div className="text-right">Hit</div>
          <div className="text-right">Hit Tok</div>
          <div className="text-right">Prompt</div>
          <div>User</div>
        </div>
        <div className="max-h-[42vh] overflow-auto">
          {turns.length === 0 ? (
            <div className="px-3 py-4 text-[12.5px] text-muted">暂无 KVCache 记录。</div>
          ) : (
            turns.map((t) => (
              <div
                key={t.id}
                className="grid items-center border-b border-border px-3 py-2 text-[12.5px] last:border-b-0 hover:bg-surface-2"
                style={{ gridTemplateColumns: TABLE_GRID, columnGap: "10px" }}
              >
                <div className="truncate font-mono tabular-nums text-muted" title={t.session_key}>{t.session_key}</div>
                <div className="font-mono tabular-nums text-muted">{_shortTs(t.ts)}</div>
                <div className={`text-right font-mono tabular-nums ${_hitTone(t.hit_rate)}`}>{_formatRate(t.hit_rate)}</div>
                <div className="text-right font-mono tabular-nums text-fg">{_formatNumber(t.hit_tokens)}</div>
                <div className="text-right font-mono tabular-nums text-muted">{_formatNumber(t.prompt_tokens)}</div>
                <div className="truncate text-fg">{t.user_preview || "（无内容）"}</div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

window.AkashicDashboard.registerPlugin({
  id: "status_commands",
  label: "KV Cache",
  viewLabel: "kv cache",
  layout: "workbench",
  pageSize: 25,
  rowKey: "id",

  countTitle(total: number): string {
    return `${total} 轮 KVCache`;
  },

  columns: [
    { key: "session_key", label: "Session", width: 108, fmt: "mono-session", cellClass: "mono cell-session", rawTitle: true },
    { key: "ts", label: "Time", width: 96, fmt: "mono-time", cellClass: "mono cell-time", rawTitle: true },
    { key: "hit_rate", label: "Hit", width: 72, fmt: "cache-rate", cellClass: "mono cell-metric", align: "right" },
    { key: "user_preview", label: "User", flex: true, fmt: "text-preview", cellClass: "content-preview" },
  ],

  async getCount(): Promise<number | null> {
    try {
      const summary = await api<KVCacheSummary>("/api/dashboard/status-commands/kvcache/overview");
      return summary.tracked_turn_count || 0;
    } catch {
      return null;
    }
  },

  async fetchPage({ page, pageSize }: { page: number; pageSize: number }) {
    const params = new URLSearchParams();
    params.set("page", String(page));
    params.set("page_size", String(pageSize));
    const data = await api<{ items: Record<string, unknown>[]; total: number }>(
      `/api/dashboard/status-commands/kvcache/turns?${params.toString()}`,
    );
    return { items: data.items || [], total: data.total || 0 };
  },

  Main: KvMain,

  formatters: {
    "cache-rate": (value: unknown) => _formatRate(value),
    number: (value: unknown) => _formatNumber(value),
  },
});

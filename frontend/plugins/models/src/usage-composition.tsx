import { useState, type ReactNode } from "react";

// 上下文构成：类别 -> 来源 -> token（按 provider 实报输入换算的估算）；颜色只给色块，文字保持正文墨色。
export type Composition = Record<string, Record<string, number>>;
export interface ComposedTurn { call_id: string; recorded_at: string; context_tokens: number; context_window: number; prompt_tokens: number; hit_tokens: number; output_tokens: number; compacted: boolean; composition: Composition | null }

export const CATEGORIES = [
  ["system", "系统提示"], ["tools", "工具定义"], ["user", "用户消息"], ["injected", "注入材料"], ["assistant", "助手消息"], ["tool_results", "工具结果"],
] as const;
const fmt = (value: number) => value >= 1e6 ? `${(value / 1e6).toFixed(1)}M` : value >= 1e4 ? `${Math.round(value / 1e3)}k` : value >= 1e3 ? `${(value / 1e3).toFixed(1)}k` : String(value);
const pct = (value: number) => `${Math.round(value * 100)}%`;
const sum = (group: Record<string, number> | undefined) => Object.values(group ?? {}).reduce((total, value) => total + value, 0);
const totals = (parts: Composition) => CATEGORIES.map(([key, label]) => ({ key, label, value: sum(parts[key]) }));
const LABELS: Record<string, string> = { "": "合计" };

// 横向构成条；给了窗口时剩余部分留空，表示还能装多少。
export function CompositionBar({ parts, window = 0 }: { parts: Composition; window?: number }) {
  const items = totals(parts);
  const used = items.reduce((total, item) => total + item.value, 0);
  return <div className="usage-stack" role="img" aria-label={items.filter((item) => item.value).map((item) => `${item.label} ${fmt(item.value)}`).join("，")}>
    {items.map((item) => item.value ? <span key={item.key} data-cat={item.key} style={{ flexGrow: item.value }} /> : null)}
    {window > used ? <span className="free" style={{ flexGrow: window - used }} /> : null}
  </div>;
}

// 图例兼明细：每类可展开看来源（哪个插件的材料、哪个工具）。
export function CompositionLegend({ parts, compact = false }: { parts: Composition; compact?: boolean }) {
  const items = totals(parts);
  const all = items.reduce((total, item) => total + item.value, 0) || 1;
  return <div className="usage-legend-grid" data-compact={compact || undefined}>
    {items.filter((item) => item.value).map((item) => {
      const sources = Object.entries(parts[item.key] ?? {}).filter(([label]) => label).sort((a, b) => b[1] - a[1]);
      const head = <><i data-cat={item.key} /><em>{item.label}</em><span>{compact ? pct(item.value / all) : `≈${fmt(item.value)} · ${pct(item.value / all)}`}</span></>;
      return compact || !sources.length ? <div key={item.key} className="usage-lg">{head}</div>
        : <details key={item.key} className="usage-lg"><summary>{head}</summary>
          <ul>{sources.map(([label, value]) => <li key={label}><span>{LABELS[label] ?? label}</span><span>≈{fmt(value)}</span></li>)}</ul>
        </details>;
    })}
  </div>;
}

// 环形图：最近一轮的构成，中心是实报上下文。
export function Donut({ parts, center, sub }: { parts: Composition; center: string; sub: string }) {
  const items = totals(parts).filter((item) => item.value);
  const all = items.reduce((total, item) => total + item.value, 0) || 1;
  const radius = 52, length = 2 * Math.PI * radius, gap = items.length > 1 ? 2 : 0;
  let offset = 0;
  return <div className="usage-donut">
    <svg viewBox="0 0 124 124" role="img" aria-label="上下文构成">
      {items.map((item) => {
        const arc = Math.max(item.value / all * length - gap, 0.5);
        const node = <circle key={item.key} data-cat={item.key} r={radius} cx="62" cy="62" fill="none" strokeWidth="12"
          strokeDasharray={`${arc} ${length - arc}`} strokeDashoffset={-offset} transform="rotate(-90 62 62)"><title>{`${item.label} ≈${fmt(item.value)}`}</title></circle>;
        offset += item.value / all * length;
        return node;
      })}
    </svg>
    <div className="usage-donut-mid"><b>{center}</b><span>{sub}</span></div>
  </div>;
}

// 逐轮趋势：柱按六类合计堆叠（没有构成记录的旧轮次画成灰柱），命中率单独一排点、与柱逐轮对齐，不和 token 共用纵轴；
// 点柱看这一轮明细，来源明细由调用方按需读取。
export function ContextTrend({ turns, detail }: { turns: ComposedTurn[]; detail: (turn: ComposedTurn) => ReactNode }) {
  const [picked, setPicked] = useState(turns.length - 1);
  const max = Math.max(1, ...turns.map((turn) => turn.context_tokens));
  const turn = turns[Math.min(picked, turns.length - 1)];
  return <div className="usage-trend">
    <div className="usage-trend-bars" role="group" aria-label="逐轮上下文构成">
      {turns.map((item, index) => {
        const parts = item.composition ? totals(item.composition) : [];
        const known = parts.reduce((total, part) => total + part.value, 0);
        return <button key={item.recorded_at} type="button" aria-pressed={index === picked} onClick={() => setPicked(index)}
          title={`${new Date(item.recorded_at).toLocaleString("zh-CN", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" })} · ${fmt(item.context_tokens)}`}>
          <span style={{ height: `${item.context_tokens / max * 100}%` }}>
            {known ? parts.map((part) => part.value ? <i key={part.key} data-cat={part.key} style={{ flexGrow: part.value }} /> : null) : <i className="unknown" style={{ flexGrow: 1 }} />}
          </span>
        </button>;
      })}
    </div>
    <div className="usage-trend-hit" role="img" aria-label="逐轮缓存命中率">
      {turns.map((item, index) => <span key={item.recorded_at} data-picked={index === picked || undefined}>
        {item.prompt_tokens ? <i style={{ bottom: `${item.hit_tokens / item.prompt_tokens * 100}%` }} /> : null}
      </span>)}
    </div>
    <div className="usage-row muted"><span>缓存命中率（上 100% · 下 0）</span><span>柱高为每轮最后一次请求的实报输入</span></div>
    {turn ? <div className="usage-step">
      <div className="usage-row"><b>{new Date(turn.recorded_at).toLocaleString("zh-CN", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" })}</b>
        <span>上下文 {fmt(turn.context_tokens)} · 命中 {turn.prompt_tokens ? pct(turn.hit_tokens / turn.prompt_tokens) : "—"} · 输出 {fmt(turn.output_tokens)}{turn.compacted ? " · 压缩" : ""}</span></div>
      {turn.composition ? detail(turn) : <p className="usage-note">这一轮在开始记录构成之前，只有总量。</p>}
    </div> : null}
  </div>;
}

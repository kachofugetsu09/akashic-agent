import { useEffect, useState } from "react";
import { render, type PluginUiSidebarPanelContext, type SidebarPanelProvider } from "@akashic/chat-ui-v1";
import "./usage-panel.css";

// 侧栏「用量」面板：小浮层是本会话摘要，展开后分概览、会话、上下文三页；数据全部来自 provider 实报。

interface Totals { turns: number; prompt_tokens: number; hit_tokens: number; output_tokens: number; calls: number }
interface Group extends Totals { source?: string; model?: string }
interface SessionRow {
  session_id: string; title: string; sources: string[]; last_at: string; model: string; compactions: number;
  context_tokens: number; context_window: number; recent: Totals; totals: Totals;
}
interface Insights { hours: [string, number, number, number, number, number][]; sources: Group[]; models: Group[]; sessions: SessionRow[]; total_sessions: number }
interface TurnRow {
  recorded_at: string; preview: string; prompt_tokens: number; hit_tokens: number; output_tokens: number; calls: number;
  context_tokens: number; context_window: number; model: string; compacted: boolean;
}
interface SessionDetail { session: SessionRow | null; turns: TurnRow[] }
type Tab = "overview" | "sessions" | "context";
type Query = PluginUiSidebarPanelContext["query"];

const HOT = 0.8;
const SOURCE_LABEL: Record<string, string> = { conversation: "对话", programmatic: "程序调用", wake: "主动消息", drift: "漫游", subagent: "子智能体" };
const SESSION_ROWS = 100;
const RANGES = [[1, "今天"], [7, "7 天"], [30, "30 天"]] as const;
const ICON = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M2.5 13.5h11M4.5 11V7.5M8 11V3.5M11.5 11V6"/></svg>';

const fmt = (value: number) => value >= 1e9 ? `${(value / 1e9).toFixed(2)}B` : value >= 1e6 ? `${(value / 1e6).toFixed(1)}M`
  : value >= 1e4 ? `${Math.round(value / 1e3)}k` : value >= 1e3 ? `${(value / 1e3).toFixed(1)}k` : String(value);
const pct = (ratio: number | null) => ratio === null ? "—" : `${Math.round(ratio * 100)}%`;
const rate = (item: Pick<Totals, "prompt_tokens" | "hit_tokens">) => item.prompt_tokens ? item.hit_tokens / item.prompt_tokens : null;
const fill = (row: Pick<SessionRow, "context_tokens" | "context_window">) => row.context_window ? row.context_tokens / row.context_window : null;
const sourceLabel = (source: string) => SOURCE_LABEL[source] ?? source;
const sessionTitle = (row: SessionRow) => row.title || "未命名会话";

// 本地零点往前推 days-1 天；小时桶按 UTC 存，这里换成本地时间再归桶。
function rangeStart(days: number): Date {
  const start = new Date();
  start.setHours(0, 0, 0, 0);
  start.setDate(start.getDate() - days + 1);
  return start;
}
const hourDate = (key: string) => new Date(`${key}:00:00Z`);
// 起点之后的小时桶合计；会话排行有上限，总量从桶里算。
function since(hours: Insights["hours"], start: Date): Totals {
  const total = { turns: 0, prompt_tokens: 0, hit_tokens: 0, output_tokens: 0, calls: 0 };
  for (const [key, turns, prompt, hit, output, calls] of hours) {
    if (hourDate(key) < start) continue;
    total.turns += turns; total.prompt_tokens += prompt; total.hit_tokens += hit; total.output_tokens += output; total.calls += calls;
  }
  return total;
}

function useQuery<T>(query: Query, method: string, payload: Record<string, unknown>, token: string): { data: T | null; error: string } {
  const [state, setState] = useState<{ data: T | null; error: string; key: string }>({ data: null, error: "", key: "" });
  const key = `${method}:${JSON.stringify(payload)}:${token}`;
  useEffect(() => {
    let live = true;
    let timer = 0;
    const load = (attempt: number) => query(method, payload, { cache: "none" }).then((result) => {
      if (!live) return;
      // 冷启动时扫描器每次请求推进约 2 秒：未扫完先显示进度，挂载期间退避着接着读，直到扫完或卸载。
      if (result.complete === false) {
        setState({ data: null, error: `正在统计 ${String(result.scanned_sessions)} / ${String(result.total_sessions)} 个会话…`, key });
        timer = window.setTimeout(() => { if (live) void load(attempt + 1); }, Math.min(2000, 400 * (attempt + 1)));
        return;
      }
      setState({ data: result as T, error: "", key });
    }).catch((error: unknown) => { if (live) setState({ data: null, error: error instanceof Error ? error.message : "用量读取失败", key }); });
    void load(0);
    return () => { live = false; window.clearTimeout(timer); };
  }, [key]);
  return state.key === key ? state : { data: state.data, error: "" };
}

function Bars({ items, label, turns = false }: { items: { hit: number; miss: number; title?: string; hot?: boolean }[]; label: string; turns?: boolean }) {
  const max = Math.max(1, ...items.map((item) => item.hit + item.miss));
  return <div className="usage-bars" role="img" aria-label={label} data-turns={turns || undefined}>
    {items.map((item, index) => <span key={index} title={item.title} data-hot={item.hot || undefined}>
      <i style={{ height: `${item.miss / max * 100}%` }} /><i className="soft" style={{ height: `${item.hit / max * 100}%` }} />
    </span>)}
  </div>;
}

function Metric({ value, label, warn = false }: { value: string; label: string; warn?: boolean }) {
  return <div className="usage-metric"><b data-warn={warn || undefined}>{value}</b><span>{label}</span></div>;
}

function ContextLine({ row }: { row: SessionRow }) {
  const ratio = fill(row);
  return <div className="usage-context">
    <div className="usage-row"><span>上下文 <b data-warn={ratio !== null && ratio >= HOT || undefined}>{pct(ratio)}</b></span>
      <span>{fmt(row.context_tokens)}{row.context_window ? ` / ${fmt(row.context_window)}` : ""}</span></div>
    <div className="usage-meter"><i style={{ width: `${Math.min(1, ratio ?? 0) * 100}%` }} data-hot={ratio !== null && ratio >= HOT || undefined} /></div>
    {ratio !== null && ratio >= HOT ? <p className="usage-note">接近上限，下一次请求可能触发压缩。</p> : null}
  </div>;
}

// 今天按本地小时归桶；未来的小时不画。
function todayBuckets(hours: Insights["hours"]) {
  const start = rangeStart(1).getTime(), now = Date.now();
  const items = Array.from({ length: 24 }, (_, hour) => ({ hit: 0, miss: 0, future: start + hour * 3600e3 > now }));
  for (const [key, , prompt, hit] of hours) {
    const index = Math.floor((hourDate(key).getTime() - start) / 3600e3);
    if (index >= 0 && index < 24) { items[index].hit += hit; items[index].miss += prompt - hit; }
  }
  return items.map((item, hour) => ({ ...item, title: `${hour} 时 · ${fmt(item.hit + item.miss)}` }));
}

function Summary({ query, sessionId, open, sheet }: { query: Query; sessionId?: string; open: (tab: Tab, session?: string) => void; sheet: boolean }) {
  const start = rangeStart(1);
  const own = useQuery<SessionDetail>(query, "usage.session", sessionId ? { session_id: sessionId } : {}, sessionId ?? "none");
  const all = useQuery<Insights>(query, "usage.insights", { since: start.toISOString(), limit: 3 }, "");
  const row = sessionId ? own.data?.session ?? null : null;
  const today = all.data ? since(all.data.hours, start) : null;
  return <div className="usage-summary" data-sheet={sheet || undefined}>
    {row ? <>
      <div className="usage-hero">
        <Metric value={fmt(row.totals.prompt_tokens)} label={`输入 · 输出 ${fmt(row.totals.output_tokens)}`} />
        <Metric value={pct(rate(row.totals))} label="缓存命中" />
        <Bars turns label="本会话逐轮输入" items={(own.data?.turns ?? []).slice(-40).map((turn) => ({ hit: turn.hit_tokens, miss: turn.prompt_tokens - turn.hit_tokens }))} />
      </div>
      <ContextLine row={row} />
      <div className="usage-row muted"><span>{row.totals.turns} 轮 · {row.totals.calls} 次调用 · 压缩 {row.compactions} 次</span>
        <button type="button" onClick={() => open("context")}>上下文</button></div>
    </> : <p className="usage-note">{sessionId ? own.error || (own.data ? "这个会话还没有完整的用量记录。" : "正在读取…") : "新会话发出第一条消息后才有用量。"}</p>}
    <div className="usage-row"><span>今天 <b>{today ? fmt(today.prompt_tokens) : "—"}</b> · 命中 {today ? pct(rate(today)) : "—"}</span>
      <button type="button" onClick={() => open("overview")}>概览</button></div>
    {all.data ? <Bars label="今天按小时的输入" items={todayBuckets(all.data.hours)} /> : <p className="usage-note">{all.error || "正在读取…"}</p>}
    {all.data?.sessions.length ? <div className="usage-list">
      <div className="usage-row muted"><span>今天用得最多</span><span>输入 · 命中</span></div>
      {all.data.sessions.slice(0, 3).map((item) => <button key={item.session_id} type="button" className="usage-item" onClick={() => open("sessions", item.session_id)}>
        <span className="t">{sessionTitle(item)}</span><span className="n">{fmt(item.recent.prompt_tokens)}</span><span className="n">{pct(rate(item.recent))}</span>
      </button>)}
    </div> : null}
    <p className="usage-note">数字为 provider 实报；只计全部调用都有完整用量的轮次。</p>
  </div>;
}

function Overview({ query, days, setDays, onPick }: { query: Query; days: number; setDays: (days: number) => void; onPick: (id: string) => void }) {
  const start = rangeStart(days);
  const { data, error } = useQuery<Insights>(query, "usage.insights", { since: start.toISOString(), limit: 6 }, "");
  const [group, setGroup] = useState<"source" | "model">("source");
  if (!data) return <p className="usage-note">{error || "正在读取…"}</p>;
  const total = since(data.hours, start);
  const hourly = days === 1, count = hourly ? 24 : days, step = hourly ? 3600e3 : 86400e3;
  const buckets = Array.from({ length: count }, () => ({ hit: 0, miss: 0 }));
  for (const [key, , prompt, hit] of data.hours) {
    const moment = hourDate(key);
    const index = hourly ? Math.floor((moment.getTime() - start.getTime()) / step)
      : Math.round((new Date(moment.getFullYear(), moment.getMonth(), moment.getDate()).getTime() - start.getTime()) / step);
    if (index >= 0 && index < count) { buckets[index].hit += hit; buckets[index].miss += prompt - hit; }
  }
  const groups = group === "source" ? data.sources : data.models;
  return <div className="usage-overview">
    <RangeSwitch days={days} setDays={setDays} />
    <div className="usage-kpis">
      <Metric value={fmt(total.prompt_tokens + total.output_tokens)} label={`Token · 输出 ${fmt(total.output_tokens)}`} />
      <Metric value={pct(rate(total))} label={`缓存命中 · 读取 ${fmt(total.hit_tokens)}`} />
      <Metric value={String(total.calls)} label={`调用 · ${total.turns} 轮`} />
      <Metric value={String(data.total_sessions)} label="会话" />
    </div>
    <Bars label={hourly ? "今天按小时的输入" : `近 ${days} 天按天的输入`} items={buckets.map((item, index) => ({ ...item, title: `${hourly ? `${index} 时` : `${days - 1 - index} 天前`} · ${fmt(item.hit + item.miss)}` }))} />
    <div className="usage-legend"><span><i />未命中</span><span><i className="soft" />命中</span><span>按{hourly ? "小时" : "天"}，只画输入</span></div>
    <div className="usage-columns">
      <div className="usage-list">
        <div className="usage-row muted"><span className="usage-seg" role="group" aria-label="分组">
          <button type="button" aria-pressed={group === "source"} onClick={() => setGroup("source")}>按来源</button>
          <button type="button" aria-pressed={group === "model"} onClick={() => setGroup("model")}>按模型</button></span><span>输入 · 命中</span></div>
        {groups.map((item) => <div key={item.source ?? item.model} className="usage-item">
          <span className="t">{item.source ? sourceLabel(item.source) : item.model}</span><span className="n">{fmt(item.prompt_tokens)}</span><span className="n">{pct(rate(item))}</span>
        </div>)}
      </div>
      <div className="usage-list">
        <div className="usage-row muted"><span>最耗会话</span><span>输入 · 命中</span></div>
        {data.sessions.slice(0, 6).map((row) => <button key={row.session_id} type="button" className="usage-item" onClick={() => onPick(row.session_id)}>
          <span className="t">{sessionTitle(row)}</span><span className="n">{fmt(row.recent.prompt_tokens)}</span><span className="n">{pct(rate(row.recent))}</span>
        </button>)}
      </div>
    </div>
  </div>;
}

function RangeSwitch({ days, setDays }: { days: number; setDays: (days: number) => void }) {
  return <div className="usage-seg" role="group" aria-label="时间范围">
    {RANGES.map(([value, label]) => <button key={value} type="button" aria-pressed={days === value} onClick={() => setDays(value)}>{label}</button>)}
  </div>;
}

// 会话页与上下文页共用：左边列表、右边详情；手机抽屉里列表与详情前后推入。
function SessionsPage({ query, days, setDays, selected, setSelected, sheet, context }: {
  query: Query; days: number; setDays: (days: number) => void; selected: string; setSelected: (id: string) => void; sheet: boolean; context: boolean;
}) {
  // 回包有 192 KiB 上限：只取排行前 SESSION_ROWS 个；上下文页按占比在服务端排好再截。
  const { data, error } = useQuery<Insights>(query, "usage.insights",
    { since: rangeStart(context ? 30 : days).toISOString(), limit: SESSION_ROWS, order: context ? "context" : "input" }, "");
  const [filter, setFilter] = useState("");
  const [pushed, setPushed] = useState(false);
  const needle = filter.trim().toLowerCase();
  const rows = (data?.sessions ?? []).filter((row) => !needle || sessionTitle(row).toLowerCase().includes(needle) || row.sources.some((source) => sourceLabel(source).includes(needle)));
  const active = selected || rows[0]?.session_id || "";
  const list = <div className="usage-pane">
    {context ? null : <RangeSwitch days={days} setDays={setDays} />}
    <input className="usage-filter" type="search" value={filter} placeholder="按标题或来源筛选" aria-label="筛选会话" onChange={(event) => setFilter(event.target.value)} />
    {!data ? <p className="usage-note">{error || "正在读取…"}</p> : null}
    {data && data.total_sessions > data.sessions.length ? <p className="usage-note">只列出{context ? "上下文占用最高" : "输入最多"}的 {data.sessions.length} 个会话（共 {data.total_sessions} 个），筛选只在其中查找。</p> : null}
    {rows.map((row) => <button key={row.session_id} type="button" className="usage-item" aria-current={row.session_id === active || undefined}
      onClick={() => { setSelected(row.session_id); setPushed(true); }}>
      <span className="t">{sessionTitle(row)}</span>
      <span className="n">{context ? pct(fill(row)) : fmt(row.recent.prompt_tokens)}</span>
      <span className="s">{row.sources.map(sourceLabel).join(" · ")} · {context ? row.model || "未知模型" : `${row.recent.turns} 轮`}</span>
      <span className="n">{context ? `压缩 ${row.compactions}` : pct(rate(row.recent))}</span>
    </button>)}
  </div>;
  const detail = active ? <SessionDetailView key={active} query={query} sessionId={active} context={context} /> : null;
  if (sheet) return pushed && detail ? <div className="usage-pane"><button type="button" className="usage-back" onClick={() => setPushed(false)}>← 返回列表</button>{detail}</div> : list;
  return <div className="usage-split">{list}<div className="usage-pane">{detail}</div></div>;
}

function SessionDetailView({ query, sessionId, context }: { query: Query; sessionId: string; context: boolean }) {
  const { data, error } = useQuery<SessionDetail>(query, "usage.session", { session_id: sessionId }, "");
  const row = data?.session;
  if (!row) return <p className="usage-note">{error || (data ? "这个会话没有完整的用量记录。" : "正在读取…")}</p>;
  const turns = data.turns;
  return <div className="usage-detail">
    <div><div className="usage-title">{sessionTitle(row)}</div>
      <div className="usage-note">{row.sources.map(sourceLabel).join(" · ")} · {row.model || "未知模型"}</div></div>
    {context ? <>
      <ContextLine row={row} />
      <Bars turns label="逐轮上下文占用" items={turns.map((turn) => ({ hit: 0, miss: turn.context_tokens, hot: turn.context_window ? turn.context_tokens / turn.context_window >= HOT : false,
        title: `${turn.compacted ? "压缩 · " : ""}${fmt(turn.context_tokens)}${turn.context_window ? ` / ${fmt(turn.context_window)}` : ""}` }))} />
      <p className="usage-note">柱高是每轮最后一次请求的实报输入，最近 {turns.length} 轮。各部分（系统提示词、工具、记忆、文件……）的构成之后提供。</p>
    </> : <>
      <div className="usage-kpis">
        <Metric value={fmt(row.totals.prompt_tokens)} label={`输入 · 输出 ${fmt(row.totals.output_tokens)}`} />
        <Metric value={pct(rate(row.totals))} label="缓存命中" />
        <Metric value={String(row.totals.turns)} label={`轮 · ${row.totals.calls} 次调用`} />
        <Metric value={String(row.compactions)} label="次压缩（累计）" />
      </div>
      <Bars turns label="逐轮输入" items={turns.map((turn) => ({ hit: turn.hit_tokens, miss: turn.prompt_tokens - turn.hit_tokens }))} />
      <div className="usage-list">
        {turns.slice().reverse().map((turn) => <div key={turn.recorded_at} className="usage-item">
          <span className="t">{turn.preview || "（无文字）"}{turn.compacted ? <em>压缩</em> : null}</span>
          <span className="n">{fmt(turn.prompt_tokens)}</span>
          <span className="s">{new Date(turn.recorded_at).toLocaleString("zh-CN", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" })} · 输出 {fmt(turn.output_tokens)}</span>
          <span className="n">{pct(rate(turn))}</span>
        </div>)}
      </div>
    </>}
  </div>;
}

function UsagePanel({ context }: { context: PluginUiSidebarPanelContext }) {
  const { panel } = context;
  const [size, setSize] = useState(panel.size());
  const [tab, setTab] = useState<Tab>("overview");
  const [days, setDays] = useState(1);
  const [selected, setSelected] = useState(context.sessionId ?? "");
  useEffect(() => context.onInvalidate(() => setSize(panel.size())), [context, panel]);
  const open = (next: Tab, session?: string) => { if (session) setSelected(session); setTab(next); panel.setSize("expanded"); };
  if (size === "compact") return <Summary query={context.query} sessionId={context.sessionId} open={open} sheet={panel.sheet} />;
  return <div className="usage-expanded" data-sheet={panel.sheet || undefined}>
    <div className="usage-tabs" role="tablist" aria-label="用量">
      {([["overview", "概览"], ["sessions", "会话"], ["context", "上下文"]] as const).map(([value, label]) =>
        <button key={value} type="button" role="tab" aria-selected={tab === value} onClick={() => setTab(value)}>{label}</button>)}
    </div>
    {tab === "overview" ? <Overview query={context.query} days={days} setDays={setDays} onPick={(id) => { setSelected(id); setTab("sessions"); }} />
      : <SessionsPage key={tab} query={context.query} days={days} setDays={setDays} selected={selected} setSelected={setSelected} sheet={panel.sheet} context={tab === "context"} />}
  </div>;
}

// 菜单短值：本会话上下文占比，窗口未知时退为本会话输入量；超过 80% 时挂提醒点。
export const usagePanel: SidebarPanelProvider = {
  label: "用量",
  iconSvg: ICON,
  // 只读已扫描的结果，不为一个短值推进扫描；还没扫到的会话没有短值。
  async status({ query, sessionId }) {
    if (!sessionId) return null;
    const { status } = await query("usage.status", {}, { cache: "none" }) as { status: { complete: boolean; context_tokens: number; context_window: number; prompt_tokens: number } | null };
    if (!status) return null;
    // 还没扫到最新一轮：已扫到的最后一轮可能很旧，不拿它当当前占用。
    if (!status.complete) return { text: "统计中" };
    const ratio = fill(status);
    return ratio === null ? { text: `本会话 ${fmt(status.prompt_tokens)}` } : { text: `上下文 ${pct(ratio)}`, alert: ratio >= HOT };
  },
  mount: (host, context) => render(host, <UsagePanel context={context} />),
};

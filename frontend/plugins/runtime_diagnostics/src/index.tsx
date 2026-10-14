import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import type { WebHostContextV1, WebUiDisposer } from "@akashic/web-ui-v1";
import "./style.css";

type Row = { seq: number; timestamp: number; kind: string; [key: string]: unknown };
type Allocation = { bytes: number; count: number; frames: { file: string; line: number }[] };
type Profile = { state: string; ends_at?: number; peak_bytes?: number; traced_bytes?: number;
  overhead_bytes?: number; allocations?: Allocation[]; error?: string };
type Report = { schema: string; stream_id: string; through_seq: number; samples: Row[]; events: Row[];
  reset_required: boolean; gap: { samples: boolean; events: boolean }; evicted: { samples: number; events: number };
  capacity: { samples: number; events: number }; profile: Profile };
const ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 18V6m0 12h18M6 14l4-5 4 3 6-8"/></svg>';
const clock = (value: number) => new Date(value * 1000).toLocaleTimeString([], { hour12: false });
const number = (value: unknown) => typeof value === "number" ? value.toLocaleString(undefined, { maximumFractionDigits: 1 }) : "不可用";
const bytes = (value: unknown) => typeof value !== "number" ? "不可用" : value >= 1024 ** 3
  ? `${(value / 1024 ** 3).toFixed(2)} GiB` : `${(value / 1024 ** 2).toFixed(1)} MiB`;

async function request(ctx: WebHostContextV1, path: string, init?: RequestInit): Promise<Response> {
  const response = await ctx.http.request(path, init);
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`请求失败 (${response.status})：${body.slice(0, 250)}`);
  }
  return response;
}

function Chart({ samples, selected, select }: { samples: Row[]; selected: Row | undefined; select: (seq: number | null) => void }) {
  const [showGroup, setShowGroup] = useState(false);
  const series = [{ key: "rss_bytes", label: "主进程 RSS", style: "rss" },
    { key: "anonymous_bytes", label: "主进程匿名内存", style: "anon" },
    { key: "cgroup_bytes", label: "所在 cgroup 总量", style: "group" }];
  const visible = series.filter(item => showGroup || item.style !== "group");
  const max = Math.max(1024 ** 2, ...samples.flatMap(row => visible.map(item => typeof row[item.key] === "number" ? row[item.key] as number : 0)));
  const first = samples[0]?.timestamp ?? 0;
  const span = Math.max(5, (samples.at(-1)?.timestamp ?? first) - first);
  const x = (row: Row) => 8 + (row.timestamp - first) / span * 784;
  const path = (key: string) => {
    let connected = false;
    return samples.map(row => {
      if (typeof row[key] !== "number") { connected = false; return ""; }
      const command = connected ? "L" : "M";
      connected = true;
      return `${command}${x(row).toFixed(1)},${(184 - (row[key] as number) / max * 168).toFixed(1)}`;
    }).join(" ");
  };
  return <div className="diagnostics-chart">
    <div className="diagnostics-scale"><span>{bytes(max)}</span><label><input type="checkbox" checked={showGroup} onChange={event => setShowGroup(event.target.checked)} />显示 cgroup 曲线</label></div>
    <svg viewBox="0 0 800 200" role="img" aria-label="近期主进程和 cgroup 内存曲线" preserveAspectRatio="none"
      onPointerMove={event => {
        const rect = event.currentTarget.getBoundingClientRect();
        const timestamp = first + Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)) * span;
        const nearest = samples.reduce<Row | undefined>((best, row) => !best || Math.abs(row.timestamp - timestamp) < Math.abs(best.timestamp - timestamp) ? row : best, undefined);
        if (nearest) select(nearest.seq);
      }}>
      {[16, 100, 184].map(y => <line key={y} x1="0" x2="800" y1={y} y2={y} className="diagnostics-grid" />)}
      {visible.map(item => <path key={item.key} d={path(item.key)} className={`diagnostics-line diagnostics-${item.style}`} />)}
      {selected && <line x1={x(selected)} x2={x(selected)} y1="8" y2="192" className="diagnostics-marker" />}
    </svg>
    <div className="diagnostics-times"><span>{first ? clock(first) : "等待采样"}</span><span>{samples.at(-1) ? clock(samples.at(-1)!.timestamp) : ""}</span></div>
    <label className="diagnostics-scrubber">查看时刻
      <input type="range" min="0" max={Math.max(0, samples.length - 1)} value={Math.max(0, samples.findIndex(row => row.seq === selected?.seq))}
        disabled={!samples.length} onChange={event => select(samples[Number(event.target.value)].seq)} />
      <button type="button" onClick={() => select(null)}>跟随最新</button>
    </label>
    <dl className="diagnostics-legend">{series.map(item => <div key={item.key} className={`diagnostics-${item.style}`}>
      <dt>{item.label}</dt><dd>{bytes(selected?.[item.key])}</dd>
    </div>)}</dl>
  </div>;
}

function Diagnostics({ ctx }: { ctx: WebHostContextV1 }) {
  const [report, setReport] = useState<Report>();
  const [error, setError] = useState("");
  const [actionError, setActionError] = useState("");
  const [busy, setBusy] = useState(false);
  const [seconds, setSeconds] = useState(30);
  const [selectedSeq, setSelected] = useState<number | null>(null);
  const [filter, setFilter] = useState("");
  const cursor = useRef({ stream: "", seq: 0 });
  // 本流是否曾发生漏读：reset 时按当前响应重建，普通轮询按需累计。
  const [gapSeen, setGapSeen] = useState({ samples: false, events: false });
  const controller = useRef<AbortController | null>(null);
  const actionVersion = useRef(0);
  useEffect(() => {
    const abort = new AbortController();
    controller.current = abort;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      const version = actionVersion.current;
      try {
        let next = await (await request(ctx, `/api/dashboard/runtime-diagnostics?after=${cursor.current.seq}`, { signal: abort.signal })).json() as Report;
        if (next.schema !== "akashic.diagnostics.v1") throw new Error("诊断数据版本不受支持，请更新页面");
        const reset = next.stream_id !== cursor.current.stream || next.reset_required;
        if (reset && cursor.current.seq) next = await (await request(ctx, "/api/dashboard/runtime-diagnostics", { signal: abort.signal })).json() as Report;
        if (abort.signal.aborted) return;
        if (reset) setSelected(null);
        // reset 只表示游标需要重取，不代表最终采用的响应没有缺口：先按新流清空，再并入
        // 本次响应的 gap。用两个结构化标志维护「本流曾发生」，避免后一次覆盖前一次。
        setGapSeen(previous => ({
          samples: (reset ? false : previous.samples) || next.gap.samples,
          events: (reset ? false : previous.events) || next.gap.events,
        }));
        cursor.current = { stream: next.stream_id, seq: next.through_seq };
        setReport(previous => ({ ...next,
          profile: reset || version === actionVersion.current || !previous ? next.profile : previous.profile,
          samples: [...(reset ? [] : previous?.samples ?? []), ...next.samples].slice(-next.capacity.samples),
          events: [...(reset ? [] : previous?.events ?? []), ...next.events].slice(-next.capacity.events),
        }));
        setError("");
      } catch (cause) {
        if (!abort.signal.aborted) setError(cause instanceof Error ? cause.message : String(cause));
      } finally {
        if (!abort.signal.aborted) timer = setTimeout(() => void poll(), 5000);
      }
    };
    void poll();
    return () => { abort.abort(); clearTimeout(timer); };
  }, [ctx]);
  const selected = report?.samples.find(row => row.seq === selectedSeq) ?? report?.samples.at(-1);
  const pss = report?.samples.filter(row => row.timestamp <= (selected?.timestamp ?? 0) && row.pss_status !== undefined).at(-1);
  const finished = new Set(report?.events.filter(row => ["plugin.operation.done", "plugin.operation.error", "plugin.operation.cancelled"].includes(String(row.event))).map(row => row.operation_id));
  const events = report?.events.filter(row => String(row.plugin_id ?? "").includes(filter) &&
    (row.event !== "plugin.operation.start" || !finished.has(row.operation_id)) &&
    (selectedSeq === null || !selected || Math.abs(row.timestamp - selected.timestamp) <= 5)).slice(-30).reverse() ?? [];
  const profile = report?.profile;
  const running = profile?.state === "running" || profile?.state === "finishing";
  const act = async (action: "start" | "stop" | "export") => {
    setBusy(true);
    setActionError("");
    actionVersion.current += 1;
    try {
      const response = await request(ctx, `/api/dashboard/runtime-diagnostics/${action === "export" ? "export" : action === "stop" ? "profile/stop" : "profile"}`,
        action === "export" ? { signal: controller.current?.signal } : { method: "POST", signal: controller.current?.signal,
          headers: { "Content-Type": "application/json" }, ...(action === "start" ? { body: JSON.stringify({ seconds }) } : {}) });
      if (action === "export") {
        const url = URL.createObjectURL(await response.blob());
        const link = document.createElement("a");
        link.href = url; link.download = "akashic-diagnostics.json"; link.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      } else {
        const result = await response.json() as Profile;
        actionVersion.current += 1;
        setReport(previous => previous ? { ...previous, profile: result } : previous);
      }
    } catch (cause) {
      if (!controller.current?.signal.aborted) setActionError(cause instanceof Error ? cause.message : String(cause));
    } finally { setBusy(false); }
  };
  return <main className="diagnostics-page">
    <header><div><p className="diagnostics-kicker">AKASHIC / RUNTIME</p><h1>运行诊断</h1>
      <p>查看近期资源变化，再沿同一时刻的操作定位问题。</p></div>
      <button type="button" disabled={busy || !report} onClick={() => void act("export")}>导出诊断</button></header>
    {error && <p role="alert" className="diagnostics-error">{error}。保留上次数据，稍后自动重试。</p>}
    {actionError && <p role="alert" className="diagnostics-error">操作未完成：{actionError}</p>}
    {!report ? <p role="status">正在读取运行状态…</p> : <>
      {(gapSeen.samples || gapSeen.events) && <p role="status" className="diagnostics-gap">
        本流曾发生{[gapSeen.samples ? "样本" : "", gapSeen.events ? "事件" : ""].filter(Boolean).join("、")}漏读：缺口两侧的记录不构成连续采样。
        窗口累计淘汰 {number(report.evicted.samples)} 条样本 / {number(report.evicted.events)} 条事件（当前值）。
      </p>}
      <section aria-labelledby="diagnostics-memory"><div className="diagnostics-heading"><h2 id="diagnostics-memory">内存变化</h2>
        <span>{selected ? clock(selected.timestamp) : "等待首个样本"} · 每 5 秒采样</span></div>
        <Chart samples={report.samples} selected={selected} select={setSelected} />
        {selected?.process_status !== "ok" && <p>进程内存不可用：{String(selected?.process_note ?? "等待采样")}</p>}
        {selected?.cgroup_status !== "ok" && <p>当前环境不提供 cgroup 内存指标；进程与运行指标仍可独立使用。</p>}
        <p className="diagnostics-note">RSS 是当前 Python 进程的驻留内存。cgroup 总量可能包含同组其他程序、子进程与文件缓存，两者不能相加。</p>
        <dl className="diagnostics-stats">{[["任务数", number(selected?.task_count)], ["事件循环延迟", `${number(selected?.loop_lag_ms)} ms`],
          ["GC 累计耗时", `${number(selected?.gc_seconds)} s`], ["GC 累计回收对象", number(selected?.gc_collected)],
          ["进程 Swap", bytes(selected?.swap_bytes)], ["cgroup 文件缓存", bytes(selected?.cgroup_file_bytes)],
          [`PSS${pss ? ` · ${clock(pss.timestamp)}` : ""}`, bytes(pss?.pss_bytes)], ["GC 次数", number(selected?.gc_collections)]].map(([label, value]) =>
          <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
      </section>
      <section aria-labelledby="diagnostics-operations"><div className="diagnostics-heading"><h2 id="diagnostics-operations">同期操作</h2>
        <label>筛选插件 <input value={filter} onChange={event => setFilter(event.target.value)} placeholder="插件名称" /></label></div>
        <p className="diagnostics-note">{selectedSeq === null ? "最近 30 条操作或测量" : "所选时刻前后 5 秒，最多 30 条"}。时间上的重合不代表内存归属。</p>
        {selected?.operations_enabled === false && <p>当前日志级别未启用 INFO，普通操作记录不可用。</p>}
        <ol className="diagnostics-events">{events.map(row => <li key={row.seq}>
          <time>{clock(row.timestamp)}</time><div><strong>{String(row.plugin_id ?? "运行时")}</strong><span>{String(row.operation ?? row.measurement ?? row.event)}</span></div>
          <span>{row.event === "plugin.operation.start" ? "尚无结束记录" : row.duration_ms !== undefined ? `${number(row.duration_ms)} ms` : `${number(row.measurement_value)} ${String(row.measurement_unit ?? "")}`}<small>{String(row.outcome ?? "")}</small></span>
        </li>)}</ol>{!events.length && <p>此范围没有操作记录。</p>}
      </section>
      <section aria-labelledby="diagnostics-profile"><h2 id="diagnostics-profile">内存分配剖析</h2>
        <p>临时追踪新分配的 Python 内存，找出结束时仍保留内存的代码位置。开启前的分配不在结果中，原生扩展也可能未被覆盖。</p>
        <p className="diagnostics-note">剖析会额外消耗 CPU 和内存，到期自动停止；结果不包含对象正文。分配位置不等于当前持有者。</p>
        <div className="diagnostics-controls"><label>追踪时长 <select value={seconds} disabled={busy || running} onChange={event => setSeconds(Number(event.target.value))}>
          {[10, 30, 60, 120].map(value => <option key={value} value={value}>{value} 秒</option>)}</select></label>
          <button type="button" disabled={busy || running} onClick={() => void act("start")}>开始剖析</button>
          {running && <button type="button" disabled={busy} onClick={() => void act("stop")}>停止并查看结果</button>}
          <span role="status">{profile?.state === "running" ? `正在追踪，${clock(profile.ends_at!)} 自动停止` : profile?.state === "finishing" ? "正在汇总…" : profile?.state === "complete" ? "剖析完成" : profile?.state === "failed" ? `剖析失败：${profile.error}` : profile?.state === "cancelled" ? "剖析已取消" : "未开启"}</span></div>
        {profile?.state === "complete" && <><dl className="diagnostics-stats">
          {[["结束时追踪内存", profile.traced_bytes], ["期间峰值", profile.peak_bytes], ["追踪器自身内存", profile.overhead_bytes]].map(([label, value]) => <div key={String(label)}><dt>{label}</dt><dd>{bytes(value)}</dd></div>)}
        </dl><ol className="diagnostics-allocations">{profile.allocations?.map((row, i) => <li key={i}><strong>{bytes(row.bytes)} · {number(row.count)} 个内存块</strong>
          <details><summary>{row.frames.at(-1)?.file}:{row.frames.at(-1)?.line}</summary>{row.frames.map((frame, j) => <code key={j}>{frame.file}:{frame.line}</code>)}</details></li>)}</ol></>}
      </section>
      <footer>仅保留本次运行最近 {report.capacity.samples} 个样本与 {report.capacity.events} 条操作；停止或更新诊断插件后清空。
        已淘汰 {report.evicted.samples} 个样本、{report.evicted.events} 条操作。导出包含操作身份和剖析文件位置，不含对话正文。</footer>
    </>}
  </main>;
}

export function activate(ctx: WebHostContextV1): WebUiDisposer {
  return ctx.ui.inject("shell.pages.v1", mount => mount.register({ id: "runtime-diagnostics", label: "运行诊断", route: "diagnostics", order: 90,
    iconSvg: ICON, render(host: HTMLElement) { const root = createRoot(host); root.render(<Diagnostics ctx={ctx} />); return () => root.unmount(); } }));
}

import { useEffect, useMemo, useState } from "react";
import { ChevronDown, Square } from "lucide-react";
import { render, type PluginUiContext, type PluginUiDefinition } from "@akashic/chat-ui-v1";
import { ICONS, useValue, valueStore, type ValueStore } from "../../shared/src/meta-store";
import "./plugin-ui.css";

export type JobStatus = "running" | "stopping" | "completed" | "failed" | "timeout" | "killed";

/** 后台任务只读镜像的一行；时间为 epoch 毫秒。 */
export interface JobView {
  id: string;
  kind: string;
  label: string;
  status: JobStatus;
  detail: string;
  started_at: number;
  finished_at: number | null;
  stoppable: boolean;
}

const STATUS_LABEL: Record<JobStatus, string> = {
  running: "运行中",
  stopping: "正在停止",
  completed: "已完成",
  failed: "失败",
  timeout: "超时",
  killed: "已停止",
};

const isJob = (value: unknown): value is JobView => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const row = value as Record<string, unknown>;
  return typeof row.id === "string" && row.id.length > 0
    && typeof row.kind === "string" && typeof row.label === "string" && typeof row.detail === "string"
    && typeof row.status === "string" && row.status in STATUS_LABEL
    && Number.isSafeInteger(row.started_at)
    && (row.finished_at === null || Number.isSafeInteger(row.finished_at))
    && typeof row.stoppable === "boolean";
};

function formatDuration(elapsedMs: number): string {
  const total = Math.max(0, Math.floor(elapsedMs / 1000));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor(total / 60) % 60;
  if (hours > 0) return `${hours} 小时 ${minutes} 分`;
  if (minutes > 0) return `${minutes} 分 ${total % 60} 秒`;
  return `${total} 秒`;
}

const isLive = (job: JobView) => job.status === "running" || job.status === "stopping";

// 运行中按开始时间升序，已结束按结束时间倒序。
function ordered(jobs: readonly JobView[]): JobView[] {
  return [...jobs].sort((left, right) => {
    const live = isLive(left);
    if (live !== isLive(right)) return live ? -1 : 1;
    if (live) return left.started_at - right.started_at;
    return (right.finished_at ?? right.started_at) - (left.finished_at ?? left.started_at);
  });
}

type StopPhase = "idle" | "armed" | "pending" | "failed";

// 两击停止：首击武装、3 秒内再点确认；受理后保持等待，直到 session.jobs 帧把行改成"正在停止"而移除按钮。
function StopButton({ job, onStop }: { job: JobView; onStop: (jobId: string) => Promise<boolean> }) {
  const [phase, setPhase] = useState<StopPhase>("idle");
  useEffect(() => {
    if (phase !== "armed" && phase !== "failed") return;
    const timer = window.setTimeout(() => setPhase("idle"), phase === "armed" ? 3000 : 4000);
    return () => window.clearTimeout(timer);
  }, [phase]);
  const label = phase === "armed" ? "再点一次确认停止" : phase === "failed" ? "停止失败，任务可能已结束" : `停止任务 ${job.label}`;
  return <button type="button" className="job-list__stop" data-phase={phase} disabled={phase === "pending"}
    aria-label={label} title={label}
    onClick={() => {
      if (phase !== "armed") {
        setPhase("armed");
        return;
      }
      setPhase("pending");
      void onStop(job.id).then((accepted) => { if (!accepted) setPhase("failed"); });
    }}>
    <Square size={10} aria-hidden="true" />
    {phase === "armed" ? <span>确认停止</span> : null}
  </button>;
}

function JobRow({ job, now, onStop }: { job: JobView; now: number; onStop: (jobId: string) => Promise<boolean> }) {
  const running = isLive(job);
  const duration = formatDuration((running ? now : job.finished_at ?? job.started_at) - job.started_at);
  // 正常完成只说状态；失败、超时、停止时附上 detail 便于判断原因。
  const note = job.status === "completed" || !job.detail ? STATUS_LABEL[job.status] : `${STATUS_LABEL[job.status]} · ${job.detail}`;
  const stoppable = job.status === "running" && job.stoppable;
  return <li className="job-list__row" data-status={job.status} data-stoppable={stoppable || undefined}>
    <span className="job-list__dot" data-status={job.status} aria-hidden="true" />
    <span className="job-list__label" title={job.label}>{job.label}</span>
    <span className="job-list__duration" title={running ? `已运行 ${duration}` : `耗时 ${duration}`}>{duration}</span>
    <span className="job-list__meta" title={note}>{job.kind} · {note}</span>
    {stoppable ? <StopButton job={job} onStop={onStop} /> : null}
  </li>;
}

// 面板分"进行中 / 已结束"两组：有任务运行时已结束组默认折叠；清空只隐藏本页已有的结束行，之后新结束的任务照常出现。
function JobPanel({ live, settled, now, settledOpen, onToggleSettled, onClear, onStop }: {
  onStop: (jobId: string) => Promise<boolean>;
  live: readonly JobView[];
  settled: readonly JobView[];
  now: number;
  settledOpen: boolean;
  onToggleSettled: () => void;
  onClear: () => void;
}) {
  return <div className="job-list__panel" role="group" aria-label="后台任务">
    {live.length ? <>
      <p className="job-list__section" aria-hidden="true">进行中</p>
      <ul>{live.map((job) => <JobRow key={job.id} job={job} now={now} onStop={onStop} />)}</ul>
    </> : null}
    {settled.length ? <>
      <div className="job-list__section">
        <button type="button" className="job-list__fold" aria-expanded={settledOpen} onClick={onToggleSettled}>
          <ChevronDown size={12} aria-hidden="true" />已结束 {settled.length}
        </button>
        <button type="button" className="job-list__clear" onClick={onClear}>清空</button>
      </div>
      {settledOpen ? <ul>{settled.map((job) => <JobRow key={job.id} job={job} now={now} onStop={onStop} />)}</ul> : null}
    </> : null}
  </div>;
}

// 会话信息面板里的任务列表：有任务运行时已结束组默认折叠，时钟只在有运行中任务时走动。
function JobsDetail({ store, onStop }: { store: ValueStore<readonly JobView[]>; onStop: (jobId: string) => Promise<boolean> }) {
  const jobs = useValue(store);
  const [now, setNow] = useState(() => Date.now());
  const [settledOpen, setSettledOpen] = useState<boolean | undefined>(undefined);
  const [cleared, setCleared] = useState<ReadonlySet<string>>(() => new Set());
  const rows = useMemo(() => ordered(jobs), [jobs]);
  const live = useMemo(() => rows.filter(isLive), [rows]);
  const settled = useMemo(() => rows.filter((job) => !isLive(job) && !cleared.has(job.id)), [rows, cleared]);
  useEffect(() => {
    if (live.length === 0) return;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [live.length]);
  if (live.length + settled.length === 0) return <p className="job-list__empty">没有后台任务。</p>;
  return <JobPanel live={live} settled={settled} now={now} onStop={onStop}
    settledOpen={settledOpen ?? live.length === 0}
    onToggleSettled={() => setSettledOpen(!(settledOpen ?? live.length === 0))}
    onClear={() => setCleared((current) => new Set([...current, ...settled.map((job) => job.id)]))} />;
}

// 说明行上的一项：有任务在跑时是进行中；最近一个结束的失败或超时时需要注意；只剩已结束任务时只在面板里。
function jobsItem(jobs: readonly JobView[], detail: (host: HTMLElement) => () => void) {
  const rows = ordered(jobs);
  const live = rows.filter(isLive).length;
  const latest = rows.find((job) => !isLive(job));
  const failed = !live && latest && (latest.status === "failed" || latest.status === "timeout");
  return {
    id: "jobs", scope: "session" as const, label: "后台任务", iconSvg: ICONS.terminal, detail,
    tone: live ? "live" as const : failed ? "alert" as const : "info" as const,
    text: live ? `${live} 个后台任务在跑` : failed ? `后台任务${STATUS_LABEL[latest.status]}` : undefined,
    value: live ? `${live} 个运行中` : failed ? latest.label : `${rows.length} 个已结束`,
  };
}

const NO_JOBS: readonly JobView[] = [];

// 会话任务镜像来自本插件后端的 follow 推送；null（断线、插件换代）即旧镜像失效，按空列表处理。
// 不渲染界面：只往标题下的说明行登记一项，列表在宿主的会话信息面板里展开。
function SessionJobs({ context }: { context: PluginUiContext<"header.meta"> }) {
  const store = useMemo(() => valueStore<readonly JobView[]>(NO_JOBS), []);
  useEffect(() => context.subscribe((value) => {
    const rows = value?.jobs;
    const jobs = Array.isArray(rows) && rows.every(isJob) ? rows : NO_JOBS;
    store.set(jobs);
    context.meta.publish(jobs.length ? [jobsItem(jobs, (host) => render(host, <JobsDetail store={store} onStop={stop} />))] : []);
  }), [context, store]);
  // 停止只回答是否受理；行状态由推送的快照收敛（停止中 → 已停止）。
  const stop = async (jobId: string) => {
    try { return (await context.query("job.stop", { job_id: jobId }, { cache: "none" })).accepted === true; }
    catch { return false; }
  };
  return null;
}

const definition: PluginUiDefinition = {
  slots: { "header.meta": { mount: (host, context) => render(host, <SessionJobs context={context} />) } },
};
export default definition;

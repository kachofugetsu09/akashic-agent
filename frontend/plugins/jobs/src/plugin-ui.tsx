import { useEffect, useId, useMemo, useRef, useState } from "react";
import { ChevronDown, Square } from "lucide-react";
import { render, type PluginUiContext, type PluginUiDefinition } from "@akashic/chat-ui-v1";
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
function JobPanel({ id, live, settled, now, settledOpen, onToggleSettled, onClear, onStop }: {
  id: string;
  onStop: (jobId: string) => Promise<boolean>;
  live: readonly JobView[];
  settled: readonly JobView[];
  now: number;
  settledOpen: boolean;
  onToggleSettled: () => void;
  onClear: () => void;
}) {
  return <div id={id} className="job-list__panel" role="group" aria-label="后台任务" data-escape-layer="">
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

// 标题行右侧的安静标记：有任务运行时显示呼吸点和数量，只剩已结束任务时退为淡色计数。
function JobListAction({ jobs, onStop }: { jobs: readonly JobView[]; onStop: (jobId: string) => Promise<boolean> }) {
  const [open, setOpen] = useState(false);
  const [now, setNow] = useState(() => Date.now());
  const [settledOpen, setSettledOpen] = useState<boolean | undefined>(undefined);
  const [cleared, setCleared] = useState<ReadonlySet<string>>(() => new Set());
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const panelId = useId();
  const rows = useMemo(() => ordered(jobs), [jobs]);
  const live = useMemo(() => rows.filter(isLive), [rows]);
  const settled = useMemo(() => rows.filter((job) => !isLive(job) && !cleared.has(job.id)), [rows, cleared]);
  // 1. 点击面板外或按 Esc 收起，Esc 把焦点还给触发按钮。
  useEffect(() => {
    if (!open) return;
    const onPointer = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || event.defaultPrevented) return;
      setOpen(false);
      // 焦点已经离开面板（例如在工作台里）时不抢回。
      const active = document.activeElement;
      if (active === document.body || rootRef.current?.contains(active)) triggerRef.current?.focus();
    };
    document.addEventListener("pointerdown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);
  // 2. 时钟只在面板展开且有运行中任务时走动，收起后不再触发重渲染。
  useEffect(() => {
    if (!open || live.length === 0) return;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [open, live.length]);
  // 3. 没有可见任务时不占位置，并在渲染期收起面板，免得下一个任务出现时面板自己弹开。
  const visible = live.length + settled.length;
  if (visible === 0) {
    if (open) setOpen(false);
    return null;
  }
  // 4. 没有运行中任务时，圆点沿用最近结束任务的状态：失败、超时、停止在收起时也能被看见。
  const latest = live.length ? "running" : settled[0].status;
  const summary = live.length ? `${live.length} 个任务运行中` : `${visible} 个后台任务`;
  const outcome = latest === "running" || latest === "completed" ? "" : `，最近一个${STATUS_LABEL[latest]}`;
  return <div className="job-list" ref={rootRef}>
    <button ref={triggerRef} type="button" className="job-list__trigger" data-live={live.length > 0 || undefined}
      aria-expanded={open} aria-controls={open ? panelId : undefined} aria-label={`后台任务：${summary}${outcome}`}
      onClick={() => { setNow(Date.now()); setOpen((value) => !value); }}>
      <span className="job-list__dot" data-status={latest} aria-hidden="true" />
      <span className="job-list__summary">{summary}</span>
      <span className="job-list__count" aria-hidden="true">{live.length || visible}</span>
    </button>
    {open ? <JobPanel id={panelId} live={live} settled={settled} now={now} onStop={onStop}
      settledOpen={settledOpen ?? live.length === 0}
      onToggleSettled={() => setSettledOpen(!(settledOpen ?? live.length === 0))}
      onClear={() => {
        setCleared((current) => new Set([...current, ...settled.map((job) => job.id)]));
        if (live.length) triggerRef.current?.focus();
      }} /> : null}
  </div>;
}

const NO_JOBS: readonly JobView[] = [];

// 会话任务镜像来自本插件后端的 follow 推送；null（断线、插件换代）即旧镜像失效，按空列表处理。
function SessionJobs({ context }: { context: PluginUiContext }) {
  const [jobs, setJobs] = useState<readonly JobView[]>(NO_JOBS);
  useEffect(() => context.subscribe((value) => {
    const rows = value?.jobs;
    setJobs(Array.isArray(rows) && rows.every(isJob) ? rows : NO_JOBS);
  }), [context]);
  // 停止只回答是否受理；行状态由推送的快照收敛（停止中 → 已停止）。
  const stop = async (jobId: string) => {
    try { return (await context.query("job.stop", { job_id: jobId }, { cache: "none" })).accepted === true; }
    catch { return false; }
  };
  return <JobListAction jobs={jobs} onStop={stop} />;
}

const definition: PluginUiDefinition = {
  slots: { "header.end": { mount: (host, context) => render(host, <SessionJobs context={context} />) } },
};
export default definition;

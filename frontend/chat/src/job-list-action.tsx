import { useEffect, useId, useMemo, useRef, useState } from "react";
import "./job-list-action.css";

export type JobStatus = "running" | "completed" | "failed" | "timeout" | "killed";

/** 后台任务只读镜像的一行；时间为 epoch 毫秒。 */
export interface JobView {
  id: string;
  kind: string;
  label: string;
  status: JobStatus;
  detail: string;
  started_at: number;
  finished_at: number | null;
}

export interface SessionJobs {
  sessionId: string;
  jobs: JobView[];
}

const STATUS_LABEL: Record<JobStatus, string> = {
  running: "运行中",
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
    && (row.finished_at === null || Number.isSafeInteger(row.finished_at));
};

/** 只在 WebSocket 边界检查后台任务快照；其他消息继续交给原协议解析器。 */
export function readJobsFrame(value: unknown): SessionJobs | null {
  if (typeof value !== "object" || value === null || !("type" in value) || value.type !== "session.jobs") return null;
  const row = value as Record<string, unknown>;
  if (typeof row.session_id !== "string" || !Array.isArray(row.jobs) || !row.jobs.every(isJob)) {
    throw new Error("后台任务快照无效");
  }
  return { sessionId: row.session_id, jobs: row.jobs };
}

function formatDuration(elapsedMs: number): string {
  const total = Math.max(0, Math.floor(elapsedMs / 1000));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor(total / 60) % 60;
  if (hours > 0) return `${hours} 小时 ${minutes} 分`;
  if (minutes > 0) return `${minutes} 分 ${total % 60} 秒`;
  return `${total} 秒`;
}

// 运行中按开始时间升序，已结束按结束时间倒序。
function ordered(jobs: readonly JobView[]): JobView[] {
  return [...jobs].sort((left, right) => {
    const live = left.status === "running";
    if (live !== (right.status === "running")) return live ? -1 : 1;
    if (live) return left.started_at - right.started_at;
    return (right.finished_at ?? right.started_at) - (left.finished_at ?? left.started_at);
  });
}

// 标题行右侧的安静标记：有任务运行时显示呼吸点和数量，只剩已结束任务时退为淡色计数。
export function JobListAction({ jobs }: { jobs: readonly JobView[] }) {
  const [open, setOpen] = useState(false);
  const [now, setNow] = useState(() => Date.now());
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const panelId = useId();
  const rows = useMemo(() => ordered(jobs), [jobs]);
  const live = rows.filter((job) => job.status === "running").length;
  // 1. 点击面板外或按 Esc 收起，Esc 把焦点还给触发按钮。
  useEffect(() => {
    if (!open) return;
    const onPointer = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setOpen(false);
      triggerRef.current?.focus();
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
    if (!open || live === 0) return;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [open, live]);
  // 3. 没有任何任务时不占位置。
  if (rows.length === 0) return null;
  // 4. 没有运行中任务时，圆点沿用最近结束任务的状态：失败、超时、停止在收起时也能被看见。
  const latest = live > 0 ? "running" : rows[0].status;
  const summary = live > 0 ? `${live} 个任务运行中` : `${rows.length} 个后台任务`;
  const outcome = latest === "running" || latest === "completed" ? "" : `，最近一个${STATUS_LABEL[latest]}`;
  return <div className="job-list" ref={rootRef}>
    <button ref={triggerRef} type="button" className="job-list__trigger" data-live={live > 0 || undefined}
      aria-expanded={open} aria-controls={open ? panelId : undefined} aria-label={`后台任务：${summary}${outcome}`}
      onClick={() => { setNow(Date.now()); setOpen((value) => !value); }}>
      <span className="job-list__dot" data-status={latest} aria-hidden="true" />
      <span className="job-list__summary">{summary}</span>
      <span className="job-list__count" aria-hidden="true">{live > 0 ? live : rows.length}</span>
    </button>
    {open ? <div id={panelId} className="job-list__panel" role="group" aria-label="后台任务">
      <ul>
        {rows.map((job) => {
          const running = job.status === "running";
          const duration = formatDuration((running ? now : job.finished_at ?? job.started_at) - job.started_at);
          // 正常完成只说状态；失败、超时、停止时附上 detail 便于判断原因。
          const note = job.status === "completed" || !job.detail ? STATUS_LABEL[job.status] : `${STATUS_LABEL[job.status]} · ${job.detail}`;
          return <li key={job.id} className="job-list__row" data-status={job.status}>
            <span className="job-list__dot" data-status={job.status} aria-hidden="true" />
            <span className="job-list__label" title={job.label}>{job.label}</span>
            <span className="job-list__duration">{duration}</span>
            <span className="job-list__meta" title={note}>{job.kind} · {note}</span>
          </li>;
        })}
      </ul>
    </div> : null}
  </div>;
}

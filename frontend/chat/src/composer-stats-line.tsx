import { useEffect, useState } from "react";
import type { ReplyActivity, TimelineMessage } from "./message-timeline";
import { formatModelCallStats, loadWebModelCallStats, selectModelCall, type LoadModelCallStats, type ModelCallStats } from "./model-call-stats";
import type { ChatStatus } from "./web-chat-status";

/** 终态摘要停留时长，到点后淡出收纳。 */
const STATS_SETTLED_MS = 3000;
/** 淡出过渡时长，与 CSS 的 --ak-sys-duration-short 一致；reduced-motion 下 CSS 归零即瞬时隐藏。 */
const STATS_FADE_MS = 150;
/** 等待类阶段持续这么久才换上它的文案：短于约 1 秒的切换不闪。 */
const PHASE_SETTLE_MS = 1000;

type StatsPhase = "active" | "settled" | "fading" | "gone";

/** 回复阶段只读服务端推来的事实：回复活动、草稿与已提交的工具调用。 */
type ReplyPhaseKind = "queued" | "preparing" | "waiting" | "thinking" | "output" | "tool" | "retry";

interface ReplyPhase {
  kind: ReplyPhaseKind;
  /** 阶段身份：换了工具也算新阶段。 */
  key: string;
  tool?: string;
  callId: string | null;
  firstTokenMs: number | null;
}

/** 生成中按阶段显示，结束后按调用 ID 读取一次服务端统计给出终态摘要。 */
export function ComposerStatsLine({ messages, activities, connected, status = "idle", stopping = false, load = loadWebModelCallStats }: {
  messages: readonly TimelineMessage[];
  activities: readonly ReplyActivity[];
  connected: boolean;
  status?: ChatStatus;
  /** 已请求停止、本轮还没停下（当前工具跑完才停）：与停止钮的"正在停止"同一说法。 */
  stopping?: boolean;
  load?: LoadModelCallStats;
}) {
  const live = replyPhase(activities, messages, status);
  const active = live !== null;
  const shown = useSteadyPhase(live);
  // 终态摘要只给看着它跑完的那次调用；打开旧会话不回放上一轮的统计。
  const [watchedCallId, setWatchedCallId] = useState("");
  if (live?.callId && watchedCallId !== live.callId) setWatchedCallId(live.callId);
  const settledCallId = active ? null : selectModelCall(messages, activities).callId;
  const summary = useSettledStats(connected ? settledCallId : null, load);
  const phase = useSettledPhase(active);
  if (!connected || phase === "gone") return null;
  if (active) {
    const label = stopping ? "正在停止…" : shown ? phaseLabel(shown.phase) : "";
    if (!label) return null;
    return <div className="composer-stats-line" data-reply-phase={stopping ? "stopping" : shown?.phase.kind}>
      <span aria-live="polite">{label}</span>
      {!stopping && shown ? <PhaseDetail phase={shown.phase} firstTokenMs={live.firstTokenMs ?? shown.phase.firstTokenMs} since={shown.since} /> : null}
    </div>;
  }
  if (!summary || summary.callId !== settledCallId || watchedCallId !== settledCallId) return null;
  // 统计读取失败只影响遥测，不向用户报错。
  const text = formatModelCallStats(summary.stats);
  return <div className={`composer-stats-line${phase === "fading" ? " is-fading" : ""}`} title={`${text} · ${summary.stats.model}`}>
    <span aria-live="polite">{text}</span>
  </div>;
}

// 从回复活动、草稿和已提交的工具调用推出当前阶段；没有活动也没有待接纳的输入时为空。
function replyPhase(activities: readonly ReplyActivity[], messages: readonly TimelineMessage[], status: ChatStatus): ReplyPhase | null {
  const activity = [...activities].reverse().find((item) => item.active);
  if (!activity) return status === "submitted" ? phaseOf("queued") : null;
  const preview = activity.preview;
  if (preview) {
    const facts = { callId: preview.call_record_id ?? null, firstTokenMs: preview.first_token_ms ?? null };
    // 1. 退避等待由重试卡讲原因和倒计时，这里只报阶段。
    if (preview.retry_status) return phaseOf("retry", facts);
    if (preview.text) return phaseOf("output", facts);
    if (preview.thinking) return phaseOf("thinking", facts);
    return phaseOf("waiting", facts);
  }
  // 2. 两次模型调用之间：有已提交但没有结果的工具调用就是在跑工具，否则是在准备下一次请求。
  const tool = pendingToolName(messages, activity.source);
  return tool === null ? phaseOf("preparing") : { ...phaseOf("tool"), key: `tool:${tool}`, tool };
}

function phaseOf(kind: ReplyPhaseKind, facts: { callId: string | null; firstTokenMs: number | null } = { callId: null, firstTokenMs: null }): ReplyPhase {
  return { kind, key: kind, ...facts };
}

// 从末尾往前找本来源最近一条带工具调用的输出，返回第一个还没有结果的工具名。
function pendingToolName(messages: readonly TimelineMessage[], source: string): string | null {
  const answered = new Set<string>();
  for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index];
    if (message.source !== source) continue;
    const body = message.body;
    if (body.kind === "tool_result") answered.add(`${body.call_ref.message_id}:${body.call_ref.part_index}`);
    if (body.kind === "input" || body.kind === "control") return null;
    if (body.kind !== "output") continue;
    const pending = body.parts.findIndex((part, part_index) => part.kind === "tool_call" && "name" in part
      && !answered.has(`${message.id}:${part_index}`));
    if (pending < 0) return null;
    const part = body.parts[pending];
    return "name" in part ? part.name : null;
  }
  return null;
}

function phaseLabel(phase: ReplyPhase): string {
  switch (phase.kind) {
    case "queued": return "排队中";
    case "preparing": return "准备请求";
    case "waiting": return "等待首 token";
    case "thinking": return "思考中";
    case "output": return "输出中";
    case "tool": return phase.tool ? `调用工具 ${phase.tool}` : "调用工具";
    case "retry": return "重试中";
  }
}

/** 出字阶段给首字耗时（服务端事实）；等待类阶段给已等多久，帮助判断是不是卡住了。 */
function PhaseDetail({ phase, firstTokenMs, since }: { phase: ReplyPhase; firstTokenMs: number | null; since: number }) {
  if (phase.kind === "thinking" || phase.kind === "output") {
    return firstTokenMs === null ? null : <span aria-hidden="true">· 首 token {(firstTokenMs / 1000).toFixed(1)}s</span>;
  }
  if (phase.kind === "queued" || phase.kind === "retry") return null;
  return <span aria-hidden="true">· <ElapsedText since={since} /></span>;
}

/** 已持续秒数；每秒只重渲染这一小段文字。 */
function ElapsedText({ since }: { since: number }) {
  const [now, setNow] = useState(() => Date.now());
  // 必要 effect：逐秒刷新已等待时长。
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  return <>{Math.max(1, Math.round((now - since) / 1000))}s</>;
}

// 出字阶段立即显示；等待类阶段持续满 PHASE_SETTLE_MS 才替换文案，期间保留上一个稳定阶段。
function useSteadyPhase(live: ReplyPhase | null): { phase: ReplyPhase; since: number } | null {
  const [steady, setSteady] = useState<{ phase: ReplyPhase; since: number } | null>(null);
  const key = live?.key ?? "";
  const immediate = live?.kind === "thinking" || live?.kind === "output";
  // 必要 effect：阶段切换后的定时确认只能在提交后挂计时器。
  useEffect(() => {
    if (!live) {
      const timer = window.setTimeout(() => setSteady(null), 0);
      return () => window.clearTimeout(timer);
    }
    const entered = { phase: live, since: Date.now() };
    const timer = window.setTimeout(() => setSteady(entered), immediate ? 0 : PHASE_SETTLE_MS);
    return () => window.clearTimeout(timer);
    // 只在阶段身份变化时重新计时；同一阶段内的草稿增长不重置。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  // 出字阶段不显示时长，since 用不到。
  if (live && immediate && steady?.phase.key !== key) return { phase: live, since: 0 };
  return steady;
}

// 结束后按调用 ID 读取一次统计；切页立即撤销旧读取。
function useSettledStats(callId: string | null, load: LoadModelCallStats): { callId: string; stats: ModelCallStats } | null {
  const [result, setResult] = useState<{ callId: string; stats: ModelCallStats } | null>(null);
  useEffect(() => {
    if (!callId) return;
    const controller = new AbortController();
    load(callId, controller.signal).then((stats) => {
      if (!controller.signal.aborted) setResult({ callId, stats });
    }, () => undefined);
    return () => controller.abort();
  }, [callId, load]);
  return result;
}

// 活动状态切换驱动收纳节奏：终态摘要停留 3 秒，淡出过渡后移出渲染。
function useSettledPhase(active: boolean): StatsPhase {
  const [phase, setPhase] = useState<StatsPhase>("active");
  useEffect(() => {
    if (active) {
      const timer = window.setTimeout(() => setPhase("active"), 0);
      return () => window.clearTimeout(timer);
    }
    const settleTimer = window.setTimeout(() => setPhase("settled"), 0);
    const fadeTimer = window.setTimeout(() => setPhase("fading"), STATS_SETTLED_MS);
    const goneTimer = window.setTimeout(() => setPhase("gone"), STATS_SETTLED_MS + STATS_FADE_MS);
    return () => { clearTimeout(settleTimer); clearTimeout(fadeTimer); clearTimeout(goneTimer); };
  }, [active]);
  return phase;
}

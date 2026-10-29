import { useEffect, useState } from "react";
import { Bot, X } from "lucide-react";
import { render, type PluginUiContext, type PluginUiDefinition } from "@akashic/chat-ui-v1";
import "./runner-ui.css";

type Run = { id: string; label: string; runner: string; status: string; result: string;
  progress: boolean; truncated?: boolean; activities: { kind: string; text: string }[] };
const labels: Record<string, string> = { accepted: "已接纳", running: "运行中", completed: "已完成",
  failed: "失败", interrupted: "已中断" };
function valid(value: unknown): value is Run {
  if (!value || typeof value !== "object") return false;
  const r = value as Run;
  return [r.id, r.label, r.runner, r.status, r.result].every(v => typeof v === "string")
    && typeof r.progress === "boolean" && Array.isArray(r.activities)
    && r.activities.every(e => typeof e?.text === "string" && typeof e?.kind === "string");
}
function Panel({ context }: { context: PluginUiContext }) {
  const [runs, setRuns] = useState<Run[]>([]);
  const [open, setOpen] = useState(false);
  const [error, setError] = useState("");
  const [pending, setPending] = useState<string | null>(null);
  useEffect(() => context.subscribe(value => {
    setRuns(Array.isArray(value?.runs) ? value.runs.filter(valid) : []);
  }), [context]);
  async function stop(id: string) {
    setPending(id); setError("");
    try {
      const response = await context.query("runners.stop", { id }, { cache: "none" });
      if (!response.accepted) setError("任务已结束，或当前不能停止。");
    } catch { setError("停止失败，请检查任务状态。"); }
    finally { setPending(null); }
  }
  if (!runs.length) return null;
  return <div className="agent-runs">
    <button type="button" className="agent-runs__toggle" aria-expanded={open}
      onClick={() => setOpen(!open)}><Bot size={16} aria-hidden="true" />代理任务 {runs.length}</button>
    {open && <section className="agent-runs__panel" aria-label="代理执行过程" onKeyDown={e => {
      if (e.key === "Escape") setOpen(false);
    }}>
      <header><strong>代理执行过程</strong><button type="button" aria-label="关闭代理任务" onClick={() => setOpen(false)}><X size={16} /></button></header>
      {error && <p role="alert">{error}</p>}
      {runs.map(run => <details key={run.id} data-run-id={run.id}>
        <summary><span>{run.label}</span><small>{run.runner} · {labels[run.status] ?? run.status}</small></summary>
        <div className="agent-runs__body">
          <p className="agent-runs__notice">独立执行；过程不自动加入主对话上下文。取消不会撤销已有外部效果。</p>
          {run.status === "running" && <button type="button" disabled={pending === run.id} onClick={() => void stop(run.id)}>
            {pending === run.id ? "正在停止…" : "停止任务"}</button>}
          {!run.progress && <p>此执行器仅提供最终结果。</p>}
          {run.truncated && <p>仅展示最近 200 条活动，长活动已截短。</p>}
          <ol>{run.activities.map((event, i) => <li key={i} data-kind={event.kind}>{event.text}</li>)}</ol>
          {run.result && <pre>{run.result}</pre>}
        </div>
      </details>)}
    </section>}
  </div>;
}
const definition: PluginUiDefinition = {
  slots: { "header.end": { mount: (host, context) => render(host, <Panel context={context} />) } },
};
export default definition;

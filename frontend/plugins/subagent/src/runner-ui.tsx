import { useEffect, useMemo, useState } from "react";
import { render, ToolText, type HeaderMetaItem, type PluginUiContext, type PluginUiDefinition } from "@akashic/chat-ui-v1";
import { ICONS, useValue, valueStore, type ValueStore } from "../../../../packages/akashic-plugin-kit/src/meta-store";
import "./runner-ui.css";

type Run = { id: string; label: string; runner: string; status: string; result: string;
  progress: boolean; truncated?: boolean; activities: { kind: string; text: string }[] };
const labels: Record<string, string> = { accepted: "已接纳", running: "运行中", completed: "已完成",
  failed: "失败", interrupted: "已中断" };
const busy = (run: Run) => run.status === "running" || run.status === "accepted";
function valid(value: unknown): value is Run {
  if (!value || typeof value !== "object") return false;
  const r = value as Run;
  return [r.id, r.label, r.runner, r.status, r.result].every(v => typeof v === "string")
    && typeof r.progress === "boolean" && Array.isArray(r.activities)
    && r.activities.every(e => typeof e?.text === "string" && typeof e?.kind === "string");
}

// 说明项：代理任务一项挂在本会话下，每次执行是它的子项；在跑时是进行中，全部结束后只留在面板里，失败的子项冒泡成需要注意。
function runItems(runs: Run[], store: ValueStore<Run[]>, stop: (id: string) => Promise<string>): HeaderMetaItem[] {
  if (!runs.length) return [];
  const running = runs.filter(busy).length;
  return [{
    id: "runs", scope: "session", label: "代理任务", iconSvg: ICONS.bot, tone: running ? "live" : "info",
    text: running ? `代理 ${running} 运行中` : undefined,
    value: running ? `${running} 个运行中 · ${runs.length - running} 个已结束` : `${runs.length} 个已结束`,
    detail: (host) => render(host, <p className="agent-runs__notice">独立执行；过程不自动加入主对话上下文。取消不会撤销已有外部效果。</p>),
  }, ...runs.map((run): HeaderMetaItem => ({
    id: `run:${run.id}`, parent: "runs", scope: "session", label: run.label, iconSvg: ICONS.bot,
    tone: busy(run) ? "live" : run.status === "failed" ? "alert" : "info",
    value: `${run.runner} · ${labels[run.status] ?? run.status}`,
    detail: (host) => render(host, <RunDetail store={store} id={run.id} stop={stop} />),
  }))];
}

function RunDetail({ store, id, stop }: { store: ValueStore<Run[]>; id: string; stop: (id: string) => Promise<string> }) {
  const run = useValue(store).find((item) => item.id === id);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  if (!run) return <p className="agent-runs__notice">这次执行已不在列表里。</p>;
  return <div className="agent-runs__body">
    {error && <p role="alert">{error}</p>}
    {run.status === "running" && <button type="button" disabled={pending} onClick={() => {
      setPending(true); setError("");
      void stop(run.id).then((message) => { setError(message); setPending(false); });
    }}>{pending ? "正在停止…" : "停止任务"}</button>}
    {!run.progress && <p>此执行器仅提供最终结果。</p>}
    {run.truncated && <p>仅展示最近 200 条活动，长活动已截短。</p>}
    <ol>{run.activities.map((event, i) => <li key={i} data-kind={event.kind}>{event.text}</li>)}</ol>
    {run.result && <pre>{run.result}</pre>}
  </div>;
}

interface SubagentResultViewModel {
  title: string;
  status: string;
  tone: "done" | "live" | "error";
  jobId?: string;
  body: string;
}

function parseSubagentResult(name: string, value: unknown, error: boolean): SubagentResultViewModel | null {
  if (typeof value !== "string") return null;
  if (name !== "spawn" && !name.startsWith("delegate_")) return null;
  const trimmed = value.trim();
  const bgMatch = /^已创建后台任务「(.+)」[（(]job_id=([a-f0-9]+)[)）]/u.exec(trimmed);
  if (bgMatch) {
    return { title: bgMatch[1], status: "后台已接纳", tone: "live", jobId: bgMatch[2], body: trimmed };
  }
  const syncMatch = /^\[子任务「([^」]+)」结果\]\n退出原因:\s*([^\n]+)\n*([\s\S]*)$/u.exec(trimmed);
  if (syncMatch) {
    const reason = syncMatch[2].trim();
    const tone = error || reason === "failed" ? "error" : reason === "cancelled" ? "error" : "done";
    const statusLabel = reason === "completed" ? "已完成" : reason === "cancelled" ? "已取消" : reason === "failed" ? "失败" : reason;
    return { title: syncMatch[1], status: statusLabel, tone, body: syncMatch[3].trim() || trimmed };
  }
  const extMatch = /^外部任务\s+([a-zA-Z0-9_-]+)：([^\n。]+)/u.exec(trimmed);
  if (extMatch) {
    const st = extMatch[2].trim();
    return {
      title: `外部任务 ${extMatch[1]}`,
      status: labels[st] ?? st,
      tone: st === "failed" || st === "interrupted" ? "error" : st === "completed" ? "done" : "live",
      jobId: extMatch[1],
      body: trimmed,
    };
  }
  return null;
}

function SubagentResultCard({ result }: { result: SubagentResultViewModel }) {
  return (
    <div className={`subagent-tool-result is-${result.tone}`}>
      <div className="subagent-tool-result__head">
        <span className="subagent-tool-result__title">{result.title}</span>
        <span className={`subagent-tool-result__badge is-${result.tone}`}>{result.status}</span>
        {result.jobId ? <code className="subagent-tool-result__id">{result.jobId.slice(0, 8)}</code> : null}
      </div>
      <ToolText text={result.body} tail={false} className="tool-result" />
    </div>
  );
}

// 不渲染界面：执行状态来自本插件后端的推送，登记到标题下的说明行，过程在宿主的会话信息面板里展开。
function Runs({ context }: { context: PluginUiContext<"header.meta"> }) {
  const store = useMemo(() => valueStore<Run[]>([]), []);
  useEffect(() => {
    const stop = async (id: string) => {
      try {
        const response = await context.query("runners.stop", { id }, { cache: "none" });
        return response.accepted ? "" : "任务已结束，或当前不能停止。";
      } catch { return "停止失败，请检查任务状态。"; }
    };
    return context.subscribe(value => {
      const runs = Array.isArray(value?.runs) ? value.runs.filter(valid) : [];
      store.set(runs);
      context.meta.publish(runItems(runs, store, stop));
    });
  }, [context, store]);
  return null;
}
const definition: PluginUiDefinition = {
  slots: {
    "header.meta": { mount: (host, context) => render(host, <Runs context={context} />) },
    "tool.result": {
      accepts: (block) => parseSubagentResult(block.name, block.value, block.error) !== null,
      mount(host, context) {
        const parsed = parseSubagentResult(context.block.name, context.block.value, context.block.error);
        return parsed ? render(host, <SubagentResultCard result={parsed} />) : undefined;
      },
    },
  },
};
export default definition;

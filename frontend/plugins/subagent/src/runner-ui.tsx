import { useEffect, useMemo, useState } from "react";
import { render, ToolText, type HeaderMetaItem, type PluginUiContext, type PluginUiDefinition, type ToolSummaryFacts, type ToolTaskSummary } from "@akashic/chat-ui-v1";
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

// spawn 回执的写法与 plugins/subagent 的 tools.query / runtime.completion 一致。
const GUIDANCE = ["最多重试一次；已有完整结果则直接向用户汇报。", "已经重试一次，请直接汇报已有结果。"];
const EXIT_STATES: Record<string, ToolTaskSummary["state"]> = {
  cancelled: { label: "已取消", tone: "error" },
};

interface SpawnResult {
  jobId?: string;
  reason?: string;
  body: string;
}

// 后台接纳只回 job_id；同步结算带退出原因，末尾给模型的重试提示不展示给读者；其他文字（例如并发已满）原样。
function parseSpawnResult(value: unknown): SpawnResult | null {
  if (typeof value !== "string") return null;
  const text = value.trim();
  const background = /^已创建后台任务「[\s\S]*」[（(]job_id=([a-f0-9]+)[)）]/u.exec(text);
  if (background) return { jobId: background[1], body: text };
  const settled = /^\[子任务「[\s\S]*?」结果\]\n退出原因:\s*([^\n]+)\n*([\s\S]*)$/u.exec(text);
  if (!settled) return { body: text };
  let body = settled[2].trimEnd();
  for (const line of GUIDANCE) if (body.endsWith(line)) body = body.slice(0, -line.length).trimEnd();
  return { reason: settled[1].trim(), body };
}

function firstLine(text: string): string {
  return text.split("\n", 64).map((line) => line.trim()).find(Boolean) ?? "";
}

// 时间线任务卡的文字：标题、标签、模型与工作目录来自调用参数，状态与预览来自回执。
function describeSpawn(block: ToolSummaryFacts): ToolTaskSummary | null {
  const args = block.arguments;
  if (block.name !== "spawn" || typeof args !== "object" || args === null || Array.isArray(args)) return null;
  const field = (name: string) => {
    const value = (args as Record<string, unknown>)[name];
    return typeof value === "string" ? value.trim() : "";
  };
  const title = field("label") || firstLine(field("task"));
  if (!title) return null;
  const background = (args as Record<string, unknown>).run_in_background === true;
  const tags = [...(background ? ["后台"] : []), ...(field("profile") === "research" ? ["只读"] : [])];
  const result = block.status === "running" ? null : parseSpawnResult(block.value);
  const summary: ToolTaskSummary = {
    kind: "task", title, tags,
    detail: [field("connection"), field("model"), field("reasoning_effort")].filter(Boolean).join(" · "),
    cwd: field("cwd") || undefined,
  };
  if (result?.jobId) return { ...summary, preview: "完成后会回传当前会话", state: { label: "已转后台", tone: "idle" } };
  if (!result) return summary;
  return { ...summary, preview: firstLine(result.body) || undefined, state: result.reason ? EXIT_STATES[result.reason] : undefined };
}

// 展开后的结果：任务名与状态已经在任务卡上，这里只放正文；后台任务给出 job_id 便于查询或取消。
function SpawnResultView({ result }: { result: SpawnResult }) {
  return result.jobId ? (
    <p className="subagent-tool-result">后台任务 <code>{result.jobId}</code>，完成后会回传当前会话。</p>
  ) : <ToolText text={result.body || "子任务没有输出文字。"} tail={false} className="tool-result" />;
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
      accepts: (block) => block.name === "spawn" && parseSpawnResult(block.value) !== null,
      mount(host, context) {
        const parsed = parseSpawnResult(context.block.value);
        return parsed ? render(host, <SpawnResultView result={parsed} />) : undefined;
      },
    },
    "tool.summary": { describe: describeSpawn },
  },
};
export default definition;

import { useEffect, useMemo, useState } from "react";
import { render, ToolText, type HeaderMetaItem, type PluginUiContext, type PluginUiDefinition, type ToolSummaryFacts, type ToolTaskSummary } from "@akashic/chat-ui-v1";
import { ICONS, useValue, valueStore, type ValueStore } from "../../../../packages/akashic-plugin-kit/src/meta-store";
import "./runner-ui.css";

type Run = {
  id: string;
  label: string;
  runner: string;
  status: string;
  result: string;
  progress: boolean;
  truncated?: boolean;
  activities: { kind: string; text: string }[];
};

type SubagentNode = {
  job_id: string;
  label: string;
  session_id: string;
  parent_session_id: string;
  depth: number;
  read_only: boolean;
  fork?: boolean;
  isolation?: string;
  model_id?: string | null;
  status: string;
  active: boolean;
  error?: string;
  fail_seq?: number;
};

type SubagentPayload = {
  runs: Run[];
  subagents: SubagentNode[];
};

const labels: Record<string, string> = {
  accepted: "已接纳",
  running: "运行中",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
  interrupted: "已中断",
};

const busyRun = (run: Run) => run.status === "running" || run.status === "accepted";

function validRun(value: unknown): value is Run {
  if (!value || typeof value !== "object") return false;
  const r = value as Run;
  return [r.id, r.label, r.runner, r.status, r.result].every((v) => typeof v === "string")
    && typeof r.progress === "boolean" && Array.isArray(r.activities)
    && r.activities.every((e) => typeof e?.text === "string" && typeof e?.kind === "string");
}

function validSubagent(value: unknown): value is SubagentNode {
  if (!value || typeof value !== "object") return false;
  const s = value as SubagentNode;
  return typeof s.job_id === "string" && typeof s.label === "string"
    && typeof s.session_id === "string" && typeof s.status === "string"
    && typeof s.active === "boolean" && typeof s.depth === "number";
}

function parsePayload(value: Record<string, unknown> | null): SubagentPayload {
  const runs = Array.isArray(value?.runs) ? value.runs.filter(validRun) : [];
  const subagents = Array.isArray(value?.subagents) ? value.subagents.filter(validSubagent) : [];
  return { runs, subagents };
}

function buildLineageItems(
  payload: SubagentPayload,
  store: ValueStore<SubagentPayload>,
  actions: {
    stopRunner: (id: string) => Promise<string>;
    cancelSubagent: (jobId: string) => Promise<string>;
    openWorkbench: (kind: "subagent" | "external", jobId: string, label: string) => void;
    canOpenWorkbench: boolean;
  },
): HeaderMetaItem[] {
  const { runs, subagents } = payload;
  const total = runs.length + subagents.length;
  if (!total) return [];
  const runningCount = runs.filter(busyRun).length + subagents.filter((s) => s.active).length;
  const failedCount = runs.filter((r) => r.status === "failed").length
    + subagents.filter((s) => s.status === "failed").length;
  const rootTone: HeaderMetaItem["tone"] = failedCount > 0 ? "alert" : runningCount > 0 ? "live" : "info";
  const bySession = new Map<string, SubagentNode>();
  for (const s of subagents) bySession.set(s.session_id, s);

  const rootItem: HeaderMetaItem = {
    id: "subagents-tree",
    scope: "session",
    label: "子任务血缘",
    iconSvg: ICONS.bot,
    tone: rootTone,
    text: runningCount > 0 ? `子任务 ${runningCount} 运行中` : failedCount > 0 ? `${failedCount} 个子任务失败` : undefined,
    value: [
      runningCount ? `${runningCount} 运行中` : "",
      failedCount ? `${failedCount} 失败` : "",
      `${total} 项`,
    ].filter(Boolean).join(" · "),
    detail: (host) => render(
      host,
      <p className="agent-runs__notice">展示当前会话树下的全部内部子任务与外部只读代理；点击条目可在右侧工作台查看完整实时流。</p>,
    ),
  };

  const subItems: HeaderMetaItem[] = subagents.map((node) => {
    const parentNode = bySession.get(node.parent_session_id);
    const parentId = parentNode ? `sub:${parentNode.job_id}` : "subagents-tree";
    const tags = [
      node.depth > 1 ? `L${node.depth}` : "",
      node.fork ? "fork" : "",
      node.isolation === "worktree" ? "worktree" : "",
      node.read_only ? "只读" : "",
    ].filter(Boolean).join(" · ");
    const tone: HeaderMetaItem["tone"] = node.active ? "live" : node.status === "failed" ? "alert" : "info";
    return {
      id: `sub:${node.job_id}`,
      parent: parentId,
      scope: "session",
      label: node.label,
      iconSvg: ICONS.bot,
      tone,
      value: [tags, labels[node.status] ?? node.status].filter(Boolean).join(" · "),
      detail: (host) => render(host, <SubagentNodeDetail store={store} jobId={node.job_id} actions={actions} />),
    };
  });

  const runItems: HeaderMetaItem[] = runs.map((run) => ({
    id: `run:${run.id}`,
    parent: "subagents-tree",
    scope: "session",
    label: run.label,
    iconSvg: ICONS.bot,
    tone: busyRun(run) ? "live" : run.status === "failed" ? "alert" : "info",
    value: `${run.runner} · 外部只读 · ${labels[run.status] ?? run.status}`,
    detail: (host) => render(host, <RunDetail store={store} id={run.id} actions={actions} />),
  }));

  return [rootItem, ...subItems, ...runItems];
}

function SubagentNodeDetail({
  store,
  jobId,
  actions,
}: {
  store: ValueStore<SubagentPayload>;
  jobId: string;
  actions: {
    cancelSubagent: (jobId: string) => Promise<string>;
    openWorkbench: (kind: "subagent" | "external", jobId: string, label: string) => void;
    canOpenWorkbench: boolean;
  };
}) {
  const node = useValue(store).subagents.find((item) => item.job_id === jobId);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  if (!node) return <p className="agent-runs__notice">该子任务已不在当前血缘列表。</p>;
  return (
    <div className="agent-runs__body">
      {error && <p role="alert">{error}</p>}
      <p className="agent-runs__notice">
        深度 L{node.depth}{node.read_only ? " · 只读" : ""}{node.fork ? " · fork" : ""}{node.isolation === "worktree" ? " · worktree" : ""}
      </p>
      {node.error ? <p role="alert">{node.error}</p> : null}
      <div className="agent-runs__actions">
        {actions.canOpenWorkbench ? (
          <button type="button" onClick={() => actions.openWorkbench("subagent", node.job_id, node.label)}>
            在工作台查看
          </button>
        ) : null}
        {node.active ? (
          <button
            type="button"
            disabled={pending}
            onClick={() => {
              setPending(true);
              setError("");
              void actions.cancelSubagent(node.job_id).then((msg) => {
                setError(msg);
                setPending(false);
              });
            }}
          >
            {pending ? "正在停止…" : "停止任务"}
          </button>
        ) : null}
      </div>
    </div>
  );
}

function RunDetail({
  store,
  id,
  actions,
}: {
  store: ValueStore<SubagentPayload>;
  id: string;
  actions: {
    stopRunner: (id: string) => Promise<string>;
    openWorkbench: (kind: "subagent" | "external", jobId: string, label: string) => void;
    canOpenWorkbench: boolean;
  };
}) {
  const run = useValue(store).runs.find((item) => item.id === id);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  if (!run) return <p className="agent-runs__notice">这次执行已不在列表里。</p>;
  return (
    <div className="agent-runs__body">
      {error && <p role="alert">{error}</p>}
      <div className="agent-runs__actions">
        {actions.canOpenWorkbench ? (
          <button type="button" onClick={() => actions.openWorkbench("external", run.id, run.label)}>
            在工作台查看
          </button>
        ) : null}
        {run.status === "running" && (
          <button
            type="button"
            disabled={pending}
            onClick={() => {
              setPending(true);
              setError("");
              void actions.stopRunner(run.id).then((message) => {
                setError(message);
                setPending(false);
              });
            }}
          >
            {pending ? "正在停止…" : "停止任务"}
          </button>
        )}
      </div>
      {!run.progress && <p>此执行器仅提供最终结果。</p>}
      {run.truncated && <p>仅展示最近 200 条活动，长活动已截短。</p>}
      <ol>{run.activities.map((event, i) => <li key={i} data-kind={event.kind}>{event.text}</li>)}</ol>
      {run.result && <pre>{run.result}</pre>}
    </div>
  );
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
  const flag = (name: string) => (args as Record<string, unknown>)[name] === true;
  const tags = [
    ...(background ? ["后台"] : []),
    ...(flag("read_only") || field("profile") === "research" ? ["只读"] : []),
    ...(flag("fork") ? ["fork"] : []),
    ...(field("isolation") === "worktree" ? ["worktree"] : []),
  ];
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

function LineageHeaderMeta({ context }: { context: PluginUiContext<"header.meta"> }) {
  const store = useMemo(() => valueStore<SubagentPayload>({ runs: [], subagents: [] }), []);
  useEffect(() => {
    const actions = {
      stopRunner: async (id: string) => {
        try {
          const response = await context.query("runners.stop", { id }, { cache: "none" });
          return response.accepted ? "" : "任务已结束，或当前不能停止。";
        } catch {
          return "停止失败，请检查任务状态。";
        }
      },
      cancelSubagent: async (jobId: string) => {
        try {
          const response = await context.query("subagent.cancel", { job_id: jobId }, { cache: "none" });
          return response.accepted ? "" : "任务已结束，或当前不能停止。";
        } catch {
          return "取消失败，请检查任务状态。";
        }
      },
      openWorkbench: (kind: "subagent" | "external", jobId: string, label: string) => {
        context.meta.closePanel();
        void context.workbench.open("subagent.open.v1", { kind, job_id: jobId, label });
      },
      canOpenWorkbench: context.workbench.accepts("subagent.open.v1"),
    };
    const publish = (value: SubagentPayload) => {
      store.set(value);
      context.meta.publish(buildLineageItems(value, store, {
        ...actions,
        canOpenWorkbench: context.workbench.accepts("subagent.open.v1"),
      }));
    };
    const offSub = context.subscribe((raw) => publish(parsePayload(raw)));
    const offInv = context.onInvalidate(() => publish(store.get()));
    return () => {
      offSub();
      offInv();
    };
  }, [context, store]);
  return null;
}

function readDismissed(sessionId: string | undefined): string {
  if (!sessionId) return "";
  try {
    return window.sessionStorage.getItem(`akashic:subagent-dismissed:${sessionId}`) ?? "";
  } catch {
    return "";
  }
}

function saveDismissed(sessionId: string | undefined, signature: string): void {
  if (!sessionId) return;
  try {
    window.sessionStorage.setItem(`akashic:subagent-dismissed:${sessionId}`, signature);
  } catch {
    // 忽略受限环境下的 sessionStorage 异常
  }
}

function ChatAttentionBanner({ context }: { context: PluginUiContext<"chat.notice"> }) {
  const [payload, setPayload] = useState<SubagentPayload>({ runs: [], subagents: [] });
  const [dismissedKey, setDismissedKey] = useState<string>(() => readDismissed(context.sessionId));
  const [canOpen, setCanOpen] = useState(() => context.workbench.accepts("subagent.open.v1"));

  useEffect(() => {
    setPayload({ runs: [], subagents: [] });
    setDismissedKey(readDismissed(context.sessionId));
    const offSub = context.subscribe((raw) => setPayload(parsePayload(raw)));
    const offInv = context.onInvalidate(() => setCanOpen(context.workbench.accepts("subagent.open.v1")));
    return () => {
      offSub();
      offInv();
    };
  }, [context]);

  const failures = [
    ...payload.subagents
      .filter((s) => s.status === "failed")
      .map((s) => ({
        kind: "subagent" as const,
        id: s.job_id,
        label: s.label,
        detail: s.error || "执行失败",
        failSeq: typeof s.fail_seq === "number" ? s.fail_seq : 0,
      })),
    ...payload.runs
      .filter((r) => r.status === "failed")
      .map((r) => ({
        kind: "external" as const,
        id: r.id,
        label: r.label,
        detail: r.result || "外部任务失败",
        failSeq: r.activities.length,
      })),
  ];

  const signature = failures.map((f) => `${f.kind}:${f.id}:${f.failSeq}:${f.detail}`).join("|");
  if (!failures.length || dismissedKey === signature) return null;
  const first = failures[0];

  return (
    <div className="subagent-composer-alert" role="status">
      <span className="subagent-composer-alert__dot" aria-hidden="true" />
      <span className="subagent-composer-alert__text">
        <strong>{failures.length === 1 ? `子任务「${first.label}」失败` : `${failures.length} 个子任务执行失败`}</strong>
        <span>{first.detail}</span>
      </span>
      <span className="subagent-composer-alert__actions">
        {canOpen ? (
          <button
            type="button"
            className="subagent-composer-alert__open"
            onClick={() => void context.workbench.open("subagent.open.v1", { kind: first.kind, job_id: first.id, label: first.label })}
          >
            在工作台查看
          </button>
        ) : null}
        <button
          type="button"
          className="subagent-composer-alert__dismiss"
          onClick={() => {
            saveDismissed(context.sessionId, signature);
            setDismissedKey(signature);
          }}
        >
          忽略
        </button>
      </span>
    </div>
  );
}

const definition: PluginUiDefinition = {
  slots: {
    "header.meta": { mount: (host, context) => render(host, <LineageHeaderMeta context={context} />) },
    "chat.notice": { mount: (host, context) => render(host, <ChatAttentionBanner context={context} />) },
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

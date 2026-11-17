import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import type { ConversationTabView } from "@akashic/conversation-tools-v2";
import type { WebHostContextV1 } from "@akashic/web-ui-v1";
import { renderMarkdownProse } from "../../../../packages/akashic-plugin-kit/src/markdown-prose";
import "./workbench.css";

export interface TreeSubagent {
  job_id: string;
  label: string;
  session_id: string;
  parent_session_id: string;
  depth: number;
  read_only: boolean;
  fork?: boolean;
  isolation?: string;
  model_id?: string | null;
  cwd?: string | null;
  status: string;
  active: boolean;
  error?: string;
}

export interface TreeExternalRun {
  id: string;
  label: string;
  runner: string;
  status: string;
  result: string;
  progress: boolean;
  truncated?: boolean;
  activities: { kind: string; text: string }[];
}

interface ChildMessagePart {
  kind: string;
  value?: string;
  binding_id?: string;
  tool?: string;
  arguments?: Record<string, unknown>;
  action?: string;
  reason?: string;
}

interface ChildMessage {
  seq: number;
  message_id: string;
  author: string;
  kind: string;
  timestamp: string;
  parts: ChildMessagePart[];
  call_ref?: { message_id: string; part_index: number };
  outcome?: "success" | "denied" | "error" | "interrupted";
}

interface ToolResultView {
  outcome: string;
  parts: string[];
}

interface ToolEntry {
  key: string;
  part: ChildMessagePart;
  result?: ToolResultView;
}

// 消息流按阅读块组织：连续工具调用并成一条轨道，结果挂回对应调用，正文按 Markdown 排版。
type StreamBlock =
  | { kind: "text"; key: string; text: string; steer: boolean; initial: boolean }
  | { kind: "tools"; key: string; entries: ToolEntry[] }
  | { kind: "control"; key: string; action?: string; reason?: string };

type Selection = { kind: "subagent" | "external"; id: string };

const STATUS_LABELS: Record<string, string> = {
  accepted: "已接纳",
  running: "运行中",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
  interrupted: "已中断",
};

const CONTROL_LABELS: Record<string, string> = {
  cancel: "任务已取消",
  pause: "已停止",
  resume: "已恢复",
};

const API_BASE = "/api/dashboard/subagent";
const VISIBLE_TOOL_TAIL = 3;

function statusTone(status: string, active?: boolean): "live" | "completed" | "cancelled" | "error" {
  if (active || status === "running" || status === "accepted") return "live";
  if (status === "failed") return "error";
  if (status === "cancelled" || status === "interrupted") return "cancelled";
  return "completed";
}

function summarizeToolCall(part: ChildMessagePart): { tool: string; summary: string; rawJson: string } {
  const args = part.arguments && typeof part.arguments === "object" ? part.arguments : {};
  const rawJson = JSON.stringify(args, null, 2);
  const str = (key: string) => (typeof args[key] === "string" ? (args[key] as string).trim() : "");

  const desc = str("description");
  const cmd = str("command");
  const path = str("path") || str("file_path") || str("file");
  const query = str("query") || str("pattern");
  const url = str("url");
  const skill = str("skill");

  let tool = "工具";
  if (part.tool) tool = part.tool;
  else if (cmd) tool = "Shell";
  else if (typeof args.new_text === "string" || typeof args.old_text === "string") tool = "Edit";
  else if (typeof args.content === "string" && path) tool = "Write";
  else if (path) tool = "Read";
  else if (query) tool = "Search";
  else if (url) tool = "Fetch";
  else if (skill) tool = "Skill";

  const summary = desc || cmd || path || query || url || skill || JSON.stringify(args);
  return { tool, summary, rawJson };
}

const RESULT_TEXT_KEYS = ["output", "stdout", "content", "text", "result", "message", "error", "stderr"];
const RESULT_HIDDEN_KEYS = new Set(["command", "original_token_count", "chunk_id", "wall_time_ms"]);
// 命令单独成块；description 已在摘要里，shell/login 是执行器默认项。
const ARG_HIDDEN_KEYS = new Set(["command", "description", "shell", "login"]);
const RESULT_VIEW_CHARS = 20000;

// 工具结果多为 JSON 信封：取出正文字段还原换行，其余标量字段压成一行元信息；非 JSON 的段落（如状态提示）原样保留。
function formatToolResult(result: ToolResultView): { body: string; meta: string[]; failed: boolean } {
  let failed = result.outcome !== "success";
  let record: Record<string, unknown> | null = null;
  const notes: string[] = [];
  for (const part of result.parts) {
    let parsed: unknown = null;
    try {
      parsed = JSON.parse(part);
    } catch {
      parsed = null;
    }
    if (!record && parsed && typeof parsed === "object" && !Array.isArray(parsed)) record = parsed as Record<string, unknown>;
    else notes.push(part);
  }
  if (!record) return { body: notes.join("\n\n"), meta: [], failed };
  const bodyKey = RESULT_TEXT_KEYS.find((key) => typeof record[key] === "string" && (record[key] as string).length > 0);
  const meta: string[] = [];
  for (const [key, value] of Object.entries(record)) {
    if (key === bodyKey || RESULT_HIDDEN_KEYS.has(key)) continue;
    if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") meta.push(`${key}: ${value}`);
  }
  if (typeof record.exit_code === "number" && record.exit_code !== 0) failed = true;
  if (typeof record.process_status === "string" && /fail|error|timeout|killed/i.test(record.process_status)) failed = true;
  const body = [...notes, bodyKey ? (record[bodyKey] as string) : JSON.stringify(record, null, 2)].join("\n\n");
  return { body, meta, failed };
}

const LEGACY_ACTIVITY_NAMES: Record<string, string> = {
  commandExecution: "Shell",
  fileChange: "Edit",
  mcpToolCall: "MCP",
  webSearch: "Search",
};

// 外部执行器活动行「名称 · 对象 · 状态」拆成工具行；旧版只有事件名的「开始」行直接略去。
function activityView(act: { kind: string; text: string }): { name: string; detail: string; failed: boolean } | null {
  const segments = act.text.split(" · ");
  const legacy = LEGACY_ACTIVITY_NAMES[segments[0]];
  if (legacy && segments.length === 2) return segments[1] === "开始" ? null : { name: legacy, detail: "", failed: false };
  if (act.kind !== "tool" || segments.length < 2) return { name: "", detail: act.text, failed: false };
  const last = segments[segments.length - 1];
  const failed = /^(失败|已拒绝|退出码 \d+)$/.test(last) || act.text === "上一步工具失败";
  return { name: segments[0], detail: segments.slice(1, failed ? -1 : undefined).join(" · "), failed };
}

function resultView(msg: ChildMessage): ToolResultView {
  return { outcome: msg.outcome ?? "success", parts: msg.parts.filter((p) => p.kind === "text" && p.value).map((p) => p.value!) };
}

function messageText(msg: ChildMessage): string {
  return msg.parts.filter((p) => p.kind === "text" && p.value).map((p) => p.value!).join("\n\n");
}

// 1. 结果按 call_ref 建索引 2. 依序生成块，连续调用合并、正文与控制打断合并 3. 找不到调用的结果单独成轨道。
function buildStreamBlocks(messages: ChildMessage[]): StreamBlock[] {
  const results = new Map<string, ToolResultView>();
  const callKeys = new Set<string>();
  for (const msg of messages) {
    if (msg.kind === "toolresult" && msg.call_ref) {
      results.set(`${msg.call_ref.message_id}:${msg.call_ref.part_index}`, resultView(msg));
    }
    msg.parts.forEach((part, index) => {
      if (part.kind === "tool_call") callKeys.add(`${msg.message_id}:${index}`);
    });
  }
  const blocks: StreamBlock[] = [];
  let rail: Extract<StreamBlock, { kind: "tools" }> | null = null;
  const pushTool = (entry: ToolEntry) => {
    if (!rail) {
      rail = { kind: "tools", key: `tools:${entry.key}`, entries: [] };
      blocks.push(rail);
    }
    rail.entries.push(entry);
  };
  for (const msg of messages) {
    if (msg.kind === "toolresult") {
      const ref = msg.call_ref ? `${msg.call_ref.message_id}:${msg.call_ref.part_index}` : msg.message_id;
      if (!callKeys.has(ref)) pushTool({ key: msg.message_id, part: { kind: "tool_call", tool: "更早调用的结果" }, result: results.get(ref) ?? resultView(msg) });
      continue;
    }
    const text = messageText(msg);
    if (text) {
      rail = null;
      blocks.push({ kind: "text", key: `text:${msg.message_id}`, text, steer: msg.author === "parent" && msg.seq > 0, initial: msg.author === "parent" && msg.seq === 0 });
    }
    msg.parts.forEach((part, index) => {
      if (part.kind === "tool_call") {
        const key = `${msg.message_id}:${index}`;
        pushTool({ key, part, result: results.get(key) });
      } else if (part.kind === "control") {
        rail = null;
        blocks.push({ kind: "control", key: `control:${msg.message_id}:${index}`, action: part.action, reason: part.reason });
      }
    });
  }
  return blocks;
}

// 与文件预览同一套 Markdown 渲染；同步挂载，贴底滚动在同一帧内就能量到新高度。
function MarkdownText({ text, className = "" }: { text: string; className?: string }) {
  const ref = useRef<HTMLDivElement | null>(null);
  useLayoutEffect(() => {
    ref.current?.replaceChildren(renderMarkdownProse(text, { headingIds: false }));
  }, [text]);
  return <div ref={ref} className={`subagent-wb__md${className ? ` ${className}` : ""}`} />;
}

async function apiJson<T>(ctx: WebHostContextV1, url: string, init?: RequestInit): Promise<T> {
  const res = await ctx.http.request(url, init);
  const data = await res.json().catch(() => ({})) as Record<string, unknown>;
  if (!res.ok) {
    const detail = typeof data.detail === "string" ? data.detail : `请求失败（${res.status}）`;
    throw new Error(detail);
  }
  return data as T;
}

function ChevronDownIcon({ rotated = false }: { rotated?: boolean }) {
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      style={rotated ? { transform: "rotate(180deg)" } : undefined}
    >
      <path d="M4 6l4 4 4-4" />
    </svg>
  );
}

function SwitcherChevronsIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M5.5 6.25 8 3.75l2.5 2.5M5.5 9.75 8 12.25l2.5-2.5" />
    </svg>
  );
}

function PromptDocIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M4.5 2.25h5.25L12.75 5.25v8a1 1 0 0 1-1 1h-7.25a1 1 0 0 1-1-1v-10a1 1 0 0 1 1-1Z" />
      <path d="M9.5 2.25v3h3M5.5 7.75h5M5.5 10.25h3.5" />
    </svg>
  );
}

function TerminalGlyphIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.45" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M3.5 5 6.5 8 3.5 11M8.5 11h4" />
    </svg>
  );
}

function ArrowUpIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M8 12.5V3.5M4 7.5 8 3.5l4 4" />
    </svg>
  );
}

function ToolCallRow({ entry, live }: { entry: ToolEntry; live: boolean }) {
  const [expanded, setExpanded] = useState(false);
  const { tool, summary, rawJson } = useMemo(() => summarizeToolCall(entry.part), [entry.part]);
  const formatted = useMemo(() => (entry.result ? formatToolResult(entry.result) : null), [entry.result]);
  const args = entry.part.arguments ?? {};
  const command = typeof args.command === "string" ? args.command : "";
  const rest = Object.fromEntries(Object.entries(args).filter(([key]) => !ARG_HIDDEN_KEYS.has(key)));
  const restJson = command ? (Object.keys(rest).length ? JSON.stringify(rest, null, 2) : "") : rawJson;
  const hasArgs = rawJson !== "{}";
  const expandable = hasArgs || Boolean(formatted);
  const state = formatted ? (formatted.failed ? "error" : "done") : live ? "running" : "idle";
  const body = formatted && formatted.body.length > RESULT_VIEW_CHARS
    ? `${formatted.body.slice(0, RESULT_VIEW_CHARS)}\n…（已截断，共 ${formatted.body.length.toLocaleString()} 字）`
    : formatted?.body ?? "";

  return (
    <div className="subagent-wb__tool-item" data-state={state}>
      <button
        type="button"
        className="subagent-wb__tool-trigger"
        aria-expanded={expandable ? expanded : undefined}
        disabled={!expandable}
        onClick={() => expandable && setExpanded((prev) => !prev)}
      >
        <TerminalGlyphIcon />
        <span className="subagent-wb__tool-name">{tool}</span>
        <span className="subagent-wb__tool-summary">{summary}</span>
        {state === "error" ? <span className="subagent-wb__tool-flag">失败</span> : null}
        {state === "running" ? <span className="subagent-wb__tool-flag is-live">运行中</span> : null}
        {expandable ? <span className="subagent-wb__tool-caret"><ChevronDownIcon rotated={expanded} /></span> : null}
      </button>
      {expanded ? (
        <div className="subagent-wb__tool-detail">
          {command ? (
            <section>
              <h4 className="subagent-wb__tool-label">命令</h4>
              <pre className="subagent-wb__tool-raw">{command}</pre>
            </section>
          ) : null}
          {restJson && restJson !== "{}" ? (
            <section>
              <h4 className="subagent-wb__tool-label">参数</h4>
              <pre className="subagent-wb__tool-raw">{restJson}</pre>
            </section>
          ) : null}
          {formatted ? (
            <section>
              <h4 className="subagent-wb__tool-label">
                结果{formatted.meta.length ? <span className="subagent-wb__tool-meta">{formatted.meta.join(" · ")}</span> : null}
              </h4>
              {body ? <pre className="subagent-wb__tool-raw is-output">{body}</pre> : <p className="subagent-wb__tool-empty">无输出</p>}
            </section>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function ToolCallRail({ entries, live }: { entries: ToolEntry[]; live: boolean }) {
  const [showAll, setShowAll] = useState(false);
  const hiddenCount = Math.max(0, entries.length - VISIBLE_TOOL_TAIL);
  const visible = showAll || hiddenCount === 0 ? entries : entries.slice(-VISIBLE_TOOL_TAIL);

  return (
    <div className="subagent-wb__tool-rail">
      {hiddenCount > 0 ? (
        <button
          type="button"
          className="subagent-wb__fold-row"
          aria-expanded={showAll}
          onClick={() => setShowAll((prev) => !prev)}
        >
          <span>{showAll ? `收起更早 ${hiddenCount} 步工具调用` : `已完成 ${hiddenCount} 步更早工具调用`}</span>
          <ChevronDownIcon rotated={showAll} />
        </button>
      ) : null}
      {visible.map((entry) => (
        <ToolCallRow key={entry.key} entry={entry} live={live} />
      ))}
    </div>
  );
}

export function SubagentWorkbench({ ctx, view }: { ctx: WebHostContextV1; view: ConversationTabView }) {
  const [sessionId, setSessionId] = useState<string | null>(view.session);
  const [active, setActive] = useState<boolean>(view.active);
  const [subagents, setSubagents] = useState<TreeSubagent[]>([]);
  const [runs, setRuns] = useState<TreeExternalRun[]>([]);
  const [selected, setSelected] = useState<Selection | null>(null);
  const [messages, setMessages] = useState<ChildMessage[]>([]);
  const [hasMore, setHasMore] = useState(false);
  const [loadingOlder, setLoadingOlder] = useState(false);
  const [childMeta, setChildMeta] = useState<{ status?: string; active?: boolean; cwd?: string | null; model_id?: string | null } | null>(null);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [error, setError] = useState("");
  const [loadError, setLoadError] = useState("");
  const [switcherOpen, setSwitcherOpen] = useState(false);
  const [promptExpanded, setPromptExpanded] = useState(false);
  const [activitiesExpanded, setActivitiesExpanded] = useState(false);

  const selectedRef = useRef<Selection | null>(null);
  const lastSeqRef = useRef(-1);
  const streamRef = useRef<HTMLDivElement | null>(null);
  const switcherRef = useRef<HTMLDivElement | null>(null);
  const composerRef = useRef<HTMLTextAreaElement | null>(null);
  const treeReqSeq = useRef(0);
  const msgReqSeq = useRef(0);
  const selectionGenRef = useRef(0);
  const oldestExhaustedRef = useRef(false);
  const stickBottomRef = useRef(true);
  const prevTailKeyRef = useRef("");

  const selectChild = useCallback((next: Selection | null) => {
    msgReqSeq.current += 1;
    selectionGenRef.current += 1;
    oldestExhaustedRef.current = false;
    stickBottomRef.current = true;
    prevTailKeyRef.current = "";
    lastSeqRef.current = -1;
    selectedRef.current = next;
    setSelected(next);
    setMessages([]);
    setHasMore(false);
    setChildMeta(null);
    setError("");
    setLoadError("");
    setSwitcherOpen(false);
    setPromptExpanded(false);
    setActivitiesExpanded(false);
  }, []);

  useEffect(() => {
    const offSession = view.onSessionChange((next) => {
      treeReqSeq.current += 1;
      setSessionId(next);
      setSubagents([]);
      setRuns([]);
      selectChild(null);
    });
    const offActive = view.onActiveChange(setActive);
    const offIntent = view.onIntent((intent) => {
      if (intent.name !== "subagent.open.v1") return "invalid";
      const payload = intent.payload as { kind?: unknown; job_id?: unknown } | null;
      if (!payload || (payload.kind !== "subagent" && payload.kind !== "external") || typeof payload.job_id !== "string") {
        return "invalid";
      }
      selectChild({ kind: payload.kind, id: payload.job_id });
      return "handled";
    });
    return () => {
      offSession();
      offActive();
      offIntent();
    };
  }, [selectChild, view]);

  useEffect(() => {
    if (!switcherOpen) return;
    const onPointerDown = (event: PointerEvent) => {
      if (switcherRef.current && !switcherRef.current.contains(event.target as Node)) {
        setSwitcherOpen(false);
      }
    };
    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") setSwitcherOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [switcherOpen]);

  const refreshTree = useCallback(async () => {
    const seq = ++treeReqSeq.current;
    if (!sessionId) {
      setSubagents([]);
      setRuns([]);
      view.setIndicator(null);
      return;
    }
    try {
      const data = await apiJson<{ subagents?: TreeSubagent[]; runs?: TreeExternalRun[] }>(
        ctx,
        `${API_BASE}/tree?session=${encodeURIComponent(sessionId)}`,
      );
      if (seq !== treeReqSeq.current) return;
      const nextSub = Array.isArray(data.subagents) ? data.subagents : [];
      const nextRuns = Array.isArray(data.runs) ? data.runs : [];
      setSubagents(nextSub);
      setRuns(nextRuns);
      const running = nextSub.filter((s) => s.active).length
        + nextRuns.filter((r) => r.status === "running" || r.status === "accepted").length;
      view.setIndicator(running > 0 ? { busy: true, detail: `${running} 运行中` } : null);
      setLoadError("");
      if (!selectedRef.current) {
        if (nextSub.length > 0) selectChild({ kind: "subagent", id: nextSub[0].job_id });
        else if (nextRuns.length > 0) selectChild({ kind: "external", id: nextRuns[0].id });
      }
    } catch (err) {
      if (seq !== treeReqSeq.current) return;
      setLoadError(err instanceof Error ? err.message : "读取子任务列表失败");
    }
  }, [ctx, selectChild, sessionId, view]);

  const refreshMessages = useCallback(async () => {
    if (!sessionId || !selected || selected.kind !== "subagent") return;
    const seq = ++msgReqSeq.current;
    const afterSeq = lastSeqRef.current;
    const query = afterSeq >= 0 ? `after_seq=${afterSeq}&limit=200` : "limit=100";
    try {
      const data = await apiJson<{
        messages?: ChildMessage[];
        has_more?: boolean;
        status?: string;
        active?: boolean;
        cwd?: string | null;
        model_id?: string | null;
      }>(
        ctx,
        `${API_BASE}/messages?session=${encodeURIComponent(sessionId)}&job_id=${encodeURIComponent(selected.id)}&${query}`,
      );
      if (seq !== msgReqSeq.current) return;
      const latest = Array.isArray(data.messages) ? data.messages : [];
      if (latest.length) lastSeqRef.current = Math.max(lastSeqRef.current, latest[latest.length - 1].seq);
      if (afterSeq < 0) {
        setMessages(latest);
        setHasMore(!oldestExhaustedRef.current && Boolean(data.has_more));
      } else if (latest.length) {
        setMessages((prev) => [...prev, ...latest.filter((m) => m.seq > (prev.length ? prev[prev.length - 1].seq : -1))]);
      }
      setLoadError("");
      setChildMeta({ status: data.status, active: data.active, cwd: data.cwd, model_id: data.model_id });
    } catch (err) {
      if (seq !== msgReqSeq.current) return;
      setLoadError(err instanceof Error ? err.message : "读取子任务消息失败");
    }
  }, [ctx, sessionId, selected]);

  const loadOlder = async () => {
    if (!sessionId || !selected || selected.kind !== "subagent" || !messages.length || loadingOlder) return;
    const beforeSeq = messages[0].seq;
    const gen = selectionGenRef.current;
    setLoadingOlder(true);
    try {
      const data = await apiJson<{ messages?: ChildMessage[]; has_more?: boolean }>(
        ctx,
        `${API_BASE}/messages?session=${encodeURIComponent(sessionId)}&job_id=${encodeURIComponent(selected.id)}&before_seq=${beforeSeq}&limit=100`,
      );
      if (gen !== selectionGenRef.current) return;
      const older = Array.isArray(data.messages) ? data.messages : [];
      setMessages((prev) => [...older, ...prev.filter((m) => m.seq >= beforeSeq)]);
      if (!data.has_more) oldestExhaustedRef.current = true;
      setHasMore(Boolean(data.has_more));
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载历史消息失败");
    } finally {
      setLoadingOlder(false);
    }
  };

  useEffect(() => {
    void refreshTree();
    const timer = window.setInterval(() => {
      if (active || document.visibilityState === "visible") {
        void refreshTree();
      }
    }, active ? 1500 : 4000);
    return () => window.clearInterval(timer);
  }, [active, refreshTree]);

  const currentSub = selected?.kind === "subagent" ? subagents.find((s) => s.job_id === selected.id) : undefined;
  const childLive = Boolean(currentSub?.active || childMeta?.active);

  useEffect(() => {
    if (!active || !selected || selected.kind !== "subagent") return;
    void refreshMessages();
    if (!childLive) return;
    const timer = window.setInterval(() => {
      void refreshMessages();
    }, 1200);
    return () => window.clearInterval(timer);
  }, [active, childLive, selected, refreshMessages]);

  const currentRun = selected?.kind === "external" ? runs.find((r) => r.id === selected.id) : undefined;
  const runActivities = useMemo(
    () => (currentRun?.activities ?? []).map(activityView).filter((item): item is NonNullable<typeof item> => item !== null),
    [currentRun?.activities],
  );
  const isRunning = currentSub ? currentSub.active : currentRun ? (currentRun.status === "running" || currentRun.status === "accepted") : false;

  const tailKey = selected?.kind === "subagent"
    ? `sub:${selected.id}:${messages.length ? messages[messages.length - 1].seq : -1}`
    : selected?.kind === "external"
      ? `ext:${selected.id}:${currentRun?.activities.length ?? 0}:${currentRun?.result.length ?? 0}`
      : "";

  useEffect(() => {
    const el = streamRef.current;
    if (!el || !tailKey || tailKey === prevTailKeyRef.current) return;
    prevTailKeyRef.current = tailKey;
    if (stickBottomRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [tailKey]);

  const handleStreamScroll = () => {
    const el = streamRef.current;
    if (!el) return;
    stickBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
  };

  const submitDraft = async () => {
    if (!sessionId || !currentSub || !draft.trim() || sending) return;
    setSending(true);
    setError("");
    stickBottomRef.current = true;
    try {
      await apiJson(ctx, `${API_BASE}/send`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session: sessionId, job_id: currentSub.job_id, message: draft.trim() }),
      });
      setDraft("");
      if (composerRef.current) composerRef.current.style.height = "";
      await refreshTree();
      await refreshMessages();
    } catch (err) {
      setError(err instanceof Error ? err.message : "发送失败");
    } finally {
      setSending(false);
    }
  };

  const handleSend = (event: FormEvent) => {
    event.preventDefault();
    void submitDraft();
  };

  const handleComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      void submitDraft();
    }
  };

  const handleStop = async () => {
    if (!sessionId || !selected || stopping) return;
    setStopping(true);
    setError("");
    try {
      const endpoint = selected.kind === "subagent" ? `${API_BASE}/cancel` : `${API_BASE}/runner-stop`;
      await apiJson(ctx, endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session: sessionId, job_id: selected.id }),
      });
      await refreshTree();
      if (selected.kind === "subagent") await refreshMessages();
    } catch (err) {
      setError(err instanceof Error ? err.message : "停止失败");
    } finally {
      setStopping(false);
    }
  };

  // 长委派指令折叠进顶部说明，不再进消息流。
  const blocks = useMemo(() => {
    const first = messages[0];
    const firstText = first && first.author === "parent" && first.seq === 0 ? messageText(first) : "";
    return buildStreamBlocks(firstText.length > 140 ? messages.slice(1) : messages);
  }, [messages]);

  if (!sessionId) {
    return <div className="subagent-wb__empty">发送首条消息后，这里将实时展示该会话树派出的全部子任务。</div>;
  }

  const totalTasks = subagents.length + runs.length;
  const activeIndex = currentSub
    ? subagents.findIndex((item) => item.job_id === currentSub.job_id)
    : currentRun
      ? subagents.length + runs.findIndex((run) => run.id === currentRun.id)
      : -1;
  const activeLabel = currentSub?.label ?? currentRun?.label ?? "选择子任务";
  const activeStatus = currentSub
    ? (childMeta?.status ?? currentSub.status)
    : currentRun?.status ?? "";
  const activeTone = statusTone(activeStatus, currentSub?.active);
  const statusLabel = STATUS_LABELS[activeStatus] ?? activeStatus;
  const activeModel = childMeta?.model_id || currentSub?.model_id || "";
  const activeCwd = childMeta?.cwd || currentSub?.cwd || "";

  // 首条委派指令默认折叠成单行轻触发器，避免长 prompt 占满视口
  const firstMsg = messages[0];
  const firstPromptText = firstMsg && firstMsg.author === "parent" && firstMsg.seq === 0
    ? firstMsg.parts.filter((p) => p.kind === "text" && p.value).map((p) => p.value!).join("\n\n")
    : "";
  const hasFoldedBrief = firstPromptText.length > 140 || Boolean(activeModel || activeCwd);
  
  return (
    <div className="subagent-wb">
      <header className="subagent-wb__bar">
        <div ref={switcherRef} className="subagent-wb__switcher-wrap">
          <button
            type="button"
            className="subagent-wb__switcher"
            aria-expanded={totalTasks > 1 ? switcherOpen : undefined}
            aria-haspopup={totalTasks > 1 ? "listbox" : undefined}
            disabled={totalTasks === 0}
            onClick={() => totalTasks > 1 && setSwitcherOpen((prev) => !prev)}
          >
            {selected ? <span className={`subagent-wb__dot is-${activeTone}`} aria-hidden="true" /> : null}
            <span className="subagent-wb__switcher-label">{totalTasks ? activeLabel : "暂无子任务"}</span>
            {totalTasks > 1 ? (
              <>
                <span className="subagent-wb__count">{activeIndex + 1}/{totalTasks}</span>
                <span className="subagent-wb__switcher-chevron"><SwitcherChevronsIcon /></span>
              </>
            ) : null}
          </button>

          {switcherOpen && totalTasks > 1 ? (
            <div className="subagent-wb__popover" role="tablist" aria-label="子任务列表">
              {subagents.map((item) => {
                const activeTab = selected?.kind === "subagent" && selected.id === item.job_id;
                const tone = statusTone(item.status, item.active);
                return (
                  <button
                    key={`sub:${item.job_id}`}
                    type="button"
                    role="tab"
                    aria-selected={activeTab}
                    className={`subagent-wb__pill${activeTab ? " is-selected" : ""}`}
                    onClick={() => selectChild({ kind: "subagent", id: item.job_id })}
                  >
                    <span className={`subagent-wb__dot is-${tone}`} aria-hidden="true" />
                    <span className="subagent-wb__pill-label">{item.label}</span>
                    <span className="subagent-wb__pill-tags">
                      {item.depth > 1 ? <span className="subagent-wb__chip">L{item.depth}</span> : null}
                      {item.fork ? <span className="subagent-wb__chip">fork</span> : null}
                      {item.isolation === "worktree" ? <span className="subagent-wb__chip">worktree</span> : null}
                      <span className="subagent-wb__chip">{STATUS_LABELS[item.status] ?? item.status}</span>
                    </span>
                  </button>
                );
              })}
              {runs.map((run) => {
                const activeTab = selected?.kind === "external" && selected.id === run.id;
                const tone = statusTone(run.status, false);
                return (
                  <button
                    key={`ext:${run.id}`}
                    type="button"
                    role="tab"
                    aria-selected={activeTab}
                    className={`subagent-wb__pill${activeTab ? " is-selected" : ""}`}
                    onClick={() => selectChild({ kind: "external", id: run.id })}
                  >
                    <span className={`subagent-wb__dot is-${tone}`} aria-hidden="true" />
                    <span className="subagent-wb__pill-label">{run.label}</span>
                    <span className="subagent-wb__pill-tags">
                      <span className="subagent-wb__chip">{run.runner} · 只读</span>
                      <span className="subagent-wb__chip">{STATUS_LABELS[run.status] ?? run.status}</span>
                    </span>
                  </button>
                );
              })}
            </div>
          ) : null}
        </div>

        <div className="subagent-wb__bar-right">
          {currentSub ? (
            <span className="subagent-wb__status-meta">
              {statusLabel}
              {" · "}L{currentSub.depth}
              {currentSub.read_only ? " · 只读" : ""}
            </span>
          ) : currentRun ? (
            <span className="subagent-wb__status-meta">
              {statusLabel} · {currentRun.runner} · 只读
            </span>
          ) : null}
          {isRunning ? (
            <button type="button" className="subagent-wb__stop" disabled={stopping} onClick={() => void handleStop()}>
              {stopping ? "停止中…" : "停止"}
            </button>
          ) : null}
        </div>
      </header>

      {error || loadError ? <p className="subagent-wb__error" role="alert">{error || loadError}</p> : null}

      {!selected ? (
        <div className="subagent-wb__empty">当前对话还没有派出子任务或外部代理。</div>
      ) : currentSub ? (
        <>
          <div ref={streamRef} className="subagent-wb__stream" onScroll={handleStreamScroll}>
            {hasMore ? (
              <button
                type="button"
                className="subagent-wb__load-older"
                disabled={loadingOlder}
                onClick={() => void loadOlder()}
              >
                {loadingOlder ? "加载中…" : "加载更早消息"}
              </button>
            ) : null}

            {hasFoldedBrief ? (
              <section className="subagent-wb__brief">
                <button
                  type="button"
                  className="subagent-wb__brief-trigger"
                  aria-expanded={promptExpanded}
                  onClick={() => setPromptExpanded((prev) => !prev)}
                >
                  <span className="subagent-wb__brief-left">
                    <PromptDocIcon />
                    <span className="subagent-wb__brief-title">
                      {firstPromptText.length > 140 ? "委派指令" : "运行环境"}
                    </span>
                    {firstPromptText.length > 140 ? (
                      <span className="subagent-wb__brief-count">{firstPromptText.length.toLocaleString()} 字</span>
                    ) : null}
                  </span>
                  <span className="subagent-wb__brief-right">
                    <span>{promptExpanded ? "收起" : "查看"}</span>
                    <ChevronDownIcon rotated={promptExpanded} />
                  </span>
                </button>
                {promptExpanded ? (
                  <div className="subagent-wb__brief-body">
                    {activeModel || activeCwd ? (
                      <div className="subagent-wb__brief-meta">
                        {activeModel ? <span>模型：<code>{activeModel}</code></span> : null}
                        {activeCwd ? <span>目录：<code>{activeCwd}</code></span> : null}
                      </div>
                    ) : null}
                    {firstPromptText.length > 140 ? (
                      <MarkdownText text={firstPromptText} className="subagent-wb__brief-text" />
                    ) : null}
                  </div>
                ) : null}
              </section>
            ) : null}

            {blocks.length === 0 && firstPromptText.length <= 140 ? (
              <p className="subagent-wb__empty-inline">等待子任务写入首条输出…</p>
            ) : (
              blocks.map((block) => {
                if (block.kind === "tools") return <ToolCallRail key={block.key} entries={block.entries} live={childLive} />;
                if (block.kind === "control") {
                  return (
                    <div key={block.key} className={`subagent-wb__control${block.action === "cancel" || block.action === "pause" ? " is-muted" : ""}`}>
                      <strong>{CONTROL_LABELS[block.action ?? ""] ?? block.action ?? "状态变更"}</strong>
                      {block.reason ? <span>{block.reason}</span> : null}
                    </div>
                  );
                }
                return (
                  <article key={block.key} className={`subagent-wb__turn${block.steer || block.initial ? " is-steer" : ""}`}>
                    {block.steer ? <span className="subagent-wb__turn-tag">插话</span> : null}
                    <MarkdownText text={block.text} />
                  </article>
                );
              })
            )}
          </div>

          <form className="subagent-wb__composer" onSubmit={handleSend}>
            <div className="subagent-wb__composer-box">
              <textarea
                ref={composerRef}
                rows={1}
                value={draft}
                disabled={sending}
                placeholder={currentSub.active ? "给运行中的子任务发送 Steer 插话指令…" : "向已结束的子任务继续追问（沿用上下文）…"}
                onChange={(e) => {
                  setDraft(e.target.value);
                  e.target.style.height = "auto";
                  e.target.style.height = `${Math.min(e.target.scrollHeight, 120)}px`;
                }}
                onKeyDown={handleComposerKeyDown}
              />
              <button type="submit" className="subagent-wb__send" disabled={sending || !draft.trim()}>
                <span>{sending ? "发送中" : currentSub.active ? "插话" : "追问"}</span>
                <ArrowUpIcon />
              </button>
            </div>
          </form>
        </>
      ) : currentRun ? (
        <>
          <div ref={streamRef} className="subagent-wb__stream" onScroll={handleStreamScroll}>
            {runActivities.length > 0 ? (
              <div className="subagent-wb__tool-rail">
                {runActivities.length > 5 ? (
                  <button
                    type="button"
                    className="subagent-wb__fold-row"
                    aria-expanded={activitiesExpanded}
                    onClick={() => setActivitiesExpanded((prev) => !prev)}
                  >
                    <span>
                      {activitiesExpanded
                        ? `收起更早 ${runActivities.length - 3} 条活动`
                        : `已完成 ${runActivities.length - 3} 条更早活动`}
                    </span>
                    <ChevronDownIcon rotated={activitiesExpanded} />
                  </button>
                ) : null}
                {(activitiesExpanded || runActivities.length <= 5 ? runActivities : runActivities.slice(-3)).map((act, i) => (
                  act.name ? (
                    <div key={i} className="subagent-wb__tool-item" data-state={act.failed ? "error" : "done"}>
                      <div className="subagent-wb__tool-trigger is-static">
                        <TerminalGlyphIcon />
                        <span className="subagent-wb__tool-name">{act.name}</span>
                        <span className="subagent-wb__tool-summary">{act.detail}</span>
                        {act.failed ? <span className="subagent-wb__tool-flag">失败</span> : null}
                      </div>
                    </div>
                  ) : (
                    <p key={i} className="subagent-wb__activity-note">{act.detail}</p>
                  )
                ))}
              </div>
            ) : (
              <p className="subagent-wb__empty-inline">{currentRun.progress ? "暂无活动记录。" : "此执行器仅提供最终结果。"}</p>
            )}
            {currentRun.result ? <MarkdownText text={currentRun.result} /> : null}
          </div>
          <div className="subagent-wb__readonly-composer" role="note">
            外部只读执行器（{currentRun.runner}）不支持续聊，仅可查看过程或停止。
          </div>
        </>
      ) : null}
    </div>
  );
}

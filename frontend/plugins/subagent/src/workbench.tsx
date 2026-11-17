import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import type { ConversationTabView } from "@akashic/conversation-tools-v2";
import type { WebHostContextV1 } from "@akashic/web-ui-v1";
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
}

type Selection = { kind: "subagent" | "external"; id: string };

const STATUS_LABELS: Record<string, string> = {
  accepted: "已接纳",
  running: "运行中",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
  interrupted: "已中断",
};

const API_BASE = "/api/dashboard/subagent";

async function apiJson<T>(ctx: WebHostContextV1, url: string, init?: RequestInit): Promise<T> {
  const res = await ctx.http.request(url, init);
  const data = await res.json().catch(() => ({})) as Record<string, unknown>;
  if (!res.ok) {
    const detail = typeof data.detail === "string" ? data.detail : `请求失败（${res.status}）`;
    throw new Error(detail);
  }
  return data as T;
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
  const selectedRef = useRef<Selection | null>(null);
  const lastSeqRef = useRef(-1);
  const streamRef = useRef<HTMLDivElement | null>(null);
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

  const handleSend = async (event: FormEvent) => {
    event.preventDefault();
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
      await refreshTree();
      await refreshMessages();
    } catch (err) {
      setError(err instanceof Error ? err.message : "发送失败");
    } finally {
      setSending(false);
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

  if (!sessionId) {
    return <div className="subagent-wb__empty">发送首条消息后，这里将实时展示该会话树派出的全部子任务。</div>;
  }

  return (
    <div className="subagent-wb">
      <header className="subagent-wb__bar">
        <div className="subagent-wb__tabs" role="tablist" aria-label="子任务列表">
          {subagents.map((item) => {
            const activeTab = selected?.kind === "subagent" && selected.id === item.job_id;
            const tone = item.active ? "live" : item.status === "failed" ? "error" : "done";
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
                {item.depth > 1 ? <span className="subagent-wb__chip">L{item.depth}</span> : null}
                {item.fork ? <span className="subagent-wb__chip">fork</span> : null}
                {item.isolation === "worktree" ? <span className="subagent-wb__chip">worktree</span> : null}
              </button>
            );
          })}
          {runs.map((run) => {
            const activeTab = selected?.kind === "external" && selected.id === run.id;
            const tone = run.status === "running" || run.status === "accepted" ? "live" : run.status === "failed" ? "error" : "done";
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
                <span className="subagent-wb__chip">{run.runner} · 只读</span>
              </button>
            );
          })}
        </div>
        {isRunning ? (
          <button type="button" className="subagent-wb__stop" disabled={stopping} onClick={() => void handleStop()}>
            {stopping ? "正在停止…" : "停止任务"}
          </button>
        ) : null}
      </header>

      {error || loadError ? <p className="subagent-wb__error" role="alert">{error || loadError}</p> : null}

      {!selected ? (
        <div className="subagent-wb__empty">当前对话还没有派出子任务或外部代理。</div>
      ) : currentSub ? (
        <>
          <div className="subagent-wb__meta">
            <span>状态：{STATUS_LABELS[childMeta?.status ?? currentSub.status] ?? (childMeta?.status ?? currentSub.status)}</span>
            <span>深度 L{currentSub.depth}</span>
            {currentSub.read_only ? <span>只读模式</span> : null}
            {(childMeta?.model_id || currentSub.model_id) ? <span>模型：{childMeta?.model_id || currentSub.model_id}</span> : null}
            {(childMeta?.cwd || currentSub.cwd) ? <code>{childMeta?.cwd || currentSub.cwd}</code> : null}
          </div>
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
            {messages.length === 0 ? (
              <p className="subagent-wb__empty-inline">等待子任务写入首条输出…</p>
            ) : (
              messages.map((msg) => (
                <article key={msg.message_id} className={`subagent-wb__msg is-${msg.kind} is-${msg.author}`}>
                  <header className="subagent-wb__msg-head">
                    <span>{msg.author === "subagent" ? "子任务" : msg.author === "parent" ? "父对话插话" : msg.author}</span>
                    <span>#{msg.seq}</span>
                  </header>
                  <div className="subagent-wb__msg-body">
                    {msg.parts.map((part, idx) => {
                      if (part.kind === "text" && part.value) {
                        return <pre key={idx} className="subagent-wb__text">{part.value}</pre>;
                      }
                      if (part.kind === "tool_call") {
                        return (
                          <div key={idx} className="subagent-wb__tool-call">
                            <code>工具调用</code>
                            <span>{JSON.stringify(part.arguments ?? {})}</span>
                          </div>
                        );
                      }
                      if (part.kind === "control") {
                        return <p key={idx} className="subagent-wb__control">[{part.action}] {part.reason ?? ""}</p>;
                      }
                      return null;
                    })}
                  </div>
                </article>
              ))
            )}
          </div>
          <form className="subagent-wb__composer" onSubmit={(e) => void handleSend(e)}>
            <textarea
              rows={2}
              value={draft}
              disabled={sending}
              placeholder={currentSub.active ? "给运行中的子任务发送 Steer 插话指令…" : "向已结束的子任务继续追问（沿用原上下文）…"}
              onChange={(e) => setDraft(e.target.value)}
            />
            <button type="submit" disabled={sending || !draft.trim()}>
              {sending ? "发送中…" : currentSub.active ? "插话 Steer" : "继续追问"}
            </button>
          </form>
        </>
      ) : currentRun ? (
        <>
          <div className="subagent-wb__meta">
            <span>执行器：{currentRun.runner}</span>
            <span>状态：{STATUS_LABELS[currentRun.status] ?? currentRun.status}</span>
            <span>只读外部代理</span>
          </div>
          <div ref={streamRef} className="subagent-wb__stream" onScroll={handleStreamScroll}>
            {currentRun.activities.length > 0 ? (
              <ol className="subagent-wb__activities">
                {currentRun.activities.map((act, i) => (
                  <li key={i} data-kind={act.kind}>{act.text}</li>
                ))}
              </ol>
            ) : (
              <p className="subagent-wb__empty-inline">{currentRun.progress ? "暂无活动记录。" : "此执行器仅提供最终结果。"}</p>
            )}
            {currentRun.result ? <pre className="subagent-wb__text">{currentRun.result}</pre> : null}
          </div>
          <div className="subagent-wb__readonly-composer" role="note">
            外部只读执行器（{currentRun.runner}）不支持续聊，仅可查看过程或停止。
          </div>
        </>
      ) : null}
    </div>
  );
}

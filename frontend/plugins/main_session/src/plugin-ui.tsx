import "./plugin-ui.css";
import { useEffect, useMemo, useState } from "react";
import {
  render, type DestinationControls, type DestinationForm, type NavigationDestination, type NavigationDestinationsProvider,
  type PluginUiDestinationContext, type PluginUiDestinationFormContext, type PluginUiHostContext,
} from "@akashic/chat-ui-v1";

type Query = PluginUiHostContext["query"];

interface Candidate {
  id: string;
  title: string;
  updated_at: string;
  /** 近 60 天你亲口说的句数；只给消息最多的几段算。 */
  said?: number;
  /** 现在就在侧栏顶部。 */
  current?: boolean;
}

function candidateOf(value: unknown): Candidate {
  const row = typeof value === "object" && value !== null ? value as Record<string, unknown> : null;
  if (!row || typeof row.id !== "string" || typeof row.title !== "string" || typeof row.updated_at !== "string"
    || (row.said !== undefined && typeof row.said !== "number")) throw new Error("候选对话记录无效");
  return { id: row.id, title: row.title, updated_at: row.updated_at, ...(typeof row.said === "number" ? { said: row.said } : {}),
    ...(row.current === true ? { current: true } : {}) };
}

const message = (reason: unknown) => reason instanceof Error ? reason.message : String(reason);

// 设成某段对话后打开它；新建则先占空位，打开空白页，首条消息由宿主交回。
async function choose(query: Query, destination: DestinationControls, sessionId: string | null, close?: () => void) {
  if (sessionId) await query("pointer.set", { session_id: sessionId });
  else await query("pointer.new");
  close?.();
  destination.reload();
  destination.open(sessionId ?? undefined);
}

/** 选择面板：引导卡“选别的…”与 Akashic 菜单“换一段对话…”共用。 */
function Picker({ context }: { context: PluginUiDestinationFormContext }) {
  const [items, setItems] = useState<Candidate[] | null>(null);
  const [needle, setNeedle] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    let live = true;
    context.query("candidates.list").then((result) => {
      if (!live) return;
      if (!Array.isArray(result.items)) throw new Error("候选列表无效");
      setItems(result.items.map(candidateOf));
    }).catch((reason: unknown) => { if (live) setError(message(reason)); });
    return () => { live = false; };
  }, [context]);
  const shown = useMemo(() => (items ?? []).filter((item) => item.title.toLowerCase().includes(needle.trim().toLowerCase())), [items, needle]);
  const pick = (sessionId: string | null) => {
    setBusy(true);
    setError("");
    choose(context.query, context.destination, sessionId, context.close).catch((reason: unknown) => {
      setBusy(false);
      setError(message(reason));
    });
  };
  return <div className="main-session-picker" aria-busy={busy || undefined}>
    <input type="search" placeholder="搜索对话" aria-label="搜索对话" value={needle} autoFocus
      onChange={(event) => setNeedle(event.target.value)} />
    {error ? <p className="main-session-error" role="alert">{error}</p> : null}
    {items === null && !error ? <p className="main-session-muted" role="status">正在读取…</p> : null}
    <div className="main-session-list" role="list">
      {shown.map((item, index) => <div key={item.id} role="listitem"><button type="button" disabled={busy || item.current}
        aria-current={item.current || undefined} onClick={() => pick(item.id)}>
        <span>{item.title}</span>
        <small>{item.current ? "现在在侧栏顶部" : item.said ? `${index === 0 && !needle ? "最常聊 · " : ""}近 60 天你说了 ${item.said} 句`
          : `最后活跃 ${item.updated_at.slice(0, 10)}`}</small>
      </button></div>)}
    </div>
    <button type="button" className="main-session-new" disabled={busy} onClick={() => pick(null)}>新建一段对话</button>
  </div>;
}

const picker: DestinationForm = {
  title: "选一段对话放到侧栏顶部",
  mount: (host, context) => render(host, <Picker context={context} />),
};

/** 老用户第一次看到：推荐近 60 天你说得最多的那段，一键设好；关掉后不再出现。 */
function Guide({ context }: { context: PluginUiDestinationContext }) {
  const [guide, setGuide] = useState<{ suggestion: Candidate | null } | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    let live = true;
    context.query("guide.read").then((result) => {
      if (live) setGuide(result.show === true ? { suggestion: result.suggestion ? candidateOf(result.suggestion) : null } : null);
    }).catch(() => { if (live) setGuide(null); });
    return () => { live = false; };
  }, [context]);
  if (!guide) return null;
  const act = (run: () => Promise<void>) => {
    setBusy(true);
    setError("");
    run().catch((reason: unknown) => { setBusy(false); setError(message(reason)); });
  };
  const { suggestion } = guide;
  return <section className="main-session-guide" aria-label="把常聊的对话放到侧栏顶部">
    <header>
      <strong>把常聊的那段对话放到这里</strong>
      <button type="button" className="main-session-close" aria-label="以后再说" title="以后再说" disabled={busy}
        onClick={() => act(async () => { await context.query("pointer.dismiss"); setGuide(null); context.destination.reload(); })} />
    </header>
    <p>固定在侧栏最上面，点 Akashic 就能回到它。</p>
    {suggestion ? <div className="main-session-suggestion">
      <span>{suggestion.title}</span>
      <small>你最常聊的 · 近 60 天说了 {suggestion.said} 句</small>
    </div> : null}
    {error ? <p className="main-session-error" role="alert">{error}</p> : null}
    <div className="main-session-actions">
      {suggestion ? <button type="button" className="primary" disabled={busy}
        onClick={() => act(() => choose(context.query, context.destination, suggestion.id))}>就用这段</button> : null}
      <button type="button" disabled={busy} onClick={() => context.destination.openForm(picker)}>{suggestion ? "选别的…" : "选一段…"}</button>
      <button type="button" disabled={busy} onClick={() => act(() => choose(context.query, context.destination, null))}>新建</button>
    </div>
  </section>;
}

const destinations: NavigationDestinationsProvider = {
  placement: "identity",
  async load({ query }) {
    const result = await query("destination.load");
    return (result.destination ?? null) as NavigationDestination | null;
  },
  async adopt({ query }, sessionId) {
    await query("pointer.adopt", { session_id: sessionId });
  },
  notice: { mount: (host, context) => render(host, <Guide context={context} />) },
  actions: [{ label: "换一段对话…", form: picker }],
  sessionActions: (session) => [{
    label: "放到侧栏顶部",
    run: async ({ query }) => { await query("pointer.set", { session_id: session.id }); },
  }],
};

export default { slots: { "navigation.destinations": destinations } };

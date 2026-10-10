import { useEffect, useRef, useState, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import type { WebHostContextV1, WebUiDisposer } from "@akashic/web-ui-v1";
import "./configuration.css";

export interface Status {
  input_ref: string; enabled: boolean | null; ready: boolean; blocked: boolean;
  can_enable?: boolean; reason: string; has_token?: boolean;
  values: Record<string, unknown>; targets?: {channel: string; recipient: string; session_id: string; label: string}[];
}
export interface FormProps { values: Record<string, unknown>; change: (key: string, value: unknown) => void; status: Status; }
export interface FormDefinition { id: string; title: string; description: string; family?: string; familyLabel?: string; fields?: (props: FormProps) => ReactNode; }
// intent="enable"：嵌入方已替用户决定开启（如引导的能力卡），表单预选开启并收起开关。
export interface EmbedProps { embedded?: boolean; changed?: () => void; dirty?: (value: boolean) => void; intent?: "enable"; mode?: "summary"; }

export class RequestError extends Error {
  constructor(message: string, readonly status: number) { super(message); }
}

export async function request<T>(ctx: WebHostContextV1, path: string, init?: RequestInit, observe?: (response: Response) => void): Promise<T> {
  const response = await ctx.http.request(path, init);
  observe?.(response);
  if (!response.headers.get("content-type")?.includes("application/json")) throw new Error(`服务暂时不可用（${response.status}），请稍后重试`);
  const body = await response.json();
  if (!response.ok) {
    const detail = body.detail ?? body.message;
    throw new RequestError(Array.isArray(detail) ? detail.map((item: {msg: string}) => item.msg).join("；") : typeof detail === "string" ? detail : `请求失败（${response.status}）`, response.status);
  }
  return body as T;
}

export function Confirm({ title, children, accept, cancel, busy = false }: {title: string; children: ReactNode; accept: () => void; cancel: () => void; busy?: boolean}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => { const dialog = ref.current!; dialog.showModal(); return () => dialog.close(); }, []);
  return <dialog ref={ref} className="config-dialog" aria-labelledby="config-confirm-title" onCancel={event => { event.preventDefault(); if (!busy) cancel(); }}>
    <h2 id="config-confirm-title">{title}</h2><p>{children}</p>
    <footer><button type="button" autoFocus disabled={busy} className="config-primary" onClick={cancel}>继续编辑</button><button type="button" disabled={busy} onClick={accept}>放弃并离开</button></footer>
  </dialog>;
}

export function Field({label, name, value, change, type = "text", hint, required = false, inputMode}: {
  label: string; name: string; value: unknown; change: FormProps["change"]; type?: string; hint?: string; required?: boolean;
  inputMode?: "text" | "numeric" | "decimal" | "tel" | "url" | "email";
}) {
  return <label className="config-field"><span>{label}</span><input name={name} type={type} inputMode={inputMode} autoComplete={type === "password" ? "new-password" : "off"} required={required}
    value={typeof value === "string" || typeof value === "number" ? value : ""}
    onChange={event => change(name, type === "number" ? event.target.valueAsNumber : event.target.value)} />{hint && <small>{hint}</small>}</label>;
}

export function registerForm(ctx: WebHostContextV1, definition: FormDefinition): WebUiDisposer {
  return ctx.ui.inject("shell.settings-plugins.v1", mount => mount.register({
    id: `${definition.id.replaceAll("_", "-")}-settings`, label: definition.title, route: `${definition.id}-settings`, family: definition.family, familyLabel: definition.familyLabel,
    description: definition.description,
    render(host, _view, props) {
      const root = createRoot(host);
      const embed = (props ?? {}) as EmbedProps;
      root.render(embed.mode === "summary" ? <Summary ctx={ctx} definition={definition} /> : <Configuration ctx={ctx} definition={definition} embed={embed} />);
      return () => root.unmount();
    },
  }));
}

/** 设置状态的一行说法；列表卡片与表单页脚共用，避免两处措辞分叉。 */
export function statusLabel(status: Status): {text: string; tone: "on" | "off" | "attention"} {
  if (status.blocked) return {text: "暂不可用", tone: "off"};
  if (status.enabled === true) return status.ready ? {text: "已开启", tone: "on"} : {text: "需要设置", tone: "attention"};
  return {text: status.enabled === false ? "已关闭" : "未开启", tone: "off"};
}

// 插件列表卡片上的一行状态：只读当前配置，点进卡片才进入完整表单。
function Summary({ctx, definition}: {ctx: WebHostContextV1; definition: FormDefinition}) {
  const [label, setLabel] = useState<ReturnType<typeof statusLabel> | null>(null);
  useEffect(() => {
    let alive = true;
    const load = () => { void request<Status>(ctx, `/api/dashboard/${definition.id}/config`)
      .then(next => { if (alive) setLabel(statusLabel(next)); })
      .catch(() => { if (alive) setLabel({text: "状态读取失败", tone: "attention"}); }); };
    load();
    window.addEventListener("focus", load);
    return () => { alive = false; window.removeEventListener("focus", load); };
  }, [ctx, definition.id]);
  return <span className={`config-status is-${label?.tone ?? "off"}`}>{label?.text ?? "…"}</span>;
}

function Configuration({ctx, definition, embed}: {ctx: WebHostContextV1; definition: FormDefinition; embed: EmbedProps}) {
  const [status, setStatus] = useState<Status | null>(null);
  const [enabled, setEnabled] = useState<boolean | null>(null);
  const [values, setValues] = useState<Record<string, unknown>>({});
  const [dirty, setDirty] = useState(false);
  // 嵌入方预选“开启”产生的草稿只让保存可点，不算用户改动，不触发离开守卫。
  const [preset, setPreset] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [leave, setLeave] = useState<(() => void) | null>(null);
  const path = `/api/dashboard/${definition.id}/config`;
  const alive = useRef(true);
  const article = useRef<HTMLElement>(null);
  const draftEditing = useRef(dirty); draftEditing.current = dirty;
  const editing = useRef(false);
  const edits = useRef(0);
  const sentEdit = useRef<number | null>(null);
  editing.current = dirty || busy;
  const markDirty = (next: boolean): void => {
    // 事件内即刻保护草稿，异步响应不能抢在 React 提交前覆盖它。
    draftEditing.current = next; editing.current = next || busy; setDirty(next);
  };
  const loads = useRef(0);
  const pendingKey = `config-request:${definition.id}`;
  const lastKey = `config-last-request:${definition.id}`;
  const polling = useRef<AbortController | null>(null);
  const settled = useRef<string | null>(null);
  const needsRebind = useRef(false);
  const inFlightRequest = useRef<string | null>(null);
  const verifyRequired = useRef<string | null>(null);
  const read = <T,>(url: string, init?: RequestInit): Promise<T> => request<T>(ctx, url, init, response => {
    if (response.headers.get("X-Akashic-Web-Rebound") === "1") needsRebind.current = true;
  });
  const poll = async (id: string): Promise<void> => {
    const wasPending = sessionStorage.getItem(pendingKey) === id;
    polling.current?.abort();
    const controller = new AbortController(); polling.current = controller;
    const current = (): boolean => alive.current && !controller.signal.aborted && (sessionStorage.getItem(pendingKey) ?? sessionStorage.getItem(lastKey)) === id;
    for (let attempt = 0; attempt < 30 && current(); attempt += 1) {
      try {
        const receipt = await read<{state: string; error: string; selected?: boolean}>(`${path}/receipts/${id}`, {signal: controller.signal});
        if (!current()) return;
        if (receipt.state === "active" || receipt.state === "superseded") {
          if (inFlightRequest.current === id) inFlightRequest.current = null;
          if (wasPending) verifyRequired.current = id;
          sessionStorage.setItem(lastKey, id);
          if (sessionStorage.getItem(pendingKey) === id) sessionStorage.removeItem(pendingKey);
          setNotice(receipt.state === "active" ? "配置已生效，正在更新界面…" : "已有更新的配置生效，正在同步最新状态…");
          if (wasPending) setBusy(true);
          if (sentEdit.current !== null && sentEdit.current === edits.current) { markDirty(false); embed.dirty?.(false); }
          let refreshed = false;
          try {
            const sequence = ++loads.current;
            const next = await read<Status>(path, {signal: controller.signal});
            if (current() && sequence === loads.current) {
              refreshed = true;
              if (verifyRequired.current === id) verifyRequired.current = null;
              if (!draftEditing.current) { setStatus(next); setEnabled(next.enabled); setValues(next.values); }
            }
          } catch (reason) {
            if (current()) setError(`配置结果已确认，但最新表单读取失败：${reason instanceof Error ? reason.message : String(reason)}`);
          }
          if (!current()) return;
          setBusy((verifyRequired.current === id && !refreshed) || (needsRebind.current && !draftEditing.current));
          setNotice(draftEditing.current ? "上一操作已生效，当前修改尚未保存；建议重新加载以核对最新配置。" : needsRebind.current ? "配置已生效，正在刷新设置…" : receipt.state === "active" ? "配置已生效" : "已有更新的配置生效，已显示最新状态");
          if (settled.current !== id) { settled.current = id; embed.changed?.(); }
          return;
        }
        if (receipt.state === "failed") {
          if (inFlightRequest.current === id) inFlightRequest.current = null;
          if (wasPending) verifyRequired.current = id;
          sessionStorage.setItem(lastKey, id);
          settled.current = id;
          if (wasPending) setBusy(true);
          if (sentEdit.current !== null && sentEdit.current === edits.current) {
            markDirty(!receipt.selected); embed.dirty?.(!receipt.selected);
          }
          setNotice("");
          setError(`${receipt.selected ? "配置已保存，但生效失败" : "配置保存失败"}：${receipt.error || "请检查后重试"}`);
          sessionStorage.removeItem(pendingKey);
          let refreshed = false;
          try {
            const sequence = ++loads.current;
            const next = await read<Status>(path, {signal: controller.signal});
            if (current() && sequence === loads.current) {
              refreshed = true;
              if (verifyRequired.current === id) verifyRequired.current = null;
              if (!draftEditing.current) { setStatus(next); setEnabled(next.enabled); setValues(next.values); }
            }
          } catch (reason) { if (current()) setNotice(`原操作失败已确认，但实际配置暂未核对：${reason instanceof Error ? reason.message : String(reason)}`); }
          if (current()) setBusy((verifyRequired.current === id && !refreshed) || (needsRebind.current && !draftEditing.current));
          return;
        }
        setNotice("配置已受理，正在等待新配置生效…");
        if (sentEdit.current !== null && sentEdit.current === edits.current) { markDirty(false); embed.dirty?.(false); }
      } catch (reason) {
        if (!current()) return;
        if (reason instanceof RequestError && [401, 403].includes(reason.status)) {
          setBusy(inFlightRequest.current !== null || verifyRequired.current === id); setError(`原操作回执无法读取：${reason.message}`); return;
        }
        if (reason instanceof RequestError && reason.status === 404) {
          setNotice("未找到操作记录，结果待确认。建议刷新后再试。");
          setBusy(inFlightRequest.current !== null || verifyRequired.current === id); return;
        }
        setNotice("正在确认配置状态…");
      }
      await new Promise<void>(resolve => {
        const finish = (): void => { window.clearTimeout(timer); controller.signal.removeEventListener("abort", finish); resolve(); };
        const timer = window.setTimeout(finish, 500);
        controller.signal.addEventListener("abort", finish, {once: true});
      });
    }
    if (current()) { setBusy(inFlightRequest.current !== null || verifyRequired.current === id); setNotice("配置应用结果待确认，请刷新页面核对，系统不会重复提交。"); }
  };
  const load = async (preserveDraft = false): Promise<void> => {
    const sequence = ++loads.current;
    try {
      const next = await read<Status>(path);
      if (!alive.current || sequence !== loads.current || (preserveDraft && editing.current)) return;
      const preset = embed.intent === "enable" && next.enabled !== true && next.can_enable !== false;
      setStatus(next); setEnabled(preset ? true : next.enabled); setValues(next.values); markDirty(preset); setPreset(preset); setError("");
      const pending = sessionStorage.getItem(pendingKey) ?? sessionStorage.getItem(lastKey);
      if (pending) void poll(pending);
    } catch (reason) { if (alive.current && sequence === loads.current) setError(reason instanceof Error ? reason.message : String(reason)); }
  };
  useEffect(() => {
    alive.current = true;
    let visible = false;
    // Shell 保留隐藏页面；每次真正打开时重读前置，不能沿用启动时的状态。
    const refresh = () => { if (visible && !editing.current) void load(true); };
    const observer = new IntersectionObserver(entries => {
      visible = entries[0].isIntersecting;
      refresh();
    });
    observer.observe(article.current!);
    window.addEventListener("focus", refresh);
    return () => { alive.current = false; polling.current?.abort(); loads.current += 1; observer.disconnect(); window.removeEventListener("focus", refresh); };
  }, [ctx, path]);
  const edited = dirty && !preset;
  useEffect(() => { embed.dirty?.(edited); return () => embed.dirty?.(false); }, [edited, embed.dirty]);
  useEffect(() => {
    if (!edited && !busy) return;
    const unload = (event: BeforeUnloadEvent) => { event.preventDefault(); };
    const navigate = (event: Event) => {
      const detail = (event as CustomEvent<{go: () => void; reason?: string}>).detail;
      if (detail.reason === "catalog" && needsRebind.current && !draftEditing.current && sessionStorage.getItem(lastKey)) return;
      event.preventDefault(); if (detail.reason === "catalog") return; if (busy) { setNotice("配置正在生效中，请稍候；离开不会取消提交。"); return; } setLeave(() => (event as CustomEvent<{go: () => void}>).detail.go); };
    window.addEventListener("beforeunload", unload); window.addEventListener("akashic:before-navigate", navigate);
    return () => { window.removeEventListener("beforeunload", unload); window.removeEventListener("akashic:before-navigate", navigate); };
  }, [edited, busy]);
  const change = (key: string, value: unknown) => { edits.current += 1; setValues(previous => ({...previous, [key]: value})); markDirty(true); setPreset(false); setNotice(""); };
  const save = async (event: React.FormEvent): Promise<void> => {
    event.preventDefault(); if (!status || enabled === null || inFlightRequest.current !== null || busy) return;
    sentEdit.current = edits.current;
    const previous = sessionStorage.getItem(pendingKey);
    const id = previous && settled.current !== previous ? previous : crypto.randomUUID();
    const reusedPendingId = id === previous;
    inFlightRequest.current = id;
    // 发送前只保存非敏感操作 ID；响应丢失或模块撤回后仍可查原回执。
    sessionStorage.setItem(pendingKey, id);
    setBusy(true); setError(""); setNotice("正在校验并应用配置…");
    window.dispatchEvent(new CustomEvent("akashic:configuration-submitted"));
    try {
      const receipt = await request<{request_id: string; state: string; error: string}>(ctx, path, {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({request_id: id, expected_input: status.input_ref, values: {...values, enabled}}),
      });
      if (!alive.current || inFlightRequest.current !== id) return;
      if (receipt.request_id !== id) throw new Error("配置回执身份不一致");
      inFlightRequest.current = null;
      markDirty(false); embed.dirty?.(false);
      setNotice("配置已提交，正在生效…");
      await poll(id);
    } catch (reason) {
      if (!alive.current || inFlightRequest.current !== id) return;
      inFlightRequest.current = null;
      if (reason instanceof RequestError && [401, 403, 404, 409, 422].includes(reason.status)) {
        if (reusedPendingId) {
          sentEdit.current = null;
          setNotice("本次提交未被接受，请检查配置后重试。");
        } else { sessionStorage.removeItem(pendingKey); setNotice(""); }
        setError(reason.message); setBusy(false);
      } else {
        setNotice("正在确认配置生效状态，请稍候…");
        await poll(id);
      }
    } finally { if (inFlightRequest.current === id) inFlightRequest.current = null; }
  };
  return <article ref={article} className={`config-form ${embed.embedded ? "is-embedded" : ""}`} aria-busy={busy}>
    {!embed.embedded && <header><span className="config-kicker">功能设置</span><h1>{definition.title}</h1><p>{definition.description}</p></header>}
    {error && <div className="config-error" role="alert"><p>{error}</p><button type="button" disabled={busy && !sessionStorage.getItem(pendingKey) && !sessionStorage.getItem(lastKey)} onClick={() => { const id = sessionStorage.getItem(pendingKey) ?? sessionStorage.getItem(lastKey); if (busy && id) { void poll(id); return; } if (dirty) setLeave(() => () => { void load(); }); else void load(); }}>重新加载</button></div>}
    {!status ? !error && <p role="status">正在读取配置…</p> : <form onSubmit={event => void save(event)}>
      {/* 引导（intent="enable"）自己说明阻塞原因；设置详情页照常显示原因与禁用的开关。 */}
      {status.reason && !(embed.intent === "enable" && status.blocked) && <p className="config-hint" role="status">{status.reason}</p>}
      {!(embed.intent === "enable" && status.blocked) && <>
        {!(embed.intent === "enable" && status.can_enable !== false) && <label className="config-toggle">
          <span><strong>启用{/^[A-Za-z0-9]/.test(definition.title) ? " " : ""}{definition.title}</strong><small>{status.can_enable === false ? "前置条件满足后才能开启" : "关闭后停用，已有数据保留"}</small></span>
          <input type="checkbox" role="switch" checked={enabled === true} disabled={busy || (enabled !== true && status.can_enable === false)}
            onChange={event => { edits.current += 1; setEnabled(event.target.checked); markDirty(true); setPreset(false); }} />
        </label>}
        {enabled === true && definition.fields && <fieldset disabled={busy} className="config-fields"><legend className="sr-only">连接配置</legend>{definition.fields({values, change, status})}</fieldset>}
        <footer className="config-actions"><span>{dirty ? "有未保存的修改" : statusLabel(status).text}</span><button className="config-primary" type="submit" disabled={busy || enabled === null || !dirty}>{busy ? "正在保存…" : "保存配置"}</button></footer>
      </>}
      {notice && <div role="status" className="config-hint">{notice}{sessionStorage.getItem(pendingKey) && <button type="button" onClick={() => { const id = sessionStorage.getItem(pendingKey); if (id) void poll(id); }}>查看状态</button>}</div>}
    </form>}
    {leave && <Confirm title="未保存的修改将丢失" accept={() => { markDirty(false); const go = leave; setLeave(null); go(); }} cancel={() => setLeave(null)}>当前页面的修改尚未保存，离开后将恢复为原有设置。</Confirm>}
  </article>;
}

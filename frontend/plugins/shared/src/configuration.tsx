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
export interface FormDefinition { id: string; title: string; description: string; fields?: (props: FormProps) => ReactNode; }
export interface EmbedProps { embedded?: boolean; changed?: () => void; dirty?: (value: boolean) => void; }
export const settingsIcon = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M5 5h14v14H5zM8 9h8M8 13h5"/></svg>';

export class RequestError extends Error {
  constructor(message: string, readonly status: number) { super(message); }
}

export async function request<T>(ctx: WebHostContextV1, path: string, init?: RequestInit): Promise<T> {
  const response = await ctx.http.request(path, init);
  if (!response.headers.get("content-type")?.includes("application/json")) throw new Error(`服务暂时不可用（${response.status}），请稍后重试`);
  const body = await response.json();
  if (!response.ok) {
    const detail = body.detail ?? body.message;
    throw new RequestError(Array.isArray(detail) ? detail.map((item: {msg: string}) => item.msg).join("；") : typeof detail === "string" ? detail : `请求失败（${response.status}）`, response.status);
  }
  return body as T;
}

export function Confirm({ title, children, accept, cancel }: {title: string; children: ReactNode; accept: () => void; cancel: () => void}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => { const dialog = ref.current!; dialog.showModal(); return () => dialog.close(); }, []);
  return <dialog ref={ref} className="config-dialog" aria-labelledby="config-confirm-title" onCancel={event => { event.preventDefault(); cancel(); }}>
    <h2 id="config-confirm-title">{title}</h2><p>{children}</p>
    <footer><button type="button" autoFocus className="config-primary" onClick={cancel}>继续填写</button><button type="button" onClick={accept}>放弃修改并离开</button></footer>
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
  return ctx.ui.inject("shell.pages.v1", mount => mount.register({
    id: `${definition.id.replaceAll("_", "-")}-settings`, label: definition.title, route: `${definition.id}-settings`, section: "settings", iconSvg: settingsIcon,
    render(host, _view, props) {
      const root = createRoot(host);
      root.render(<Configuration ctx={ctx} definition={definition} embed={(props ?? {}) as EmbedProps} />);
      return () => root.unmount();
    },
  }));
}

function Configuration({ctx, definition, embed}: {ctx: WebHostContextV1; definition: FormDefinition; embed: EmbedProps}) {
  const [status, setStatus] = useState<Status | null>(null);
  const [enabled, setEnabled] = useState<boolean | null>(null);
  const [values, setValues] = useState<Record<string, unknown>>({});
  const [dirty, setDirty] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [leave, setLeave] = useState<(() => void) | null>(null);
  const path = `/api/dashboard/${definition.id}/config`;
  const alive = useRef(true);
  const article = useRef<HTMLElement>(null);
  const editing = useRef(false);
  const edits = useRef(0);
  const sentEdit = useRef<number | null>(null);
  editing.current = dirty || busy;
  const loads = useRef(0);
  const pendingKey = `config-request:${definition.id}`;
  const polling = useRef<AbortController | null>(null);
  const settled = useRef<string | null>(null);
  const poll = async (id: string): Promise<void> => {
    polling.current?.abort();
    const controller = new AbortController(); polling.current = controller;
    const current = (): boolean => alive.current && !controller.signal.aborted && sessionStorage.getItem(pendingKey) === id;
    for (let attempt = 0; attempt < 30 && current(); attempt += 1) {
      try {
        const receipt = await request<{state: string; error: string}>(ctx, `${path}/receipts/${id}`, {signal: controller.signal});
        if (!current()) return;
        if (receipt.state === "active" || receipt.state === "superseded") {
          setNotice(receipt.state === "active" ? "配置已生效" : "原操作已被较新的配置替代，当前显示最新状态");
          setBusy(false);
          if (sentEdit.current !== null && sentEdit.current === edits.current) { setDirty(false); embed.dirty?.(false); }
          try {
            const sequence = ++loads.current;
            const next = await request<Status>(ctx, path, {signal: controller.signal});
            if (current() && sequence === loads.current && !editing.current) {
              setStatus(next); setEnabled(next.enabled); setValues(next.values);
            }
          } catch (reason) {
            if (current()) setError(`配置结果已确认，但最新表单读取失败：${reason instanceof Error ? reason.message : String(reason)}`);
          }
          if (!current()) return;
          if (settled.current !== id) { settled.current = id; embed.changed?.(); }
          return;
        }
        if (receipt.state === "failed") {
          setBusy(false); setDirty(true); setNotice(""); setError(receipt.error || "配置未能生效，请检查后重试");
          sessionStorage.removeItem(pendingKey); return;
        }
        setNotice("配置已受理，正在等待新配置生效…");
        if (sentEdit.current !== null && sentEdit.current === edits.current) { setDirty(false); embed.dirty?.(false); }
      } catch (reason) {
        if (!current()) return;
        if (reason instanceof RequestError && [401, 403].includes(reason.status)) {
          setBusy(false); setError(`原操作回执无法读取：${reason.message}`); return;
        }
        if (reason instanceof RequestError && reason.status === 404) {
          setNotice("尚未找到原操作回执，结果未确认。请重新核对后再决定是否提交。");
          setBusy(false); return;
        }
        setNotice("原操作结果仍在核对，设置服务可能正在更新…");
      }
      await new Promise<void>(resolve => {
        const finish = (): void => { window.clearTimeout(timer); controller.signal.removeEventListener("abort", finish); resolve(); };
        const timer = window.setTimeout(finish, 500);
        controller.signal.addEventListener("abort", finish, {once: true});
      });
    }
    if (current()) { setBusy(false); setNotice("原操作尚未确认；请重新读取以核对回执，不会自动重复提交。"); }
  };
  const load = async (preserveDraft = false): Promise<void> => {
    const sequence = ++loads.current;
    try {
      const next = await request<Status>(ctx, path);
      if (!alive.current || sequence !== loads.current || (preserveDraft && editing.current)) return;
      setStatus(next); setEnabled(next.enabled); setValues(next.values); setDirty(false); setError("");
      const pending = sessionStorage.getItem(pendingKey);
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
  useEffect(() => { embed.dirty?.(dirty); return () => embed.dirty?.(false); }, [dirty, embed.dirty]);
  useEffect(() => {
    if (!dirty && !busy) return;
    const unload = (event: BeforeUnloadEvent) => { event.preventDefault(); };
    const navigate = (event: Event) => { event.preventDefault(); if (busy) { setNotice("原操作正在核对，请等待结果；离开不会撤销已提交配置。"); return; } setLeave(() => (event as CustomEvent<{go: () => void}>).detail.go); };
    window.addEventListener("beforeunload", unload); window.addEventListener("akashic:before-navigate", navigate);
    return () => { window.removeEventListener("beforeunload", unload); window.removeEventListener("akashic:before-navigate", navigate); };
  }, [dirty, busy]);
  const change = (key: string, value: unknown) => { edits.current += 1; setValues(previous => ({...previous, [key]: value})); setDirty(true); setNotice(""); };
  const saving = useRef(false);
  const save = async (event: React.FormEvent): Promise<void> => {
    event.preventDefault(); if (!status || enabled === null || saving.current || busy) return;
    saving.current = true; sentEdit.current = edits.current;
    const previous = sessionStorage.getItem(pendingKey);
    const id = previous && settled.current !== previous ? previous : crypto.randomUUID();
    // 发送前只保存非敏感操作 ID；响应丢失或模块撤回后仍可查原回执。
    sessionStorage.setItem(pendingKey, id);
    setBusy(true); setError(""); setNotice("正在校验并应用配置…");
    window.dispatchEvent(new CustomEvent("akashic:configuration-submitted"));
    try {
      const receipt = await request<{request_id: string; state: string; error: string}>(ctx, path, {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({request_id: id, expected_input: status.input_ref, values: {...values, enabled}}),
      });
      if (!alive.current) return;
      if (receipt.request_id !== id) throw new Error("配置回执身份不一致");
      setDirty(false); embed.dirty?.(false);
      setNotice("配置已受理，正在等待新配置生效…");
      await poll(id);
    } catch (reason) {
      if (!alive.current) return;
      if (reason instanceof RequestError && [401, 403, 404, 409, 422].includes(reason.status)) {
        sessionStorage.removeItem(pendingKey); setNotice("");
        setError(reason.message); setBusy(false);
      } else {
        setNotice("提交响应未确认，正在查询原操作回执；不会自动重复提交。");
        await poll(id);
      }
    } finally { saving.current = false; }
  };
  return <article ref={article} className={`config-form ${embed.embedded ? "is-embedded" : ""}`} aria-busy={busy}>
    {!embed.embedded && <header><span className="config-kicker">功能设置</span><h1>{definition.title}</h1><p>{definition.description}</p></header>}
    {error && <div className="config-error" role="alert"><p>{error}</p><button type="button" disabled={busy} onClick={() => { if (dirty) setLeave(() => () => { void load(); }); else void load(); }}>重新读取</button></div>}
    {!status ? !error && <p role="status">正在读取配置…</p> : <form onSubmit={event => void save(event)}>
      {status.reason && !(embed.embedded && status.blocked) && <p className="config-hint" role="status">{status.reason}</p>}
      {!(embed.embedded && status.blocked) && <>
        <fieldset className="config-choices" disabled={busy}><legend>是否开启{definition.title}？</legend>
          <label className={enabled === true ? "is-selected" : ""}><input type="radio" name="enabled" checked={enabled === true} disabled={status.can_enable === false} onChange={() => { edits.current += 1; setEnabled(true); setDirty(true); }} /><strong>开启</strong><span>配置并使用此功能</span></label>
          <label className={enabled === false ? "is-selected" : ""}><input type="radio" name="enabled" checked={enabled === false} onChange={() => { edits.current += 1; setEnabled(false); setDirty(true); }} /><strong>关闭</strong><span>保留已有配置和数据</span></label>
        </fieldset>
        {enabled === true && definition.fields && <fieldset disabled={busy} className="config-fields"><legend className="sr-only">连接配置</legend>{definition.fields({values, change, status})}</fieldset>}
        <footer className="config-actions"><span>{!dirty && (status.enabled === false ? "已关闭" : status.ready ? "已开启" : "尚未完成配置")}</span><button className="config-primary" type="submit" disabled={busy || enabled === null || !dirty}>{busy ? "正在应用…" : "保存配置"}</button></footer>
      </>}
      {notice && <div role="status" className="config-hint">{notice}{!busy && sessionStorage.getItem(pendingKey) && <button type="button" onClick={() => { const id = sessionStorage.getItem(pendingKey); if (id) void poll(id); }}>核对原操作</button>}</div>}
    </form>}
    {leave && <Confirm title="放弃尚未保存的修改？" accept={() => { setDirty(false); const go = leave; setLeave(null); go(); }} cancel={() => setLeave(null)}>本页修改还没有保存，已有配置保持不变。</Confirm>}
  </article>;
}

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

export async function request<T>(ctx: WebHostContextV1, path: string, init?: RequestInit): Promise<T> {
  const response = await ctx.http.request(path, init);
  if (!response.headers.get("content-type")?.includes("application/json")) throw new Error(`服务暂时不可用（${response.status}），请稍后重试`);
  const body = await response.json();
  if (!response.ok) {
    const detail = body.detail ?? body.message;
    throw new Error(Array.isArray(detail) ? detail.map((item: {msg: string}) => item.msg).join("；") : typeof detail === "string" ? detail : `请求失败（${response.status}）`);
  }
  return body as T;
}

export function Confirm({ title, children, accept, cancel }: {title: string; children: ReactNode; accept: () => void; cancel: () => void}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => { const dialog = ref.current!; dialog.showModal(); return () => dialog.close(); }, []);
  return <dialog ref={ref} className="config-dialog" aria-labelledby="config-confirm-title" onCancel={event => { event.preventDefault(); cancel(); }}>
    <h2 id="config-confirm-title">{title}</h2><p>{children}</p>
    <footer><button type="button" autoFocus onClick={cancel}>继续填写</button><button className="config-primary" type="button" onClick={accept}>放弃修改并离开</button></footer>
  </dialog>;
}

export function Field({label, name, value, change, type = "text", hint, required = false}: {
  label: string; name: string; value: unknown; change: FormProps["change"]; type?: string; hint?: string; required?: boolean;
}) {
  return <label className="config-field"><span>{label}</span><input name={name} type={type} autoComplete={type === "password" ? "new-password" : "off"} required={required}
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
  editing.current = dirty || busy;
  const load = async () => {
    try { const next = await request<Status>(ctx, path); if (!alive.current) return;
      setStatus(next); setEnabled(next.enabled); setValues(next.values); setDirty(false); setError("");
      const pending = sessionStorage.getItem(`config-request:${definition.id}`);
      if (pending) {
        const receipt = await request<{state: string; error: string}>(ctx, `${path}/receipts/${pending}`);
        if (receipt.state === "active") { setNotice("配置已生效"); embed.changed?.(); }
        else if (receipt.state === "failed") { setError(receipt.error || "配置未能生效，请检查后重试"); sessionStorage.removeItem(`config-request:${definition.id}`); }
        else if (receipt.state === "superseded") { setNotice("已读取较新的配置"); sessionStorage.removeItem(`config-request:${definition.id}`); }
        else setNotice("配置仍在应用中，请稍后刷新状态");
      }
    } catch (reason) { if (alive.current) setError(reason instanceof Error ? reason.message : String(reason)); }
  };
  useEffect(() => {
    alive.current = true;
    let visible = false;
    // Shell 保留隐藏页面；每次真正打开时重读前置，不能沿用启动时的状态。
    const refresh = () => { if (visible && !editing.current) void load(); };
    const observer = new IntersectionObserver(entries => {
      visible = entries[0].isIntersecting;
      refresh();
    });
    observer.observe(article.current!);
    window.addEventListener("focus", refresh);
    return () => { alive.current = false; observer.disconnect(); window.removeEventListener("focus", refresh); };
  }, [ctx, path]);
  useEffect(() => { embed.dirty?.(dirty); return () => embed.dirty?.(false); }, [dirty, embed.dirty]);
  useEffect(() => {
    if (!dirty) return;
    const unload = (event: BeforeUnloadEvent) => { event.preventDefault(); };
    const navigate = (event: Event) => { event.preventDefault(); setLeave(() => (event as CustomEvent<{go: () => void}>).detail.go); };
    window.addEventListener("beforeunload", unload); window.addEventListener("akashic:before-navigate", navigate);
    return () => { window.removeEventListener("beforeunload", unload); window.removeEventListener("akashic:before-navigate", navigate); };
  }, [dirty]);
  const change = (key: string, value: unknown) => { setValues(previous => ({...previous, [key]: value})); setDirty(true); setNotice(""); };
  const save = async (event: React.FormEvent) => {
    event.preventDefault(); if (!status || enabled === null || busy) return;
    setBusy(true); setError(""); setNotice("正在校验并应用配置…");
    try {
      const receipt = await request<{request_id: string; state: string; error: string}>(ctx, path, {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({request_id: crypto.randomUUID(), expected_input: status.input_ref, values: {...values, enabled}}),
      });
      if (!alive.current) return;
      if (receipt.state === "failed") throw new Error(receipt.error);
      setDirty(false); embed.dirty?.(false);
      // 配置换代会撤回旧模块；新模块从正式输入恢复，不在旧页面伪报成功。
      setNotice("配置已受理，正在等待新配置生效…");
      sessionStorage.setItem(`config-request:${definition.id}`, receipt.request_id);
      window.dispatchEvent(new CustomEvent("akashic:configuration-submitted"));
      for (let attempt = 0; attempt < 30 && alive.current; attempt += 1) {
        await new Promise(resolve => window.setTimeout(resolve, 500));
        if (!alive.current) return;
        const result = await request<{state: string; error: string}>(ctx, `${path}/receipts/${receipt.request_id}`);
        if (result.state === "failed") { setDirty(true); embed.dirty?.(true); throw new Error(result.error); }
        if (result.state === "active") { await load(); setNotice("配置已生效"); embed.changed?.(); return; }
      }
      if (alive.current) setNotice("配置仍在应用，可刷新查看实际状态。");
    } catch (reason) { if (alive.current) { setNotice(""); setError(reason instanceof Error ? reason.message : String(reason)); } }
    finally { if (alive.current) setBusy(false); }
  };
  return <article ref={article} className={`config-form ${embed.embedded ? "is-embedded" : ""}`} aria-busy={busy}>
    {!embed.embedded && <header><span className="config-kicker">功能设置</span><h1>{definition.title}</h1><p>{definition.description}</p></header>}
    {error && <div className="config-error" role="alert"><p>{error}</p><button type="button" disabled={busy} onClick={() => { if (dirty) setLeave(() => () => { void load(); }); else void load(); }}>重新读取</button></div>}
    {!status ? !error && <p role="status">正在读取配置…</p> : <form onSubmit={event => void save(event)}>
      {status.reason && !(embed.embedded && status.blocked) && <p className="config-hint" role="status">{status.reason}</p>}
      {!(embed.embedded && status.blocked) && <>
        <fieldset className="config-choices" disabled={busy}><legend>是否开启{definition.title}？</legend>
          <label className={enabled === true ? "is-selected" : ""}><input type="radio" name="enabled" checked={enabled === true} disabled={status.can_enable === false} onChange={() => { setEnabled(true); setDirty(true); }} /><strong>开启</strong><span>配置并使用此功能</span></label>
          <label className={enabled === false ? "is-selected" : ""}><input type="radio" name="enabled" checked={enabled === false} onChange={() => { setEnabled(false); setDirty(true); }} /><strong>关闭</strong><span>保留已有配置和数据</span></label>
        </fieldset>
        {enabled === true && definition.fields && <fieldset disabled={busy} className="config-fields"><legend className="sr-only">连接配置</legend>{definition.fields({values, change, status})}</fieldset>}
        <footer className="config-actions"><span>{!dirty && (status.enabled === false ? "已关闭" : status.ready ? "已开启" : "尚未完成配置")}</span><button className="config-primary" type="submit" disabled={busy || enabled === null || !dirty}>{busy ? "正在应用…" : "保存配置"}</button></footer>
      </>}
      {notice && <p role="status" className="config-hint">{notice}</p>}
    </form>}
    {leave && <Confirm title="放弃尚未保存的修改？" accept={() => { setDirty(false); const go = leave; setLeave(null); go(); }} cancel={() => setLeave(null)}>本页修改还没有保存，已有配置保持不变。</Confirm>}
  </article>;
}

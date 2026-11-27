import { useCallback, useEffect, useRef, useState } from "react";
import { call, type Http, type Task } from "./plugin-installable";
import { kickInstallHold } from "./install-hold";

interface Update { plugin_id: string; name: string; source: string; version: string; head: string; remote_head: string }
interface Updates { updates: Update[]; unreachable: string[]; official: string }
interface Source { name: string; url: string; official: boolean }

const message = (reason: unknown) => reason instanceof Error ? reason.message : String(reason);
const OUTCOME_FAILED = new Set(["failed", "in-progress", "inactive"]);

// 插件更新条：打开插件页时联网检查一次；按范围更新（全部、只官方、只某个私有源、单个），与安装共用一个任务位。
export function PluginUpdates({ http, active, onDone }: { http: Http; active: boolean; onDone: () => Promise<void> }): React.ReactElement | null {
  const [data, setData] = useState<Updates | null>(null);
  const [error, setError] = useState("");
  const [task, setTask] = useState<Task | null>(null);
  const [busy, setBusy] = useState(false);
  const running = task?.state === "running";
  const seen = useRef(false);

  const check = useCallback(() => call<Updates>(http, "/api/dashboard/plugins/updates")
    .then((body) => { setData(body); setError(""); }, (reason: unknown) => setError(message(reason))), [http]);
  const poll = useCallback(() => call<{ task: Task | null }>(http, "/api/dashboard/plugins/install")
    .then((body) => setTask(body.task), () => undefined), [http]);
  useEffect(() => { if (active) void check(); }, [active, check]);
  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => void poll(), 2000);
    return () => window.clearInterval(timer);
  }, [running, poll]);
  // 更新任务结束：重读插件列表与更新状态。
  useEffect(() => {
    if (seen.current && !running) void Promise.all([onDone(), check()]);
    seen.current = running;
  }, [running, onDone, check]);

  const upgrade = async (source: string | null, names: string[] = []) => {
    setBusy(true);
    setError("");
    try {
      setTask(await call<Task>(http, "/api/dashboard/plugins/upgrade", {
        method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ source, names }) }));
      kickInstallHold();
    } catch (reason) { setError(message(reason)); } finally { setBusy(false); }
  };

  if (!data && !error) return null;
  const updates = data?.updates ?? [];
  const sources = [...new Set(updates.map((item) => item.source))];
  const failed = task && !running ? task.outcomes.filter((item) => OUTCOME_FAILED.has(item.result)) : [];
  if (!updates.length && !running && !error && !failed.length && !data?.unreachable.length) return null;
  const disabled = running || busy;
  return <section className="shell-inventory-group shell-updates" aria-label="插件更新">
    {updates.length > 0 && <div className="shell-updates__bar">
      <p><strong>{updates.length} 个插件有更新</strong>
        {sources.length > 1 && <span>{sources.map((name) => `${name === data?.official ? "官方" : name} ${updates.filter((item) => item.source === name).length}`).join("，")}</span>}</p>
      <div className="shell-updates__actions">
        <button type="button" className="shell-setup-primary" disabled={disabled} onClick={() => void upgrade(null)}>全部更新</button>
        {sources.length > 1 && sources.map((name) => <button key={name} type="button" disabled={disabled} onClick={() => void upgrade(name)}>
          {name === data?.official ? "只更新官方" : `只更新 ${name}`}</button>)}
      </div>
    </div>}
    {running && <p className="shell-installable-note" role="status">正在更新{task?.name ? `：${task.name}` : ""}。更新在后台进行，装完自动换上，不打断正在进行的对话。</p>}
    {error && <p className="shell-inventory-error" role="alert">{error}</p>}
    {failed.length > 0 && <p className="shell-inventory-error" role="alert">
      {failed.map((item) => `${item.name}：${item.detail || item.result}`).join("；")}</p>}
    {data?.unreachable.map((name) => <p key={name} className="shell-installable-note" role="status">连不上私有源 {name}，它的插件这次没有检查更新。</p>)}
    {updates.length > 0 && <ul className="shell-inventory-rows">
      {updates.map((item) => <li key={item.plugin_id} className="shell-installable-row">
        <span className="shell-inventory-name"><strong>{item.name}{item.source !== data?.official && <span className="shell-source-chip">{item.source}</span>}</strong>
          <small>{item.version} · {item.head.slice(0, 7)} → {item.remote_head.slice(0, 7)}</small></span>
        <button type="button" disabled={disabled} onClick={() => void upgrade(item.source, [item.name])}>更新</button>
      </li>)}
    </ul>}
  </section>;
}

// 「来源」页：官方源固定在最前；私有源可以添加多个，添加时服务端先确认地址真的是插件源。
export function PluginSources({ http, active }: { http: Http; active: boolean }): React.ReactElement {
  const [sources, setSources] = useState<Source[] | null>(null);
  const [error, setError] = useState("");
  const [form, setForm] = useState({ name: "", url: "", token: "" });
  const [busy, setBusy] = useState(false);
  const apply = (body: { sources: Source[] }) => { setSources(body.sources); setError(""); };
  const load = useCallback(() => call<{ sources: Source[] }>(http, "/api/dashboard/plugins/sources")
    .then((body) => { setSources(body.sources); setError(""); }, (reason: unknown) => setError(message(reason))), [http]);
  useEffect(() => { if (active) void load(); }, [active, load]);

  const run = async (init: () => Promise<{ sources: Source[] }>, after?: () => void) => {
    setBusy(true);
    try { apply(await init()); after?.(); } catch (reason) { setError(message(reason)); } finally { setBusy(false); }
  };
  const add = (event: React.FormEvent) => {
    event.preventDefault();
    void run(() => call(http, "/api/dashboard/plugins/sources", {
      method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(form) }),
    () => setForm({ name: "", url: "", token: "" }));
  };
  const remove = (name: string) => {
    if (!window.confirm(`移除私有源 ${name}？已装的插件保留，之后不再随它更新。`)) return;
    void run(() => call(http, `/api/dashboard/plugins/sources/${encodeURIComponent(name)}`, { method: "DELETE" }));
  };
  const field = (key: keyof typeof form) => ({
    value: form[key], onChange: (event: React.ChangeEvent<HTMLInputElement>) => setForm({ ...form, [key]: event.currentTarget.value }),
  });

  return <div className="shell-inventory">
    <p className="shell-installable-note">插件从这些来源安装和更新。官方源总在最前；同名插件按顺序取第一个有它的来源。私有源是和 akashic-plugins 同样结构的仓库，私有仓库也可以。</p>
    {error && <p className="shell-inventory-error" role="alert">{error}</p>}
    {sources && <ul className="shell-inventory-rows">
      {sources.map((item) => <li key={item.name} className="shell-installable-row">
        <span className="shell-inventory-name"><strong>{item.official ? "官方" : item.name}{!item.official && <span className="shell-source-chip">私有</span>}</strong>
          <small>{item.url}</small></span>
        {!item.official && <button type="button" disabled={busy} onClick={() => remove(item.name)}>移除</button>}
      </li>)}
    </ul>}
    <form className="shell-sources-form" onSubmit={add}>
      <h4>添加私有源</h4>
      <label><span>仓库地址</span><input required placeholder="https://github.com/me/akashic-fleet.git" {...field("url")} /></label>
      <label><span>名称<small>插件 ID 的后缀，装过插件后不要改</small></span>
        <input required pattern="[a-z0-9][a-z0-9._\-]{0,39}" placeholder="fleet" {...field("name")} /></label>
      <label><span>只读 token<small>公开仓库不用填；私有仓库用只能读这一个仓库的 token</small></span>
        <input type="password" autoComplete="off" {...field("token")} /></label>
      <button type="submit" className="shell-setup-primary" disabled={busy}>{busy ? "正在确认…" : "添加"}</button>
    </form>
  </div>;
}

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import type { WebUiDisposer } from "@akashic/web-ui-v1";
import { kickInstallHold } from "./install-hold";
import { BAD, call, needText, outcomeText, sizeText, type Group, type Http, type Package, type Task } from "./plugin-installable";

/** 首装推迟到网页时，新手引导内嵌的“选择功能”表单；路由归插件页所有。 */
export const FIRST_INSTALL_ROUTE = "plugins/setup";

interface EmbedHooks { changed?: () => void }
interface Listing { groups: Group[]; first_install: boolean }

// 由 renderSettingsRoute 挂载；引导传入的 changed 在提交与任务结束时回调，让它重读这一步的状态。
export function mountFirstInstall(http: Http, host: HTMLElement, props: unknown): WebUiDisposer {
  const root = createRoot(host);
  root.render(<FirstInstall http={http} changed={(props as EmbedHooks | undefined)?.changed} />);
  return () => root.unmount();
}

// 选中包加上它们（传递）依赖的包，像 pacman 一样自动带上；返回 名字 → 需要它的已选包。
function pulled(packages: Map<string, Package>, selected: ReadonlySet<string>): Map<string, string> {
  const result = new Map<string, string>();
  const stack = [...selected].map(name => [name, name] as const);
  while (stack.length) {
    const [name, owner] = stack.pop()!;
    for (const dep of packages.get(name)?.requires ?? []) {
      if (selected.has(dep) || result.has(dep) || !packages.has(dep)) continue;
      result.set(dep, owner);
      stack.push([dep, owner]);
    }
  }
  return result;
}

function FirstInstall({ http, changed }: { http: Http; changed?: () => void }) {
  const [listing, setListing] = useState<Listing | null>(null);
  const [task, setTask] = useState<Task | null>(null);
  const [error, setError] = useState("");
  const running = task?.state === "running";
  const seen = useRef(false);
  // 任务状态先到先用：换代后回到这一步只看结果，不必等读官方仓库的列表。
  const load = useCallback(async () => {
    try {
      const status = call<{ task: Task | null }>(http, "/api/dashboard/plugins/install").then(body => setTask(body.task));
      const [next] = await Promise.all([call<Listing>(http, "/api/dashboard/plugins/installable"), status]);
      setListing(next); setError("");
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
  }, [http]);
  useEffect(() => { void load(); }, [load]);
  // 1. 装着时每 2 秒读一次进度；2. 结束的那一下重读列表并通知引导。
  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => void call<{ task: Task | null }>(http, "/api/dashboard/plugins/install")
      .then(body => setTask(body.task), () => undefined), 2000);
    return () => window.clearInterval(timer);
  }, [running, http]);
  useEffect(() => {
    if (seen.current && !running) { void load(); changed?.(); }
    seen.current = running;
  }, [running, load, changed]);
  const submit = async (names: string[]) => {
    setError("");
    try {
      if (names.length) setTask(await call<Task>(http, "/api/dashboard/plugins/install", {
        method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ names, first_install: true }) }));
      if (names.length) kickInstallHold();
      else await call(http, "/api/dashboard/plugins/first-install", { method: "DELETE" });
      await load(); changed?.();
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
  };
  // 列表还没到但已有首装任务：先显示进度或结果（重试按钮等列表到了再出现）。
  if (!listing && !error && task?.first_install) return <div className="shell-setup">
    <Progress task={task} groups={[]} retry={submit} />
  </div>;
  // 列表读不到（官方仓库暂时不可达）时仍可重试，或者直接只用核心。
  if (!listing) return error ? <div className="shell-setup">
    <p className="shell-inventory-error" role="alert">读不到扩展包列表：{error}</p>
    <div className="shell-setup-actions">
      <button type="button" className="shell-setup-primary" onClick={() => void load()}>重试</button>
      <button type="button" className="shell-setup-link" onClick={() => void submit([])}>只用核心，不装扩展包</button>
    </div>
  </div> : <p role="status">正在读取官方扩展包…</p>;
  return <div className="shell-setup">
    {error && <p className="shell-inventory-error" role="alert">{error}</p>}
    {listing.first_install && !running ? <Picker groups={listing.groups} submit={submit} />
      : task && (task.first_install || running) ? <Progress task={task} groups={listing.groups} retry={submit} />
      : <p className="shell-installable-note">首次安装的扩展包已经处理过了。以后需要时，在设置的「插件」页里安装。</p>}
  </div>;
}

// 挑选：默认全选；分组可整组勾选，单个包可取消；被已选包依赖的包自动带上并说明原因。
function Picker({ groups, submit }: { groups: Group[]; submit: (names: string[]) => Promise<void> }) {
  const packages = useMemo(() => new Map(groups.flatMap(group => group.packages.map(item => [item.name, item] as const))), [groups]);
  const [selected, setSelected] = useState(() => new Set(packages.keys()));
  const [busy, setBusy] = useState(false);
  const deps = pulled(packages, selected);
  const names = [...packages.keys()].filter(name => selected.has(name) || deps.has(name));
  const size = names.reduce((sum, name) => sum + (packages.get(name)?.size_bytes ?? 0), 0);
  const flip = (items: string[], on: boolean) => setSelected(current => {
    const next = new Set(current); for (const name of items) { if (on) next.add(name); else next.delete(name); } return next;
  });
  const go = async (list: string[]) => { setBusy(true); try { await submit(list); } finally { setBusy(false); } };
  // 操作放在列表上方：直接装全部不用滚到底；下面的清单供核对与取消。
  return <>
    <div className="shell-setup-actions">
      <button type="button" className="shell-setup-primary" disabled={busy || names.length === 0} onClick={() => void go(names)}>
        {names.length ? `安装 ${names.length} 个包` : "没有选择任何包"}</button>
      <button type="button" className="shell-setup-link" disabled={busy} onClick={() => void go([])}>只用核心，不装扩展包</button>
    </div>
    <div className="shell-setup-bar">
      <span>已选 <strong>{names.length}</strong> / {packages.size} 个包{size > 0 && ` · 约 ${sizeText(size)}`}</span>
      <button type="button" className="shell-setup-link" onClick={() => flip([...packages.keys()], selected.size < packages.size)}>
        {selected.size < packages.size ? "全选" : "全不选"}</button>
    </div>
    {groups.map(group => <GroupBlock key={group.id} group={group} selected={selected} deps={deps} flip={flip} />)}
  </>;
}

function GroupBlock({ group, selected, deps, flip }: {
  group: Group; selected: ReadonlySet<string>; deps: ReadonlyMap<string, string>; flip: (items: string[], on: boolean) => void;
}) {
  const members = group.packages.map(item => item.name);
  const count = members.filter(name => selected.has(name)).length;
  const size = group.packages.reduce((sum, item) => sum + item.size_bytes, 0);
  return <fieldset className="shell-setup-group">
    <legend>
      <label><input type="checkbox" checked={count === members.length} ref={box => { if (box) box.indeterminate = count > 0 && count < members.length; }}
        onChange={event => flip(members, event.currentTarget.checked)} />
        <strong>{group.title}</strong><em>推荐</em><small>{members.length} 个包{size > 0 && ` · ${sizeText(size)}`}</small></label>
      {group.description && <p>{group.description}</p>}
    </legend>
    <ul>
      {group.packages.map(item => {
        const owner = deps.get(item.name);
        const meta = [sizeText(item.size_bytes), ...item.needs.map(need => `需要 ${needText(need)}，不满足时装上但保持关闭`),
          ...(item.requires.length ? [`依赖 ${item.requires.join("、")}`] : [])].filter(Boolean);
        return <li key={item.name}><label>
          <input type="checkbox" checked={selected.has(item.name) || !!owner} disabled={!!owner && !selected.has(item.name)}
            onChange={event => flip([item.name], event.currentTarget.checked)} />
          <span><code>{item.name}</code><small>{item.desc}</small>
            {meta.length > 0 && <small>{meta.join(" · ")}</small>}
            {owner && !selected.has(item.name) && <small className="shell-setup-pulled">{owner} 需要它，会一并安装</small>}</span>
        </label></li>;
      })}
    </ul>
  </fieldset>;
}

// 进度与结果：只显示真实事件（第几个、正在装谁）；结束后逐包列出结果，失败的可以重试，日志随时可看。
function Progress({ task, groups, retry }: { task: Task; groups: Group[]; retry: (names: string[]) => Promise<void> }) {
  const running = task.state === "running";
  const step = task.progress;
  const available = new Set(groups.flatMap(group => group.packages.map(item => item.name)));
  const failed = task.outcomes.filter(item => BAD.has(item.result));
  // 失败的排在最前，长清单里不用滚动去找。
  const outcomes = [...failed, ...task.outcomes.filter(item => !BAD.has(item.result))];
  const again = task.names.filter(name => available.has(name));
  return <div className="shell-setup-progress" aria-live="polite">
    {running ? <>
      <p><strong>{step ? `正在安装（${step.done + 1}/${step.total}）${step.current}` : "正在准备安装…"}</strong></p>
      <progress max={step?.total ?? 1} value={step?.done ?? 0} aria-label="安装进度" />
      <p className="shell-installable-note">按依赖顺序逐个安装，装好的功能会在这一步结束后出现。也可以先关掉引导去聊天，安装不会中断。</p>
    </> : <p><strong>{failed.length ? `${failed.length} 个包没有装好` : task.state === "succeeded" ? `已安装 ${task.outcomes.length} 个包` : "安装结果未知，请到「插件」页查看"}</strong></p>}
    {task.outcomes.length > 0 && <ul className="shell-setup-results">
      {outcomes.map(item => <li key={item.name} className={BAD.has(item.result) ? "is-bad" : ""}>
        <code>{item.name}</code><span>{BAD.has(item.result) ? `失败：${item.detail}` : outcomeText(item) || (item.result === "restart-required" ? "需要重启实例后生效" : "已安装")}</span>
      </li>)}
    </ul>}
    {!running && again.length > 0 && <button type="button" className="shell-setup-primary" onClick={() => void retry(again)}>重试没装上的 {again.length} 个包</button>}
    <details><summary>安装日志</summary><pre className="shell-installable-log">{task.tail.join("\n")}</pre></details>
  </div>;
}

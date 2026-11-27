import { useCallback, useEffect, useRef, useState } from "react";
import type { WebHostContextV1 } from "@akashic/web-ui-v1";
import { kickInstallHold } from "./install-hold";

export interface Package { name: string; desc: string; requires: string[]; needs: string[]; size_bytes: number }
export interface Group { id: string; title: string; description: string; packages: Package[] }
export interface Outcome { name: string; result: string; detail: string }
export interface Task {
  name: string; names: string[]; state: "running" | "succeeded" | "failed" | "unknown"; exit_code: number | null; first_install: boolean;
  outcomes: Outcome[]; progress: { done: number; total: number; current: string } | null; tail: string[];
}
export type Http = WebHostContextV1["http"];

const UNMET = "已安装但未启用";
export const BAD = new Set(["failed", "in-progress", "needs-switch", "inactive"]);
export const needText = (need: string) => need === "docker" ? "Docker" : `命令 ${need.replace(/^command:/, "")}`;
export const sizeText = (bytes: number) => bytes <= 0 ? "" : bytes < 52429 ? "小于 0.1 MB" : `${(bytes / 1048576).toFixed(1)} MB`;
// 一包的结果说明：装上但因前置条件保持关闭时，给出原因。
export const outcomeText = (item: Outcome) => item.detail.includes(UNMET) ? item.detail.slice(item.detail.indexOf(UNMET)) : "";

export async function call<T>(http: Http, path: string, init?: RequestInit): Promise<T> {
  const response = await http.request(path, init);
  const body = await response.json().catch(() => ({})) as { detail?: unknown };
  if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `请求失败（${response.status}）`);
  return body as T;
}

// 安装结果：成功时说明未启用原因或需要重启；仍在进行和结果未知不按失败显示；失败给输出尾部。
function Result({ task }: { task: Task }): React.ReactElement {
  const outcome = task.outcomes[task.outcomes.length - 1];
  const held = outcome ? outcomeText(outcome) : "";
  const note = (text: string) => <p className="shell-installable-note" role="status">{text}</p>;
  if (task.state === "unknown") return note(`${task.name}：结果未知，请查看已装列表。`);
  if (task.state === "failed" && outcome?.result === "in-progress") return note(`${task.name}：仍在进行，稍后查看已装列表。`);
  if (task.state === "succeeded") {
    if (outcome?.result === "restart-required") return note(`${task.name} 已安装，需要重启实例后生效。`);
    return note(held ? `${task.name}：${held}。满足条件后在上方启用。` : `${task.name} 已安装，在上方列表里。`);
  }
  return <div role="alert"><p className="shell-inventory-error">{task.name} 安装失败（退出码 {task.exit_code ?? "无"}）</p>
    <pre className="shell-installable-log">{task.tail.join("\n")}</pre></div>;
}

// 可安装分组：来自 extra 仓库的未装包；一次只装一个，装完刷新已装列表。
export function InstallablePackages({ http, active, onInstalled }: {
  http: Http;
  active: boolean;
  onInstalled: () => Promise<void>;
}): React.ReactElement | null {
  const [groups, setGroups] = useState<Group[] | null>(null);
  const [error, setError] = useState("");
  const [task, setTask] = useState<Task | null>(null);
  const [failure, setFailure] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const running = task?.state === "running";
  const seen = useRef(running);
  // 提交代次：提交安装时和提交返回时各加一，提交之前与提交期间发出的状态读取回来时作废
  // （服务端那时可能还没落盘任务，会读到空），不能覆盖刚提交的任务。
  const epoch = useRef(0);
  // 状态读取串行：上一次还没回来就不再发，避免慢回包互相淘汰。
  const polling = useRef(false);

  const load = useCallback(() => call<{ groups: Group[] }>(http, "/api/dashboard/plugins/installable")
    .then((body) => { setGroups(body.groups); setError(""); }, (reason: unknown) => setError(reason instanceof Error ? reason.message : String(reason))), [http]);
  const poll = useCallback(async () => {
    if (polling.current) return;
    polling.current = true;
    const ticket = epoch.current;
    try {
      const body = await call<{ task: Task | null }>(http, "/api/dashboard/plugins/install");
      if (ticket === epoch.current) setTask(body.task);
    } catch { /* 下一轮再读 */ } finally { polling.current = false; }
  }, [http]);

  // 列表读取较慢，只在变为可见时读一次；同时接上页面重开前已在进行的安装。
  useEffect(() => { if (active) { void load(); void poll(); } }, [active, load, poll]);
  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => void poll(), 2000);
    return () => window.clearInterval(timer);
  }, [running, poll]);
  // 任务结束的瞬间刷新两个列表。
  useEffect(() => {
    if (seen.current && !running) void Promise.all([onInstalled(), load()]);
    seen.current = running;
  }, [running, onInstalled, load]);

  // 提交期间禁用所有安装按钮；被拒绝（已有任务在装）时重新读状态，接上实际在跑的任务。
  const install = async (name: string) => {
    setFailure("");
    setSubmitting(true);
    epoch.current += 1;
    try {
      const started = await call<Task>(http, "/api/dashboard/plugins/install", {
        method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ names: [name] }),
      });
      epoch.current += 1;
      setTask(started);
      kickInstallHold();
    } catch (reason) {
      setFailure(reason instanceof Error ? reason.message : String(reason));
      polling.current = false;
      await poll();
    } finally { setSubmitting(false); }
  };

  if (error) return <section className="shell-inventory-group"><h4>可安装</h4>
    <p className="shell-inventory-error" role="alert">读不到扩展包列表：{error}</p></section>;
  if (groups === null) return null;
  return <>
    {task && task.state !== "running" && <section className="shell-inventory-group"><Result task={task} /></section>}
    {groups.length > 0 && <section className="shell-inventory-group">
      <h4>可安装 <span>{groups.reduce((sum, group) => sum + group.packages.length, 0)}</span></h4>
      {failure && <p className="shell-inventory-error" role="alert">{failure}</p>}
      {groups.map((group) => <div key={group.id} className="shell-installable-group">
        <h5>{group.title}{group.description && <small>{group.description}</small>}</h5>
        <ul className="shell-inventory-rows">
          {group.packages.map((pkg) => {
            const meta = [sizeText(pkg.size_bytes), ...pkg.needs.map((need) => `需要 ${needText(need)}`),
              ...(pkg.requires.length ? [`会一并安装 ${pkg.requires.join("、")}`] : [])].filter(Boolean);
            const mine = running && !!task?.names.includes(pkg.name);
            return <li key={pkg.name} className="shell-installable-row">
              <span className="shell-inventory-name"><strong>{pkg.name}</strong>
                <small>{pkg.desc}</small>{meta.length > 0 && <small>{meta.join(" · ")}</small>}</span>
              <button type="button" disabled={running || submitting} aria-busy={mine} onClick={() => void install(pkg.name)}>
                {mine ? "安装中…" : "安装"}</button>
            </li>;
          })}
        </ul>
      </div>)}
    </section>}
  </>;
}

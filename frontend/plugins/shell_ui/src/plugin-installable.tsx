import { useCallback, useEffect, useRef, useState } from "react";
import type { WebHostContextV1 } from "@akashic/web-ui-v1";

interface Package { name: string; desc: string; requires: string[]; needs: string[]; size_bytes: number }
interface Group { id: string; title: string; description: string; packages: Package[] }
interface Task { name: string; state: "running" | "succeeded" | "failed"; exit_code: number | null; tail: string[] }
type Http = WebHostContextV1["http"];

const UNMET = "已安装但未启用";
const needText = (need: string) => need === "docker" ? "Docker" : `命令 ${need.replace(/^command:/, "")}`;
const sizeText = (bytes: number) => bytes > 0 ? `${(bytes / 1048576).toFixed(1)} MB` : "";

async function call<T>(http: Http, path: string, init?: RequestInit): Promise<T> {
  const response = await http.request(path, init);
  const body = await response.json().catch(() => ({})) as { detail?: unknown };
  if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `请求失败（${response.status}）`);
  return body as T;
}

// 安装结果：成功时若输出带“已安装但未启用：原因”就把原因说出来；失败给输出尾部。
function Result({ task }: { task: Task }): React.ReactElement {
  const held = task.tail.find((line) => line.includes(UNMET));
  if (task.state === "succeeded") {
    return <p className="shell-installable-note" role="status">
      {held ? `${task.name}：${held.slice(held.indexOf(UNMET))}。满足条件后在上方启用。` : `${task.name} 已安装，在上方列表里。`}
    </p>;
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
  const running = task?.state === "running";
  const seen = useRef(running);

  const load = useCallback(() => call<{ groups: Group[] }>(http, "/api/dashboard/plugins/installable")
    .then((body) => { setGroups(body.groups); setError(""); }, (reason: unknown) => setError(reason instanceof Error ? reason.message : String(reason))), [http]);
  const poll = useCallback(() => call<{ task: Task | null }>(http, "/api/dashboard/plugins/install")
    .then((body) => setTask(body.task), () => undefined), [http]);

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

  const install = async (name: string) => {
    setFailure("");
    try { setTask(await call<Task>(http, "/api/dashboard/plugins/install", {
      method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name }),
    })); } catch (reason) { setFailure(reason instanceof Error ? reason.message : String(reason)); }
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
            const mine = running && task?.name === pkg.name;
            return <li key={pkg.name} className="shell-installable-row">
              <span className="shell-inventory-name"><strong>{pkg.name}</strong>
                <small>{pkg.desc}</small>{meta.length > 0 && <small>{meta.join(" · ")}</small>}</span>
              <button type="button" disabled={running} aria-busy={mine} onClick={() => void install(pkg.name)}>
                {mine ? "安装中…" : "安装"}</button>
            </li>;
          })}
        </ul>
      </div>)}
    </section>}
  </>;
}

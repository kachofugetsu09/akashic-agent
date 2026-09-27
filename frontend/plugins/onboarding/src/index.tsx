import { useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import type { WebHostContextV1, WebMountView, WebUiDisposer } from "@akashic/web-ui-v1";
import { Confirm, request, settingsIcon, type Status } from "../../shared/src/configuration";
import "./style.css";

interface Step { id: string; title: string; group: string; group_title: string; route: string; }
interface StepStatus extends Partial<Status> { fault?: string; }
interface Catalog { steps: Step[]; }
function done(status?: StepStatus): boolean { return !!status && !status.fault && (status.ready === true || status.enabled === false || status.blocked === true); }
function label(status?: StepStatus): string {
  if (!status) return "读取中";
  if (status.fault) return "读取失败";
  if (status.blocked) return "前置不可用";
  if (status.enabled === false) return "已关闭";
  if (status.ready) return "已开启";
  return "待配置";
}

export function activate(ctx: WebHostContextV1): WebUiDisposer {
  return ctx.ui.inject("shell.pages.v1", mount => mount.register({
    id: "onboarding", label: "初始配置", route: "onboarding", iconSvg: settingsIcon,
    render(host, _view, props) {
      const pages = (props as {pages: WebMountView}).pages;
      const root = createRoot(host); root.render(<Onboarding ctx={ctx} pages={pages} />);
      return () => root.unmount();
    },
  }));
}

function Onboarding({ctx, pages}: {ctx: WebHostContextV1; pages: WebMountView}) {
  const [steps, setSteps] = useState<Step[]>([]);
  const [states, setStates] = useState<Record<string, StepStatus>>({});
  const [selected, setSelected] = useState(() => sessionStorage.getItem("onboarding-page") ?? "");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [dirty, setDirty] = useState(false);
  const [leave, setLeave] = useState<(() => void) | null>(null);
  const [finished, setFinished] = useState(false);
  const formHost = useRef<HTMLDivElement>(null);
  const heading = useRef<HTMLHeadingElement>(null);
  const invitation = useRef<HTMLDialogElement>(null);
  const alive = useRef(true);
  const refresh = useCallback(async () => {
    try {
      const catalog = await request<Catalog>(ctx, "/api/dashboard/onboarding/catalog");
      const statuses = await Promise.all(catalog.steps.map(async step => {
        try { return [step.id, await request<StepStatus>(ctx, `/api/dashboard/onboarding/status/${encodeURIComponent(step.id)}`)] as const; }
        catch (reason) { return [step.id, {fault: reason instanceof Error ? reason.message : String(reason)}] as const; }
      }));
      if (!alive.current) return;
      const next = Object.fromEntries(statuses);
      setSteps(catalog.steps); setStates(next); setError("");
      setSelected(current => catalog.steps.some(step => step.id === current) ? current : (catalog.steps.find(step => !done(next[step.id])) ?? catalog.steps[0])?.id ?? "");
      if (catalog.steps.length && statuses.every(([, status]) => "enabled" in status && status.enabled === null)
          && !window.location.hash && !sessionStorage.getItem("onboarding-invited")) {
        sessionStorage.setItem("onboarding-invited", "1"); invitation.current?.showModal();
      }
      return {steps: catalog.steps, states: next};
    } catch (reason) { if (alive.current) setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { if (alive.current) setLoading(false); }
    return undefined;
  }, [ctx]);
  useEffect(() => {
    alive.current = true; void refresh();
    const change = () => { void refresh(); };
    window.addEventListener("focus", change);
    return () => { alive.current = false; window.removeEventListener("focus", change); };
  }, [refresh]);
  const changed = useCallback(() => { void refresh(); }, [refresh]);
  const current = steps.find(step => step.id === selected);
  const index = steps.findIndex(step => step.id === selected);
  const state = states[selected];
  const firstPending = steps.findIndex(step => !done(states[step.id]));
  const allDone = steps.length > 0 && firstPending < 0;
  useEffect(() => {
    if (!current || !formHost.current || finished) return;
    sessionStorage.setItem("onboarding-page", current.id);
    const page = pages.entries.find(entry => entry.route === current.route);
    const host = formHost.current;
    if (!page) { host.textContent = "此插件的设置页面尚未就绪，请刷新或检查插件状态。"; return; }
    return pages.render(page.id, host, {embedded: true, changed, dirty: setDirty});
  }, [current?.id, current?.route, pages, finished, changed]);
  const navigate = (go: () => void) => { if (dirty) setLeave(() => go); else go(); };
  const choose = (step: Step) => navigate(() => { setSelected(step.id); setFinished(false); heading.current?.focus(); });
  const next = async () => {
    const fresh = await refresh();
    if (!fresh || !done(fresh.states[selected])) return;
    const at = fresh.steps.findIndex(step => step.id === selected);
    if (at >= 0 && at + 1 < fresh.steps.length) choose(fresh.steps[at + 1]);
    else if (fresh.steps.every(step => done(fresh.states[step.id]))) navigate(() => setFinished(true));
  };
  return <main className="onboarding-page">
    <header className="onboarding-header"><div><span className="config-kicker">开始使用 AKASHIC</span><h1>让它按你的方式工作</h1><p>逐项决定开启或关闭。配置由各功能保存，之后也能随时修改。</p></div><button type="button" disabled={loading || dirty} onClick={() => void refresh()}>刷新状态</button></header>
    {error && <div className="config-error" role="alert">{error}</div>}
    {loading ? <p role="status">正在读取已安装的功能…</p> : !steps.length && !error ? <div className="config-hint">当前没有需要配置的插件。你仍可使用功能设置。</div> : finished && allDone ?
      <section className="onboarding-complete"><h2 tabIndex={-1}>配置已完成</h2><p>已保存你的选择。前置关闭的功能保持不可用，已有数据会保留。</p><ul>{steps.map(step => <li key={step.id}><span>{step.title}</span><strong>{label(states[step.id])}</strong></li>)}</ul><button type="button" onClick={() => setFinished(false)}>查看配置</button><a className="onboarding-chat" href="#">开始对话</a></section> :
      <div className="onboarding-layout"><nav aria-label="配置步骤" className="onboarding-steps">{steps.map((step, i) => <button key={step.id} type="button" aria-current={step.id === selected ? "step" : undefined} disabled={firstPending >= 0 && i > firstPending} onClick={() => choose(step)}>
        <span className="onboarding-number" aria-hidden="true">{done(states[step.id]) ? "✓" : i + 1}</span><span><strong>{step.title}</strong><small>{label(states[step.id])}</small></span>
      </button>)}</nav><section className="onboarding-content" aria-labelledby="onboarding-step-title"><header><span className="config-kicker">{index + 1} / {steps.length} · {current?.group_title}</span><h2 id="onboarding-step-title" ref={heading} tabIndex={-1}>{current?.title}</h2></header>
        {state?.fault && <div role="alert" className="config-error">{state.fault}<button type="button" onClick={() => void refresh()}>重试读取</button></div>}
        {state?.blocked && <p className="config-hint">{state.reason}。此项目前不可开启，可以继续下一步；不会记录为你主动关闭。</p>}
        <div key={current?.id} ref={formHost} />
        <footer className="onboarding-footer"><button type="button" disabled={index <= 0} onClick={() => choose(steps[index - 1])}>上一步</button><span>{dirty ? "请先保存本页选择" : label(state)}</span><button className="config-primary" type="button" disabled={!done(state) || dirty} onClick={() => void next()}>{index === steps.length - 1 ? "查看完成情况" : "下一步"}</button></footer>
      </section></div>}
    {leave && <Confirm title="离开前要放弃修改吗？" accept={() => { setDirty(false); const go = leave; setLeave(null); go(); }} cancel={() => setLeave(null)}>本页尚有未保存的修改。离开不会改变已保存的配置。</Confirm>}
    <dialog ref={invitation} className="config-dialog" aria-labelledby="onboarding-welcome"><h2 id="onboarding-welcome">欢迎使用 Akashic</h2><p>先连接模型，再选择渠道、情景记忆和主动联系。每一项由你决定是否开启。</p><footer><button type="button" onClick={() => invitation.current?.close()}>关闭窗口</button><button autoFocus className="config-primary" type="button" onClick={() => { invitation.current?.close(); window.location.hash = "onboarding"; }}>开始配置</button></footer></dialog>
  </main>;
}

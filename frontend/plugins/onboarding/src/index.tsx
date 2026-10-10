import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { createPortal } from "react-dom";
import type { WebHostContextV1, WebUiDisposer } from "@akashic/web-ui-v1";
import type { RenderSettingsRoute, ShellSettingsRenderProps } from "@akashic/shell-ui-v1";
import { Confirm, request, type EmbedProps, type Status } from "../../shared/src/configuration";
import { akashicBrandIcon } from "../../shell_ui/src/brand";
import "./style.css";

/** 初始配置是引导清单：勾选列表图标，不复用通用设置齿轮。 */
const onboardingIcon = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m3 17 2 2 4-4"/><path d="m3 7 2 2 4-4"/><path d="M13 6h8"/><path d="M13 12h8"/><path d="M13 18h8"/></svg>';
/** 看过引导（完成或“稍后再说”）后不再自动弹出；设置里的“新手引导”仍可随时打开。 */
const SEEN_KEY = "akashic.onboarding.seen";
const brandMask = { WebkitMaskImage: `url(${akashicBrandIcon})`, maskImage: `url(${akashicBrandIcon})` };

interface Step { id: string; title: string; group: string; group_title: string; route: string; }
interface PreviewLine { speaker: string; text: string; }
interface Group { key: string; title: string; required: boolean; pitch: string; benefit: string; preview: PreviewLine[]; }
interface Catalog { steps: Step[]; groups: Group[]; }
interface StepStatus extends Partial<Status> { fault?: string; }
type Statuses = Record<string, StepStatus>;
type Phase = { kind: "welcome" } | { kind: "group"; key: string } | { kind: "abilities" } | { kind: "done" };

function ready(status?: StepStatus): boolean {
  return !!status && !status.fault && status.enabled === true && status.ready === true;
}
function settled(status?: StepStatus): boolean {
  return !!status && !status.fault && (status.ready === true || status.enabled === false || status.blocked === true);
}
function readSeen(): boolean {
  try { return window.localStorage.getItem(SEEN_KEY) === "1"; } catch { return false; }
}
function markSeen(): void {
  try { window.localStorage.setItem(SEEN_KEY, "1"); } catch { /* 存储不可用时下次仍会弹出，不影响使用 */ }
}

async function readCatalog(ctx: WebHostContextV1): Promise<{catalog: Catalog; statuses: Statuses}> {
  const catalog = await request<Catalog>(ctx, "/api/dashboard/onboarding/catalog");
  const entries = await Promise.all(catalog.steps.map(async step => {
    try { return [step.id, await request<StepStatus>(ctx, `/api/dashboard/onboarding/status/${encodeURIComponent(step.id)}`)] as const; }
    catch (reason) { return [step.id, {fault: reason instanceof Error ? reason.message : String(reason)}] as const; }
  }));
  return {catalog, statuses: Object.fromEntries(entries)};
}

// 首次打开且必需能力未就绪时，经 hash 深链接让 Shell 打开本分节；不抢已有的深链接。
async function openOnFirstRun(ctx: WebHostContextV1): Promise<void> {
  if (readSeen() || window.location.hash) return;
  const {catalog, statuses} = await readCatalog(ctx);
  const required = new Set(catalog.groups.filter(group => group.required).map(group => group.key));
  const missing = catalog.steps.some(step => required.has(step.group) && !ready(statuses[step.id]) && !statuses[step.id]?.fault);
  if (missing && !window.location.hash) window.location.hash = "onboarding";
}

export function activate(ctx: WebHostContextV1): WebUiDisposer {
  void openOnFirstRun(ctx).catch(() => { /* 首跑探测失败只是不自动弹出，入口仍在设置里 */ });
  return ctx.ui.inject("shell.settings.v1", mount => mount.register({
    id: "onboarding", label: "新手引导", route: "onboarding", iconSvg: onboardingIcon, order: -10,
    render(host, _view, props) {
      const { pages, renderRoute, close } = (props ?? {}) as ShellSettingsRenderProps;
      // 引导是整屏图层，放进独立容器；容器复用本 entry 的样式归属。
      const surfaceHost = document.createElement("div");
      const releaseStyle = pages.style("onboarding", surfaceHost);
      document.body.appendChild(surfaceHost);
      const root = createRoot(host);
      root.render(<Onboarding ctx={ctx} renderRoute={renderRoute} close={close} surfaceHost={surfaceHost} />);
      return () => { root.unmount(); releaseStyle(); surfaceHost.remove(); };
    },
  }));
}

// 由目录推导流程：欢迎 → 必需分组 → 能力挑选 → 待设置的分组 → 完成。
// planned 在离开能力页时定格，保存成功后不会因分组变为就绪而跳步。
function phasesOf(catalog: Catalog, statuses: Statuses, chosen: ReadonlySet<string>, planned: ReadonlySet<string> | null): Phase[] {
  const optional = catalog.groups.filter(group => !group.required);
  const pending = optional.filter(group => planned ? planned.has(group.key) : chosen.has(group.key) && !groupReady(catalog, statuses, group.key));
  return [
    {kind: "welcome"},
    ...catalog.groups.filter(group => group.required).map(group => ({kind: "group", key: group.key}) as const),
    ...(optional.length ? [{kind: "abilities"} as const] : []),
    ...pending.map(group => ({kind: "group", key: group.key}) as const),
    {kind: "done"},
  ];
}

// 默认选中：未被明确关闭的可选能力；已关闭的保持关闭，尊重之前的决定。
function initialChoice(catalog: Catalog, statuses: Statuses): Set<string> {
  return new Set(catalog.groups.filter(group => !group.required && !catalog.steps
    .filter(step => step.group === group.key).some(step => statuses[step.id]?.enabled === false)).map(group => group.key));
}

// 读取目录与各步状态；打开、保存后与窗口回到前台时重读，不轮询。
function useCatalog(ctx: WebHostContextV1) {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [statuses, setStatuses] = useState<Statuses>({});
  const [chosen, setChosen] = useState<Set<string> | null>(null);
  const [error, setError] = useState("");
  const alive = useRef(true);
  const refresh = useCallback(async () => {
    try {
      const next = await readCatalog(ctx);
      if (!alive.current) return;
      setCatalog(next.catalog); setStatuses(next.statuses); setError("");
      setChosen(current => current ?? initialChoice(next.catalog, next.statuses));
    } catch (reason) {
      if (alive.current) setError(`读取引导失败：${reason instanceof Error ? reason.message : String(reason)}`);
    }
  }, [ctx]);
  useEffect(() => {
    alive.current = true; void refresh();
    const onFocus = () => { void refresh(); };
    window.addEventListener("focus", onFocus);
    return () => { alive.current = false; window.removeEventListener("focus", onFocus); };
  }, [refresh]);
  return {catalog, statuses, chosen, setChosen, error, refresh};
}

function Onboarding({ctx, renderRoute, close, surfaceHost}: {
  ctx: WebHostContextV1; renderRoute?: RenderSettingsRoute; close?: () => void; surfaceHost: HTMLElement;
}) {
  const {catalog, statuses, chosen, setChosen, error, refresh} = useCatalog(ctx);
  const [planned, setPlanned] = useState<Set<string> | null>(null);
  const [index, setIndex] = useState(0);
  const [dirty, setDirty] = useState<Record<string, boolean>>({});
  const [leave, setLeave] = useState<(() => void) | null>(null);
  const phases = useMemo(() => catalog && chosen ? phasesOf(catalog, statuses, chosen, planned) : [{kind: "welcome"} as Phase], [catalog, statuses, chosen, planned]);
  const at = Math.min(index, phases.length - 1);
  const phase = phases[at];
  const editing = Object.values(dirty).some(Boolean);
  const guard = (go: () => void) => { if (editing) setLeave(() => go); else go(); };
  const move = (delta: number) => guard(() => {
    setDirty({});
    // 1. 离开能力页时定格待设置分组；2. 回到能力页时解除定格，允许重新挑选。
    if (phase.kind === "abilities" && delta > 0 && catalog && chosen) {
      const plan = new Set([...chosen].filter(key => !groupReady(catalog, statuses, key)));
      setPlanned(plan);
      setIndex(Math.min(phasesOf(catalog, statuses, chosen, plan).length - 1, at + 1));
      return;
    }
    const target = Math.max(0, Math.min(phases.length - 1, at + delta));
    if (phases[target]?.kind === "abilities") setPlanned(null);
    setIndex(target);
  });
  const finish = () => guard(() => { markSeen(); close?.(); });
  const markDirty = useCallback((route: string, value: boolean) => setDirty(current => current[route] === value ? current : {...current, [route]: value}), []);
  const toggle = (key: string) => setChosen(current => {
    const next = new Set(current); if (next.has(key)) next.delete(key); else next.add(key); return next;
  });
  return <>
    <section className="onboarding-placeholder">
      <h1>新手引导</h1>
      <p>引导已在全屏打开。完成或关闭后会回到对话。</p>
    </section>
    {createPortal(<Surface onCancel={finish}>
      <TopBar phases={phases} at={at} onLater={phase.kind === "welcome" || phase.kind === "done" ? undefined : finish} />
      <div className="onboarding-scroll">
        <div className="onboarding-column" ref={enterStep} key={`${phase.kind}:${phase.kind === "group" ? phase.key : ""}`}>
          {error && <div className="config-error" role="alert"><p>{error}</p><button type="button" onClick={() => void refresh()}>重试</button></div>}
          {!catalog ? !error && <p role="status">正在检查已安装的功能…</p> : <PhaseView phase={phase} catalog={catalog} statuses={statuses}
            chosen={chosen ?? new Set()} toggle={toggle} renderRoute={renderRoute} changed={refresh} markDirty={markDirty}
            start={() => move(1)} skipAll={finish} />}
        </div>
      </div>
      {catalog && phase.kind !== "welcome" && <Footer phase={phase} at={at} catalog={catalog} statuses={statuses} chosen={chosen ?? new Set()}
        editing={editing} back={() => move(-1)} next={() => phase.kind === "done" ? finish() : move(1)} />}
      {leave && <Confirm title="未保存的修改将会丢失，确定离开吗？" accept={() => { const go = leave; setLeave(null); setDirty({}); go(); }}
        cancel={() => setLeave(null)}>当前页包含未保存的内容，离开后修改将不会生效。</Confirm>}
    </Surface>, surfaceHost)}
  </>;
}

// 换步时正文从可见态出发做一次短位移；样式表不能声明 @keyframes，动效走 Web Animations。
function enterStep(element: HTMLDivElement | null): void {
  if (!element || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  element.animate([{opacity: 0.001, transform: "translateY(8px)"}, {opacity: 1, transform: "none"}],
    {duration: 360, easing: "cubic-bezier(0.22, 1, 0.36, 1)"});
}

// 整屏图层：原生模态对话框，压在设置工作区之上；Esc 等同“稍后再说”。
function Surface({children, onCancel}: {children: React.ReactNode; onCancel: () => void}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => { const dialog = ref.current!; dialog.showModal(); return () => dialog.close(); }, []);
  return <dialog ref={ref} className="onboarding-surface" aria-label="Akashic 新手引导"
    onCancel={event => { event.preventDefault(); onCancel(); }}>{children}</dialog>;
}

function TopBar({phases, at, onLater}: {phases: Phase[]; at: number; onLater?: () => void}) {
  const middle = phases.length - 2;
  const showProgress = at > 0 && at < phases.length - 1;
  return <header className="onboarding-top">
    <span className="onboarding-brand"><i className="onboarding-mark" style={brandMask} aria-hidden="true" /><span>Akashic</span></span>
    {showProgress ? <div className="onboarding-progress" role="progressbar" aria-label="引导进度" aria-valuemin={1} aria-valuemax={middle} aria-valuenow={at}>
      {Array.from({length: middle}, (_, i) => <i key={i} className={i + 1 < at ? "is-done" : i + 1 === at ? "is-now" : ""} />)}
    </div> : <span />}
    {onLater ? <button type="button" className="onboarding-text-button" onClick={onLater}>稍后再说</button> : <span />}
  </header>;
}

interface PhaseProps {
  phase: Phase; catalog: Catalog; statuses: Statuses; chosen: ReadonlySet<string>; toggle: (key: string) => void;
  renderRoute?: RenderSettingsRoute; changed: () => void; markDirty: (route: string, value: boolean) => void;
  start: () => void; skipAll: () => void;
}

function PhaseView(props: PhaseProps) {
  const {phase, catalog} = props;
  if (phase.kind === "welcome") return <Welcome catalog={catalog} start={props.start} skip={props.skipAll} />;
  if (phase.kind === "abilities") return <Abilities {...props} />;
  if (phase.kind === "done") return <Done catalog={catalog} statuses={props.statuses} chosen={props.chosen} />;
  const group = catalog.groups.find(item => item.key === phase.key);
  if (!group) return <p className="config-hint">这一项已被移除，可以直接继续。</p>;
  return <GroupStep group={group} steps={catalog.steps.filter(step => step.group === group.key)} statuses={props.statuses}
    renderRoute={props.renderRoute} changed={props.changed} markDirty={props.markDirty} />;
}

function Welcome({catalog, start, skip}: {catalog: Catalog; start: () => void; skip: () => void}) {
  const optional = catalog.groups.filter(group => !group.required).length;
  const required = catalog.groups.filter(group => group.required);
  return <div className="onboarding-welcome">
    <i className="onboarding-mark is-large" style={brandMask} aria-hidden="true" />
    <h2 tabIndex={-1}>你好，我是 Akashic</h2>
    <p className="onboarding-lead">先花两三分钟让我能说话，其余能力随时可以再开。</p>
    <ol className="onboarding-promise">
      {required.map(group => <li key={group.key}><span>{group.pitch || group.title}</span><em>必需</em></li>)}
      {optional > 0 && <li><span>挑选你想要的能力</span><em>{optional} 项可选</em></li>}
      <li><span>开始第一段对话</span><em>马上</em></li>
    </ol>
    <button type="button" className="onboarding-primary" autoFocus onClick={start}>开始</button>
    <button type="button" className="onboarding-quiet" onClick={skip}>我已经配好了，直接聊天</button>
  </div>;
}

function Heading({eyebrow, title, lead}: {eyebrow: string; title: string; lead: string}) {
  const ref = useRef<HTMLHeadingElement>(null);
  // 换步后焦点落在新标题，读屏从这一步开始读。
  useEffect(() => { ref.current?.focus({preventScroll: true}); }, []);
  return <header className="onboarding-heading">
    <span className="onboarding-eyebrow">{eyebrow}</span>
    <h2 ref={ref} tabIndex={-1}>{title}</h2>
    {lead && <p className="onboarding-lead">{lead}</p>}
  </header>;
}

function Abilities({catalog, statuses, chosen, toggle}: PhaseProps) {
  const optional = catalog.groups.filter(group => !group.required);
  const pending = optional.filter(group => chosen.has(group.key) && !groupReady(catalog, statuses, group.key)).length;
  return <>
    <Heading eyebrow="可选 · 随时可改" title="还想让它做什么？" lead="每一项都来自一个已安装的插件。打开的项接下来做简单设置，没打开的不会打扰你。" />
    <div className="onboarding-abilities">
      {optional.map(group => <AbilityCard key={group.key} group={group} on={chosen.has(group.key)}
        done={groupReady(catalog, statuses, group.key)} blocked={blockedReason(catalog, statuses, group.key)} toggle={() => toggle(group.key)} />)}
    </div>
    <p className="onboarding-note">{pending ? `接下来还有 ${pending} 个小设置。` : "不需要再设置什么了。"}安装新插件后，这里会自动多出对应的一项。</p>
  </>;
}

function groupReady(catalog: Catalog, statuses: Statuses, key: string): boolean {
  const steps = catalog.steps.filter(step => step.group === key);
  return steps.length > 0 && steps.every(step => ready(statuses[step.id]));
}
function blockedReason(catalog: Catalog, statuses: Statuses, key: string): string {
  const step = catalog.steps.find(item => item.group === key && statuses[item.id]?.blocked);
  return step ? statuses[step.id]?.reason ?? "" : "";
}

function AbilityCard({group, on, done, blocked, toggle}: {group: Group; on: boolean; done: boolean; blocked: string; toggle: () => void}) {
  const id = `onboarding-ability-${group.key}`;
  return <article className={`onboarding-ability${on || done ? " is-on" : ""}`}>
    <h3 id={id}>{group.pitch || group.title}</h3>
    {done ? <span className="onboarding-chip">已开启</span>
      : <button type="button" role="switch" className="onboarding-switch" aria-checked={on} aria-labelledby={id} onClick={toggle} />}
    {group.benefit && <p>{group.benefit}</p>}
    <div className="onboarding-source"><span>来自 {group.title}</span>{blocked && <span className="is-warning">{blocked}</span>}</div>
    {(on || done) && group.preview.length > 0 && <div className="onboarding-preview" aria-label="示例">
      {group.preview.map((line, i) => <div key={i}><span>{line.speaker}</span><p>{line.text}</p></div>)}
    </div>}
  </article>;
}

function GroupStep({group, steps, statuses, renderRoute, changed, markDirty}: {
  group: Group; steps: Step[]; statuses: Statuses; renderRoute?: RenderSettingsRoute;
  changed: () => void; markDirty: (route: string, value: boolean) => void;
}) {
  const fault = steps.map(step => statuses[step.id]?.fault).find(Boolean);
  return <>
    <Heading eyebrow={group.required ? "必需" : group.title} title={group.pitch || group.title} lead={group.benefit} />
    {fault && <div className="config-error" role="alert"><p>{fault}</p></div>}
    {steps.map(step => <section key={step.id} className="onboarding-form">
      {steps.length > 1 && <h3>{step.title}</h3>}
      {/* 嵌入表单在前置未满足时不渲染内容，阻塞原因由引导说明。 */}
      {statuses[step.id]?.blocked && <p className="onboarding-blocked" role="status">
        <strong>暂时还不能开启</strong><span>{statuses[step.id]?.reason || "它依赖的能力还没准备好"}</span><span>可以先继续，之后在功能设置里再开。</span>
      </p>}
      <StepForm route={step.route} intent={group.required ? undefined : "enable"} renderRoute={renderRoute} changed={changed} markDirty={markDirty} />
    </section>)}
  </>;
}

// 步骤表单按 route 内嵌对应设置分节；子插件拥有独立 React root，销毁延后到父提交结束。
function StepForm({route, intent, renderRoute, changed, markDirty}: {
  route: string; intent?: "enable"; renderRoute?: RenderSettingsRoute; changed: () => void; markDirty: (route: string, value: boolean) => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const element = host.current!;
    const props: EmbedProps = {embedded: true, intent, changed, dirty: value => markDirty(route, value)};
    const dispose = renderRoute?.(route, element, props);
    if (!dispose) { element.textContent = "这项设置的页面暂时不可用，可以先跳过，稍后在功能设置里完成。"; return; }
    return () => { markDirty(route, false); queueMicrotask(dispose); };
  }, [route, intent, renderRoute, changed, markDirty]);
  return <div ref={host} />;
}

function Done({catalog, statuses, chosen}: {catalog: Catalog; statuses: Statuses; chosen: ReadonlySet<string>}) {
  const missing = catalog.groups.some(group => group.required && !groupReady(catalog, statuses, group.key));
  return <>
    <Heading eyebrow="准备好了" title={missing ? "还差一步才能对话" : "说第一句话吧"}
      lead={missing ? "还没有可用的对话模型。可以随时在功能设置里连接。" : "这些设置都可以在功能设置里随时修改，新装的插件也会出现在那里。"} />
    <ul className="onboarding-summary">
      {catalog.groups.map(group => {
        const on = groupReady(catalog, statuses, group.key);
        const state = on ? "已开启" : group.required ? "未连接" : chosen.has(group.key) ? "未完成" : "未开启";
        return <li key={group.key} className={on ? "is-on" : group.required ? "is-missing" : ""}>
          <span aria-hidden="true">{on ? "✓" : group.required ? "!" : "–"}</span><span>{group.pitch || group.title}</span><em>{state}</em>
        </li>;
      })}
    </ul>
  </>;
}

function Footer({phase, at, catalog, statuses, chosen, editing, back, next}: {
  phase: Phase; at: number; catalog: Catalog; statuses: Statuses; chosen: ReadonlySet<string>;
  editing: boolean; back: () => void; next: () => void;
}) {
  // 1. 必需分组必须就绪；2. 可选分组可以跳过；3. 有未保存修改时先保存。
  const group = phase.kind === "group" ? catalog.groups.find(item => item.key === phase.key) : undefined;
  const steps = group ? catalog.steps.filter(step => step.group === group.key) : [];
  const complete = steps.every(step => group?.required ? ready(statuses[step.id]) : settled(statuses[step.id]));
  const blocked = !!group && (editing || !complete);
  const optional = catalog.groups.filter(item => !item.required && chosen.has(item.key)).length;
  const label = phase.kind === "done" ? "开始对话" : phase.kind === "abilities" && optional === 0 ? "先不用，直接开始" : "继续";
  const hint = editing ? "先保存当前设置" : group?.required && !complete ? "连接成功后继续" : "";
  return <footer className="onboarding-footer">
    <button type="button" className="onboarding-text-button" hidden={at <= 1} onClick={back}>上一步</button>
    <span className="onboarding-hint">
      {group && !group.required && !complete && !editing ? <button type="button" className="onboarding-text-button" onClick={next}>跳过这项</button> : hint}
    </span>
    <button type="button" className="onboarding-primary" disabled={blocked} onClick={next}>{label}</button>
  </footer>;
}

import { ChevronLeft, LayoutGrid, Maximize2, Minimize2, X } from "lucide-react";
import { useEffect, useLayoutEffect, useMemo, useRef, useState, useSyncExternalStore, type CSSProperties } from "react";
import { createPortal } from "react-dom";
import type { SidebarPanelControls, SidebarPanelStatus } from "@akashic/chat-ui-v1";

import { chatPortalRoot } from "./chat-layout";
import { SidebarPanelMount, sidebarPanelSources, usePluginUiCatalogVersion, type SidebarPanelSource } from "./plugin-ui-runtime";
import { reportShellOverlay } from "./shell-rail-actions";
import "./sidebar-panels.css";

// 侧栏顶部启动器：插件面板收进一个图标，插件再多也只多菜单行，不多占侧栏位置。
// 状态放在模块里而不是侧栏组件里：手机上侧栏在抽屉中，抽屉关掉后面板仍要留在页面上。

type PanelSize = "compact" | "expanded";
interface Surface {
  key: string | null;
  size: PanelSize;
  sheet: boolean;
  anchor: HTMLElement | null;
}

const FILTER_FROM = 7;
let surface: Surface | null = null;
let statuses = new Map<string, SidebarPanelStatus | null>();
const listeners = new Set<() => void>();
let statusAbort: AbortController | null = null;
let statusSession: string | undefined;

function publish(next: Surface | null) {
  surface = next;
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

/** 打开启动器菜单；手机抽屉里没有锚点，改为底部抽屉。已打开时再点一次收起。 */
export function toggleSidebarPanels(anchor: HTMLElement | null, sheet: boolean): void {
  publish(surface ? null : { key: null, size: "compact", sheet, anchor });
}

// 每次重读都中止上一轮：切换会话后旧会话的短值不会晚到覆盖新值，换会话时先清掉旧值。
function refreshStatuses(sources: readonly SidebarPanelSource[], sessionId: string | undefined) {
  statusAbort?.abort();
  const abort = statusAbort = new AbortController();
  if (statusSession !== sessionId) {
    statusSession = sessionId;
    statuses = new Map();
    for (const listener of listeners) listener();
  }
  void Promise.all(sources.map(async (source) => {
    try {
      const status = await source.provider.status?.({ ...source.context(abort.signal, sessionId), sessionId }) ?? null;
      return [source.key, status] as const;
    } catch (error) {
      if (!abort.signal.aborted) console.debug(`[sidebar-panels] status failed: ${source.pluginId}`, error);
      return [source.key, null] as const;
    }
  })).then((entries) => {
    if (abort.signal.aborted) return;
    statuses = new Map(entries);
    for (const listener of listeners) listener();
  });
}

function useSidebarPanels() {
  const { version } = usePluginUiCatalogVersion();
  const current = useSyncExternalStore(subscribe, () => surface);
  const shown = useSyncExternalStore(subscribe, () => statuses);
  const sources = useMemo(() => { void version; return sidebarPanelSources(); }, [version]);
  const alert = sources.some((source) => shown.get(source.key)?.alert);
  return { sources, statuses: shown, surface: current, alert };
}

/** 有提醒的面板时为 true；手机上导航按钮据此挂同一个提醒点。 */
export function useSidebarPanelAlert(): boolean {
  return useSidebarPanels().alert;
}

/** 品牌行里的启动器按钮；没有任何插件提供面板时不出现。 */
export function SidebarPanelLauncher({ sheet, onOpenSheet }: {
  sheet: boolean;
  /** 手机抽屉里：先关抽屉再打开底部抽屉，由抽屉组件安排顺序。 */
  onOpenSheet?: (open: () => void) => void;
}) {
  const { sources, surface: open, alert } = useSidebarPanels();
  const ref = useRef<HTMLButtonElement>(null);
  if (!sources.length) return null;
  return <button ref={ref} type="button" className="chat-sidebar__new chat-sidebar__panels" data-alert={alert || undefined}
    aria-label={alert ? "插件面板，有提醒" : "插件面板"} title="插件面板（Alt+U）" aria-haspopup="dialog"
    aria-expanded={!!open && open.sheet === sheet}
    onClick={() => sheet && onOpenSheet ? onOpenSheet(() => toggleSidebarPanels(null, true)) : toggleSidebarPanels(ref.current, false)}>
    <LayoutGrid size={18} strokeWidth={1.75} aria-hidden="true" />
  </button>;
}

/** 页面里唯一的面板外框；随会话、一轮回答结束与页面重新可见重读各面板短值。 */
export function SidebarPanelHost({ sessionId, refreshToken }: { sessionId?: string; refreshToken?: string }) {
  const { sources, statuses: shown, surface: current } = useSidebarPanels();
  const open = current !== null;
  useEffect(() => {
    refreshStatuses(sources, sessionId);
  }, [sources, sessionId, refreshToken, open]);
  useEffect(() => () => { statusAbort?.abort(); }, []);
  useEffect(() => {
    const onShown = () => { if (document.visibilityState === "visible") refreshStatuses(sources, sessionId); };
    const onKey = (event: KeyboardEvent) => {
      if (!event.altKey || event.ctrlKey || event.metaKey || event.code !== "KeyU" || !sources.length) return;
      event.preventDefault();
      if (surface) { publish(null); return; }
      // 与点击启动器同一条路：手机导航抽屉里的启动器会先关抽屉再弹底部抽屉；没有可见启动器时直接用底部抽屉。
      const launcher = [...document.querySelectorAll<HTMLElement>(".chat-sidebar__panels")].find((node) => node.getClientRects().length);
      if (launcher) launcher.click();
      else toggleSidebarPanels(null, true);
    };
    window.addEventListener("focus", onShown);
    document.addEventListener("visibilitychange", onShown);
    document.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("focus", onShown);
      document.removeEventListener("visibilitychange", onShown);
      document.removeEventListener("keydown", onKey);
    };
  }, [sources, sessionId]);
  useEffect(() => {
    if (!open) return;
    reportShellOverlay(true);
    return () => reportShellOverlay(false);
  }, [open]);
  if (!current || !sources.length) return null;
  return createPortal(<PanelSurface current={current} sources={sources} statuses={shown} sessionId={sessionId} />, chatPortalRoot());
}

// 1. 菜单与面板共用一个外框，返回键回到菜单；2. 点外部、Esc 关闭并把焦点还给启动器；
// 3. 展开态与手机抽屉带遮罩，浮层态不挡页面。
function PanelSurface({ current, sources, statuses: shown, sessionId }: {
  current: Surface;
  sources: readonly SidebarPanelSource[];
  statuses: ReadonlyMap<string, SidebarPanelStatus | null>;
  sessionId?: string;
}) {
  const rootRef = useRef<HTMLDivElement>(null);
  const source = sources.find((item) => item.key === current.key) ?? null;
  const expanded = current.size === "expanded";
  // 关闭时焦点还给启动器（手机上是导航按钮）；布局阶段的清理在节点移除前运行，还能判断焦点是否在面板里。
  useLayoutEffect(() => {
    const root = rootRef.current;
    const returnTo = current.anchor ?? document.querySelector<HTMLElement>(".compact-navigation-trigger");
    return () => {
      const active = document.activeElement;
      if (!surface && returnTo?.isConnected && (active === document.body || root?.contains(active))) returnTo.focus({ preventScroll: true });
    };
  }, [current.anchor]);
  useEffect(() => {
    const onPointer = (event: PointerEvent) => {
      const target = event.target as Node;
      if (!rootRef.current?.contains(target) && !current.anchor?.contains(target) && !current.sheet && !expanded) publish(null);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Tab" && (current.sheet || expanded)) { trapTab(event, rootRef.current); return; }
      if (event.key !== "Escape" || event.defaultPrevented) return;
      event.preventDefault();
      publish(null);
    };
    document.addEventListener("pointerdown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [current.anchor, current.sheet, expanded]);
  const style = current.sheet || expanded ? undefined : anchorStyle(current.anchor);
  return <>
    {current.sheet || expanded ? <div className="sidebar-panels-scrim" onClick={() => publish(null)} /> : null}
    <div ref={rootRef} className="sidebar-panels" role="dialog" aria-label={source?.provider.label ?? "插件面板"}
      aria-modal={current.sheet || expanded || undefined}
      data-size={current.size} data-sheet={current.sheet || undefined} data-escape-layer="" style={style}>
      {current.sheet ? <button type="button" className="sidebar-panels__grab" aria-label={expanded ? "降低" : "拉高"}
        onClick={() => publish({ ...current, size: expanded ? "compact" : "expanded" })}><i /></button> : null}
      <PanelHead current={current} source={source} />
      {source ? <PanelBody key={source.key} current={current} source={source} sessionId={sessionId} />
        : <PanelMenu sources={sources} statuses={shown} onPick={(key) => publish({ ...current, key })} />}
    </div>
  </>;
}

// 带遮罩的形态是模态的：Tab 在面板里循环，不落到遮罩后面的会话与输入框。
function trapTab(event: KeyboardEvent, root: HTMLElement | null) {
  if (!root) return;
  const items = [...root.querySelectorAll<HTMLElement>("button, input, select, textarea, a[href], [tabindex]:not([tabindex='-1'])")]
    .filter((node) => !node.matches(":disabled") && node.getClientRects().length);
  if (!items.length) return;
  const first = items[0], last = items[items.length - 1], active = document.activeElement;
  if (!root.contains(active)) { event.preventDefault(); first.focus(); }
  else if (event.shiftKey && active === first) { event.preventDefault(); last.focus(); }
  else if (!event.shiftKey && active === last) { event.preventDefault(); first.focus(); }
}

function anchorStyle(anchor: HTMLElement | null): CSSProperties | undefined {
  if (!anchor) return undefined;
  const rect = anchor.getBoundingClientRect();
  const width = Math.min(360, window.innerWidth - 24);
  return { top: rect.bottom + 6, left: Math.max(12, Math.min(window.innerWidth - width - 12, rect.left)), width };
}

function PanelHead({ current, source }: { current: Surface; source: SidebarPanelSource | null }) {
  const expanded = current.size === "expanded";
  return <div className="sidebar-panels__head">
    {source ? <button type="button" className="sidebar-panels__icon" aria-label="返回插件面板"
      onClick={() => publish({ ...current, key: null, size: current.sheet ? current.size : "compact" })}><ChevronLeft size={18} aria-hidden="true" /></button> : null}
    <span className="sidebar-panels__title">{source?.provider.label ?? "插件面板"}</span>
    {source && !current.sheet ? <button type="button" className="sidebar-panels__icon" aria-label={expanded ? "收起" : "展开"}
      title={expanded ? "收起" : "展开"} onClick={() => publish({ ...current, size: expanded ? "compact" : "expanded" })}>
      {expanded ? <Minimize2 size={15} aria-hidden="true" /> : <Maximize2 size={15} aria-hidden="true" />}
    </button> : null}
    <button type="button" className="sidebar-panels__icon" aria-label="关闭" onClick={() => publish(null)}><X size={16} aria-hidden="true" /></button>
  </div>;
}

// 插件超过 6 个时给筛选框；短值只在插件给出 alert 时变色。
function PanelMenu({ sources, statuses: shown, onPick }: {
  sources: readonly SidebarPanelSource[];
  statuses: ReadonlyMap<string, SidebarPanelStatus | null>;
  onPick: (key: string) => void;
}) {
  const [query, setQuery] = useState("");
  const listRef = useRef<HTMLDivElement>(null);
  useEffect(() => { listRef.current?.querySelector<HTMLElement>("input, button")?.focus({ preventScroll: true }); }, []);
  const needle = query.trim().toLowerCase();
  const items = sources.filter((source) => !needle || source.provider.label.toLowerCase().includes(needle));
  return <div className="sidebar-panels__menu" ref={listRef}>
    {sources.length >= FILTER_FROM ? <input className="sidebar-panels__filter" type="search" value={query}
      placeholder="筛选面板" aria-label="筛选面板" onChange={(event) => setQuery(event.target.value)} /> : null}
    {items.map((source) => {
      const status = shown.get(source.key);
      return <button key={source.key} type="button" className="sidebar-panels__item" onClick={() => onPick(source.key)}>
        <span className="sidebar-panels__item-icon" aria-hidden="true" dangerouslySetInnerHTML={{ __html: source.provider.iconSvg }} />
        <span className="sidebar-panels__item-label">{source.provider.label}</span>
        {status ? <span className="sidebar-panels__item-value" data-alert={status.alert || undefined}>{status.text}</span> : null}
      </button>;
    })}
    {!items.length ? <p className="sidebar-panels__empty">没有匹配的面板</p> : null}
  </div>;
}

// 外框能力的身份只随面板与形态（浮层或抽屉）变化；尺寸经 refreshToken 通知，不重新挂载插件。
function PanelBody({ current, source, sessionId }: { current: Surface; source: SidebarPanelSource; sessionId?: string }) {
  const panel = useMemo<SidebarPanelControls>(() => ({
    size: () => surface?.size ?? "compact",
    setSize: (size) => { if (surface && surface.key === source.key) publish({ ...surface, size }); },
    sheet: current.sheet,
    close: () => publish(null),
  }), [source.key, current.sheet]);
  // 从菜单进到面板时选中的菜单行已卸载：焦点落到头部返回键，键盘可以接着往下走。
  const bodyRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const root = bodyRef.current?.closest(".sidebar-panels");
    if (root && !root.contains(document.activeElement)) root.querySelector<HTMLElement>(".sidebar-panels__head button")?.focus({ preventScroll: true });
  }, []);
  return <div className="sidebar-panels__body" ref={bodyRef}>
    <SidebarPanelMount source={source} sessionId={sessionId || undefined} panel={panel} size={current.size} />
  </div>;
}

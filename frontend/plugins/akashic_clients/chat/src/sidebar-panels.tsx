import { ChevronLeft, LayoutGrid, Maximize2, Minimize2, X } from "lucide-react";
import { useEffect, useLayoutEffect, useMemo, useRef, useState, useSyncExternalStore, type CSSProperties, type RefObject } from "react";
import { createPortal } from "react-dom";
import { cssDurationMs } from "./motion";
import type { SidebarPanelControls, SidebarPanelStatus } from "@akashic/chat-ui-v1";

import { chatPortalRoot } from "./chat-layout";
import { IconSwap } from "./icon-swap";
import { onPageShown, SidebarPanelMount, sidebarPanelSources, usePluginUiCatalogVersion, type SidebarPanelSource } from "./plugin-ui-runtime";
import { useExitPresence } from "./presence";
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
  const open = current !== null && sources.length > 0;
  const presence = useExitPresence(open, 200);
  const [lastSurface, setLastSurface] = useState<Surface | null>(current);
  if (current !== null && current !== lastSurface) setLastSurface(current);
  const shownSurface = current ?? (presence.present ? lastSurface : null);
  useEffect(() => {
    refreshStatuses(sources, sessionId);
  }, [sources, sessionId, refreshToken, open]);
  useEffect(() => () => { statusAbort?.abort(); }, []);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (!event.altKey || event.ctrlKey || event.metaKey || event.code !== "KeyU" || !sources.length) return;
      event.preventDefault();
      if (surface) { publish(null); return; }
      // 与点击启动器同一条路：手机导航抽屉里的启动器会先关抽屉再弹底部抽屉；没有可见启动器时直接用底部抽屉。
      const launcher = [...document.querySelectorAll<HTMLElement>(".chat-sidebar__panels")].find((node) => node.getClientRects().length);
      if (launcher) launcher.click();
      else toggleSidebarPanels(null, true);
    };
    const stopShown = onPageShown(() => refreshStatuses(sources, sessionId));
    document.addEventListener("keydown", onKey);
    return () => {
      stopShown();
      document.removeEventListener("keydown", onKey);
    };
  }, [sources, sessionId]);
  useEffect(() => {
    if (!open) return;
    reportShellOverlay(true);
    return () => reportShellOverlay(false);
  }, [open]);
  if (!presence.present || !shownSurface || !sources.length) return null;
  return createPortal(<PanelSurface open={open} current={shownSurface} sources={sources} statuses={shown} sessionId={sessionId} onTransitionEnd={presence.onTransitionEnd} />, chatPortalRoot());
}

// 1. 菜单与面板共用一个外框，返回键回到菜单；2. 点外部、Esc 关闭并把焦点还给启动器；
// 3. 展开态与手机抽屉带遮罩，浮层态不挡页面。
function PanelSurface({ open, current, sources, statuses: shown, sessionId, onTransitionEnd }: {
  open: boolean;
  current: Surface;
  sources: readonly SidebarPanelSource[];
  statuses: ReadonlyMap<string, SidebarPanelStatus | null>;
  sessionId?: string;
  onTransitionEnd: (event: React.TransitionEvent<HTMLElement>) => void;
}) {
  const rootRef = useRef<HTMLDivElement>(null);
  const source = sources.find((item) => item.key === current.key) ?? null;
  const expanded = current.size === "expanded";
  useSheetHeightFlip(rootRef, current.sheet, open, `${current.key ?? ""}|${current.size}`);
  // 关闭时立刻把焦点还给启动器（退场期间面板会标 inert）。
  useLayoutEffect(() => {
    if (open) return;
    const root = rootRef.current;
    const returnTo = current.anchor ?? document.querySelector<HTMLElement>(".compact-navigation-trigger");
    const active = document.activeElement;
    if (returnTo?.isConnected && (active === document.body || root?.contains(active))) returnTo.focus({ preventScroll: true });
  }, [open, current.anchor]);
  useEffect(() => {
    if (!open) return;
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
  }, [open, current.anchor, current.sheet, expanded]);
  // 退场途中重开不会重挂子树，菜单/面板挂载时的聚焦不再执行：在这里补一次。
  const wasOpen = useRef(open);
  useEffect(() => {
    const reopened = open && !wasOpen.current;
    wasOpen.current = open;
    const root = rootRef.current;
    if (!reopened || !root || root.contains(document.activeElement)) return;
    const target = root.querySelector<HTMLElement>(".sidebar-panels__menu input, .sidebar-panels__menu button")
      ?? root.querySelector<HTMLElement>(".sidebar-panels__head button");
    target?.focus({ preventScroll: true });
  }, [open]);
  const style = current.sheet || expanded ? undefined : anchorStyle(current.anchor);
  const state = open ? "open" : "closed";
  return <>
    {current.sheet || expanded ? <div className="sidebar-panels-scrim" data-state={state} onClick={() => publish(null)} /> : null}
    <div ref={rootRef} className="sidebar-panels" role="dialog" aria-label={source?.provider.label ?? "插件面板"}
      aria-modal={current.sheet || expanded || undefined}
      data-state={state} data-size={current.size} data-sheet={current.sheet || undefined} data-escape-layer=""
      inert={!open ? true : undefined} style={style} onTransitionEnd={onTransitionEnd}>
      {current.sheet ? <button type="button" className="sidebar-panels__grab" aria-label={expanded ? "降低" : "拉高"}
        onClick={() => publish({ ...current, size: expanded ? "compact" : "expanded" })}><i /></button> : null}
      <PanelHead current={current} source={source} />
      {source ? <PanelBody key={source.key} current={current} source={source} sessionId={sessionId} />
        : <PanelMenu sources={sources} statuses={shown} onPick={(key) => publish({ ...current, key })} />}
    </div>
  </>;
}

// 手机抽屉换高（拉高/降低、菜单与面板互切）：高度一步到位，上沿用 transform 从旧位置滑到新位置（FLIP）。
// 途中再切换时从当前的视觉位置接着滑，不跳回起点。
function useSheetHeightFlip(rootRef: RefObject<HTMLDivElement | null>, sheet: boolean, open: boolean, shape: string): void {
  const last = useRef<{ height: number; animation: Animation | null } | null>(null);
  useLayoutEffect(() => {
    const root = rootRef.current;
    const previous = last.current;
    const running = previous?.animation?.playState === "running" ? previous.animation : null;
    if (!root || !sheet) {
      running?.cancel();
      last.current = null;
      return;
    }
    // 1. 布局高度不含 transform；关闭时让进行中的滑动走完，交给退场过渡。
    const height = root.offsetHeight;
    if (!open) {
      last.current = { height, animation: running };
      return;
    }
    // 2. 进行中的滑动按当前位移折进起点。
    const carried = running ? new DOMMatrixReadOnly(getComputedStyle(root).transform).m42 : 0;
    // 3. cancel 事件异步才到：垫层在这里同步撤掉，新一轮读到的是 CSS 本来的阴影。
    running?.cancel();
    root.style.boxShadow = "";
    const offset = previous ? height - previous.height + carried : 0;
    last.current = { height, animation: Math.abs(offset) >= 1 ? slideSheetEdge(root, offset) : null };
  }, [rootRef, sheet, open, shape]);
  // 地址栏伸缩、窗口改高时 60dvh 直接重排：持续写回布局高度（不含 transform），下一次 FLIP 的起点不陈旧。
  // ResizeObserver 回调晚于提交时的 layout effect，不会抢先覆盖本次要读的旧值。
  useEffect(() => {
    const root = rootRef.current;
    if (!root || !sheet) return;
    const observer = new ResizeObserver(() => { if (last.current) last.current.height = root.offsetHeight; });
    observer.observe(root);
    return () => observer.disconnect();
  }, [rootRef, sheet]);
  useEffect(() => () => { last.current?.animation?.cancel(); }, []);
}

const backingOwner = new WeakMap<HTMLElement, Animation>();

// 抽屉贴着视口底：变高时从下压 offset 处升上来，变矮时从上提处落下；
// 变矮那段抽屉底下会露出一截，用一层画死的纸色投影垫住（静态，不参与动画），结束即撤。
// 垫层 = 外框按 gap/2 外扩再下移 gap/2 + 24：上沿落在圆角以下，下沿盖到原来的视口底；抽屉横向铺满视口，外扩的两侧在屏外。
function slideSheetEdge(root: HTMLElement, offset: number): Animation | null {
  const styles = getComputedStyle(root);
  const phase = offset > 0 ? "open" : "close";
  const duration = cssDurationMs(styles.getPropertyValue(`--ak-sys-duration-${phase}`));
  if (!(duration > 0)) return null;
  const gap = -offset;
  if (gap > 0) root.style.boxShadow = `0 ${gap / 2 + 24}px 0 ${gap / 2}px ${styles.backgroundColor}, ${styles.boxShadow}`;
  const animation = root.animate([{ transform: `translateY(${offset}px)` }, { transform: "translateY(0)" }],
    { duration, easing: styles.getPropertyValue(`--ak-sys-motion-${phase}`).trim() || "ease" });
  if (gap > 0) backingOwner.set(root, animation);
  // 被打断的旧动画晚到的 cancel 不能撤掉新一轮的垫层：只有垫层的主人能撤。
  const release = () => {
    if (backingOwner.get(root) !== animation) return;
    backingOwner.delete(root);
    root.style.boxShadow = "";
  };
  animation.addEventListener("finish", release);
  animation.addEventListener("cancel", release);
  return animation;
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
      <IconSwap state={expanded} a={<Maximize2 size={15} aria-hidden="true" />} b={<Minimize2 size={15} aria-hidden="true" />} />
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

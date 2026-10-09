import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";
import { akashicBrandIcon } from "./brand";
import type {
  WebEntry,
  WebHostContextV1,
  WebMountView,
  WebUiDisposer,
} from "@akashic/web-ui-v1";
import type {
  RenderSettingsRoute,
  ShellRailAction,
  ShellSettingsPlugin,
  ShellSettingsSection,
} from "@akashic/shell-ui-v1";

type ShellPage = WebEntry & {
  label: string;
  route: string;
  iconSvg: string;
};

const PAGES_MOUNT = "shell.pages.v1";
const RAIL_ACTIONS_MOUNT = "shell.rail-actions.v1";
const SETTINGS_MOUNT = "shell.settings.v1";
const SETTINGS_PLUGINS_MOUNT = "shell.settings-plugins.v1";

/** “插件”是设置工作区的内置配置区：条目来自 settings-plugins 目录，导航位置固定在初始配置（-10）与模型（30）之间。 */
const SETTINGS_PLUGINS_ORDER = 20;
const SETTINGS_PLUGINS_LABEL = "插件配置";
const SETTINGS_PLUGINS_ICON = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M19.439 7.85c-.049.322.059.648.289.878l1.568 1.568c.47.47.706 1.087.706 1.704s-.235 1.233-.706 1.704l-1.611 1.611a.98.98 0 0 1-.837.276c-.47-.07-.802-.48-.968-.925a2.501 2.501 0 1 0-3.214 3.214c.446.166.855.497.925.968a.979.979 0 0 1-.276.837l-1.61 1.61a2.404 2.404 0 0 1-1.705.707 2.402 2.402 0 0 1-1.704-.706l-1.568-1.568a1.026 1.026 0 0 0-.877-.29c-.493.074-.84.504-1.02.968a2.5 2.5 0 1 1-3.237-3.237c.464-.18.894-.527.967-1.02a1.026 1.026 0 0 0-.289-.877l-1.568-1.568A2.402 2.402 0 0 1 1.998 12c0-.617.236-1.234.706-1.704L4.23 8.77c.24-.24.581-.353.917-.303.515.077.877.528 1.073 1.01a2.5 2.5 0 1 0 3.259-3.259c-.482-.196-.933-.558-1.01-1.073-.05-.336.062-.676.303-.917l1.525-1.525A2.402 2.402 0 0 1 12 1.998c.617 0 1.234.236 1.704.706l1.568 1.568c.23.23.556.338.877.29.493-.074.84-.504 1.02-.968a2.5 2.5 0 1 1 3.237 3.237c-.464.18-.894.527-.967 1.02Z"/></svg>';

const SETTINGS_ICON = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>';

/** 设置工作区的当前位置：要么停在某个顶层分节，要么停在插件配置区的某个成员。 */
type SettingsTarget =
  | { kind: "section"; id: string }
  | { kind: "plugins"; memberId: string };

type SettingsNavItem =
  | { kind: "section"; entry: ShellSettingsSection }
  | { kind: "plugins" };

type SettingsTab =
  | { kind: "entry"; entry: ShellSettingsPlugin }
  | { kind: "family"; id: string; label: string; entries: ShellSettingsPlugin[] };

/** 插件配置区分节折叠成 tab：≥2 个成员的 family 合成一个组合 tab，标签取自成员自带的 familyLabel。 */
function buildSettingsTabs(entries: ShellSettingsPlugin[]): SettingsTab[] {
  const familyCount = new Map<string, number>();
  for (const entry of entries) {
    if (entry.family) familyCount.set(entry.family, (familyCount.get(entry.family) ?? 0) + 1);
  }
  const tabs: SettingsTab[] = [];
  const familyAt = new Map<string, number>();
  for (const entry of entries) {
    if (entry.family && (familyCount.get(entry.family) ?? 0) > 1) {
      const at = familyAt.get(entry.family);
      if (at === undefined) {
        familyAt.set(entry.family, tabs.length);
        tabs.push({ kind: "family", id: entry.family, label: entry.familyLabel ?? entry.family, entries: [entry] });
      } else {
        (tabs[at] as { kind: "family"; entries: ShellSettingsPlugin[] }).entries.push(entry);
      }
    } else {
      tabs.push({ kind: "entry", entry });
    }
  }
  return tabs;
}

function settingsTargetForRoute(
  sections: readonly ShellSettingsSection[],
  plugins: readonly ShellSettingsPlugin[],
  route: string,
): SettingsTarget | undefined {
  const section = sections.find((entry) => entry.route === route);
  if (section) return { kind: "section", id: section.id };
  const member = plugins.find((entry) => entry.route === route);
  if (member) return { kind: "plugins", memberId: member.id };
  return undefined;
}

/** 对话框留在 Shell 组件内；底栏动作与快捷键经这个模块级入口打开它。 */
const settingsLauncher = { open: () => {} };

/** Register the ordinary Shell plugin as the only owner of the outer frame. */
export function activate(ctx: WebHostContextV1): WebUiDisposer {
  const disposers = [
    ctx.ui.inject("web.root.v1", (mount) => mount.register({
      id: "shell",
      children: [
        { id: PAGES_MOUNT, cardinality: "list" },
        { id: RAIL_ACTIONS_MOUNT, cardinality: "list" },
        { id: SETTINGS_MOUNT, cardinality: "list" },
        { id: SETTINGS_PLUGINS_MOUNT, cardinality: "list" },
      ],
      render(host, view) {
        const root = createRoot(host);
        root.render(<Shell
          pages={view.child(PAGES_MOUNT)}
          railActions={view.child(RAIL_ACTIONS_MOUNT)}
          settings={view.child(SETTINGS_MOUNT)}
          settingsPlugins={view.child(SETTINGS_PLUGINS_MOUNT)}
        />);
        return () => root.unmount();
      },
    })),
    ctx.ui.inject(RAIL_ACTIONS_MOUNT, (mount) => mount.register({
      id: "shell.settings",
      order: 100,
      label: "功能设置",
      iconSvg: SETTINGS_ICON,
      onActivate: () => settingsLauncher.open(),
    })),
  ];
  return () => {
    for (const dispose of disposers.reverse()) dispose();
  };
}

function Shell({ pages, railActions, settings, settingsPlugins }: {
  pages: WebMountView;
  railActions: WebMountView;
  settings: WebMountView;
  settingsPlugins: WebMountView;
}): React.ReactElement {
  const bandEntries = useMemo(() => checkPages(pages.entries), [pages.entries]);
  const railActionEntries = useMemo(() => checkRailActions(railActions.entries), [railActions.entries]);
  const sectionEntries = useMemo(() => checkSettingsSections(settings.entries), [settings.entries]);
  const pluginEntries = useMemo(() => checkSettingsPlugins(settingsPlugins.entries), [settingsPlugins.entries]);
  // 三个目录共用同一个 hash 路由命名空间，跨目录撞 route 是合同错误。
  useMemo(() => {
    const routes = [...bandEntries, ...sectionEntries, ...pluginEntries].map((entry) => entry.route);
    if (new Set(routes).size !== routes.length) throw new Error("Shell 页面与设置分节的 route 不能重复");
  }, [bandEntries, sectionEntries, pluginEntries]);

  // 左栏导航是顶层分节按 order 排列；插件配置区有条目时按固定 order 插入。
  const settingsNav = useMemo(() => {
    const items: { order: number; item: SettingsNavItem }[] = sectionEntries.map((entry) => ({
      order: typeof entry.order === "number" ? entry.order : 0,
      item: { kind: "section", entry },
    }));
    if (pluginEntries.length > 0) items.push({ order: SETTINGS_PLUGINS_ORDER, item: { kind: "plugins" } });
    return items.sort((left, right) => left.order - right.order).map(({ item }) => item);
  }, [sectionEntries, pluginEntries]);
  const settingsTabs = useMemo(() => buildSettingsTabs(pluginEntries), [pluginEntries]);

  const defaultPage = bandEntries.find((entry) => entry.route === "") ?? bandEntries[0];
  const requestedRoute = window.location.hash.slice(1);
  const requestedTarget = settingsTargetForRoute(sectionEntries, pluginEntries, requestedRoute);
  const [withdrawn, setWithdrawn] = useState(() =>
    !!requestedRoute && !bandEntries.some((entry) => entry.route === requestedRoute) && !requestedTarget);
  const [settingsOpen, setSettingsOpen] = useState(!!requestedTarget);
  const [settingsTarget, setSettingsTarget] = useState<SettingsTarget | undefined>(requestedTarget);
  const [activeId, setActiveId] = useState(() => (requestedTarget ? defaultPage : pageFromLocation(bandEntries, defaultPage))?.id ?? "");
  const pageHosts = useRef(new Map<string, HTMLElement>());
  const settingsDialog = useRef<HTMLDialogElement>(null);
  const settingsContent = useRef<HTMLDivElement>(null);
  const bandTrack = useRef<HTMLDivElement>(null);
  const focusAfterNavigation = useRef(false);

  // 激活指示器：量当前按钮的内容区（扣除 12px padding），写入 track 的 CSS 变量滑动过去。
  useLayoutEffect(() => {
    const track = bandTrack.current;
    if (!track) return;
    const measure = () => {
      const current = track.querySelector<HTMLElement>('[aria-current="page"]');
      if (current) {
        track.style.setProperty("--band-indicator-x", `${current.offsetLeft + 12}px`);
        track.style.setProperty("--band-indicator-w", `${Math.max(0, current.offsetWidth - 24)}px`);
      } else {
        track.style.setProperty("--band-indicator-w", "0px");
      }
      requestAnimationFrame(() => { track.dataset.ready = "true"; });
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(track);
    track.querySelectorAll<HTMLElement>("[data-band-item]").forEach((item) => observer.observe(item));
    return () => observer.disconnect();
  }, [activeId, bandEntries]);

  // 已卸载插件的分节不留在当前位置：落回导航第一项。
  const firstTarget = useMemo<SettingsTarget | undefined>(() => {
    const first = settingsNav[0];
    if (!first) return undefined;
    return first.kind === "section"
      ? { kind: "section", id: first.entry.id }
      : { kind: "plugins", memberId: pluginEntries[0].id };
  }, [settingsNav, pluginEntries]);
  const currentTarget = useMemo<SettingsTarget | undefined>(() => {
    if (settingsTarget?.kind === "section" && sectionEntries.some((entry) => entry.id === settingsTarget.id)) return settingsTarget;
    if (settingsTarget?.kind === "plugins" && pluginEntries.some((entry) => entry.id === settingsTarget.memberId)) return settingsTarget;
    return firstTarget;
  }, [settingsTarget, sectionEntries, pluginEntries, firstTarget]);

  const activeSettingsTab = currentTarget?.kind === "plugins"
    ? settingsTabs.find((tab) =>
      tab.kind === "entry" ? tab.entry.id === currentTarget.memberId : tab.entries.some((entry) => entry.id === currentTarget.memberId))
    : undefined;
  // 组合 tab 渲染全部成员，普通 tab 只渲染当前分节。
  const settingsMembers = useMemo(
    () => activeSettingsTab ? (activeSettingsTab.kind === "family" ? activeSettingsTab.entries : [activeSettingsTab.entry]) : [],
    [activeSettingsTab],
  );

  const openPage = useCallback((entry: ShellPage): void => {
    if (entry.id === activeId) { setWithdrawn(false); setSettingsOpen(false); return; }
    const go = () => {
      focusAfterNavigation.current = true;
      setActiveId(entry.id); setWithdrawn(false); setSettingsOpen(false);
      const base = `${window.location.pathname}${window.location.search}`;
      window.history.replaceState(null, "", entry.route ? `${base}#${entry.route}` : base);
    };
    if (window.dispatchEvent(new CustomEvent("akashic:before-navigate", { cancelable: true, detail: { go } }))) go();
  }, [activeId]);

  // 设置页不再整页切换:功能设置是对话框内的双栏工作区,左栏分节导航,右栏就地渲染。
  // 打开、换节、关闭都先经 before-navigate,当前分节里未保存的表单可以拦截并走自己的确认。
  const bandRoute = useCallback(() => {
    const entry = bandEntries.find((item) => item.id === activeId) ?? defaultPage;
    const base = `${window.location.pathname}${window.location.search}`;
    return entry?.route ? `${base}#${entry.route}` : base;
  }, [bandEntries, activeId, defaultPage]);

  const openSettings = useCallback((target?: SettingsTarget): void => {
    const next = target ?? currentTarget;
    if (!next) return;
    const route = next.kind === "section"
      ? sectionEntries.find((entry) => entry.id === next.id)?.route
      : pluginEntries.find((entry) => entry.id === next.memberId)?.route;
    const go = () => {
      setSettingsTarget(next);
      setSettingsOpen(true);
      const base = `${window.location.pathname}${window.location.search}`;
      window.history.replaceState(null, "", route ? `${base}#${route}` : base);
    };
    if (window.dispatchEvent(new CustomEvent("akashic:before-navigate", { cancelable: true, detail: { go } }))) go();
  }, [currentTarget, sectionEntries, pluginEntries]);

  const closeSettings = useCallback((): void => {
    const go = () => {
      setSettingsOpen(false);
      window.history.replaceState(null, "", bandRoute());
    };
    if (window.dispatchEvent(new CustomEvent("akashic:before-navigate", { cancelable: true, detail: { go } }))) go();
  }, [bandRoute]);

  useLayoutEffect(() => {
    if (withdrawn) {
      const entry = bandEntries.find(item => item.id === activeId);
      const base = `${window.location.pathname}${window.location.search}`;
      window.history.replaceState(window.history.state, "", entry?.route ? `${base}#${entry.route}` : base);
    }
  }, []);

  useLayoutEffect(() => {
    if (!focusAfterNavigation.current) return;
    focusAfterNavigation.current = false;
    // 页面可见性已提交；弹窗不能在旧页面上猜测导航后的焦点。
    document.querySelector<HTMLButtonElement>('.product-band__track button[aria-current="page"]')?.focus();
  }, [activeId]);

  // 底栏动作入口与 Ctrl/Cmd+, 都打开同一个设置工作区；触发按钮在页面 iframe 内，
  // 宿主不直接管理它的焦点，对话框关闭后焦点回到触发侧是页面自己的职责。
  useEffect(() => {
    settingsLauncher.open = () => openSettings();
    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key === ",") {
        event.preventDefault();
        if (settingsOpen) closeSettings();
        else openSettings();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      settingsLauncher.open = () => {};
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [settingsOpen, openSettings, closeSettings]);

  // React 状态是对话框唯一 owner；原生 Esc 经 onCancel 走 closeSettings 的守卫。
  useEffect(() => {
    const dialog = settingsDialog.current;
    if (!dialog) return;
    if (settingsOpen && !dialog.open) dialog.showModal();
    else if (!settingsOpen && dialog.open) dialog.close();
  }, [settingsOpen]);

  // 设置工作区内的跨目录渲染：初始配置按 route 内嵌任意分节的表单，不关心它属于哪个目录。
  const renderSettingsRoute = useCallback<RenderSettingsRoute>((route, host, props) => {
    const section = settings.entries.find((entry) => entry.route === route);
    if (section) return settings.render(section.id, host, props);
    const member = settingsPlugins.entries.find((entry) => entry.route === route);
    if (member) return settingsPlugins.render(member.id, host, props);
    return null;
  }, [settings, settingsPlugins]);

  // 分节内容就地渲染进对话框；关闭或换节即销毁，同一分节不会同时挂载在两处。
  // 顶层分节由 settings 目录渲染；插件配置区把当前 tab 的成员挂进同一滚动页，
  // 成员间切换不重挂载，各自草稿与守卫独立。dispose 延迟到 microtask：layout cleanup
  // 阶段 React 无法同步 flush 子 root 的卸载提交，同步 dispose 会与渲染器竞争而崩。
  useLayoutEffect(() => {
    if (!settingsOpen || !currentTarget) return;
    const container = settingsContent.current;
    if (!container) return;
    if (currentTarget.kind === "section") {
      const host = document.createElement("div");
      host.className = "shell-settings-entry";
      container.replaceChildren(host);
      const dispose = settings.render(currentTarget.id, host, {
        pages: settings, railActions: railActionEntries, embedded: true, renderRoute: renderSettingsRoute,
      });
      return () => queueMicrotask(() => { dispose(); host.remove(); });
    }
    if (settingsMembers.length === 0) return;
    const mounts = settingsMembers.map((entry) => {
      const wrapper = document.createElement(settingsMembers.length > 1 ? "section" : "div");
      wrapper.className = settingsMembers.length > 1 ? "shell-settings-family-member" : "shell-settings-entry";
      wrapper.dataset.settingsMember = entry.id;
      if (settingsMembers.length > 1) {
        const heading = document.createElement("h3");
        heading.className = "shell-settings-member-title";
        heading.textContent = entry.label;
        wrapper.append(heading);
      }
      const host = document.createElement("div");
      wrapper.append(host);
      return { entry, wrapper, host };
    });
    container.replaceChildren(...mounts.map((item) => item.wrapper));
    const disposers = mounts.map((item) => settingsPlugins.render(item.entry.id, item.host, {
      pages: settingsPlugins, railActions: railActionEntries, embedded: true,
    }));
    return () => {
      queueMicrotask(() => {
        for (const dispose of disposers) dispose();
        for (const item of mounts) item.wrapper.remove();
      });
    };
  }, [settingsOpen, currentTarget, settingsMembers, settings, settingsPlugins, railActionEntries, renderSettingsRoute]);

  // 组合页内深链（如 #telegram_sender-settings）滚动到对应成员；单成员页天然在顶部。
  // 必须等 showModal 的被动 effect 之后执行：对话框未 open 时没有可滚动的布局。
  // 成员表单异步加载后才撑开高度，首帧滚动会被钳制；内容稳定后再对齐一次。
  useEffect(() => {
    if (!settingsOpen || currentTarget?.kind !== "plugins" || settingsMembers.length < 2) return;
    const scroll = () => {
      const target = settingsContent.current?.querySelector(`[data-settings-member="${CSS.escape(currentTarget.memberId)}"]`);
      target?.scrollIntoView({ block: "start" });
    };
    scroll();
    const timer = window.setTimeout(scroll, 800);
    return () => window.clearTimeout(timer);
  }, [settingsOpen, currentTarget, settingsMembers]);

  useLayoutEffect(() => {
    const disposers: WebUiDisposer[] = [];
    for (const entry of bandEntries) {
      const target = pageHosts.current.get(entry.id);
      if (target) disposers.push(pages.render(entry.id, target, { pages, railActions: railActionEntries }));
    }
    return () => {
      for (const dispose of disposers.reverse()) dispose();
    };
  }, [bandEntries, pages, railActionEntries]);

  // 页面切换的进入感：150ms 淡入并上浮 4px。插件样式不得声明全局 @keyframes，
  // 用 WAAPI 表达；时长与曲线仍读 motion token，reduced-motion 下 token 归零即瞬时。
  useLayoutEffect(() => {
    const target = activeId ? pageHosts.current.get(activeId) : undefined;
    if (!target) return;
    const styles = getComputedStyle(target);
    const duration = Number.parseFloat(styles.getPropertyValue("--ak-sys-duration-short"));
    const easing = styles.getPropertyValue("--ak-sys-motion-standard").trim();
    if (!Number.isFinite(duration) || duration <= 0) return;
    const animation = target.animate(
      [{ opacity: 0, transform: "translateY(4px)" }, { opacity: 1, transform: "none" }],
      { duration, easing: easing || "ease-out" },
    );
    return () => animation.cancel();
  }, [activeId]);

  useEffect(() => {
    const syncLocation = (): void => {
      const requested = window.location.hash.slice(1);
      // 深链接（如首次配置的"开始配置"、刷新的 #models）落到设置工作区对应分节。
      const target = settingsTargetForRoute(sectionEntries, pluginEntries, requested);
      if (target) { setWithdrawn(false); openSettings(target); return; }
      const entry = pageFromLocation(bandEntries, defaultPage);
      if (!entry) return;
      const missing = !!requested && !bandEntries.some((item) => item.route === requested);
      if (entry.id === activeId) {
        if (missing) window.history.replaceState(window.history.state, "", `${window.location.pathname}${window.location.search}${entry.route ? `#${entry.route}` : ""}`);
        setWithdrawn(missing); return;
      }
      const previous = bandEntries.find(item => item.id === activeId);
      const base = `${window.location.pathname}${window.location.search}`;
      const restore = () => window.history.replaceState(window.history.state, "", previous?.route ? `${base}#${previous.route}` : base);
      const go = () => {
        window.history.replaceState(window.history.state, "", entry.route ? `${base}#${entry.route}` : base);
        focusAfterNavigation.current = true;
        setActiveId(entry.id); setWithdrawn(missing); setSettingsOpen(false);
      };
      if (window.dispatchEvent(new CustomEvent("akashic:before-navigate", {cancelable:true, detail:{go}}))) go();
      else restore();
    };
    // hash 路由只监听一次变化；同一次跳转的 popstate 会重复清除撤回提示。
    window.addEventListener("hashchange", syncLocation);
    return () => {
      window.removeEventListener("hashchange", syncLocation);
    };
  }, [activeId, defaultPage, bandEntries, sectionEntries, pluginEntries, openSettings]);

  const onBandKeyDown = (event: KeyboardEvent<HTMLElement>): void => {
    if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
    const buttons = [...event.currentTarget.querySelectorAll<HTMLButtonElement>("[data-band-item]")];
    const current = buttons.indexOf(event.target as HTMLButtonElement);
    if (current < 0) return;
    event.preventDefault();
    const next = (current + (event.key === "ArrowRight" ? 1 : -1) + buttons.length) % buttons.length;
    buttons[next].focus();
  };

  // 插件配置区 tab 条与顶栏同一套方向键漫游；tab 只是导航钮，不持有分节状态。
  const onTabsKeyDown = (event: KeyboardEvent<HTMLElement>): void => {
    if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
    const buttons = [...event.currentTarget.querySelectorAll<HTMLButtonElement>("button")];
    const current = buttons.indexOf(event.target as HTMLButtonElement);
    if (current < 0) return;
    event.preventDefault();
    const next = (current + (event.key === "ArrowRight" ? 1 : -1) + buttons.length) % buttons.length;
    buttons[next].focus();
  };

  return <div className="unified-shell">
    {withdrawn && <p role="status" className="config-hint">所选页面暂不可用，已自动切换至默认页面。可在功能设置中查看已安装功能。</p>}
    <header className="product-band" aria-label="Akashic 主导航">
      <div className="product-band__brand" title="Akashic">
        <img src={akashicBrandIcon} alt="" />
        <strong>Akashic</strong>
      </div>
      <nav
        className="product-band__nav"
        aria-label="主要功能"
        onKeyDown={onBandKeyDown}
      >
        <div className="product-band__track" ref={bandTrack}>
          {bandEntries.map((entry) => <button
            key={entry.id}
            type="button"
            data-band-item=""
            data-band-id={entry.id}
            className="product-band__item"
            aria-label={entry.label}
            title={entry.label}
            aria-current={activeId === entry.id ? "page" : undefined}
            onClick={() => openPage(entry)}
          >
            <span className="shell-page-icon" aria-hidden="true" dangerouslySetInnerHTML={{ __html: entry.iconSvg }} />
            <span>{entry.label}</span>
          </button>)}
          <div className="product-band__indicator" aria-hidden="true" />
        </div>
      </nav>
      {currentTarget && <button
        type="button"
        className="product-band__item shell-settings-button"
        aria-label="功能设置"
        aria-haspopup="dialog"
        aria-expanded={settingsOpen}
        title="功能设置 (Ctrl+,)"
        onClick={() => openSettings()}
      >
        <span className="shell-page-icon" aria-hidden="true" dangerouslySetInnerHTML={{ __html: SETTINGS_ICON }} />
      </button>}
    </header>
    <dialog
      ref={settingsDialog}
      className="shell-settings-dialog"
      aria-label="功能设置"
      onCancel={(event) => { event.preventDefault(); closeSettings(); }}
      onClose={() => setSettingsOpen(false)}
    >
      <nav className="shell-settings-nav" aria-label="设置分节">
        <h2>功能设置</h2>
        {settingsNav.map((item) => {
          if (item.kind === "plugins") {
            return <button
              key="plugins"
              type="button"
              aria-current={currentTarget?.kind === "plugins" ? "true" : undefined}
              onClick={() => openSettings({ kind: "plugins", memberId: pluginEntries[0].id })}
            >
              <span className="shell-page-icon" aria-hidden="true" dangerouslySetInnerHTML={{ __html: SETTINGS_PLUGINS_ICON }} />
              <span>{SETTINGS_PLUGINS_LABEL}</span>
            </button>;
          }
          const entry = item.entry;
          return <button
            key={entry.id}
            type="button"
            aria-current={currentTarget?.kind === "section" && currentTarget.id === entry.id ? "true" : undefined}
            onClick={() => openSettings({ kind: "section", id: entry.id })}
          >
            <span className="shell-page-icon" aria-hidden="true" dangerouslySetInnerHTML={{ __html: entry.iconSvg }} />
            <span>{entry.label}</span>
          </button>;
        })}
      </nav>
      <div className="shell-settings-body">
        <header>
          <button type="button" onClick={closeSettings} aria-label="关闭设置">关闭</button>
        </header>
        {currentTarget?.kind === "plugins" && settingsTabs.length > 1 && (
          <div className="shell-settings-tabs" role="group" aria-label={SETTINGS_PLUGINS_LABEL} onKeyDown={onTabsKeyDown}>
            {settingsTabs.map((tab) => {
              if (tab.kind === "entry") {
                return <button
                  key={tab.entry.id}
                  type="button"
                  aria-current={currentTarget.memberId === tab.entry.id ? "true" : undefined}
                  onClick={() => openSettings({ kind: "plugins", memberId: tab.entry.id })}
                >{tab.entry.label}</button>;
              }
              return <button
                key={`family:${tab.id}`}
                type="button"
                aria-current={tab.entries.some((entry) => entry.id === currentTarget.memberId) ? "true" : undefined}
                onClick={() => openSettings({ kind: "plugins", memberId: tab.entries[0].id })}
              >{tab.label}</button>;
            })}
          </div>
        )}
        <div ref={settingsContent} className="shell-settings-page" />
      </div>
    </dialog>
    <div className="shell-view-stack">
      {bandEntries.map((entry) => <section
        key={entry.id}
        ref={(node) => {
          if (node) pageHosts.current.set(entry.id, node);
          else pageHosts.current.delete(entry.id);
        }}
        className={`shell-view ${activeId === entry.id ? "is-active" : ""}`}
        aria-hidden={activeId !== entry.id}
      />)}
    </div>
  </div>;
}

function checkPages(entries: readonly WebEntry[]): ShellPage[] {
  const pages = entries.map((entry) => {
    if ("section" in entry || "group" in entry || "family" in entry) {
      throw new Error(`Shell 页面合同无效: ${entry.id} 仍使用已退役的 section/group/family 字段`);
    }
    if (
      typeof entry.label !== "string"
      || typeof entry.route !== "string"
      || typeof entry.iconSvg !== "string"
      || !entry.iconSvg.startsWith("<svg")
    ) {
      throw new Error(`Shell 页面合同无效: ${entry.id}`);
    }
    return entry as ShellPage;
  });
  if (new Set(pages.map((entry) => entry.route)).size !== pages.length) {
    throw new Error("Shell 页面 route 不能重复");
  }
  return pages;
}

function checkSettingsSections(entries: readonly WebEntry[]): ShellSettingsSection[] {
  const sections = entries.map((entry) => {
    if (
      typeof entry.label !== "string"
      || typeof entry.route !== "string"
      || typeof entry.iconSvg !== "string"
      || !entry.iconSvg.startsWith("<svg")
    ) {
      throw new Error(`Shell 设置分节合同无效: ${entry.id}`);
    }
    return entry as ShellSettingsSection;
  });
  if (new Set(sections.map((entry) => entry.route)).size !== sections.length) {
    throw new Error("Shell 设置分节 route 不能重复");
  }
  return sections;
}

function checkSettingsPlugins(entries: readonly WebEntry[]): ShellSettingsPlugin[] {
  const members = entries.map((entry) => {
    if (
      typeof entry.label !== "string"
      || typeof entry.route !== "string"
      || (entry.family !== undefined && typeof entry.family !== "string")
      || (entry.familyLabel !== undefined && typeof entry.familyLabel !== "string")
      || (entry.family !== undefined && !entry.familyLabel)
    ) {
      throw new Error(`Shell 插件配置分节合同无效: ${entry.id}`);
    }
    return entry as ShellSettingsPlugin;
  });
  // family 组合标签由成员自带，同一 family 的 familyLabel 必须一致。
  const familyLabels = new Map<string, string>();
  for (const entry of members) {
    if (!entry.family) continue;
    const label = familyLabels.get(entry.family);
    if (label === undefined) familyLabels.set(entry.family, entry.familyLabel!);
    else if (label !== entry.familyLabel) throw new Error(`Shell 插件配置组合 ${entry.family} 的 familyLabel 不一致`);
  }
  if (new Set(members.map((entry) => entry.route)).size !== members.length) {
    throw new Error("Shell 插件配置分节 route 不能重复");
  }
  return members;
}

function checkRailActions(entries: readonly WebEntry[]): ShellRailAction[] {
  const actions = entries.map((entry) => {
    if (
      typeof entry.label !== "string"
      || typeof entry.iconSvg !== "string"
      || !entry.iconSvg.startsWith("<svg")
      || typeof entry.onActivate !== "function"
    ) {
      throw new Error(`Shell 底栏动作合同无效: ${entry.id}`);
    }
    return entry as ShellRailAction;
  });
  if (new Set(actions.map((entry) => entry.id)).size !== actions.length) {
    throw new Error("Shell 底栏动作 id 不能重复");
  }
  return actions;
}

function pageFromLocation(entries: ShellPage[], fallback: ShellPage | undefined): ShellPage | undefined {
  const route = window.location.hash.slice(1);
  return entries.find((entry) => entry.route === route) ?? fallback;
}

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
import type { ShellRailAction } from "@akashic/shell-ui-v1";

type ShellPage = WebEntry & {
  label: string;
  route: string;
  iconSvg: string;
  section?: string;
  group?: string;
  family?: string;
};

const RAIL_ACTIONS_MOUNT = "shell.rail-actions.v1";

/** 设置分节声明 group 即归入对应分组；分组标签与图标由 Shell 拥有，插件不各自起名。 */
const SETTINGS_GROUP_LABELS: Record<string, string> = { plugins: "插件" };
const SETTINGS_GROUP_ICONS: Record<string, string> = {
  plugins: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 7V4a2 2 0 0 0-4 0v3"/><path d="M17 7h3a2 2 0 0 1 2 2v3a2 2 0 0 1-2 2h-1"/><path d="M7 7H4a2 2 0 0 0-2 2v3a2 2 0 0 0 2 2h1"/><path d="M14 21v-3a2 2 0 0 0-4 0v3"/><rect x="7" y="7" width="10" height="10" rx="2"/></svg>',
};
/** 同组内 ≥2 个分节声明同一 family 时折叠为一个组合页；family 标签由 Shell 拥有，插件只声明归属。 */
const SETTINGS_FAMILY_LABELS: Record<string, string> = { telegram: "Telegram" };

type SettingsNavItem =
  | { kind: "entry"; entry: ShellPage }
  | { kind: "group"; id: string; entries: ShellPage[] };

type SettingsTab =
  | { kind: "entry"; entry: ShellPage }
  | { kind: "family"; id: string; entries: ShellPage[] };

/** 组内分节折叠成 tab：≥2 个成员的 family 合成一个组合 tab，单成员 family 与普通分节一样独立成 tab。 */
function buildSettingsTabs(entries: ShellPage[]): SettingsTab[] {
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
        tabs.push({ kind: "family", id: entry.family, entries: [entry] });
      } else {
        (tabs[at] as { kind: "family"; entries: ShellPage[] }).entries.push(entry);
      }
    } else {
      tabs.push({ kind: "entry", entry });
    }
  }
  return tabs;
}

const SETTINGS_ICON = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>';

/** 对话框留在 Shell 组件内；底栏动作与快捷键经这个模块级入口打开它。 */
const settingsLauncher = { open: () => {} };

/** Register the ordinary Shell plugin as the only owner of the outer frame. */
export function activate(ctx: WebHostContextV1): WebUiDisposer {
  const disposers = [
    ctx.ui.inject("web.root.v1", (mount) => mount.register({
      id: "shell",
      children: [
        { id: "shell.pages.v1", cardinality: "list" },
        { id: RAIL_ACTIONS_MOUNT, cardinality: "list" },
      ],
      render(host, view) {
        const root = createRoot(host);
        root.render(<Shell
          pages={view.child("shell.pages.v1")}
          railActions={view.child(RAIL_ACTIONS_MOUNT)}
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

function Shell({ pages, railActions }: { pages: WebMountView; railActions: WebMountView }): React.ReactElement {
  const entries = useMemo(() => checkPages(pages.entries), [pages.entries]);
  const railActionEntries = useMemo(() => checkRailActions(railActions.entries), [railActions.entries]);
  const bandEntries = useMemo(() => entries.filter((entry) => entry.section !== "settings"), [entries]);
  const settingsEntries = useMemo(() => entries.filter((entry) => entry.section === "settings"), [entries]);
  // 顶层分节与分组按声明顺序混排；组内分节保持注册顺序，在内容区以 tab 呈现。
  const settingsNav = useMemo(() => {
    const items: SettingsNavItem[] = [];
    const groupAt = new Map<string, number>();
    for (const entry of settingsEntries) {
      if (!entry.group) { items.push({ kind: "entry", entry }); continue; }
      const at = groupAt.get(entry.group);
      if (at === undefined) {
        groupAt.set(entry.group, items.length);
        items.push({ kind: "group", id: entry.group, entries: [entry] });
      } else {
        (items[at] as { kind: "group"; entries: ShellPage[] }).entries.push(entry);
      }
    }
    return items;
  }, [settingsEntries]);
  const defaultPage = bandEntries.find((entry) => entry.route === "") ?? bandEntries[0] ?? entries[0];
  const requestedRoute = window.location.hash.slice(1);
  const requestedEntry = entries.find((entry) => entry.route === requestedRoute);
  const initialSettings = requestedEntry?.section === "settings" ? requestedEntry : undefined;
  const [withdrawn, setWithdrawn] = useState(() => !!requestedRoute && !entries.some(entry => entry.route === requestedRoute));
  const [activeId, setActiveId] = useState(() => (initialSettings ? defaultPage : pageFromLocation(entries, defaultPage))?.id ?? "");
  const [settingsOpen, setSettingsOpen] = useState(!!initialSettings);
  const [settingsEntryId, setSettingsEntryId] = useState(initialSettings?.id ?? "");
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
  }, [activeId, entries]);

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

  const openSettings = useCallback((entry?: ShellPage): void => {
    const target = entry ?? settingsEntries.find((item) => item.id === settingsEntryId) ?? settingsEntries[0];
    if (!target) return;
    const go = () => {
      setSettingsEntryId(target.id);
      setSettingsOpen(true);
      const base = `${window.location.pathname}${window.location.search}`;
      window.history.replaceState(null, "", target.route ? `${base}#${target.route}` : base);
    };
    if (window.dispatchEvent(new CustomEvent("akashic:before-navigate", { cancelable: true, detail: { go } }))) go();
  }, [settingsEntries, settingsEntryId]);

  const closeSettings = useCallback((): void => {
    const go = () => {
      setSettingsOpen(false);
      window.history.replaceState(null, "", bandRoute());
    };
    if (window.dispatchEvent(new CustomEvent("akashic:before-navigate", { cancelable: true, detail: { go } }))) go();
  }, [bandRoute]);

  useLayoutEffect(() => {
    if (withdrawn) {
      const entry = entries.find(item => item.id === activeId);
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

  const currentSettingsEntry = settingsEntries.find((item) => item.id === settingsEntryId) ?? settingsEntries[0];
  const currentSettingsGroup = currentSettingsEntry?.group
    ? settingsNav.find((item): item is Extract<SettingsNavItem, { kind: "group" }> => item.kind === "group" && item.id === currentSettingsEntry.group)
    : undefined;
  const settingsTabs = useMemo(() => buildSettingsTabs(currentSettingsGroup?.entries ?? []), [currentSettingsGroup]);
  const activeSettingsTab = settingsTabs.find((tab) =>
    tab.kind === "entry" ? tab.entry.id === currentSettingsEntry?.id : tab.entries.some((entry) => entry.id === currentSettingsEntry?.id));
  // 组合页渲染成员列表：family tab 渲染全部成员，普通 tab 只渲染当前分节。
  const settingsMembers = useMemo(
    () => activeSettingsTab?.kind === "family" ? activeSettingsTab.entries : currentSettingsEntry ? [currentSettingsEntry] : [],
    [activeSettingsTab, currentSettingsEntry],
  );

  // 分节页面就地渲染进对话框；关闭或换节即销毁，同一页面不会同时挂载在两处。
  // family 组合页把成员表单挂进同一滚动页，成员间切换不重挂载，各自草稿与守卫独立。
  // 每个分节挂进独立 wrapper，dispose 延迟到 microtask：layout cleanup 阶段 React 无法
  // 同步 flush 子 root 的卸载提交，同步 dispose 会与渲染器自身的 replaceChildren 竞争而崩。
  useLayoutEffect(() => {
    if (!settingsOpen) return;
    const container = settingsContent.current;
    if (!container || settingsMembers.length === 0) return;
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
    const disposers = mounts.map((item) => pages.render(item.entry.id, item.host, { pages, railActions: railActionEntries, embedded: true }));
    return () => {
      queueMicrotask(() => {
        for (const dispose of disposers) dispose();
        for (const item of mounts) item.wrapper.remove();
      });
    };
  }, [settingsOpen, settingsMembers, pages, railActionEntries]);

  // 组合页内深链（如 #telegram_sender-settings）滚动到对应成员；单成员页天然在顶部。
  // 必须等 showModal 的被动 effect 之后执行：对话框未 open 时没有可滚动的布局。
  // 成员表单异步加载后才撑开高度，首帧滚动会被钳制；内容稳定后再对齐一次。
  useEffect(() => {
    if (!settingsOpen || settingsMembers.length < 2) return;
    const scroll = () => {
      const target = settingsContent.current?.querySelector(`[data-settings-member="${CSS.escape(settingsEntryId)}"]`);
      target?.scrollIntoView({ block: "start" });
    };
    scroll();
    const timer = window.setTimeout(scroll, 800);
    return () => window.clearTimeout(timer);
  }, [settingsOpen, settingsEntryId, settingsMembers]);

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
      const entry = pageFromLocation(entries, defaultPage);
      if (!entry) return;
      const requested = window.location.hash.slice(1);
      const missing = !!requested && !entries.some(item => item.route === requested);
      // 深链接（如首次配置的"开始配置"、刷新的 #models）落到设置工作区对应分节。
      if (entry.section === "settings") { setWithdrawn(missing); openSettings(entry); return; }
      if (entry.id === activeId) {
        if (missing) window.history.replaceState(window.history.state, "", `${window.location.pathname}${window.location.search}${entry.route ? `#${entry.route}` : ""}`);
        setWithdrawn(missing); return;
      }
      const previous = entries.find(item => item.id === activeId);
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
  }, [activeId, defaultPage, entries, openSettings]);

  const onBandKeyDown = (event: KeyboardEvent<HTMLElement>): void => {
    if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
    const buttons = [...event.currentTarget.querySelectorAll<HTMLButtonElement>("[data-band-item]")];
    const current = buttons.indexOf(event.target as HTMLButtonElement);
    if (current < 0) return;
    event.preventDefault();
    const next = (current + (event.key === "ArrowRight" ? 1 : -1) + buttons.length) % buttons.length;
    buttons[next].focus();
  };

  // 设置组内 tab 条与顶栏同一套方向键漫游；tab 只是导航钮，不持有分节状态。
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
    {withdrawn && <p role="status" className="config-hint">原页面已撤回或暂不可用，已打开当前可用页面。可以从功能设置查看已安装功能。</p>}
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
          if (item.kind === "entry") {
            const entry = item.entry;
            return <button
              key={entry.id}
              type="button"
              aria-current={entry.id === currentSettingsEntry?.id ? "true" : undefined}
              onClick={() => openSettings(entry)}
            >
              <span className="shell-page-icon" aria-hidden="true" dangerouslySetInnerHTML={{ __html: entry.iconSvg }} />
              <span>{entry.label}</span>
            </button>;
          }
          const active = item.entries.some((entry) => entry.id === currentSettingsEntry?.id);
          return <button
            key={item.id}
            type="button"
            aria-current={active ? "true" : undefined}
            onClick={() => openSettings(item.entries[0])}
          >
            <span className="shell-page-icon" aria-hidden="true" dangerouslySetInnerHTML={{ __html: SETTINGS_GROUP_ICONS[item.id] ?? SETTINGS_ICON }} />
            <span>{SETTINGS_GROUP_LABELS[item.id] ?? item.id}</span>
          </button>;
        })}
      </nav>
      <div className="shell-settings-body">
        <header>
          <button type="button" onClick={closeSettings} aria-label="关闭设置">关闭</button>
        </header>
        {settingsTabs.length > 1 && currentSettingsGroup && (
          <div className="shell-settings-tabs" role="group" aria-label={SETTINGS_GROUP_LABELS[currentSettingsGroup.id] ?? currentSettingsGroup.id} onKeyDown={onTabsKeyDown}>
            {settingsTabs.map((tab) => {
              if (tab.kind === "entry") {
                return <button
                  key={tab.entry.id}
                  type="button"
                  aria-current={tab.entry.id === currentSettingsEntry?.id ? "true" : undefined}
                  onClick={() => openSettings(tab.entry)}
                >{tab.entry.label}</button>;
              }
              return <button
                key={`family:${tab.id}`}
                type="button"
                aria-current={tab.entries.some((entry) => entry.id === currentSettingsEntry?.id) ? "true" : undefined}
                onClick={() => openSettings(tab.entries[0])}
              >{SETTINGS_FAMILY_LABELS[tab.id] ?? tab.id}</button>;
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
    if (
      typeof entry.label !== "string"
      || typeof entry.route !== "string"
      || typeof entry.iconSvg !== "string"
      || !entry.iconSvg.startsWith("<svg")
      || (entry.family !== undefined && typeof entry.family !== "string")
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

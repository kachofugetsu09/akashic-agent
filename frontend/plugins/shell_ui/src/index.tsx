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
};

const RAIL_ACTIONS_MOUNT = "shell.rail-actions.v1";

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

  // 分节页面就地渲染进对话框；关闭或换节即销毁，同一页面不会同时挂载在两处。
  useLayoutEffect(() => {
    if (!settingsOpen) return;
    const host = settingsContent.current;
    const entry = settingsEntries.find((item) => item.id === settingsEntryId) ?? settingsEntries[0];
    if (!host || !entry) return;
    return pages.render(entry.id, host, { pages, railActions: railActionEntries, embedded: true });
  }, [settingsOpen, settingsEntryId, settingsEntries, pages, railActionEntries]);

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
        {settingsEntries.map((entry) => {
          const current = (settingsEntries.find((item) => item.id === settingsEntryId) ?? settingsEntries[0])?.id;
          return <button
            key={entry.id}
            type="button"
            aria-current={entry.id === current ? "true" : undefined}
            onClick={() => openSettings(entry)}
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

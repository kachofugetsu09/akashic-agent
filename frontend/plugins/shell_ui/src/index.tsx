import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { createRoot } from "react-dom/client";
import { SunMoon } from "lucide-react";
import "./style.css";
import { akashicBrandIcon } from "./brand";
import type {
  WebEntry,
  WebHostContextV1,
  WebMountView,
  WebUiDisposer,
} from "@akashic/web-ui-v1";
import { cycleTheme, themes, useTheme } from "@akashic/web-ui-v1";

type ShellPage = WebEntry & {
  label: string;
  route: string;
  iconSvg: string;
  section?: string;
};

/** Register the ordinary Shell plugin as the only owner of the outer frame. */
export function activate(ctx: WebHostContextV1): WebUiDisposer {
  return ctx.ui.inject("web.root.v1", (mount) => mount.register({
    id: "shell",
    children: [{ id: "shell.pages.v1", cardinality: "list" }],
    render(host, view) {
      const root = createRoot(host);
      root.render(<Shell pages={view.child("shell.pages.v1")} />);
      return () => root.unmount();
    },
  }));
}

function Shell({ pages }: { pages: WebMountView }): React.ReactElement {
  const entries = useMemo(() => checkPages(pages.entries), [pages.entries]);
  const bandEntries = useMemo(() => entries.filter((entry) => entry.section !== "settings"), [entries]);
  const settingsEntries = useMemo(() => entries.filter((entry) => entry.section === "settings"), [entries]);
  const defaultPage = bandEntries.find((entry) => entry.route === "") ?? bandEntries[0] ?? entries[0];
  const requestedRoute = window.location.hash.slice(1);
  const [withdrawn, setWithdrawn] = useState(() => !!requestedRoute && !entries.some(entry => entry.route === requestedRoute));
  const [activeId, setActiveId] = useState(() => pageFromLocation(entries, defaultPage)?.id ?? "");
  const pageHosts = useRef(new Map<string, HTMLElement>());
  const settingsDialog = useRef<HTMLDialogElement>(null);
  const settingsTrigger = useRef<HTMLButtonElement>(null);
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
    if (entry.id === activeId) { setWithdrawn(false); settingsDialog.current?.close(); return; }
    const go = () => {
      focusAfterNavigation.current = true;
      setActiveId(entry.id); setWithdrawn(false);
      const base = `${window.location.pathname}${window.location.search}`;
      window.history.replaceState(null, "", entry.route ? `${base}#${entry.route}` : base);
      settingsDialog.current?.close();
    };
    if (window.dispatchEvent(new CustomEvent("akashic:before-navigate", { cancelable: true, detail: { go } }))) go();
  }, [activeId]);

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
    const current = document.querySelector<HTMLButtonElement>('.product-band__track button[aria-current="page"]');
    (current ?? settingsTrigger.current)?.focus();
  }, [activeId]);

  useLayoutEffect(() => {
    const disposers: WebUiDisposer[] = [];
    for (const entry of entries) {
      const target = pageHosts.current.get(entry.id);
      if (target) disposers.push(pages.render(entry.id, target, { pages }));
    }
    return () => {
      for (const dispose of disposers.reverse()) dispose();
    };
  }, [entries, pages]);

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
        setActiveId(entry.id); setWithdrawn(missing);
        settingsDialog.current?.close();
      };
      if (window.dispatchEvent(new CustomEvent("akashic:before-navigate", {cancelable:true, detail:{go}}))) go();
      else restore();
    };
    // hash 路由只监听一次变化；同一次跳转的 popstate 会重复清除撤回提示。
    window.addEventListener("hashchange", syncLocation);
    return () => {
      window.removeEventListener("hashchange", syncLocation);
    };
  }, [activeId, defaultPage, entries]);

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
      <div className="product-band__footer">
        <button ref={settingsTrigger} type="button" className="theme-cycle-button" onClick={() => settingsDialog.current?.showModal()}>功能设置</button>
        <dialog ref={settingsDialog} className="shell-settings-dialog" aria-label="功能设置">
          <header><h2>功能设置</h2><button type="button" onClick={() => settingsDialog.current?.close()} aria-label="关闭设置目录">关闭</button></header>
          <nav>{settingsEntries.map((entry) => <button key={entry.id} type="button" onClick={() => openPage(entry)}>
            <span className="shell-page-icon" aria-hidden="true" dangerouslySetInnerHTML={{ __html: entry.iconSvg }} />
            <span>{entry.label}</span>
          </button>)}</nav>
        </dialog>
        <ThemeToggle />
      </div>
    </header>
    <div className="shell-view-stack">
      {entries.map((entry) => <section
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

function pageFromLocation(entries: ShellPage[], fallback: ShellPage | undefined): ShellPage | undefined {
  const route = window.location.hash.slice(1);
  return entries.find((entry) => entry.route === route) ?? fallback;
}

function ThemeToggle(): React.ReactElement {
  const theme = useTheme();
  const options = themes();
  const currentIndex = options.findIndex((option) => option.id === theme.id);
  const nextTheme = options[(currentIndex + 1) % options.length];
  return <button
    type="button"
    onClick={() => cycleTheme()}
    title={`当前主题：${theme.label}；切换到${nextTheme.label}`}
    aria-label={`切换主题，当前为${theme.label}，下一主题为${nextTheme.label}`}
    className="theme-cycle-button"
  >
    <SunMoon size={20} strokeWidth={2} aria-hidden="true" />
    <span>主题 · {theme.label}</span>
  </button>;
}

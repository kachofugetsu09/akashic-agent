import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { SunMoon } from "lucide-react";
import "../plugin-styles/shell.css";
import { akashicBrandIcon } from "../../../chat/src/akashic-brand";
import { cycleTheme, themes, useTheme } from "../../../theme/src/theme-runtime";
import type {
  WebEntry,
  WebHostContextV1,
  WebMountView,
  WebUiDisposer,
} from "@akashic/web-ui-v1";

type ShellStatus = "needs_setup" | "starting" | "ready";

type ShellPage = WebEntry & {
  label: string;
  route: string;
  iconSvg: string;
  setup?: boolean;
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
  const theme = useTheme();
  const entries = useMemo(() => checkPages(pages.entries), [pages.entries]);
  const defaultPage = entries.find((entry) => entry.route === "") ?? entries[0];
  const setupPage = entries.find((entry) => entry.setup);
  const [activeId, setActiveId] = useState(() => pageFromLocation(entries, defaultPage)?.id ?? "");
  const [shellStatus, setShellStatus] = useState<ShellStatus>("starting");
  const pageHosts = useRef(new Map<string, HTMLElement>());
  const serviceOrigin = window.location.origin;

  const openPage = useCallback((entry: ShellPage): void => {
    setActiveId(entry.id);
    const base = `${window.location.pathname}${window.location.search}`;
    window.history.replaceState(null, "", entry.route ? `${base}#${entry.route}` : base);
  }, []);

  useLayoutEffect(() => {
    const disposers: WebUiDisposer[] = [];
    for (const entry of entries) {
      const target = pageHosts.current.get(entry.id);
      if (target) disposers.push(pages.render(entry.id, target));
    }
    return () => {
      for (const dispose of disposers.reverse()) dispose();
    };
  }, [entries, pages]);

  useEffect(() => {
    const syncLocation = (): void => {
      const entry = pageFromLocation(entries, defaultPage);
      if (entry) setActiveId(entry.id);
    };
    window.addEventListener("hashchange", syncLocation);
    window.addEventListener("popstate", syncLocation);
    return () => {
      window.removeEventListener("hashchange", syncLocation);
      window.removeEventListener("popstate", syncLocation);
    };
  }, [defaultPage, entries]);

  useEffect(() => {
    let active = true;
    const refresh = async (): Promise<void> => {
      try {
        const response = await fetch("/api/shell/state", { cache: "no-store" });
        const state = await response.json() as { status?: unknown; chatReady?: unknown };
        if (!response.ok || typeof state.status !== "string" || typeof state.chatReady !== "boolean") {
          throw new Error("/api/shell/state 返回了无效状态");
        }
        if (active) setShellStatus(state.chatReady ? "ready" : state.status as ShellStatus);
      } catch (error) {
        console.error("[shell-ui] shell readiness failed", error);
        if (active) setShellStatus("starting");
      }
    };
    void refresh();
    const timer = window.setInterval(refresh, 1_500);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (shellStatus === "needs_setup" && setupPage && activeId !== setupPage.id) {
      openPage(setupPage);
    }
  }, [activeId, openPage, setupPage, shellStatus]);

  useEffect(() => {
    for (const frame of pageHosts.current.values()) {
      frame.querySelectorAll("iframe").forEach((iframe) => iframe.contentWindow?.postMessage(
        { type: "akashic.theme", themeId: theme.id },
        serviceOrigin,
      ));
    }
  }, [serviceOrigin, theme.id]);

  return <div className="unified-shell">
    <header className="primary-band" aria-label="Akashic 主导航">
      <div className="primary-band-brand" title="Akashic">
        <img src={akashicBrandIcon} alt="" />
        <strong>Akashic</strong>
      </div>
      <nav className="primary-band-nav" aria-label="主要功能">
        {entries.map((entry) => <button
          key={entry.id}
          type="button"
          className={`primary-rail-button ${activeId === entry.id ? "is-active" : ""}`}
          aria-label={entry.label}
          title={entry.label}
          aria-current={activeId === entry.id ? "page" : undefined}
          onClick={() => openPage(shellStatus === "needs_setup" && setupPage ? setupPage : entry)}
        >
          <span className="shell-page-icon" aria-hidden="true" dangerouslySetInnerHTML={{ __html: entry.iconSvg }} />
          <span>{entry.label}</span>
        </button>)}
      </nav>
      <div className="primary-band-footer"><ThemeToggle /></div>
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
  if (pages.filter((entry) => entry.setup).length > 1) {
    throw new Error("Shell 只能注册一个 setup 页面");
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

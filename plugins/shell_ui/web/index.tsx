import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
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
  const defaultPage = entries.find((entry) => entry.route === "") ?? entries[0];
  const [activeId, setActiveId] = useState(() => pageFromLocation(entries, defaultPage)?.id ?? "");
  const pageHosts = useRef(new Map<string, HTMLElement>());

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
          onClick={() => openPage(entry)}
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

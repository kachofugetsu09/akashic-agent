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

// beUI Wheel Picker 的落格手感：松手按减速度滑行，再吸到最近一格，并带一点回弹。
const DECELERATION = 0.00042;
const MAX_VELOCITY = 0.18;
const VELOCITY_WINDOW = 90;
const WHEEL_SENS = 0.012;
const WHEEL_SETTLE = 110;
const BACK = 1.35;

const clamp = (value: number, lo: number, hi: number) => Math.max(lo, Math.min(value, hi));
const easeOutBack = (progress: number) => 1 + (BACK + 1) * (progress - 1) ** 3 + BACK * (progress - 1) ** 2;

function prefersReducedMotion(): boolean {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function readCenters(track: HTMLElement): number[] {
  return [...track.querySelectorAll<HTMLElement>("[data-band-item]")].map(
    (item) => item.offsetLeft + item.offsetWidth / 2,
  );
}

function slotSpan(centers: number[]): number {
  if (centers.length < 2) return 80;
  return Math.max(48, (centers[centers.length - 1] - centers[0]) / (centers.length - 1));
}

function offsetAt(centers: number[], index: number): number {
  const last = centers.length - 1;
  const span = slotSpan(centers);
  if (index <= 0) return centers[0] + index * span;
  if (index >= last) return centers[last] + (index - last) * span;
  const lo = Math.floor(index);
  const mix = index - lo;
  return centers[lo] * (1 - mix) + centers[lo + 1] * mix;
}

function fadeFor(index: number, last: number): string {
  const start = index > 0.08;
  const end = index < last - 0.08;
  if (start && end) return "both";
  if (start) return "start";
  if (end) return "end";
  return "none";
}

/** 正中的一格放大、加粗、去模糊；旁边的格子按距离略微发虚。 */
function paintFocus(items: HTMLElement[], index: number) {
  const focus = Math.round(index);
  for (const [itemIndex, item] of items.entries()) {
    const distance = Math.abs(itemIndex - index);
    const near = Math.max(0, 1 - distance);
    const mark = itemIndex === focus ? "1" : "0";
    if (item.dataset.focus !== mark) item.dataset.focus = mark;
    const blur = distance < 0.15 ? 0 : distance < 1.8 ? Math.min(1, (distance - 0.15) * 0.7) : 0;
    item.style.transform = `scale(${(1 + 0.18 * near).toFixed(3)})`;
    item.style.filter = blur > 0.04 ? `blur(${blur.toFixed(2)}px)` : "";
    item.style.opacity = (0.48 + 0.52 * Math.max(0, 1 - distance * 0.62)).toFixed(3);
  }
}

function useBandScale(count: number, onSettle: (index: number) => void) {
  const scrollerRef = useRef<HTMLElement>(null);
  const trackRef = useRef<HTMLDivElement>(null);
  const indexRef = useRef(0);
  const rafRef = useRef(0);
  const movedRef = useRef(false);
  const settleRef = useRef(onSettle);
  settleRef.current = onSettle;

  const paint = (index: number) => {
    const scroller = scrollerRef.current;
    const track = trackRef.current;
    if (!scroller || !track) return;
    const items = [...track.querySelectorAll<HTMLElement>("[data-band-item]")];
    if (!items.length) return;
    paintFocus(items, index);
    const centers = readCenters(track);
    const x = offsetAt(centers, index);
    track.style.transform = `translate3d(${scroller.clientWidth / 2 - x}px, 0, 0)`;
    const fade = fadeFor(index, centers.length - 1);
    if (scroller.dataset.fade !== fade) scroller.dataset.fade = fade;
  };

  const stop = () => cancelAnimationFrame(rafRef.current);

  const glide = (to: number, duration: number, done?: () => void) => {
    stop();
    const from = indexRef.current;
    const distance = to - from;
    const finish = () => {
      indexRef.current = to;
      paint(to);
      settleRef.current(to);
      done?.();
    };
    if (!distance || duration <= 0 || prefersReducedMotion()) {
      finish();
      return;
    }
    const start = performance.now();
    const tick = (now: number) => {
      const progress = (now - start) / duration;
      if (progress >= 1) {
        finish();
        return;
      }
      indexRef.current = from + distance * easeOutBack(progress);
      paint(indexRef.current);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  };

  const fling = (velocity: number) => {
    const last = Math.max(0, count - 1);
    const from = indexRef.current;
    if (from < 0 || from > last) {
      glide(clamp(Math.round(from), 0, last), 260);
      return;
    }
    const direction = Math.sign(velocity);
    const coast = ((velocity * velocity) / (2 * DECELERATION)) * direction;
    const to = clamp(Math.round(from + coast), 0, last);
    const duration = clamp(Math.sqrt(Math.abs(to - from)) * 300 + 240, 280, 1700);
    glide(to, duration);
  };

  useLayoutEffect(() => {
    paint(indexRef.current);
  });

  useEffect(() => {
    const scroller = scrollerRef.current;
    const track = trackRef.current;
    if (!scroller || !track) return;
    const drag = {
      x: 0,
      index: 0,
      span: 80,
      pts: [] as [number, number][],
      active: false,
    };
    let frame = 0;
    let latestX = 0;
    let wheelTimer = 0;

    const begin = (x: number) => {
      stop();
      movedRef.current = false;
      drag.active = true;
      drag.x = x;
      drag.index = indexRef.current;
      drag.span = slotSpan(readCenters(track));
      drag.pts = [[x, performance.now()]];
      scroller.dataset.grabbing = "true";
    };
    const move = (x: number) => {
      if (!drag.active) return;
      if (Math.abs(x - drag.x) > 6) movedRef.current = true;
      latestX = x;
      drag.pts.push([x, performance.now()]);
      if (drag.pts.length > 8) drag.pts.shift();
      if (frame) return;
      frame = requestAnimationFrame(() => {
        frame = 0;
        if (!drag.active) return;
        let next = drag.index + (drag.x - latestX) / drag.span;
        const last = Math.max(0, count - 1);
        if (next < 0) next *= 0.3;
        else if (next > last) next = last + (next - last) * 0.3;
        indexRef.current = next;
        paint(next);
      });
    };
    const end = () => {
      if (!drag.active) return;
      drag.active = false;
      delete scroller.dataset.grabbing;
      if (frame) cancelAnimationFrame(frame);
      frame = 0;
      const pts = drag.pts;
      let velocity = 0;
      if (pts.length > 1) {
        const latest = pts[pts.length - 1];
        let ref = pts[0];
        for (const point of pts) {
          if (latest[1] - point[1] <= VELOCITY_WINDOW) {
            ref = point;
            break;
          }
        }
        const dt = latest[1] - ref[1];
        if (dt > 0) velocity = clamp((ref[0] - latest[0]) / drag.span / dt, -MAX_VELOCITY, MAX_VELOCITY);
      }
      if (!movedRef.current) return;
      fling(velocity);
    };
    const onPointerDown = (event: PointerEvent) => {
      if (event.pointerType === "touch" || prefersReducedMotion()) return;
      begin(event.clientX);
      scroller.setPointerCapture(event.pointerId);
    };
    const onPointerMove = (event: PointerEvent) => {
      if (event.pointerType === "touch") return;
      move(event.clientX);
    };
    const onPointerUp = (event: PointerEvent) => {
      if (event.pointerType === "touch") return;
      if (scroller.hasPointerCapture(event.pointerId)) scroller.releasePointerCapture(event.pointerId);
      end();
    };
    const onWheel = (event: WheelEvent) => {
      if (prefersReducedMotion()) return;
      event.preventDefault();
      stop();
      const raw = Math.abs(event.deltaX) > Math.abs(event.deltaY) ? event.deltaX : event.deltaY;
      const px = event.deltaMode === 1 ? raw * 16 : raw;
      const last = Math.max(0, count - 1);
      indexRef.current = clamp(indexRef.current + px * WHEEL_SENS, 0, last);
      paint(indexRef.current);
      window.clearTimeout(wheelTimer);
      wheelTimer = window.setTimeout(() => glide(clamp(Math.round(indexRef.current), 0, last), 240), WHEEL_SETTLE);
    };
    scroller.addEventListener("pointerdown", onPointerDown);
    scroller.addEventListener("pointermove", onPointerMove);
    scroller.addEventListener("pointerup", onPointerUp);
    scroller.addEventListener("pointercancel", onPointerUp);
    scroller.addEventListener("wheel", onWheel, { passive: false });
    const onTouchStart = (event: TouchEvent) => {
      const touch = event.touches[0];
      if (touch) begin(touch.clientX);
    };
    const onTouchMove = (event: TouchEvent) => {
      const touch = event.touches[0];
      if (!touch || !drag.active) return;
      event.preventDefault();
      move(touch.clientX);
    };
    scroller.addEventListener("touchstart", onTouchStart, { passive: true });
    scroller.addEventListener("touchmove", onTouchMove, { passive: false });
    scroller.addEventListener("touchend", end);
    scroller.addEventListener("touchcancel", end);
    const observer = new ResizeObserver(() => paint(indexRef.current));
    observer.observe(scroller);
    return () => {
      stop();
      window.clearTimeout(wheelTimer);
      scroller.removeEventListener("pointerdown", onPointerDown);
      scroller.removeEventListener("pointermove", onPointerMove);
      scroller.removeEventListener("pointerup", onPointerUp);
      scroller.removeEventListener("pointercancel", onPointerUp);
      scroller.removeEventListener("wheel", onWheel);
      scroller.removeEventListener("touchstart", onTouchStart);
      scroller.removeEventListener("touchmove", onTouchMove);
      scroller.removeEventListener("touchend", end);
      scroller.removeEventListener("touchcancel", end);
      observer.disconnect();
    };
  }, [count]);

  return {
    scrollerRef,
    trackRef,
    glideTo(index: number, done?: () => void) {
      const last = Math.max(0, count - 1);
      glide(clamp(index, 0, last), 320, done);
    },
    step(by: number) {
      const last = Math.max(0, count - 1);
      glide(clamp(Math.round(indexRef.current) + by, 0, last), 300);
    },
    dragged() {
      return movedRef.current;
    },
  };
}

function Shell({ pages }: { pages: WebMountView }): React.ReactElement {
  const entries = useMemo(() => checkPages(pages.entries), [pages.entries]);
  const bandEntries = useMemo(() => entries.filter((entry) => entry.section !== "settings"), [entries]);
  const settingsEntries = useMemo(() => entries.filter((entry) => entry.section === "settings"), [entries]);
  const defaultPage = bandEntries.find((entry) => entry.route === "") ?? bandEntries[0] ?? entries[0];
  const [activeId, setActiveId] = useState(() => pageFromLocation(entries, defaultPage)?.id ?? "");
  // 视觉焦点必须落在刻度带内的真实条目上；深链接进设置页时停在第一个刻度。
  const [focusId, setFocusId] = useState(() => bandEntries.some((entry) => entry.id === activeId) ? activeId : bandEntries[0]?.id ?? "");
  const pageHosts = useRef(new Map<string, HTMLElement>());
  const settingsDialog = useRef<HTMLDialogElement>(null);

  const rail = useBandScale(bandEntries.length, (index) => {
    const entry = bandEntries[index];
    if (!entry) return;
    setFocusId(entry.id);
    // 键盘操作时让真实焦点跟上视觉中心，屏幕阅读器才能读出落格结果。
    const scroller = rail.scrollerRef.current;
    if (scroller?.contains(document.activeElement)) {
      scroller.querySelector<HTMLElement>(`[data-band-id="${entry.id}"]`)?.focus();
    }
  });

  const openPage = useCallback((entry: ShellPage): void => {
    const go = () => {
      setActiveId(entry.id);
      const base = `${window.location.pathname}${window.location.search}`;
      window.history.replaceState(null, "", entry.route ? `${base}#${entry.route}` : base);
      closeSettings(settingsDialog.current);
    };
    if (window.dispatchEvent(new CustomEvent("akashic:before-navigate", { cancelable: true, detail: { go } }))) go();
  }, []);

  const pick = (entry: ShellPage, index: number): void => {
    if (rail.dragged()) return;
    rail.glideTo(index, () => openPage(entry));
  };

  // 外部定位（hash、返回键）变化时把对应页吸到刻度中间。
  const activeIndex = bandEntries.findIndex((entry) => entry.id === activeId);
  useEffect(() => {
    if (activeIndex >= 0) rail.glideTo(activeIndex);
  }, [activeIndex]);

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

  const onBandKeyDown = (event: KeyboardEvent<HTMLElement>): void => {
    if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
    event.preventDefault();
    rail.step(event.key === "ArrowRight" ? 1 : -1);
  };

  return <div className="unified-shell">
    <header className="primary-band" aria-label="Akashic 主导航">
      <div className="primary-band-brand" title="Akashic">
        <img src={akashicBrandIcon} alt="" />
        <strong>Akashic</strong>
      </div>
      <nav
        ref={rail.scrollerRef}
        className="primary-band-nav"
        aria-label="主要功能"
        onKeyDown={onBandKeyDown}
      >
        <div ref={rail.trackRef} className="primary-band-track">
          {bandEntries.map((entry, index) => <button
            key={entry.id}
            type="button"
            data-band-item=""
            data-band-id={entry.id}
            className="primary-rail-button"
            aria-label={entry.label}
            title={entry.label}
            tabIndex={focusId === entry.id ? 0 : -1}
            aria-current={activeId === entry.id ? "page" : undefined}
            onClick={() => pick(entry, index)}
          >
            <span className="shell-page-icon" aria-hidden="true" dangerouslySetInnerHTML={{ __html: entry.iconSvg }} />
            <span>{entry.label}</span>
          </button>)}
        </div>
      </nav>
      <div className="primary-band-footer">
        <button type="button" className="theme-cycle-button" onClick={() => settingsDialog.current?.showModal()}>功能设置</button>
        <dialog ref={settingsDialog} className="shell-settings-dialog" aria-label="功能设置">
          <header><h2>功能设置</h2><button type="button" onClick={() => closeSettings(settingsDialog.current)} aria-label="关闭设置目录">关闭</button></header>
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

/** 关闭动画结束后再真正收起对话框；reduced motion 下立即关闭。 */
function closeSettings(dialog: HTMLDialogElement | null): void {
  if (!dialog?.open) return;
  if (prefersReducedMotion()) {
    dialog.close();
    return;
  }
  dialog.dataset.state = "closed";
  window.setTimeout(() => {
    dialog.close();
    delete dialog.dataset.state;
  }, 180);
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

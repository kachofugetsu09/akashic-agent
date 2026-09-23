import { Bot, Gauge, Palette, Puzzle, SlidersHorizontal } from "lucide-react";
import { useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent, type MouseEvent, type ReactNode } from "react";
import { akashicBrandIcon } from "./akashic-brand";

export interface ChatProductBandProps {
  chatReady: boolean;
  themeLabel: string;
  onCycleTheme: () => void;
}

interface BandItem {
  id: string;
  label: string;
  icon: ReactNode;
  href?: string;
  disabled?: boolean;
  onSelect?: () => void;
}

const PREVIEW_LABELS = [
  "记忆", "唤醒", "日程", "技能", "频道", "附件", "引用", "终端",
  "笔记", "邮件", "浏览", "搜索", "任务", "监控", "日志", "插件",
];

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

function previewItems(onSelect: (id: string) => void): BandItem[] {
  if (!import.meta.env.DEV || typeof window === "undefined") return [];
  if (new URLSearchParams(window.location.search).get("nav") !== "many") return [];
  return Array.from({ length: 48 }, (_, index) => ({
    id: `preview-${index}`,
    label: PREVIEW_LABELS[index] ?? `插件 ${index + 1}`,
    icon: <Puzzle size={16} aria-hidden="true" />,
    onSelect: () => onSelect(`preview-${index}`),
  }));
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

function BandDestination({
  item,
  index,
  current,
  onPick,
}: {
  item: BandItem;
  index: number;
  current: boolean;
  onPick: (item: BandItem, index: number) => void;
}) {
  const className = `chat-product-band__item${item.disabled ? " is-disabled" : ""}`;
  const shared = {
    "data-band-item": "",
    "data-band-id": item.id,
    "data-band-index": index,
    className,
    tabIndex: item.disabled ? -1 : current ? 0 : -1,
    "aria-current": current ? "page" as const : undefined,
    onClick: (event: MouseEvent) => {
      event.preventDefault();
      if (!item.disabled) onPick(item, index);
    },
  };
  if (item.disabled || !item.href) {
    const Tag = item.onSelect || !item.disabled ? "button" : "span";
    return (
      <Tag {...shared} type={Tag === "button" ? "button" : undefined} aria-disabled={item.disabled || undefined}>
        {item.icon}
        <span>{item.label}</span>
      </Tag>
    );
  }
  return (
    <a {...shared} href={item.href}>
      {item.icon}
      <span>{item.label}</span>
    </a>
  );
}

/** 顶部目的地。滚动手感对齐 beUI Wheel Picker：惯性滑行，松手落在最近一格，并让该格居中。 */
export function ChatProductBand({ chatReady, themeLabel, onCycleTheme }: ChatProductBandProps) {
  const [previewId, setPreviewId] = useState<string | null>(null);
  const [focusId, setFocusId] = useState("chat");
  const dashboardHref = chatReady ? "/" : undefined;
  const items: BandItem[] = [
    { id: "chat", label: "对话", icon: <Bot size={16} aria-hidden="true" />, href: "/chat" },
    {
      id: "workbench",
      label: "工作台",
      icon: <Gauge size={16} aria-hidden="true" />,
      href: dashboardHref,
      disabled: !dashboardHref,
    },
    { id: "models", label: "模型", icon: <SlidersHorizontal size={16} aria-hidden="true" />, href: "/#models" },
    ...previewItems(setPreviewId),
  ];
  const rail = useBandScale(items.length, (index) => {
    const item = items[index];
    if (item) setFocusId(item.id);
  });
  const itemsRef = useRef(items);
  itemsRef.current = items;

  const pick = (item: BandItem, index: number) => {
    if (rail.dragged()) return;
    rail.glideTo(index, () => {
      if (item.href && item.id !== "chat") window.location.assign(item.href);
      else item.onSelect?.();
    });
  };

  const onKeyDown = (event: KeyboardEvent<HTMLElement>) => {
    if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
    event.preventDefault();
    rail.step(event.key === "ArrowRight" ? 1 : -1);
  };

  return (
    <header className="chat-product-band" aria-label="Akashic 主导航">
      <div className="chat-product-band__brand" title="Akashic">
        <span
          className="chat-product-band__mark"
          style={{ WebkitMaskImage: `url(${akashicBrandIcon})`, maskImage: `url(${akashicBrandIcon})` }}
          aria-hidden="true"
        />
        <strong>Akashic</strong>
      </div>
      <nav
        ref={rail.scrollerRef}
        className="chat-product-band__nav"
        aria-label="主要功能"
        onKeyDown={onKeyDown}
      >
        <div ref={rail.trackRef} className="chat-product-band__track">
          {items.map((item, index) => (
            <BandDestination
              key={item.id}
              item={item}
              index={index}
              current={(previewId ?? focusId) === item.id}
              onPick={pick}
            />
          ))}
        </div>
      </nav>
      <div className="chat-product-band__footer">
        <button type="button" className="chat-product-band__item" onClick={onCycleTheme} title={`主题 · ${themeLabel}`}>
          <Palette size={16} aria-hidden="true" />
          <span className="chat-product-band__theme-label">{themeLabel}</span>
        </button>
      </div>
    </header>
  );
}

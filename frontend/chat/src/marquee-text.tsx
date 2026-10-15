import { useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";

// 往返一趟的阅读速度（px/s）与两端停顿（以越界距离折算，见 conversation-navigation.css 的 clamp）。
const SPEED = 30;
const OVERSHOOT = 6;
// 往返滚动靠注册过的 --marquee-pos 插值；不支持注册属性的浏览器自定义属性只会离散跳变，只保留静态渐隐。
const ANIMATABLE = typeof CSS !== "undefined" && "registerProperty" in CSS;

type Measured = { viewport: HTMLElement; content: HTMLElement; report: (overflow: number) => void };

// 侧栏一屏可能有上百行标题：共用一个 ResizeObserver，按元素找回各自的测量回调。
const watched = new Map<Element, Measured>();
let sharedObserver: ResizeObserver | null = null;

function observer(): ResizeObserver {
  sharedObserver ??= new ResizeObserver((entries) => {
    const seen = new Set<Measured>();
    for (const entry of entries) {
      const item = watched.get(entry.target);
      if (!item || seen.has(item)) continue;
      seen.add(item);
      item.report(item.content.offsetWidth - item.viewport.clientWidth);
    }
  });
  return sharedObserver;
}

// 一行标题：放不下时右缘渐隐代替省略号；所在行悬停或键盘聚焦时，以恒定速度往返滚出尾部（触屏与减少动态时只渐隐）。
// still 用于副标题一类只需渐隐、不该跟着标题一起动的文字。
export function MarqueeText({ children, still = false }: { children: ReactNode; still?: boolean }) {
  const viewportRef = useRef<HTMLSpanElement>(null);
  const contentRef = useRef<HTMLSpanElement>(null);
  const [overflow, setOverflow] = useState(0);

  // 必要 effect：宽度取决于侧栏拖宽、字体加载、标题改写，只能由 ResizeObserver 观测。
  useLayoutEffect(() => {
    const viewport = viewportRef.current;
    const content = contentRef.current;
    if (!viewport || !content) return;
    const item: Measured = { viewport, content, report: (value) => setOverflow(value > 1 ? Math.ceil(value) : 0) };
    const shared = observer();
    for (const element of [viewport, content]) {
      watched.set(element, item);
      shared.observe(element);
    }
    return () => {
      for (const element of [viewport, content]) {
        watched.delete(element);
        shared.unobserve(element);
      }
    };
  }, []);

  const style = overflow ? {
    "--marquee-travel": `${overflow}px`,
    "--marquee-duration": `${Math.round(((overflow + 2 * OVERSHOOT) * 2000) / SPEED)}ms`,
  } as CSSProperties : undefined;
  return <span ref={viewportRef} className={`marquee${overflow ? ` is-truncated${ANIMATABLE ? " is-animatable" : ""}` : ""}${still ? " is-still" : ""}`} style={style}>
    <span ref={contentRef} className="marquee__content">{children}</span>
  </span>;
}

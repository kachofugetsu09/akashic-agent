import { useLayoutEffect, useRef, type ReactNode } from "react";

// 往返一趟的阅读速度（px/s）与两端停顿（以越界距离折算，见 conversation-navigation.css 的 clamp）。
const SPEED = 30;
const OVERSHOOT = 6;
// 往返滚动靠注册过的 --marquee-pos 插值；不支持注册属性的浏览器自定义属性只会离散跳变，只保留静态渐隐。
const ANIMATABLE = typeof CSS !== "undefined" && "registerProperty" in CSS;

type Measured = { viewport: HTMLElement; content: HTMLElement; still: boolean };

// 侧栏一屏可能有上百行标题：共用一个 ResizeObserver，按元素找回各自的测量回调。
const watched = new Map<Element, Measured>();
let sharedObserver: ResizeObserver | null = null;

// 测量结果直接写 DOM（class 与自定义属性），不走 React state：目录规模下每行一次
// setState 会汇成整列重渲染， marquee 子树在高亮继承之外再付一整轮样式解析。
// React 之后重渲染时 style/className prop 始终是同一引用，不会覆盖这里写下的值。
function apply(item: Measured): void {
  const overflow = item.content.offsetWidth - item.viewport.clientWidth;
  const truncated = overflow > 1;
  const viewport = item.viewport;
  // 拖拽侧栏等连续 resize 里同一元素每帧都会回报，值没变就不写：同值写
  // class/属性同样触发该子树的样式失效。
  const wasTruncated = viewport.classList.contains("is-truncated");
  if (truncated === wasTruncated) {
    if (!truncated) return;
    const travel = `${Math.ceil(overflow)}px`;
    if (viewport.style.getPropertyValue("--marquee-travel") === travel) return;
  }
  viewport.classList.toggle("is-truncated", truncated);
  viewport.classList.toggle("is-animatable", truncated && ANIMATABLE && !item.still);
  if (truncated) {
    const value = Math.ceil(overflow);
    viewport.style.setProperty("--marquee-travel", `${value}px`);
    viewport.style.setProperty("--marquee-duration", `${Math.round(((value + 2 * OVERSHOOT) * 2000) / SPEED)}ms`);
  } else {
    viewport.style.removeProperty("--marquee-travel");
    viewport.style.removeProperty("--marquee-duration");
  }
}

function observer(): ResizeObserver {
  sharedObserver ??= new ResizeObserver((entries) => {
    const seen = new Set<Measured>();
    for (const entry of entries) {
      const item = watched.get(entry.target);
      if (!item || seen.has(item)) continue;
      seen.add(item);
      apply(item);
    }
  });
  return sharedObserver;
}

// 一行标题：放不下时右缘渐隐代替省略号；所在行悬停或键盘聚焦时，以恒定速度往返滚出尾部（触屏与减少动态时只渐隐）。
// still 用于副标题一类只需渐隐、不该跟着标题一起动的文字。
export function MarqueeText({ children, still = false }: { children: ReactNode; still?: boolean }) {
  const viewportRef = useRef<HTMLSpanElement>(null);
  const contentRef = useRef<HTMLSpanElement>(null);

  // 必要 effect：宽度取决于侧栏拖宽、字体加载、标题改写，只能由 ResizeObserver 观测。
  useLayoutEffect(() => {
    const viewport = viewportRef.current;
    const content = contentRef.current;
    if (!viewport || !content) return;
    const item: Measured = { viewport, content, still };
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
  }, [still]);

  return <span ref={viewportRef} className={`marquee${still ? " is-still" : ""}`}>
    <span ref={contentRef} className="marquee__content">{children}</span>
  </span>;
}

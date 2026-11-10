import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties } from "react";

/** 停留这么久才出提示：毫秒级掠过的指针不该被打断。 */
const SHOW_DELAY_MS = 400;
/** 气泡与目标之间的缝隙。 */
const GAP_PX = 9;
/** 贴边留白：气泡水平方向夹在视口里，垂直方向放不下就翻到目标下方。 */
const EDGE_PX = 8;

type Hint = { text: string; rect: DOMRect };
type Placed = { style: CSSProperties; place: "above" | "below" };

/**
 * 纸气泡提示：控件上写 `data-tip="文案"` 即可，不需要包一层 DOM。
 *
 * 原来的原生 `title` 延迟和样式都不受控，触控上还会粘住。这里改成挂在 document 上的
 * 一个浮层：进入延迟 400ms、带 3px 位移淡入，离开立即消失（卸载即瞬时）。
 * 可访问名称仍由控件自己的 `aria-label` 承担，气泡只是视觉补充，不重复播报。
 */
export function PaperTooltipLayer() {
  const [hint, setHint] = useState<Hint | null>(null);
  const [placed, setPlaced] = useState<Placed | null>(null);
  const bubbleRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let timer: number | null = null;
    const clearTimer = () => {
      if (timer !== null) window.clearTimeout(timer);
      timer = null;
    };
    const hide = () => { clearTimer(); setHint(null); };

    // 指针在控件的子元素之间移动会连续触发 pointerover，所以每次都按最近的 data-tip 归位。
    const anchorOf = (node: EventTarget | null): HTMLElement | null =>
      node instanceof Element ? node.closest<HTMLElement>("[data-tip]") : null;

    const show = (el: HTMLElement) => {
      const text = el.dataset.tip;
      if (text) setHint({ text, rect: el.getBoundingClientRect() });
    };

    const onPointerOver = (event: PointerEvent) => {
      // 触屏没有「悬停」，提示只在指针设备上出现；键盘走下面的 focusin。
      if (event.pointerType !== "mouse") return;
      const el = anchorOf(event.target);
      if (!el) return;
      clearTimer();
      timer = window.setTimeout(() => show(el), SHOW_DELAY_MS);
    };
    const onPointerOut = (event: PointerEvent) => { if (anchorOf(event.target)) hide(); };
    const onFocusIn = (event: FocusEvent) => {
      const el = anchorOf(event.target);
      if (el?.matches(":focus-visible")) { clearTimer(); show(el); }
    };
    const onFocusOut = (event: FocusEvent) => { if (anchorOf(event.target)) hide(); };
    const onKeyDown = (event: KeyboardEvent) => { if (event.key === "Escape") hide(); };

    document.addEventListener("pointerover", onPointerOver, true);
    document.addEventListener("pointerout", onPointerOut, true);
    document.addEventListener("focusin", onFocusIn, true);
    document.addEventListener("focusout", onFocusOut, true);
    document.addEventListener("keydown", onKeyDown, true);
    // 目标一旦滚动或改尺寸，记下的矩形就失效了：直接收起来，不追着跑。
    window.addEventListener("scroll", hide, true);
    window.addEventListener("resize", hide);
    return () => {
      document.removeEventListener("pointerover", onPointerOver, true);
      document.removeEventListener("pointerout", onPointerOut, true);
      document.removeEventListener("focusin", onFocusIn, true);
      document.removeEventListener("focusout", onFocusOut, true);
      document.removeEventListener("keydown", onKeyDown, true);
      window.removeEventListener("scroll", hide, true);
      window.removeEventListener("resize", hide);
      clearTimer();
    };
  }, []);

  // 先量气泡自身尺寸再定位，所以未定位的那一帧只用来量尺寸（CSS 里 visibility: hidden）。
  // 定位与方向在同一次提交里定下来，入场动画才只跑一遍、且朝着正确的方向。
  useLayoutEffect(() => {
    const el = bubbleRef.current;
    if (!hint || !el) {
      setPlaced(null);
      return;
    }
    const box = el.getBoundingClientRect();
    const above = hint.rect.top - box.height - GAP_PX >= EDGE_PX;
    const half = box.width / 2;
    const center = Math.min(
      Math.max(hint.rect.left + hint.rect.width / 2, half + EDGE_PX),
      Math.max(half + EDGE_PX, window.innerWidth - half - EDGE_PX),
    );
    setPlaced({
      place: above ? "above" : "below",
      style: {
        top: above ? hint.rect.top - box.height - GAP_PX : hint.rect.bottom + GAP_PX,
        left: center,
        visibility: "visible",
      },
    });
  }, [hint]);

  if (!hint) return null;

  return (
    <div
      ref={bubbleRef}
      className="paper-tooltip"
      data-place={placed?.place}
      data-ready={placed ? "true" : undefined}
      // 文案与目标的 aria-label 同一份，浮层只是视觉补充，不参与播报。
      aria-hidden="true"
      style={placed?.style}
    >
      {hint.text}
    </div>
  );
}

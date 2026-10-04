import { useCallback, useMemo, useRef, useState, type CSSProperties } from "react";

export const SIDEBAR_RAIL_MIN_REM = 15;
export const SIDEBAR_RAIL_MAX_REM = 26;
export const SIDEBAR_RAIL_STEP_REM = 0.5;
const RAIL_WIDTH_KEY = "akashic.chat.rail-width";

function clampRail(rem: number): number {
  return Math.min(SIDEBAR_RAIL_MAX_REM, Math.max(SIDEBAR_RAIL_MIN_REM, rem));
}

/** 存储值按 0.25rem 取整，避免指针拖拽写入亚像素级噪声。 */
function snapRail(rem: number): number {
  return clampRail(Math.round(rem * 4) / 4);
}

function readRailWidth(): number | null {
  try {
    const value = Number(localStorage.getItem(RAIL_WIDTH_KEY));
    return Number.isFinite(value) && value > 0 ? clampRail(value) : null;
  } catch {
    return null;
  }
}

function rootFontPx(): number {
  return parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
}

export interface SidebarRailControl {
  /** 用户覆盖值（rem）；null 表示回到 CSS 的 clamp 默认。 */
  widthRem: number | null;
  /** 唯一写入点：grid 容器（.chat-shell-body）上的 --chat-rail-width。 */
  style: CSSProperties | undefined;
  /** 指针拖拽锚点：按下时记录列宽与指针 x，移动中按差值换算并钳制。 */
  dragStart: (clientX: number, widthPx: number) => void;
  dragTo: (clientX: number) => void;
  /** 键盘步进：以当前实际列宽为基准增减 0.5rem。 */
  stepBy: (deltaRem: number, widthPx: number) => void;
  /** 双击复位：移除 inline 覆盖与存储偏好。 */
  reset: () => void;
}

/** 侧栏宽度偏好：只写 --chat-rail-width 一个值；首次渲染前同步读取，挂载无回跳。 */
export function useSidebarRail(): SidebarRailControl {
  const [widthRem, setWidthRem] = useState<number | null>(readRailWidth);
  const drag = useRef<{ x: number; widthPx: number } | null>(null);

  const commit = useCallback((next: number | null) => {
    setWidthRem(next);
    try {
      if (next === null) localStorage.removeItem(RAIL_WIDTH_KEY);
      else localStorage.setItem(RAIL_WIDTH_KEY, String(next));
    } catch {
      /* 存储禁用时仍允许本次浏览的调整。 */
    }
  }, []);

  const dragStart = useCallback((clientX: number, widthPx: number) => {
    drag.current = { x: clientX, widthPx };
  }, []);

  const dragTo = useCallback((clientX: number) => {
    const start = drag.current;
    if (!start) return;
    commit(snapRail((start.widthPx + clientX - start.x) / rootFontPx()));
  }, [commit]);

  const stepBy = useCallback((deltaRem: number, widthPx: number) => {
    commit(snapRail(widthPx / rootFontPx() + deltaRem));
  }, [commit]);

  const reset = useCallback(() => {
    drag.current = null;
    commit(null);
  }, [commit]);

  const style = useMemo<CSSProperties | undefined>(() => widthRem === null
    ? undefined
    : { "--chat-rail-width": `${widthRem}rem` } as CSSProperties, [widthRem]);

  return { widthRem, style, dragStart, dragTo, stepBy, reset };
}

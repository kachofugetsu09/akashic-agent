import { useCallback, useMemo, useRef, useState, type CSSProperties, type RefObject } from "react";

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
  /** 静置写入点：grid 容器上的 --chat-rail-width。 */
  style: CSSProperties | undefined;
  /** 指针拖拽锚点：按下时记录列宽与指针 x，移动中按差值换算并钳制。 */
  dragStart: (clientX: number, widthPx: number) => void;
  /** 拖拽中直写容器列宽并返回当前值（供 aria-valuenow 等展示），不经过 React state。 */
  dragTo: (clientX: number) => number | null;
  /** 拖拽结束：清掉直写的列宽，把最终值提交进 state（一次性全量重算）。 */
  dragEnd: () => void;
  /** 键盘步进：以当前实际列宽为基准增减 0.5rem。 */
  stepBy: (deltaRem: number, widthPx: number) => void;
  /** 双击复位：移除 inline 覆盖与存储偏好。 */
  reset: () => void;
}

/** 侧栏宽度偏好：静置时只写 --chat-rail-width 一个值；首次渲染前同步读取，挂载无回跳。 */
export function useSidebarRail(shellRef: RefObject<HTMLDivElement | null>): SidebarRailControl {
  const [widthRem, setWidthRem] = useState<number | null>(readRailWidth);
  const drag = useRef<{ x: number; widthPx: number; initial: number; next: number | null } | null>(null);

  const commit = useCallback((next: number | null) => {
    setWidthRem(next);
    try {
      if (next === null) localStorage.removeItem(RAIL_WIDTH_KEY);
      else localStorage.setItem(RAIL_WIDTH_KEY, String(next));
    } catch {
      /* 存储禁用时仍允许本次浏览的调整。 */
    }
  }, []);

  // 清拖拽期直写的列宽：dragEnd 与 reset 共用，pointer capture 丢失时也走这里兜底。
  const clearDirect = useCallback(() => {
    const shell = shellRef.current;
    if (shell) shell.style.gridTemplateColumns = "";
  }, [shellRef]);

  const dragStart = useCallback((clientX: number, widthPx: number) => {
    drag.current = { x: clientX, widthPx, initial: snapRail(widthPx / rootFontPx()), next: null };
  }, []);

  const dragTo = useCallback((clientX: number): number | null => {
    const start = drag.current;
    if (!start) return null;
    const next = snapRail((start.widthPx + clientX - start.x) / rootFontPx());
    start.next = next;
    // 拖拽中不经过 React state，也不在容器上写自定义属性：自定义属性可被后代继承，
    // 每帧改写会让 24k 后代的样式全量重解析（实测拖拽一帧一次 42k 元素的重算）。
    // 列宽只被容器自己的 grid-template-columns 消费，直写只重算容器一个元素。
    const shell = shellRef.current;
    if (shell) shell.style.gridTemplateColumns = `${next}rem minmax(0, 1fr)`;
    return next;
  }, [shellRef]);

  const dragEnd = useCallback(() => {
    const start = drag.current;
    if (!start) return;
    drag.current = null;
    clearDirect();
    // 只在真的拖到了新值才提交：8px 热区上单击（或亚档抖动）不得把当前 clamp 档冻成持久偏好。
    if (start.next !== null && start.next !== start.initial) commit(start.next);
  }, [commit, clearDirect]);

  const stepBy = useCallback((deltaRem: number, widthPx: number) => {
    commit(snapRail(widthPx / rootFontPx() + deltaRem));
  }, [commit]);

  const reset = useCallback(() => {
    drag.current = null;
    clearDirect();
    commit(null);
  }, [commit, clearDirect]);

  const style = useMemo<CSSProperties | undefined>(() => widthRem === null
    ? undefined
    : { "--chat-rail-width": `${widthRem}rem` } as CSSProperties, [widthRem]);

  return { widthRem, style, dragStart, dragTo, dragEnd, stepBy, reset };
}

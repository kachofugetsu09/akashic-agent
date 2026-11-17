import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type RefObject } from "react";

export const SIDEBAR_RAIL_MIN_REM = 15;
export const SIDEBAR_RAIL_MAX_REM = 26;
export const SIDEBAR_RAIL_STEP_REM = 0.5;
const RAIL_WIDTH_KEY = "akashic.chat.rail-width";
const RAIL_SAMPLES_KEY = "akashic.chat.rail-samples";

export interface RailSample {
  widthPx: number;
  at: number;
}

export interface RailSnapTuning {
  /** 高斯核带宽（px）：落在 2*spread 内的样本互相叠高。 */
  spread: number;
  /** 半衰期（按松手次数计，不按天数）：每过 halfLife 次拖拽旧样本权重减半。 */
  halfLife: number;
  /** 形成磁吸点的最小累积权重（约 3 次近期一致拖拽）。 */
  minWeight: number;
  /** 吸附半径（px）：指针进入 [point - snap, point + snap] 时吸附到偏好宽度。 */
  snap: number;
  /** 这么快之内的再次松手视为同一次微调，覆盖上一条样本。 */
  correctionMs: number;
  /** 最多保留的历史样本条数。 */
  keep: number;
  /** 在吸附区内停顿超过此时长（ms）自动解锁磁吸，进入 1:1 精细微调。 */
  dwellMs: number;
}

export const RAIL_SNAP_DEFAULTS: RailSnapTuning = {
  spread: 8,
  halfLife: 5,
  minWeight: 2.5,
  snap: 10,
  correctionMs: 2000,
  keep: 40,
  dwellMs: 320,
};

export function sampleWeights(samples: readonly RailSample[], halfLife: number): number[] {
  const n = samples.length;
  return samples.map((_, index) => 0.5 ** ((n - 1 - index) / halfLife));
}

export function sampleDensity(
  samples: readonly RailSample[],
  weights: readonly number[],
  widthPx: number,
  spread: number,
): number {
  let sum = 0;
  for (let index = 0; index < samples.length; index += 1) {
    const delta = widthPx - samples[index].widthPx;
    sum += weights[index] * Math.exp(-(delta * delta) / (2 * spread * spread));
  }
  return sum;
}

function sampleSupport(
  samples: readonly RailSample[],
  weights: readonly number[],
  widthPx: number,
  spread: number,
): number {
  return samples.reduce(
    (sum, sample, index) => (Math.abs(sample.widthPx - widthPx) <= 2 * spread ? sum + weights[index] : sum),
    0,
  );
}

function tallestPeak(samples: readonly RailSample[], tuning: RailSnapTuning): { widthPx: number; support: number } {
  const weights = sampleWeights(samples, tuning.halfLife);
  const from = Math.floor(Math.min(...samples.map((sample) => sample.widthPx)));
  const to = Math.ceil(Math.max(...samples.map((sample) => sample.widthPx)));
  let best = from;
  let top = -1;
  for (let x = from; x <= to; x += 1) {
    const d = sampleDensity(samples, weights, x, tuning.spread);
    if (d > top) {
      top = d;
      best = x;
    }
  }
  return { widthPx: best, support: sampleSupport(samples, weights, best, tuning.spread) };
}

/**
 * 计算用户习惯的磁吸点（px）：
 * 只有当最高峰 support >= minWeight 时才形成磁吸点；一旦形成，只要后续最高峰仍在当前吸附区内，
 * 磁吸点像素就保持不动（避免每次拖拽偏移 1px）。
 */
export function favouriteRailWidth(
  samples: readonly RailSample[],
  tuning: RailSnapTuning = RAIL_SNAP_DEFAULTS,
): { widthPx: number; support: number } | null {
  let current: { widthPx: number; support: number } | null = null;
  for (let n = 1; n <= samples.length; n += 1) {
    const seen = samples.slice(0, n);
    const peak = tallestPeak(seen, tuning);
    if (peak.support < tuning.minWeight) {
      current = null;
    } else if (current && Math.abs(peak.widthPx - current.widthPx) <= tuning.snap) {
      current = {
        widthPx: current.widthPx,
        support: sampleSupport(seen, sampleWeights(seen, tuning.halfLife), current.widthPx, tuning.spread),
      };
    } else {
      current = peak;
    }
  }
  return current;
}

export function recordRailSample(
  samples: readonly RailSample[],
  widthPx: number,
  at: number,
  tuning: RailSnapTuning = RAIL_SNAP_DEFAULTS,
): RailSample[] {
  const last = samples[samples.length - 1];
  const corrected = Boolean(last && at - last.at < tuning.correctionMs);
  const next = corrected
    ? [...samples.slice(0, -1), { widthPx, at }]
    : [...samples, { widthPx, at }];
  return next.slice(-tuning.keep);
}

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

function readRailSamples(): RailSample[] {
  try {
    const raw = JSON.parse(localStorage.getItem(RAIL_SAMPLES_KEY) ?? "[]") as unknown;
    if (!Array.isArray(raw)) return [];
    return raw.filter((item): item is RailSample => Boolean(item)
      && typeof item === "object"
      && typeof (item as RailSample).widthPx === "number"
      && Number.isFinite((item as RailSample).widthPx)
      && (item as RailSample).widthPx > 0
      && typeof (item as RailSample).at === "number"
      && Number.isFinite((item as RailSample).at)).slice(-RAIL_SNAP_DEFAULTS.keep);
  } catch {
    return [];
  }
}

function rootFontPx(): number {
  return parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
}

export interface SidebarRailControl {
  /** 用户覆盖值（rem）；null 表示回到 CSS 的 clamp 默认。 */
  widthRem: number | null;
  /** 已学习的习惯磁吸宽度（px），未达到共识门槛时为 null。 */
  snapPointPx: number | null;
  /** 静置写入点：grid 容器上的 --chat-rail-width。 */
  style: CSSProperties | undefined;
  /** 指针拖拽锚点：按下时记录列宽与指针 x，移动中按差值换算并钳制。 */
  dragStart: (clientX: number, widthPx: number, resizerEl?: HTMLElement | null) => void;
  /** 拖拽中直写容器列宽并返回当前值（供 aria-valuenow 等展示），不经过 React state。 */
  dragTo: (clientX: number) => number | null;
  /** 拖拽结束：清掉直写的列宽，把最终值提交进 state（一次性全量重算）。 */
  dragEnd: () => void;
  /** 键盘步进：以当前实际列宽为基准增减 0.5rem（不触发磁吸）。 */
  stepBy: (deltaRem: number, widthPx: number) => void;
  /** 双击复位：移除 inline 覆盖与存储偏好。 */
  reset: () => void;
}

/** 侧栏宽度偏好：静置时只写 --chat-rail-width 一个值；首次渲染前同步读取，挂载无回跳。 */
export function useSidebarRail(shellRef: RefObject<HTMLDivElement | null>): SidebarRailControl {
  const [widthRem, setWidthRem] = useState<number | null>(readRailWidth);
  const [samples, setSamples] = useState<RailSample[]>(readRailSamples);
  const favourite = useMemo(() => favouriteRailWidth(samples), [samples]);
  const drag = useRef<{
    x: number;
    widthPx: number;
    initial: number;
    next: number | null;
    edgePx: number;
    holding: boolean;
    freeFromPx: number | null;
    stillPx: number;
    dwellTimer: number;
    resizerEl: HTMLElement | null;
  } | null>(null);

  useEffect(() => () => {
    if (drag.current?.dwellTimer) window.clearTimeout(drag.current.dwellTimer);
  }, []);

  const commit = useCallback((next: number | null) => {
    setWidthRem(next);
    try {
      if (next === null) localStorage.removeItem(RAIL_WIDTH_KEY);
      else localStorage.setItem(RAIL_WIDTH_KEY, String(next));
    } catch {
      /* 存储禁用时仍允许本次浏览的调整。 */
    }
  }, []);

  const commitSample = useCallback((edgePx: number) => {
    setSamples((current) => {
      const next = recordRailSample(current, Math.round(edgePx), Date.now());
      try {
        localStorage.setItem(RAIL_SAMPLES_KEY, JSON.stringify(next));
      } catch {
        /* 忽略存储配额限制 */
      }
      return next;
    });
  }, []);

  // 清拖拽期直写的列宽：dragEnd 与 reset 共用，pointer capture 丢失时也走这里兜底。
  const clearDirect = useCallback(() => {
    const shell = shellRef.current;
    if (shell) shell.style.gridTemplateColumns = "";
  }, [shellRef]);

  const applyDirectRem = useCallback((rem: number) => {
    // 拖拽中不经过 React state，也不在容器上写自定义属性：自定义属性可被后代继承，
    // 每帧改写会让 24k 后代的样式全量重解析（实测拖拽一帧一次 42k 元素的重算）。
    // 列宽只被容器自己的 grid-template-columns 消费，直写只重算容器一个元素。
    const shell = shellRef.current;
    if (shell) shell.style.gridTemplateColumns = `${rem}rem minmax(0, 1fr)`;
  }, [shellRef]);

  const setDirectSnapped = useCallback((el: HTMLElement | null, value: boolean) => {
    if (!el) return;
    if (value) el.setAttribute("data-snapped", "true");
    else el.removeAttribute("data-snapped");
  }, []);

  const dragStart = useCallback((clientX: number, widthPx: number, resizerEl: HTMLElement | null = null) => {
    const fontPx = rootFontPx();
    if (drag.current?.dwellTimer) window.clearTimeout(drag.current.dwellTimer);
    setDirectSnapped(resizerEl, false);
    drag.current = {
      x: clientX,
      widthPx,
      initial: snapRail(widthPx / fontPx),
      next: null,
      edgePx: widthPx,
      holding: false,
      freeFromPx: null,
      stillPx: widthPx,
      dwellTimer: 0,
      resizerEl,
    };
  }, [setDirectSnapped]);

  const dragTo = useCallback((clientX: number): number | null => {
    const start = drag.current;
    if (!start) return null;
    const fontPx = rootFontPx();
    const minPx = SIDEBAR_RAIL_MIN_REM * fontPx;
    const maxPx = SIDEBAR_RAIL_MAX_REM * fontPx;
    const unclampedPx = start.widthPx + clientX - start.x;
    const rawPx = Math.min(maxPx, Math.max(minPx, unclampedPx));
    const snapTarget = favourite?.widthPx ?? null;

    // 悬停解锁后的精细微调：从吸附点按手移动增量跟随（仍落在侧栏 0.25rem 档上），位移取未钳制值，两端拖得到边界。
    if (start.freeFromPx !== null && snapTarget !== null) {
      const finePx = Math.min(maxPx, Math.max(minPx, snapTarget + (unclampedPx - start.freeFromPx)));
      const next = snapRail(finePx / fontPx);
      start.next = next;
      start.edgePx = next * fontPx;
      applyDirectRem(next);
      return next;
    }

    const wasHolding = start.holding;
    const inSnapZone = snapTarget !== null && Math.abs(rawPx - snapTarget) <= RAIL_SNAP_DEFAULTS.snap;
    start.holding = inSnapZone;

    if (!inSnapZone) {
      if (start.dwellTimer) {
        window.clearTimeout(start.dwellTimer);
        start.dwellTimer = 0;
      }
      if (wasHolding) setDirectSnapped(start.resizerEl, false);
      const next = snapRail(rawPx / fontPx);
      start.next = next;
      start.edgePx = next * fontPx;
      applyDirectRem(next);
      return next;
    }

    if (!wasHolding) setDirectSnapped(start.resizerEl, true);
    const snappedRem = snapRail(snapTarget / fontPx);
    start.next = snappedRem;
    start.edgePx = snapTarget;
    applyDirectRem(snappedRem);

    if (!wasHolding || Math.abs(rawPx - start.stillPx) > RAIL_SNAP_DEFAULTS.snap / 6) {
      start.stillPx = rawPx;
      if (start.dwellTimer) window.clearTimeout(start.dwellTimer);
      start.dwellTimer = window.setTimeout(() => {
        const active = drag.current;
        if (!active || !active.holding) return;
        active.freeFromPx = unclampedPx;
        active.holding = false;
        active.dwellTimer = 0;
        setDirectSnapped(active.resizerEl, false);
      }, RAIL_SNAP_DEFAULTS.dwellMs);
    }
    return snappedRem;
  }, [applyDirectRem, favourite, setDirectSnapped]);

  const dragEnd = useCallback(() => {
    const start = drag.current;
    if (!start) return;
    if (start.dwellTimer) window.clearTimeout(start.dwellTimer);
    setDirectSnapped(start.resizerEl, false);
    drag.current = null;
    clearDirect();
    // 只在真的拖到了新值才提交：8px 热区上单击（或亚档抖动）不得把当前 clamp 档冻成持久偏好。
    if (start.next !== null && start.next !== start.initial) {
      commit(start.next);
      commitSample(start.edgePx);
    }
  }, [commit, commitSample, clearDirect, setDirectSnapped]);

  const stepBy = useCallback((deltaRem: number, widthPx: number) => {
    commit(snapRail(widthPx / rootFontPx() + deltaRem));
  }, [commit]);

  const reset = useCallback(() => {
    if (drag.current?.dwellTimer) window.clearTimeout(drag.current.dwellTimer);
    drag.current = null;
    clearDirect();
    commit(null);
  }, [commit, clearDirect]);

  const style = useMemo<CSSProperties | undefined>(() => widthRem === null
    ? undefined
    : { "--chat-rail-width": `${widthRem}rem` } as CSSProperties, [widthRem]);

  return { widthRem, snapPointPx: favourite?.widthPx ?? null, style, dragStart, dragTo, dragEnd, stepBy, reset };
}

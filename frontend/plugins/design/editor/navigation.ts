import { reactive } from "vue";
import type { Editor } from "@open-pencil/core";

// 画布导航：滚轮、触控板、右键拖动与键盘缩放。open-pencil 自带的滚轮处理里触控板平移是手指位移的 2 倍、
// 单次还有上限，捏合也比手指快；只有鼠标时缩放必须按住 Ctrl/⌘。这里接管画布上的滚轮，按 Figma、Miro 的做法：
// 触控板平移与捏合都跟手，鼠标滚轮默认缩放，右键或空格拖动平移，滚轮行为可以在缩放菜单里改。

export type WheelMode = "auto" | "zoom" | "pan";
export const WHEEL_MODES: { mode: WheelMode; label: string; hint: string }[] = [
  { mode: "auto", label: "自动识别", hint: "鼠标滚轮缩放，触控板双指平移" },
  { mode: "zoom", label: "滚轮缩放", hint: "滚轮总是缩放，按住 Shift 横向平移" },
  { mode: "pan", label: "滚轮平移", hint: "滚轮总是平移，按住 Ctrl/⌘ 缩放" },
];

const STORAGE_KEY = "akashic.design.wheel";
// 一格鼠标滚轮（约 100px）缩放 1.2 倍；Chromium 的捏合按 deltaY = -100·ln(比例) 上报，除以 100 就与手指一致。
const WHEEL_ZOOM_PER_DELTA = Math.log(1.2) / 100;
const PINCH_ZOOM_PER_DELTA = 1 / 100;
const MAX_ZOOM_STEP = Math.log(2);
const LINE_PX = 40;
const PAGE_PX = 800;
// 两次滚动事件相隔这么久就算新的一串：一串只判断一次是鼠标还是触控板，中途不来回切换。
const BURST_GAP_MS = 250;
const SETTLE_MS = 120;
const DRAG_SLOP_PX = 3;
export const ZOOM_STEP = 2;

export const navigation = reactive({ wheel: readWheelMode() });

function readWheelMode(): WheelMode {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    return saved === "zoom" || saved === "pan" ? saved : "auto";
  } catch {
    return "auto";
  }
}

export function setWheelMode(mode: WheelMode): void {
  navigation.wheel = mode;
  try {
    localStorage.setItem(STORAGE_KEY, mode);
  } catch {
    // 存不下只影响下次打开的默认值
  }
}

// 鼠标滚轮一格：Firefox 按行或页上报；Chromium 在 Windows、Linux 上是不小于 50 的整数 deltaY，没有横向分量。
// 触控板一串的第一下是连续的小步长，常带横向分量。macOS 上的鼠标步长也小，会被认成触控板，可在菜单里固定为“滚轮缩放”。
function looksLikeMouse(event: WheelEvent): boolean {
  if (event.deltaMode !== WheelEvent.DOM_DELTA_PIXEL) return true;
  return event.deltaX === 0 && Number.isInteger(event.deltaY) && Math.abs(event.deltaY) >= 50;
}

function pixels(event: WheelEvent): { dx: number; dy: number } {
  const scale = event.deltaMode === WheelEvent.DOM_DELTA_LINE ? LINE_PX : event.deltaMode === WheelEvent.DOM_DELTA_PAGE ? PAGE_PX : 1;
  return { dx: event.deltaX * scale, dy: event.deltaY * scale };
}

// 以视口中心为基准按倍数缩放，给缩放菜单与 +/- 键用
export function zoomBy(editor: Editor, factor: number): void {
  editor.zoomToLevel(editor.state.zoom * factor);
}

// 接管画布上的滚轮与右键拖动；返回解绑函数。监听挂在画布外层的捕获阶段，open-pencil 挂在画布上的滚轮处理就收不到了。
export function bindNavigation(stage: HTMLElement, editor: Editor, enabled: () => boolean): () => void {
  let settle: ReturnType<typeof setTimeout> | null = null;
  let lastWheelAt = Number.NEGATIVE_INFINITY;
  let burstIsMouse = false;
  const moved = (phase: "pan" | "zoom") => {
    editor.setHoveredNode(null);
    editor.setNavigationPhase(phase, performance.now());
    if (settle !== null) clearTimeout(settle);
    settle = setTimeout(() => {
      settle = null;
      editor.setNavigationPhase("idle");
      editor.requestRepaint();
    }, SETTLE_MS);
  };
  const onCanvas = (target: EventTarget | null) =>
    target instanceof Element && (target.classList.contains("design-canvas") || target.closest(".design-dock") !== null);

  // 1. 滚轮：一串滚动只判断一次来源；Ctrl/⌘（含 Chromium 的捏合）总是缩放
  const onWheel = (event: WheelEvent) => {
    if (!onCanvas(event.target) || !enabled()) return;
    event.preventDefault();
    event.stopPropagation();
    if (event.timeStamp - lastWheelAt > BURST_GAP_MS) burstIsMouse = looksLikeMouse(event);
    lastWheelAt = event.timeStamp;
    const { dx, dy } = pixels(event);
    const modified = event.ctrlKey || event.metaKey;
    const zoom = modified || (!event.shiftKey && (navigation.wheel === "zoom" || (navigation.wheel === "auto" && burstIsMouse)));
    if (zoom) {
      const perDelta = event.ctrlKey && !burstIsMouse ? PINCH_ZOOM_PER_DELTA : WHEEL_ZOOM_PER_DELTA;
      const step = Math.max(-MAX_ZOOM_STEP, Math.min(MAX_ZOOM_STEP, -dy * perDelta));
      const rect = stage.getBoundingClientRect();
      editor.setZoomAroundPoint(editor.state.zoom * Math.exp(step), event.clientX - rect.left, event.clientY - rect.top);
      moved("zoom");
      return;
    }
    // 2. 平移跟手；鼠标按住 Shift 时竖向滚轮改横向
    const horizontal = event.shiftKey && dx === 0;
    editor.pan(horizontal ? -dy : -dx, horizontal ? 0 : -dy);
    moved("pan");
  };

  // 3. 右键拖动平移；没拖动时右键什么也不做，画布上不弹浏览器菜单
  const onMouseDown = (event: MouseEvent) => {
    if (event.button !== 2 || !onCanvas(event.target) || !enabled()) return;
    event.preventDefault();
    event.stopPropagation();
    const surface = event.target as HTMLElement;
    const cursor = surface.style.cursor;
    let lastX = event.clientX;
    let lastY = event.clientY;
    let dragging = false;
    const move = (next: MouseEvent) => {
      if (!dragging && Math.hypot(next.clientX - event.clientX, next.clientY - event.clientY) < DRAG_SLOP_PX) return;
      dragging = true;
      surface.style.cursor = "grabbing";
      editor.pan(next.clientX - lastX, next.clientY - lastY);
      lastX = next.clientX;
      lastY = next.clientY;
      moved("pan");
    };
    const up = () => {
      if (dragging) surface.style.cursor = cursor;
      window.removeEventListener("mousemove", move, true);
      window.removeEventListener("mouseup", up, true);
    };
    window.addEventListener("mousemove", move, true);
    window.addEventListener("mouseup", up, true);
  };
  const onContextMenu = (event: MouseEvent) => {
    if (onCanvas(event.target)) event.preventDefault();
  };

  stage.addEventListener("wheel", onWheel, { capture: true, passive: false });
  stage.addEventListener("mousedown", onMouseDown, true);
  stage.addEventListener("contextmenu", onContextMenu, true);
  return () => {
    if (settle !== null) clearTimeout(settle);
    stage.removeEventListener("wheel", onWheel, true);
    stage.removeEventListener("mousedown", onMouseDown, true);
    stage.removeEventListener("contextmenu", onContextMenu, true);
  };
}

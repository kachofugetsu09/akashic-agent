import type { ReactNode } from "react";

/**
 * 同一个格子里的两枚图标：切到哪一枚，另一枚就缩小、模糊着让位。
 *
 * 两枚都常驻 DOM 并叠在同一个 grid 单元，所以切换过程不改变按钮尺寸，
 * 也不会把控件移出 Tab 序。样式在 styles.css 的 .icon-swap，只动
 * transform / opacity / filter，不参与布局。
 */
export function IconSwap({ state, a, b }: { state: boolean; a: ReactNode; b: ReactNode }) {
  return (
    <span className="icon-swap" data-state={state ? "b" : "a"} aria-hidden="true">
      <span className="icon-swap__slot">{a}</span>
      <span className="icon-swap__slot">{b}</span>
    </span>
  );
}

/**
 * 复制完成的勾：一次画完的路径，`pathLength={1}` 让描线进度与路径实际长度无关。
 * 描线只是「完成了」的视觉说法，本身不含位移。
 */
export function CheckGlyph() {
  return (
    <svg className="icon-swap__tick" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path pathLength={1} d="M4 12.5 9.5 18 20 6.5" />
    </svg>
  );
}

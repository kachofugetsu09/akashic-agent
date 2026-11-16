import { useCallback, useEffect, useState, type TransitionEvent } from "react";
import { prefersReducedMotion } from "./tool-result-view";

export interface ExitPresence {
  /** 元素是否应保留在 DOM 中（打开或正在退场）。 */
  present: boolean;
  /** 当前语义状态是否处于打开态。 */
  open: boolean;
  /** 正在播放退场过渡。 */
  closing: boolean;
  /** 挂在过渡根节点上，退场的全部过渡结束立即卸载，不等兜底定时器。 */
  onTransitionEnd: (event: TransitionEvent<HTMLElement>) => void;
}

/**
 * 为挂载/卸载型浮层提供可打断的进退场生命周期：
 * 打开立即挂载并触发 CSS transition；关闭先切到 closed 态播完柔和退场再卸载；
 * 若退场中途再次打开，直接在当前位置平滑折返。
 */
export function useExitPresence(open: boolean, fallbackMs = 240): ExitPresence {
  const [mounted, setMounted] = useState(open);
  const [prevOpen, setPrevOpen] = useState(open);
  const [closing, setClosing] = useState(false);

  if (open !== prevOpen) {
    setPrevOpen(open);
    if (open) {
      setMounted(true);
      setClosing(false);
    } else if (mounted) {
      if (prefersReducedMotion()) {
        setMounted(false);
        setClosing(false);
      } else {
        setClosing(true);
      }
    }
  }

  useEffect(() => {
    if (!closing) return;
    const timer = window.setTimeout(() => {
      setMounted(false);
      setClosing(false);
    }, fallbackMs);
    return () => window.clearTimeout(timer);
  }, [closing, fallbackMs]);

  // 各属性退场时长不同（如透明度 180ms、位移 220ms），先结束的那条不能把还在走的位移截断。
  const onTransitionEnd = useCallback((event: TransitionEvent<HTMLElement>) => {
    const element = event.currentTarget;
    if (event.target !== element || open) return;
    if (element.getAnimations().some((animation) => animation instanceof CSSTransition && animation.playState === "running")) return;
    setMounted(false);
    setClosing(false);
  }, [open]);

  return { present: open || mounted, open, closing, onTransitionEnd };
}

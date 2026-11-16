import { useEffect, useState } from "react";
import { prefersReducedMotion } from "./tool-result-view";

/** 旧文案留在这里退场，与 CSS 的 --ak-sys-duration-instant 一致。 */
const SWAP_OUT_MS = 80;

type Phase = "idle" | "out" | "in";

/**
 * 同一格里的文案换挡：旧的那句先向上 4px 淡出，新的一句从下方 4px 补上来。
 *
 * 全程只维护一个文本节点——`aria-live` 区域如果同时留着新旧两份，读屏会把同一件事播两遍；
 * 进出都走纯 CSS transition（不用固定时间轴 @keyframes），连续变化时可平滑打断。
 */
export function SwapText({ text }: { text: string }) {
  const [wanted, setWanted] = useState(text);
  const [shown, setShown] = useState(text);
  const [phase, setPhase] = useState<Phase>("idle");

  if (wanted !== text) {
    setWanted(text);
    if (prefersReducedMotion()) {
      setShown(text);
      setPhase("idle");
    } else {
      setPhase("out");
    }
  }

  useEffect(() => {
    if (phase !== "out") return;
    const timer = window.setTimeout(() => {
      setShown(wanted);
      setPhase("in");
    }, SWAP_OUT_MS);
    return () => window.clearTimeout(timer);
  }, [phase, wanted]);

  useEffect(() => {
    if (phase !== "in") return;
    let f1 = 0;
    let f2 = 0;
    f1 = window.requestAnimationFrame(() => {
      f2 = window.requestAnimationFrame(() => setPhase("idle"));
    });
    return () => {
      window.cancelAnimationFrame(f1);
      window.cancelAnimationFrame(f2);
    };
  }, [phase]);

  return <span className="swap-text" data-swap={phase}>{shown}</span>;
}

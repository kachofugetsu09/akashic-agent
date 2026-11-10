import { useEffect, useState } from "react";

/** 旧文案留在这里退场，与 CSS 的 --ak-sys-duration-instant 一致。 */
const SWAP_OUT_MS = 80;
/** 新文案落位，与 CSS 的 --ak-sys-duration-short 一致。 */
const SWAP_IN_MS = 150;

type Phase = "idle" | "out" | "in";

/**
 * 同一格里的文案换挡：旧的那句先向上 4px 淡出，新的一句从下方 4px 补上来。
 *
 * 全程只维护一个文本节点——`aria-live` 区域如果同时留着新旧两份，读屏会把同一件事播两遍；
 * 所以这里是「先出后进」而不是两句话交叠。位移只有 4px，属于状态切换，不是装饰。
 */
export function SwapText({ text }: { text: string }) {
  // 目标是渲染期就调到最新，不等一次 effect 再渲染：相位变化和文案变化同一帧发生。
  const [wanted, setWanted] = useState(text);
  const [shown, setShown] = useState(text);
  const [phase, setPhase] = useState<Phase>("idle");

  if (wanted !== text) {
    setWanted(text);
    setPhase("out");
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
    const timer = window.setTimeout(() => setPhase("idle"), SWAP_IN_MS);
    return () => window.clearTimeout(timer);
  }, [phase]);

  return <span className="swap-text" data-swap={phase}>{shown}</span>;
}

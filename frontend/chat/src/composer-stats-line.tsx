import { useEffect, useState } from "react";
import type { ReplyActivity, TimelineMessage } from "./message-timeline";
import { formatModelCallStats, loadWebModelCallStats, selectModelCall, type LoadModelCallStats, type ModelCallStats } from "./model-call-stats";

/** 终态摘要停留时长，到点后淡出收纳。 */
const STATS_SETTLED_MS = 3000;
/** 淡出过渡时长，与 CSS 的 --ak-sys-duration-short 一致；reduced-motion 下 CSS 归零即瞬时隐藏。 */
const STATS_FADE_MS = 150;

type StatsPhase = "active" | "settled" | "fading" | "gone";

/** 按调用 ID 读取服务端统计；切页立即撤销旧读取，重连不重新计时。 */
export function ComposerStatsLine({ messages, activities, connected, load = loadWebModelCallStats }: {
  messages: readonly TimelineMessage[];
  activities: readonly ReplyActivity[];
  connected: boolean;
  load?: LoadModelCallStats;
}) {
  const { callId, active } = selectModelCall(messages, activities);
  const [result, setResult] = useState<{ callId: string; stats?: ModelCallStats; error?: string } | null>(null);
  const [phase, setPhase] = useState<StatsPhase>("active");
  useEffect(() => {
    if (!callId || !connected) return;
    const selectedId = callId;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    async function read() {
      try {
        const stats = await load(selectedId, controller.signal);
        if (controller.signal.aborted) return;
        setResult({ callId: selectedId, stats });
        if (active && stats.state === "started") timer = setTimeout(() => void read(), 1000);
      } catch {
        if (!controller.signal.aborted) {
          setResult({ callId: selectedId, error: "统计暂不可用" });
          if (active) timer = setTimeout(() => void read(), 1000);
        }
      }
    }
    void read();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [callId, active, connected, load]);
  // 活动状态切换驱动收纳节奏：终态摘要停留 3 秒，淡出过渡后移出渲染。
  useEffect(() => {
    if (active) {
      setPhase("active");
      return;
    }
    setPhase("settled");
    const fadeTimer = setTimeout(() => setPhase("fading"), STATS_SETTLED_MS);
    const goneTimer = setTimeout(() => setPhase("gone"), STATS_SETTLED_MS + STATS_FADE_MS);
    return () => { clearTimeout(fadeTimer); clearTimeout(goneTimer); };
  }, [active]);
  if (!callId || !connected || phase === "gone") return null;
  const current = result?.callId === callId ? result : null;
  if (!active && !current) return null;
  const summary = current?.stats ? formatModelCallStats(current.stats, false) : current?.error ?? "";
  // aria-live 只在活动切换时改文案：开始固定"正在生成…"，结束给一条终态摘要；逐秒数据进 aria-hidden 呈现层。
  const liveText = active ? "正在生成…" : summary;
  const detail = active && current?.stats ? formatModelCallStats(current.stats, true) : "";
  const titleLine = current?.stats ? formatModelCallStats(current.stats, active) : undefined;
  return (
    <div className={`composer-stats-line${phase === "fading" ? " is-fading" : ""}`}
      title={titleLine && current?.stats ? `${titleLine} · ${current.stats.model}` : current?.stats?.model}>
      <span aria-live="polite">{liveText}</span>
      {detail ? <span aria-hidden="true">{detail}</span> : null}
    </div>
  );
}

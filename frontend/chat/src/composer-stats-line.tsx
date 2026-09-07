import { useEffect, useState } from "react";
import type { ReplyActivity, TimelineMessage } from "./message-timeline";
import { formatModelCallStats, loadWebModelCallStats, selectModelCall, type LoadModelCallStats, type ModelCallStats } from "./model-call-stats";

/** 按调用 ID 读取服务端统计；切页立即撤销旧读取，重连不重新计时。 */
export function ComposerStatsLine({ messages, activities, connected, load = loadWebModelCallStats }: {
  messages: readonly TimelineMessage[];
  activities: readonly ReplyActivity[];
  connected: boolean;
  load?: LoadModelCallStats;
}) {
  const { callId, active } = selectModelCall(messages, activities);
  const [result, setResult] = useState<{ callId: string; stats?: ModelCallStats; error?: string } | null>(null);
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
  if (!callId || !connected) return null;
  const current = result?.callId === callId ? result : null;
  const line = current?.stats ? formatModelCallStats(current.stats, active) : current?.error;
  if (!line) return null;
  return <div className="composer-stats-line" aria-live="polite" title={current?.stats?.model}>
    <span>{line}</span>
  </div>;
}

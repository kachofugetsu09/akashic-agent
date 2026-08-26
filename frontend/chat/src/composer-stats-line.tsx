import { useSyncExternalStore } from "react";
import {
  ClientTurnMetricsTracker,
  formatComposerStatsLine,
  type TurnMetricsSnapshot,
} from "./stream-turn-metrics";

/** Subscribe to live TTFT / TPS under the composer (dsh StatsLine placement). */
export function ComposerStatsLine({ tracker }: { tracker: ClientTurnMetricsTracker }) {
  const snapshot = useSyncExternalStore(
    tracker.subscribe,
    tracker.getSnapshot,
    tracker.getSnapshot,
  );
  const line = formatComposerStatsLine(snapshot);
  if (!line) return null;
  return (
    <div className="composer-stats-line" aria-live="polite">
      <span>{line}</span>
      {estimateHint(snapshot)}
    </div>
  );
}

function estimateHint(snapshot: TurnMetricsSnapshot | null) {
  if (snapshot === null || snapshot.phase !== "decoding") return null;
  if (snapshot.tokensPerSecond === undefined) return null;
  return <span className="composer-stats-line__hint">估</span>;
}

// Client-side turn latency/throughput fold aligned with dsh turn-metrics:
// TTFT = attempt start → first non-empty delta; TPS = outputTokens / (decodeMs/1000).

export type TurnMetricsPhase = "idle" | "waiting" | "decoding" | "settled";

/** Display reading for the composer stats strip. */
export type TurnMetricsSnapshot = {
  phase: TurnMetricsPhase;
  ttftMs?: number;
  tokensPerSecond?: number;
};

/**
 * Sub-turn latency figure: one decimal under ten seconds, whole seconds beyond.
 * Unit-less so the template owns the second suffix (dsh formatLatencySeconds).
 */
export function formatLatencySeconds(ms: number): string {
  const s = Math.max(0, ms) / 1000;
  return s < 10 ? String(Math.round(s * 10) / 10) : String(Math.round(s));
}

/**
 * Decode-throughput figure: whole tokens from ten up, one decimal below
 * (dsh formatTokensPerSecond).
 */
export function formatTokensPerSecond(tps: number): string {
  const clamped = Math.max(0, tps);
  return clamped >= 10 ? String(Math.round(clamped)) : String(Math.round(clamped * 10) / 10);
}

/** Rough token estimate when provider usage is absent (Latin ~4 chars, CJK ~1.5). */
export function estimateOutputTokens(text: string): number {
  let units = 0;
  for (const ch of text) {
    const code = ch.codePointAt(0) ?? 0;
    units += code > 0x7f ? 1.5 : 0.25;
  }
  return Math.max(0, Math.round(units));
}

/** dsh rule: only report TPS when decode window and token count are both present. */
export function deriveTokensPerSecond(
  outputTokens: number,
  decodeMs: number,
): number | undefined {
  if (!(decodeMs > 0) || !(outputTokens > 0)) return undefined;
  return outputTokens / (decodeMs / 1000);
}

function readingFromParts(
  phase: TurnMetricsPhase,
  startedAt: number,
  firstTokenAt: number | null,
  decodeText: string,
  outputTokens: number | null,
  now: number,
): TurnMetricsSnapshot {
  if (firstTokenAt === null) return { phase };
  const ttftMs = Math.max(0, firstTokenAt - startedAt);
  const decodeMs = Math.max(0, now - firstTokenAt);
  const tokens = outputTokens ?? estimateOutputTokens(decodeText);
  return {
    phase,
    ttftMs,
    tokensPerSecond: deriveTokensPerSecond(tokens, decodeMs),
  };
}

/** Mutable tracker fed by WebSocket frames; React reads via subscribe/getSnapshot. */
export class ClientTurnMetricsTracker {
  private startedAt: number | null = null;
  private firstTokenAt: number | null = null;
  private decodeText = "";
  private turnId: string | null = null;
  private phase: TurnMetricsPhase = "idle";
  private lastSettled: TurnMetricsSnapshot | null = null;
  private cached: TurnMetricsSnapshot | null = null;
  private readonly listeners = new Set<() => void>();

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  getSnapshot = (): TurnMetricsSnapshot | null => this.cached;

  onTurnStarted(turnId: string, now = Date.now()): void {
    this.turnId = turnId;
    this.startedAt = now;
    this.firstTokenAt = null;
    this.decodeText = "";
    this.phase = "waiting";
    this.cached = { phase: "waiting" };
    this.emit();
  }

  /** Start the clock once per turn; ignore duplicate starts for the same id. */
  ensureTurnStarted(turnId: string, now = Date.now()): void {
    if (
      this.turnId === turnId
      && (this.phase === "waiting" || this.phase === "decoding")
    ) {
      return;
    }
    this.onTurnStarted(turnId, now);
  }

  onDelta(turnId: string, delta: string, now = Date.now()): void {
    if (!delta) return;
    if (this.turnId !== turnId) {
      if (this.phase === "waiting" || this.phase === "decoding") return;
      this.onTurnStarted(turnId, now);
    }
    if (this.startedAt === null) this.startedAt = now;
    if (this.firstTokenAt === null) this.firstTokenAt = now;
    this.decodeText += delta;
    this.phase = "decoding";
    this.cached = readingFromParts(
      "decoding",
      this.startedAt,
      this.firstTokenAt,
      this.decodeText,
      null,
      now,
    );
    this.emit();
  }

  onSettled(turnId: string, outputTokens: number | null = null, now = Date.now()): void {
    if (this.turnId !== null && this.turnId !== turnId) return;
    if (this.startedAt === null) {
      this.phase = "idle";
      this.emit();
      return;
    }
    if (this.firstTokenAt !== null) {
      const decodeMs = Math.max(0, now - this.firstTokenAt);
      const tokens = outputTokens ?? estimateOutputTokens(this.decodeText);
      this.lastSettled = {
        phase: "settled",
        ttftMs: Math.max(0, this.firstTokenAt - this.startedAt),
        tokensPerSecond: deriveTokensPerSecond(tokens, decodeMs),
      };
      this.cached = this.lastSettled;
    } else {
      this.cached = this.lastSettled;
    }
    this.phase = "settled";
    this.turnId = null;
    this.startedAt = null;
    this.firstTokenAt = null;
    this.decodeText = "";
    this.emit();
  }

  onInterrupted(): void {
    if (this.phase === "waiting" || this.phase === "decoding") {
      if (this.startedAt !== null && this.firstTokenAt !== null) {
        this.onSettled(this.turnId ?? "", null);
        return;
      }
      this.phase = "idle";
      this.turnId = null;
      this.startedAt = null;
      this.firstTokenAt = null;
      this.decodeText = "";
      this.cached = this.lastSettled;
      this.emit();
    }
  }

  private emit(): void {
    for (const listener of this.listeners) listener();
  }
}

/** Format the composer strip: `首 token 0.8s · 20 tok/s`. */
export function formatComposerStatsLine(snapshot: TurnMetricsSnapshot | null): string | null {
  if (snapshot === null) return null;
  if (snapshot.phase === "waiting") return "等待首 token…";
  const parts: string[] = [];
  if (snapshot.ttftMs !== undefined) {
    parts.push(`首 token ${formatLatencySeconds(snapshot.ttftMs)}s`);
  }
  if (snapshot.tokensPerSecond !== undefined) {
    parts.push(`${formatTokensPerSecond(snapshot.tokensPerSecond)} tok/s`);
  }
  if (parts.length === 0) return null;
  return parts.join(" · ");
}

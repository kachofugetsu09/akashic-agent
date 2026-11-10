/** 桌面 WebUI turn 级观测：记录无正文、有界、可关联的渲染里程碑。 */

export const WEB_TURN_TRACE_MAX_TRACKED = 64;

export const WEB_TURN_TRACE_EVENTS = [
  "webui.frame_received",
  "webui.projection_published",
  "webui.react_committed",
  "webui.next_frame_ready",
] as const;
const WEB_TURN_TRACE_MAX_RECORDS = WEB_TURN_TRACE_MAX_TRACKED * WEB_TURN_TRACE_EVENTS.length * 3;

export type WebTurnTraceEvent = (typeof WEB_TURN_TRACE_EVENTS)[number];
export type WebTurnTraceKind = "thinking" | "answer" | "terminal";

export interface WebTurnTraceRecord {
  event: WebTurnTraceEvent;
  session_id: string;
  turn_id: string;
  wall_ms: number;
  performance_ms: number;
  kind: WebTurnTraceKind;
  origin: string;
}

interface WebTurnTraceEntry {
  sessionId: string;
  turnId: string;
  pendingKinds: Set<WebTurnTraceKind>;
  milestones: Set<string>;
}

export type WebTurnTraceEmit = (record: WebTurnTraceRecord) => void;

export const webTurnTraceEmit: WebTurnTraceEmit = (record) => {
  console.log(`[akashic-trace] ${JSON.stringify(record)}`);
};

/** 将一个 turn 的 transport、projection 与 React 帧里程碑关联起来。 */
export class WebTurnTraceRegistry {
  private readonly entries = new Map<string, WebTurnTraceEntry>();
  private readonly records: WebTurnTraceRecord[] = [];
  private readonly emit: WebTurnTraceEmit;

  constructor(emit: WebTurnTraceEmit = webTurnTraceEmit) {
    this.emit = emit;
  }

  observeFrame(sessionId: string, turnId: string, kind: WebTurnTraceKind): void {
    const entry = this.entry(sessionId, turnId);
    if (this.mark(entry, "webui.frame_received", kind, "websocket")) {
      entry.pendingKinds.add(kind);
    }
  }

  markProjection(turnId: string): void {
    const entry = this.findByTurnId(turnId);
    if (entry === undefined) return;
    for (const kind of entry.pendingKinds) {
      this.mark(entry, "webui.projection_published", kind, "stream-projection");
    }
  }

  markReactCommit(turnId: string): WebTurnTraceKind[] {
    const entry = this.findByTurnId(turnId);
    if (entry === undefined) return [];
    const committed: WebTurnTraceKind[] = [];
    for (const kind of entry.pendingKinds) {
      if (this.mark(entry, "webui.react_committed", kind, "message-row")) committed.push(kind);
    }
    return committed;
  }

  markNextFrame(turnId: string, kinds: readonly WebTurnTraceKind[]): void {
    const entry = this.findByTurnId(turnId);
    if (entry === undefined) return;
    for (const kind of kinds) {
      this.mark(entry, "webui.next_frame_ready", kind, "requestAnimationFrame");
      entry.pendingKinds.delete(kind);
    }
  }

  snapshot(): readonly WebTurnTraceRecord[] {
    return [...this.records];
  }

  reset(): void {
    this.entries.clear();
    this.records.length = 0;
  }

  private entry(sessionId: string, turnId: string): WebTurnTraceEntry {
    const key = `${sessionId}\u001f${turnId}`;
    let entry = this.entries.get(key);
    if (entry !== undefined) return entry;
    if (this.entries.size >= WEB_TURN_TRACE_MAX_TRACKED) {
      const oldest = this.entries.keys().next().value;
      if (oldest !== undefined) this.entries.delete(oldest);
    }
    entry = { sessionId, turnId, pendingKinds: new Set(), milestones: new Set() };
    this.entries.set(key, entry);
    return entry;
  }

  private findByTurnId(turnId: string): WebTurnTraceEntry | undefined {
    for (const entry of this.entries.values()) {
      if (entry.turnId === turnId) return entry;
    }
    return undefined;
  }

  private mark(
    entry: WebTurnTraceEntry,
    event: WebTurnTraceEvent,
    kind: WebTurnTraceKind,
    origin: string,
  ): boolean {
    const milestone = `${event}\u001f${kind}`;
    if (entry.milestones.has(milestone)) return false;
    entry.milestones.add(milestone);
    const record = {
      event,
      session_id: entry.sessionId,
      turn_id: entry.turnId,
      wall_ms: Date.now(),
      performance_ms: performance.now(),
      kind,
      origin,
    } satisfies WebTurnTraceRecord;
    this.records.push(record);
    if (this.records.length > WEB_TURN_TRACE_MAX_RECORDS) this.records.shift();
    this.emit(record);
    return true;
  }
}

export const webTurnTrace = new WebTurnTraceRegistry();

declare global {
  interface Window {
    __akashicWebTrace?: {
      snapshot: () => readonly WebTurnTraceRecord[];
      reset: () => void;
    };
  }
}

if (
  typeof window !== "undefined"
  && new URLSearchParams(window.location.search).get("akashic_perf") === "1"
) {
  window.__akashicWebTrace = {
    snapshot: () => webTurnTrace.snapshot(),
    reset: () => webTurnTrace.reset(),
  };
}

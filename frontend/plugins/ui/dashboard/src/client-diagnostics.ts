// 真实使用中的交互耗时与前端报错：逐条记下每次交互从按下到出帧的耗时、三段拆分和长帧归因，
// 攒批发给服务端，夜间巡检按「动作 × 版本」算 p75、找回归。对照 web-vitals 的 INP 归因口径，
// 但不取页面级单值：这里的页面一开就是几天，要的是每一次交互。

const ENDPOINT = "/api/client/diagnostics";
const KEY_THRESHOLD_MS = 50;
const FLUSH_SIZE = 50;
const FLUSH_DELAY_MS = 60_000;
const SETTLE_MS = 300;
const FRAME_WINDOW_MS = 10_000;
const BATCH_LIMIT = 100;
const INTERACTIVE = "[data-perf], button, a[href], summary, input, textarea, select, label, [role=button], [role=tab], [role=menuitem], [role=option], [role=switch], [role=checkbox], [tabindex]";
const NOISE = /ResizeObserver loop|^Script error\.?$/;
// Tailwind 工具类不表达「点的是什么」，取动作名时跳过。
const UTILITY = /^(w|h|size|min|max|p[xytrbl]?|m[xytrbl]?|text|bg|border|rounded|flex|grid|gap|items|justify|font|leading|tracking|overflow|shadow|ring|outline|opacity|z|top|left|right|bottom|inset|col|row|self|place|space|divide|cursor|select|resize|transition|duration|ease|animate|sr|truncate|shrink|grow|basis|order|align|whitespace|break|fill|stroke|field-sizing)(-|$)/;

interface EventTiming extends PerformanceEntry {
  readonly interactionId?: number;
  readonly processingStart: number;
  readonly processingEnd: number;
  readonly target: Node | null;
}

interface ScriptTiming {
  readonly duration: number;
  readonly invoker: string;
  readonly sourceURL: string;
  readonly sourceFunctionName: string;
}

interface AnimationFrameTiming extends PerformanceEntry {
  readonly blockingDuration: number;
  readonly renderStart: number;
  readonly styleAndLayoutStart: number;
  readonly scripts: readonly ScriptTiming[];
}

interface PendingInteraction {
  start: number;
  duration: number;
  processingStart: number;
  processingEnd: number;
  event: string;
  action: string;
}

interface InteractionRecord {
  kind: "interaction";
  at: number;
  action: string;
  event: string;
  duration: number;
  input_delay: number;
  processing: number;
  presentation: number;
  script?: number;
  style_layout?: number;
  render?: number;
  blocking?: number;
  script_source?: string;
  dom: number;
  width: number;
  touch: boolean;
  page_release?: string;
}

interface ErrorRecord {
  kind: "error";
  at: number;
  message: string;
  stack: string;
  source: string;
  count: number;
  page_release?: string;
}

type DiagnosticRecord = InteractionRecord | ErrorRecord;

const round = (value: number): number => Math.round(value * 10) / 10;

// 动作名：优先显式的 data-perf，其次最近可交互元素的语义 class（跳过 Tailwind 变体与状态类），再退到无障碍名称。
function describe(node: Node | null): string {
  const element = node instanceof Element ? node : node?.parentElement ?? null;
  if (!element) return "";
  const labelled = element.closest<HTMLElement>("[data-perf]");
  if (labelled?.dataset.perf) return labelled.dataset.perf;
  const owner = element.closest(INTERACTIVE) ?? element;
  const tokens = [...owner.classList].filter((token) => /^[a-z][\w-]*$/i.test(token) && !UTILITY.test(token) && !/^(is|has)-|^(open|active|selected|disabled)$/.test(token));
  const named = tokens.find((token) => /[-_]/.test(token)) ?? tokens[0];
  if (named) return `${owner.tagName.toLowerCase()}.${named}`;
  const label = owner.getAttribute("aria-label")?.trim().slice(0, 40);
  return label ? `${owner.tagName.toLowerCase()}[${label}]` : owner.tagName.toLowerCase();
}

function shortSource(url: string): string {
  try {
    const parts = new URL(url, location.href).pathname.split("/").filter(Boolean);
    return parts.slice(-2).join("/");
  } catch {
    return "";
  }
}

// 长帧归因：与交互时间窗相交的帧里，脚本、样式布局、渲染各占多少，最重的一段脚本是谁。
function attribute(frames: readonly AnimationFrameTiming[], start: number, end: number): Partial<InteractionRecord> {
  const hit = frames.filter((frame) => frame.startTime < end && frame.startTime + frame.duration > start);
  if (!hit.length) return {};
  let script = 0, styleLayout = 0, render = 0, blocking = 0;
  let longest: ScriptTiming | null = null;
  for (const frame of hit) {
    const frameEnd = frame.startTime + frame.duration;
    blocking += frame.blockingDuration;
    if (frame.renderStart > 0) render += frameEnd - frame.renderStart;
    if (frame.styleAndLayoutStart > 0) styleLayout += frameEnd - frame.styleAndLayoutStart;
    for (const entry of frame.scripts) {
      script += entry.duration;
      if (!longest || entry.duration > longest.duration) longest = entry;
    }
  }
  const source = longest ? `${longest.invoker} @ ${shortSource(longest.sourceURL)}${longest.sourceFunctionName ? `:${longest.sourceFunctionName}` : ""}` : undefined;
  return { script: round(script), style_layout: round(styleLayout), render: round(render), blocking: round(blocking), script_source: source?.slice(0, 240) };
}

class DiagnosticsQueue {
  private records: DiagnosticRecord[] = [];
  private errors = new Map<string, ErrorRecord>();
  private timer = 0;
  release: string | undefined;

  push(record: DiagnosticRecord): void {
    this.records.push(record);
    if (this.records.length >= FLUSH_SIZE) this.flush(false);
    else if (!this.timer) this.timer = window.setTimeout(() => this.flush(false), FLUSH_DELAY_MS);
  }

  // 同一页面里重复的报错只发一条并计数，避免一个循环里的异常刷满日志。
  error(message: string, stack: string, source: string): void {
    if (!message || NOISE.test(message) || /^(chrome|moz)-extension:/.test(source)) return;
    const key = `${message}\n${source}`;
    const known = this.errors.get(key);
    if (known) {
      known.count += 1;
      return;
    }
    const record: ErrorRecord = { kind: "error", at: Date.now(), message: message.slice(0, 1000), stack: stack.slice(0, 6000), source: source.slice(0, 240), count: 1 };
    this.errors.set(key, record);
    this.push(record);
  }

  // 页面转入后台时用 sendBeacon，平时用 keepalive 的 fetch；失败就丢，打点不重试也不阻塞界面。
  flush(leaving: boolean): void {
    window.clearTimeout(this.timer);
    this.timer = 0;
    const records = this.records.splice(0);
    for (let index = 0; index < records.length; index += BATCH_LIMIT) {
      const batch = records.slice(index, index + BATCH_LIMIT).map((record) => ({ ...record, page_release: record.page_release ?? this.release }));
      const body = JSON.stringify({ records: batch });
      if (leaving && navigator.sendBeacon?.(ENDPOINT, new Blob([body], { type: "application/json" }))) continue;
      void fetch(ENDPOINT, { method: "POST", body, keepalive: body.length < 60_000, headers: { "Content-Type": "application/json" } }).catch(() => undefined);
    }
  }
}

// 记下最近几次输入的目标：交互结束时目标节点可能已被卸载，Event Timing 的 target 会是 null。
function watchInputs(): (start: number) => string {
  const recent: { time: number; action: string }[] = [];
  const remember = (event: Event) => {
    recent.push({ time: event.timeStamp, action: describe(event.target as Node | null) });
    if (recent.length > 8) recent.shift();
  };
  for (const type of ["pointerdown", "keydown", "click"]) window.addEventListener(type, remember, { capture: true, passive: true });
  return (start) => recent.find((input) => Math.abs(input.time - start) < 2)?.action ?? "";
}

function watchInteractions(queue: DiagnosticsQueue): void {
  const supported = PerformanceObserver.supportedEntryTypes;
  if (!supported.includes("event")) return;
  const frames: AnimationFrameTiming[] = [];
  const pending = new Map<number, PendingInteraction>();
  const actionAt = watchInputs();
  let settleTimer = 0;

  // 1. 同一次交互的多个事件合并（pointerdown/up/click 共享 interactionId）
  // 2. 等长帧条目到齐后再结算，挂上归因
  const settle = () => {
    settleTimer = 0;
    const horizon = performance.now() - FRAME_WINDOW_MS;
    while (frames.length && frames[0].startTime < horizon) frames.shift();
    const dom = document.getElementsByTagName("*").length;
    const touch = window.matchMedia("(pointer: coarse)").matches;
    for (const item of pending.values()) {
      if (item.event === "keyboard" && item.duration < KEY_THRESHOLD_MS) continue;
      const end = item.start + item.duration;
      queue.push({
        kind: "interaction", at: Math.round(performance.timeOrigin + item.start), action: item.action || item.event, event: item.event,
        duration: round(item.duration), input_delay: round(item.processingStart - item.start),
        processing: round(Math.min(item.processingEnd, end) - item.processingStart), presentation: round(Math.max(0, end - item.processingEnd)),
        ...attribute(frames, item.start, end), dom, width: window.innerWidth, touch,
      });
    }
    pending.clear();
  };

  new PerformanceObserver((list) => {
    for (const entry of list.getEntries() as EventTiming[]) {
      const id = entry.interactionId ?? 0;
      if (!id) continue;
      const known = pending.get(id);
      if (known) {
        known.duration = Math.max(known.duration, entry.duration);
        known.processingStart = Math.min(known.processingStart, entry.processingStart);
        known.processingEnd = Math.max(known.processingEnd, entry.processingEnd);
        if (!known.action) known.action = describe(entry.target);
        continue;
      }
      const action = (entry.target?.isConnected ? describe(entry.target) : "") || actionAt(entry.startTime);
      const event = entry.name.startsWith("key") ? "keyboard" : "pointer";
      pending.set(id, { start: entry.startTime, duration: entry.duration, processingStart: entry.processingStart, processingEnd: entry.processingEnd, event, action });
    }
    if (pending.size && !settleTimer) settleTimer = window.setTimeout(settle, SETTLE_MS);
  }).observe({ type: "event", buffered: true, durationThreshold: 16 } as PerformanceObserverInit);

  if (supported.includes("long-animation-frame")) {
    new PerformanceObserver((list) => {
      frames.push(...(list.getEntries() as AnimationFrameTiming[]));
    }).observe({ type: "long-animation-frame", buffered: true });
  }
}

export function startClientDiagnostics(): void {
  if (typeof PerformanceObserver === "undefined") return;
  const queue = new DiagnosticsQueue();
  window.addEventListener("error", (event) => {
    const error: unknown = event.error;
    queue.error(event.message || String(error), error instanceof Error ? error.stack ?? "" : "", `${event.filename}:${event.lineno}:${event.colno}`);
  });
  window.addEventListener("unhandledrejection", (event) => {
    const reason: unknown = event.reason;
    queue.error(reason instanceof Error ? `${reason.name}: ${reason.message}` : `unhandledrejection: ${String(reason)}`, reason instanceof Error ? reason.stack ?? "" : "", "");
  });
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") queue.flush(true);
  });
  window.addEventListener("pagehide", () => queue.flush(true));
  watchInteractions(queue);
  window.setTimeout(() => {
    void fetch(ENDPOINT, { headers: { Accept: "application/json" } })
      .then((response) => response.ok ? response.json() as Promise<{ release?: string }> : null)
      .then((info) => { queue.release = info?.release || undefined; })
      .catch(() => undefined);
  }, 3_000);
}

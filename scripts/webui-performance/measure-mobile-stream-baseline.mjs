import { writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

const cdpEndpoint = requiredEnvironment("AKASHIC_PLAYWRIGHT_CDP");
const timeoutMs = numberEnvironment("AKASHIC_WEBUI_STREAM_TIMEOUT_MS", 130_000);
const prompt = process.env.AKASHIC_WEBUI_STREAM_PROMPT ?? "请重放性能基准对话";
const cdp = await connectCdp(cdpEndpoint);

try {
  await cdp.call("Runtime.enable");
  await waitFor(cdp, "document.querySelector(\"textarea[placeholder='输入消息']\") !== null", true, 10_000);
  await waitFor(cdp, "document.querySelector('.mobile-message-anchor.streaming') === null", true, 10_000);
  await installProbe(cdp);
  const startedAt = await cdp.evaluate("window.__resetMobileStreamBaseline(); performance.now()", true);

  await cdp.evaluate(`(() => {
    const textarea = document.querySelector("textarea[placeholder='输入消息']");
    if (!(textarea instanceof HTMLTextAreaElement)) throw new Error("mobile composer textarea is missing");
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value").set;
    setter.call(textarea, ${JSON.stringify(prompt)});
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
    const send = [...document.querySelectorAll("button")].find((button) => button.getAttribute("aria-label") === "发送消息");
    if (!(send instanceof HTMLButtonElement) || send.disabled) throw new Error("mobile send button is unavailable");
    send.click();
  })()`);
  await waitFor(cdp, "document.querySelector('.mobile-message-anchor.streaming') !== null", true, 10_000);
  await waitFor(cdp, "document.querySelector('.mobile-message-anchor.streaming') === null", true, timeoutMs);

  const metric = await cdp.evaluate(`window.__readMobileStreamBaseline(${startedAt})`, true);
  const pageState = await cdp.evaluate(`({
    pageUrl: location.href,
    messageRows: document.querySelectorAll(".mobile-message-anchor").length,
    domElements: document.querySelectorAll("*").length,
  })`, true);
  const report = {
    schemaVersion: 1,
    capturedAt: new Date().toISOString(),
    pageUrl: pageState.pageUrl,
    metric: {
      ...metric,
      messageRows: pageState.messageRows,
      domElements: pageState.domElements,
    },
  };
  const path = resolve(tmpdir(), `akashic-mobile-stream-${Date.now()}.json`);
  writeFileSync(path, `${JSON.stringify(report, null, 2)}\n`, { mode: 0o600 });
  console.log(JSON.stringify({ report: path, ...report }));
} finally {
  cdp.close();
}
process.exit(0);

async function installProbe(cdp) {
  await cdp.evaluate(`(() => {
    const state = {
      bridgeTimes: [],
      bridgeContentCharacters: 0,
      bridgeThinkingCharacters: 0,
      domTimes: [],
      frameGaps: [],
      longTasks: [],
      longAnimationFrames: [],
      resizeTimes: [],
      resizePixels: 0,
      scrollTimes: [],
      previousFrame: 0,
      observedRow: null,
      observedRowHeight: 0,
    };
    new PerformanceObserver((list) => {
      state.longTasks.push(...list.getEntries().map((entry) => ({
        at: entry.startTime,
        duration: entry.duration,
      })));
    }).observe({ type: "longtask" });
    if (PerformanceObserver.supportedEntryTypes.includes("long-animation-frame")) {
      new PerformanceObserver((list) => {
        state.longAnimationFrames.push(...list.getEntries().map((entry) => ({
          at: entry.startTime,
          duration: entry.duration,
          blockingDuration: entry.blockingDuration,
          scriptDuration: entry.scripts.reduce((sum, script) => sum + script.duration, 0),
        })));
      }).observe({ type: "long-animation-frame" });
    }
    window.addEventListener("message", (event) => {
      if (typeof event.data !== "string") return;
      let envelope;
      try {
        envelope = JSON.parse(event.data);
      } catch {
        return;
      }
      if (envelope?.type !== "mobile.stream-patch") return;
      state.bridgeTimes.push(performance.now());
      const payload = envelope.payload;
      state.bridgeContentCharacters += payload?.contentAppend?.length ?? 0;
      state.bridgeThinkingCharacters += payload?.thinkingAppend?.delta?.length ?? 0;
    });
    const rowObserver = new ResizeObserver((entries) => {
      const height = entries[0]?.borderBoxSize?.[0]?.blockSize ?? entries[0]?.contentRect.height ?? 0;
      if (state.observedRowHeight > 0 && height !== state.observedRowHeight) {
        state.resizeTimes.push(performance.now());
        state.resizePixels += Math.abs(height - state.observedRowHeight);
      }
      state.observedRowHeight = height;
    });
    const observeStreamingRow = () => {
      const row = document.querySelector(".mobile-message-anchor.streaming")?.closest(".mobile-virtual-row") ?? null;
      if (row === state.observedRow) return;
      if (state.observedRow) rowObserver.unobserve(state.observedRow);
      state.observedRow = row;
      state.observedRowHeight = row?.getBoundingClientRect().height ?? 0;
      if (row) rowObserver.observe(row, { box: "border-box" });
    };
    new MutationObserver((mutations) => {
      observeStreamingRow();
      if (!state.observedRow) return;
      if (mutations.some((mutation) => state.observedRow.contains(mutation.target))) {
        state.domTimes.push(performance.now());
      }
    }).observe(document.body, { characterData: true, childList: true, subtree: true });
    document.querySelector(".mobile-conversation")?.addEventListener("scroll", () => {
      state.scrollTimes.push(performance.now());
    }, { passive: true });
    const frame = (timestamp) => {
      if (state.previousFrame > 0) {
        state.frameGaps.push({ at: timestamp, duration: timestamp - state.previousFrame });
      }
      state.previousFrame = timestamp;
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
    window.__resetMobileStreamBaseline = () => {
      state.frameGaps.length = 0;
      state.longTasks.length = 0;
      state.longAnimationFrames.length = 0;
      state.bridgeTimes.length = 0;
      state.bridgeContentCharacters = 0;
      state.bridgeThinkingCharacters = 0;
      state.domTimes.length = 0;
      state.resizeTimes.length = 0;
      state.resizePixels = 0;
      state.scrollTimes.length = 0;
      state.previousFrame = 0;
    };
    window.__readMobileStreamBaseline = (startedAt) => ({
      durationMs: performance.now() - startedAt,
      frameCount: state.frameGaps.length,
      frameGapP50Ms: percentile(state.frameGaps.map((entry) => entry.duration), 0.50),
      frameGapP95Ms: percentile(state.frameGaps.map((entry) => entry.duration), 0.95),
      frameGapP99Ms: percentile(state.frameGaps.map((entry) => entry.duration), 0.99),
      frameGapMaxMs: Math.max(0, ...state.frameGaps.map((entry) => entry.duration)),
      frameGapsOver33Ms: state.frameGaps.filter((entry) => entry.duration > 33).length,
      frameGapsOver50Ms: state.frameGaps.filter((entry) => entry.duration > 50).length,
      longTaskCount: state.longTasks.length,
      longTaskTotalMs: sum(state.longTasks.map((entry) => entry.duration)),
      longTaskMaxMs: Math.max(0, ...state.longTasks.map((entry) => entry.duration)),
      longAnimationFrameCount: state.longAnimationFrames.length,
      longAnimationFrameTotalMs: sum(state.longAnimationFrames.map((entry) => entry.duration)),
      longAnimationFrameBlockingMs: sum(state.longAnimationFrames.map((entry) => entry.blockingDuration)),
      longAnimationFrameScriptMs: sum(state.longAnimationFrames.map((entry) => entry.scriptDuration)),
      longAnimationFrameMaxMs: Math.max(0, ...state.longAnimationFrames.map((entry) => entry.duration)),
      bridgePatchCount: state.bridgeTimes.length,
      bridgeCharacters: state.bridgeContentCharacters + state.bridgeThinkingCharacters,
      bridgeContentCharacters: state.bridgeContentCharacters,
      bridgeThinkingCharacters: state.bridgeThinkingCharacters,
      bridgeIntervalP50Ms: percentile(intervals(state.bridgeTimes), 0.50),
      bridgeIntervalP95Ms: percentile(intervals(state.bridgeTimes), 0.95),
      bridgeIntervalMaxMs: Math.max(0, ...intervals(state.bridgeTimes)),
      domMutationBatchCount: state.domTimes.length,
      domMutationIntervalP50Ms: percentile(intervals(state.domTimes), 0.50),
      domMutationIntervalP95Ms: percentile(intervals(state.domTimes), 0.95),
      rowResizeCount: state.resizeTimes.length,
      rowResizePixels: state.resizePixels,
      rowResizeIntervalP50Ms: percentile(intervals(state.resizeTimes), 0.50),
      rowResizeIntervalP95Ms: percentile(intervals(state.resizeTimes), 0.95),
      scrollEventCount: state.scrollTimes.length,
      scrollIntervalP50Ms: percentile(intervals(state.scrollTimes), 0.50),
      scrollIntervalP95Ms: percentile(intervals(state.scrollTimes), 0.95),
      startup: {
        firstBridgeMs: offsetOf(state.bridgeTimes[0], startedAt),
        firstDomMutationMs: offsetOf(state.domTimes[0], startedAt),
        firstResizeMs: offsetOf(state.resizeTimes[0], startedAt),
        windows: [
          frameWindow(startedAt, 0, 2_000),
          frameWindow(startedAt, 2_000, 5_000),
          frameWindow(startedAt, 5_000, 15_000),
          frameWindow(startedAt, 15_000, Number.POSITIVE_INFINITY),
        ],
      },
      jsHeapBytes: performance.memory?.usedJSHeapSize ?? null,
    });
    function frameWindow(startedAt, fromMs, toMs) {
      const frames = state.frameGaps.filter((entry) => {
        const offset = entry.at - startedAt;
        return offset >= fromMs && offset < toMs;
      });
      const durations = frames.map((entry) => entry.duration);
      const longTasks = state.longTasks.filter((entry) => {
        const offset = entry.at - startedAt;
        return offset >= fromMs && offset < toMs;
      });
      return {
        fromMs,
        toMs: Number.isFinite(toMs) ? toMs : null,
        frameCount: durations.length,
        p95Ms: percentile(durations, 0.95),
        p99Ms: percentile(durations, 0.99),
        maxMs: Math.max(0, ...durations),
        over33Ms: durations.filter((value) => value > 33).length,
        over50Ms: durations.filter((value) => value > 50).length,
        longTaskCount: longTasks.length,
        longTaskTotalMs: sum(longTasks.map((entry) => entry.duration)),
      };
    }
    function offsetOf(value, startedAt) {
      return value === undefined ? null : value - startedAt;
    }
    function percentile(values, ratio) {
      if (values.length === 0) return 0;
      const sorted = [...values].sort((left, right) => left - right);
      return sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * ratio) - 1)];
    }
    function sum(values) {
      return values.reduce((total, value) => total + value, 0);
    }
    function intervals(values) {
      return values.slice(1).map((value, index) => value - values[index]);
    }
  })()`);
}

async function waitFor(cdp, expression, expected, timeout) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (await cdp.evaluate(expression, true) === expected) return;
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 100));
  }
  throw new Error(`timed out waiting for: ${expression}`);
}

async function connectCdp(endpoint) {
  const socket = new WebSocket(endpoint);
  await new Promise((resolvePromise, reject) => {
    socket.addEventListener("open", resolvePromise, { once: true });
    socket.addEventListener("error", () => reject(new Error(`cannot connect to CDP endpoint ${endpoint}`)), { once: true });
  });
  let nextId = 1;
  const pending = new Map();
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (!message.id) return;
    const request = pending.get(message.id);
    if (!request) return;
    pending.delete(message.id);
    if (message.error) request.reject(new Error(message.error.message));
    else request.resolve(message.result);
  });
  const call = (method, params = {}) => new Promise((resolvePromise, reject) => {
    const id = nextId++;
    pending.set(id, { resolve: resolvePromise, reject });
    socket.send(JSON.stringify({ id, method, params }));
  });
  return {
    call,
    close: () => socket.close(),
    async evaluate(expression, returnByValue = false) {
      const result = await call("Runtime.evaluate", { expression, awaitPromise: true, returnByValue });
      if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description ?? result.exceptionDetails.text);
      return result.result.value;
    },
  };
}

function requiredEnvironment(name) {
  const value = process.env[name];
  if (!value) throw new Error(`${name} is required`);
  return value;
}

function numberEnvironment(name, fallback) {
  const raw = process.env[name];
  if (raw === undefined) return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0) throw new Error(`${name} must be a positive number`);
  return value;
}

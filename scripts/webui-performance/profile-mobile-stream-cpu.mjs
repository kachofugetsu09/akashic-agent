import { writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

const endpoint = requiredEnvironment("AKASHIC_PLAYWRIGHT_CDP");
const sampleMillis = numberEnvironment("AKASHIC_WEBUI_CPU_PROFILE_MS", 20_000);
const prompt = process.env.AKASHIC_WEBUI_STREAM_PROMPT ?? "请重放性能基准对话";
const cdp = await connectCdp(endpoint);

try {
  await cdp.call("Runtime.enable");
  await waitFor(cdp, "document.querySelector(\"textarea[placeholder='输入消息']\") !== null", true, 10_000);
  await waitFor(cdp, "document.querySelector('.mobile-message-anchor.streaming') === null", true, 10_000);
  await cdp.call("Profiler.enable");
  await cdp.call("Profiler.setSamplingInterval", { interval: 100 });
  await cdp.call("Profiler.start");
  await cdp.evaluate(`(() => {
    const textarea = document.querySelector("textarea[placeholder='输入消息']");
    if (!(textarea instanceof HTMLTextAreaElement)) throw new Error("mobile composer textarea is missing");
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value").set;
    setter.call(textarea, ${JSON.stringify(prompt)});
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
    const send = [...document.querySelectorAll("button")].find(
      (button) => button.getAttribute("aria-label") === "发送消息",
    );
    if (!(send instanceof HTMLButtonElement) || send.disabled) throw new Error("mobile send button is unavailable");
    send.click();
  })()`);
  await waitFor(cdp, "document.querySelector('.mobile-message-anchor.streaming') !== null", true, 10_000);
  await new Promise((resolveWait) => setTimeout(resolveWait, sampleMillis));
  const { profile } = await cdp.call("Profiler.stop");
  const summary = summarizeProfile(profile);
  const path = resolve(tmpdir(), `akashic-mobile-cpu-${Date.now()}.json`);
  writeFileSync(path, `${JSON.stringify({ profile, summary }, null, 2)}\n`, { mode: 0o600 });
  console.log(JSON.stringify({ report: path, sampleMillis, summary }));
} finally {
  cdp.close();
}

function summarizeProfile(profile) {
  const nodes = new Map(profile.nodes.map((node) => [node.id, node.callFrame]));
  const totals = new Map();
  profile.samples?.forEach((nodeId, index) => {
    const frame = nodes.get(nodeId);
    if (!frame) return;
    const key = `${frame.url}:${frame.lineNumber + 1}:${frame.functionName || "(anonymous)"}`;
    const current = totals.get(key) ?? { functionName: frame.functionName || "(anonymous)", url: frame.url, line: frame.lineNumber + 1, selfMicros: 0 };
    current.selfMicros += profile.timeDeltas?.[index] ?? 0;
    totals.set(key, current);
  });
  return [...totals.values()]
    .sort((left, right) => right.selfMicros - left.selfMicros)
    .slice(0, 30)
    .map((entry) => ({ ...entry, selfMillis: entry.selfMicros / 1_000 }));
}

async function waitFor(cdp, expression, expected, timeout) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (await cdp.evaluate(expression, true) === expected) return;
    await new Promise((resolveWait) => setTimeout(resolveWait, 100));
  }
  throw new Error(`timed out waiting for: ${expression}`);
}

async function connectCdp(endpoint) {
  const socket = new WebSocket(endpoint);
  await new Promise((resolveOpen, rejectOpen) => {
    socket.addEventListener("open", resolveOpen, { once: true });
    socket.addEventListener("error", () => rejectOpen(new Error(`cannot connect to CDP endpoint ${endpoint}`)), { once: true });
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
  const call = (method, params = {}) => new Promise((resolveCall, rejectCall) => {
    const id = nextId++;
    pending.set(id, { resolve: resolveCall, reject: rejectCall });
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

import { parentPort, workerData } from "node:worker_threads";

const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;

let nextRpcId = 1;
const pending = new Map();

parentPort.on("message", (msg) => {
  if (msg?.kind !== "reply") return;
  const entry = pending.get(msg.id);
  if (!entry) return;
  pending.delete(msg.id);
  if (msg.error) entry.reject(new Error(msg.error));
  else entry.resolve(msg.value);
});

function callHost(method, args) {
  const id = nextRpcId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    parentPort.postMessage({ kind: "rpc", id, method, args });
  });
}

function serializeArg(arg) {
  return typeof arg === "function" ? arg.toString() : arg;
}

const page = {
  goto: (url, opts) => callHost("goto", [url, opts]),
  waitForSelector: (selector, timeoutMs) => callHost("waitForSelector", [selector, timeoutMs]),
  waitForFunction: (fnOrExpr, timeoutMs, arg) =>
    callHost("waitForFunction", [typeof fnOrExpr === "function" ? fnOrExpr.toString() : fnOrExpr, typeof fnOrExpr === "function", timeoutMs, serializeArg(arg)]),
  evaluate: (fnOrExpr, arg) =>
    callHost("evaluate", [typeof fnOrExpr === "function" ? fnOrExpr.toString() : fnOrExpr, typeof fnOrExpr === "function", serializeArg(arg)]),
  click: (selector) => callHost("click", [selector]),
  fill: (selector, value) => callHost("fill", [selector, value]),
  title: () => callHost("title", []),
  url: () => callHost("url", []),
  content: () => callHost("content", []),
  sleep: (ms) => callHost("sleep", [ms]),
};

const sleep = (ms) => callHost("sleep", [ms]);

(async () => {
  try {
    const fn = new AsyncFunction("page", "sleep", "space", `"use strict";\n${workerData.script}`);
    const result = await fn(page, sleep, workerData.space);
    parentPort.postMessage({ kind: "result", value: result });
  } catch (error) {
    parentPort.postMessage({
      kind: "error",
      error: error instanceof Error ? error.message : String(error),
    });
  }
})();

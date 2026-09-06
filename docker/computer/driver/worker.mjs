import { create as createDomain } from "node:domain";
import { AsyncLocalStorage } from "node:async_hooks";
import { parentPort, workerData } from "node:worker_threads";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import { start as startRepl } from "node:repl";
import { readFile, writeFile } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import { setTimeout as sleep } from "node:timers/promises";

let nextId = 1;
const pending = new Map();
let context;
const callScope = new AsyncLocalStorage();
let output = [];
function requireCall() {
  if (!context || callScope.getStore() !== context)
    throw new Error("Computer output is outside a live call");
}
const hooks = [],
  turnHooks = [];
function call(kind, method, params) {
  if (!context || callScope.getStore() !== context)
    return Promise.reject(
      new Error("Computer operation is outside a live call"),
    );
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    parentPort.postMessage({
      kind,
      id,
      callId: context.call_id,
      method,
      params,
    });
  });
}
const pipes = new Set();
class BrowserPipe extends EventEmitter {
  buffer = Buffer.alloc(0);
  constructor() {
    super();
    pipes.add(this);
  }
  send(message) {
    const data = Buffer.from(JSON.stringify(message)),
      header = Buffer.alloc(4);
    header.writeUInt32LE(data.length);
    this.emit("data", Buffer.concat([header, data]));
  }
  write(data) {
    this.buffer = Buffer.concat([this.buffer, data]);
    while (
      this.buffer.length >= 4 &&
      this.buffer.length >= 4 + this.buffer.readUInt32LE(0)
    ) {
      const size = this.buffer.readUInt32LE(0);
      const message = JSON.parse(this.buffer.subarray(4, size + 4));
      this.buffer = this.buffer.subarray(size + 4);
      call("browser", message.method, message.params ?? {}).then(
        (result) => {
          if (message.id != null)
            this.send({ jsonrpc: "2.0", id: message.id, result });
        },
        (error) => {
          if (message.id != null)
            this.send({
              jsonrpc: "2.0",
              id: message.id,
              error: { code: -32000, message: error.message },
            });
        },
      );
    }
    return true;
  }
  end() {
    pipes.delete(this);
    this.emit("close");
  }
}
const configPath = (path) => {
  if (typeof path !== "string" || path.includes("..") || path.startsWith("/"))
    throw new Error("Invalid driver config key");
  return `${workerData.directory}/config-${Buffer.from(path).toString("hex")}.json`;
};
const nodeRepl = {
  cwd: workerData.directory,
  tmpDir: workerData.directory,
  env: Object.freeze({
    BROWSER_USE_DISABLE_AMBIENT_NETWORK: "1",
    BROWSER_USE_DISABLE_ROLLOUT_TRACKING: "1",
    BROWSER_USE_SECURITY_MODE: "akashic-container",
    BROWSER_USE_AVAILABLE_BACKENDS: "cdp",
    CDP_BROWSER_BACKEND_PIPE_PATH: workerData.pipePath,
    BROWSER_AUTH_EVAL_EXACT_CDP_BACKEND_SOCKET: "true",
  }),
  get requestMeta() {
    return {
      "x-codex-turn-metadata": context && {
        session_id: context.session_id,
        turn_id: context.turn_id,
      },
    };
  },
  config: {
    async read() {
      return { config: {} };
    },
    async readRequirements() {
      return null;
    },
    async readToml(path) {
      try {
        return JSON.parse(await readFile(configPath(path), "utf8"));
      } catch (error) {
        if (error.code === "ENOENT") return {};
        throw error;
      }
    },
    async writeToml(path, value) {
      await writeFile(configPath(path), JSON.stringify(value), { mode: 0o600 });
    },
  },
  nativePipe: {
    async createConnection(path) {
      if (path !== workerData.pipePath)
        throw new Error("Unknown browser backend pipe");
      return new BrowserPipe();
    },
  },
  async createElicitation(params) {
    throw new Error(
      `This operation requires an approval provider unavailable in the container: ${params.message}`,
    );
  },
  addAfterSubmittedCodeHook(hook) {
    hooks.push(hook);
  },
  addTurnEndedHandler(hook) {
    turnHooks.push(hook);
    return () => {
      turnHooks.splice(turnHooks.indexOf(hook), 1);
    };
  },
  setResponseMeta(value) {
    requireCall();
    parentPort.postMessage({ kind: "metadata", value });
  },
  emitContentItem(value) {
    requireCall();
    output.push(value);
  },
  write(value) {
    requireCall();
    output.push({
      type: "text",
      text: typeof value === "string" ? value : JSON.stringify(value),
    });
  },
  async emitImage(value) {
    requireCall();
    const bytes = value?.bytes ?? value;
    if (!(bytes instanceof Uint8Array))
      throw new TypeError("emitImage expects image bytes");
    output.push({
      type: "image",
      data: Buffer.from(bytes).toString("base64"),
      mimeType:
        value.mimeType ?? (bytes[0] === 0x89 ? "image/png" : "image/jpeg"),
    });
  },
  async fetch() {
    throw new Error(
      "Driver ambient network access is disabled; use browser page APIs",
    );
  },
};
globalThis.nodeRepl = nodeRepl;
const { handleRpc } = await import(
  "./reference/browser/scripts/browser-service.mjs"
);
nodeRepl.rpc = (service, request) => {
  if (service !== "browser") throw new Error(`Unknown service: ${service}`);
  return handleRpc(request);
};
const { setupBrowserRuntime } = await import(
  "./reference/browser/scripts/browser-client.mjs"
);
let initialized = false;
const terminal = new PassThrough();
terminal.on("data", (data) => {
  if (context && callScope.getStore() === context && data.toString().trim())
    nodeRepl.write(data.toString());
});
let evaluation;
const evaluationDomain = createDomain();
evaluationDomain.on("error", (error) => {
  if (evaluation && callScope.getStore() === context) evaluation.reject(error);
});
const repl = startRepl({
  domain: evaluationDomain,
  input: new PassThrough(),
  output: terminal,
  terminal: false,
  prompt: "",
  ignoreUndefined: true,
  useGlobal: false,
});
// Reference service 使用私有 host；用户 REPL 只获得公开的输出与临时目录接口。
repl.context.nodeRepl = Object.freeze({
  write: nodeRepl.write,
  emitImage: nodeRepl.emitImage,
  cwd: nodeRepl.cwd,
  tmpDir: nodeRepl.tmpDir,
});
const evaluate = (code) =>
  new Promise((resolve, reject) => {
    evaluation = { reject };
    repl.eval(code + "\n", repl.context, "computer.js", (error, value) =>
      error ? reject(error) : resolve(value),
    );
  }).finally(() => {
    evaluation = undefined;
  });
let settledAt = 0;
const settle = async () => {
  while (performance.now() < settledAt)
    await sleep(settledAt - performance.now());
};
const sky = { target: "linux" };
for (const method of [
  "click",
  "drag",
  "move",
  "press_key",
  "scroll",
  "type_text",
]) {
  sky[method] = async (input) => {
    await settle();
    await call("desktop", method, input);
    settledAt = performance.now() + 100;
  };
}
sky.get_screenshot = async () => {
  await settle();
  const result = await call("desktop", "get_screenshot", {});
  const bytes = Buffer.from(result.data, "base64");
  return [{ bytes, data_url: `data:${result.mimeType};base64,${result.data}` }];
};
const handles = new Set();
sky.drag_handle = () => {
  let state = "idle";
  const handle = {
    async start(input) {
      if (state !== "idle")
        throw new Error("drag handle can only be started once");
      await settle();
      await call("desktop", "drag_handle", { action: "start", ...input });
      state = "dragging";
      handles.add(handle);
      settledAt = performance.now() + 100;
    },
    async move_to(input) {
      if (state !== "dragging")
        throw new Error("drag handle must be started before moving");
      await settle();
      await call("desktop", "drag_handle", { action: "move_to", ...input });
      settledAt = performance.now() + 100;
    },
    async end() {
      if (state !== "dragging")
        throw new Error("drag handle must be started before ending");
      await settle();
      await call("desktop", "drag_handle", { action: "end" });
      state = "ended";
      handles.delete(handle);
      settledAt = performance.now() + 100;
    },
  };
  return handle;
};
repl.context.sky = sky;
repl.context.desktop = sky;
parentPort.on("message", (message) => {
  if (message.kind === "reply") {
    const item = pending.get(message.id);
    if (!item) return;
    pending.delete(message.id);
    if (message.error) item.reject(new Error(message.error));
    else item.resolve(message.result);
  } else if (message.kind === "event") {
    for (const pipe of pipes)
      callScope.run(context, () =>
        pipe.send({
          jsonrpc: "2.0",
          method: "onCDPEvent",
          params: message.event,
        }),
      );
  } else if (message.kind === "run") {
    callScope
      .run(message.context, () => run(message))
      .then(
        (content) => parentPort.postMessage({ kind: "result", content }),
        (error) =>
          parentPort.postMessage({
            kind: "result",
            content: output,
            error: error.stack ?? String(error),
          }),
      );
  }
});
async function run(message) {
  context = message.context;
  output = [];
  try {
    if (!initialized) {
      const agent = await setupBrowserRuntime({ environment: "training" });
      const docs = JSON.parse(
        await readFile(
          new URL("./reference/browser/docs/documents.json", import.meta.url),
          "utf8",
        ),
      );
      for (const doc of docs)
        if (doc.requiredFor?.length) await agent.documentation.get(doc.name);
      repl.context.agent = agent;
      repl.context.browser = await agent.browsers.get("cdp");
      await repl.context.browser.documentation();
      initialized = true;
    }
    if (message.endTurn) {
      for (const hook of turnHooks)
        await hook.run({
          session_id: context.session_id,
          turn_id: context.turn_id,
        });
    } else {
      const value = await evaluate(message.code);
      if (value !== undefined) nodeRepl.write(value);
      for (const hook of hooks) await hook.run();
    }
    return output;
  } finally {
    for (const handle of handles) await handle.end();
    context = undefined;
  }
}
parentPort.postMessage({ kind: "ready" });

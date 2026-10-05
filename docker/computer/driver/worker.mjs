import { create as createDomain } from "node:domain";
import { AsyncLocalStorage } from "node:async_hooks";
import { parentPort, workerData, Worker } from "node:worker_threads";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import { start as startRepl } from "node:repl";
import { readFile, writeFile, rm } from "node:fs/promises";
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
  constructor(browserId = workerData.browserId ?? "primary") {
    super();
    this.browserId = browserId;
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
      call("browser", message.method, {
        browserId: this.browserId,
        params: message.params ?? {},
      }).then(
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
const browserScope = new AsyncLocalStorage();
Object.defineProperty(globalThis, "nodeRepl", {
  configurable: true,
  get: () => browserScope.getStore() ?? nodeRepl,
});
const { handleRpc } =
  await import("./reference/browser/scripts/browser-service.mjs");
nodeRepl.rpc = (service, request) => {
  if (service !== "browser") throw new Error(`Unknown service: ${service}`);
  return handleRpc(request);
};
const { setupBrowserRuntime } =
  await import("./reference/browser/scripts/browser-client.mjs");

const services = new Map();
const closedBrowsers = new Map();

/** 每个匿名 service 属于可终止的 worker，关闭后连同模块缓存一起释放。 */
async function createBrowser() {
  // 1. 创建空 Context，service 的模块缓存由独立 worker 持有。
  const turn = context.turn_id;
  const browserId = await call("browser", "createBrowser", {});
  const pipePath = `${workerData.pipePath}-${browserId}`;
  let closed = false;
  let failure;
  const replies = new Map();
  let sequence = 0;
  await writeFile(pipePath, "", { mode: 0o600 });
  const worker = new Worker(new URL(import.meta.url), {
    workerData: { ...workerData, pipePath, browserId, service: true },
  });
  const request = (rpc) => {
    requireCall();
    if (closed) return Promise.reject(new Error("Anonymous browser is closed"));
    if (failure) return Promise.reject(failure);
    const id = ++sequence;
    return new Promise((resolve, reject) => {
      replies.set(id, { resolve, reject });
      worker.postMessage({ kind: "service", id, rpc, context });
    });
  };
  const hook = { run: () => context.turn_id === turn ? request({ method: "afterCode" }) : undefined };
  const turnHook = { run: (metadata) => metadata.turn_id === turn ? request({ method: "endTurn", params: metadata }) : undefined };
  const dispose = async () => {
    if (closed) return;
    closed = true;
    services.delete(browserId);
    hooks.splice(hooks.indexOf(hook), 1);
    turnHooks.splice(turnHooks.indexOf(turnHook), 1);
    await worker.terminate();
    await rm(pipePath);
  };
  // 2. RPC 只转发属于当前调用的 backend 请求；退出会拒绝所有等待者。
  worker.on("message", (message) => {
    if (message.kind === "serviceResult") {
      const reply = replies.get(message.id);
      replies.delete(message.id);
      if (!reply) return;
      if (message.error) reply.reject(new Error(message.error));
      else { output.push(...message.content); reply.resolve(message.result); }
    } else if (message.kind === "browser") {
      const work = message.callId !== context?.call_id
        ? Promise.reject(new Error("Anonymous operation is outside a live call"))
        : callScope.run(context, () => call("browser", message.method, message.params));
      work.then(
        (result) => worker.postMessage({ kind: "reply", id: message.id, result }),
        (error) => worker.postMessage({ kind: "reply", id: message.id, error: error.message }),
      );
    }
  });
  const fail = (error) => {
    failure = error;
    for (const reply of replies.values()) reply.reject(error);
    replies.clear();
  };
  worker.on("error", fail);
  worker.on("exit", (code) => fail(new Error(`Anonymous service exited: ${code}`)));
  hooks.push(hook);
  turnHooks.push(turnHook);
  services.set(browserId, { worker, turn, dispose });
  // 3. 用户获得原参考 Browser API；关闭和 Turn 结束都终止 service。
  try {
    const host = Object.create(nodeRepl);
    host.rpc = (name, rpc) => {
      if (name !== "browser") throw new Error(`Unknown service: ${name}`);
      return request(rpc);
    };
    return await browserScope.run(host, async () => {
      const client = await setupBrowserRuntime({ environment: "training" });
      const browser = await client.browsers.get("cdp");
      browser.close = async () => {
        if (closed) return;
        await call("browser", "closeBrowser", { browserId, params: {} });
        await dispose();
      };
      const exposed = new Proxy(browser, { get(target, key) {
        if (key === "browserId") return browserId;
        const value = Reflect.get(target, key, target);
        return typeof value === "function" ? value.bind(target) : value;
      } });
      services.get(browserId).browser = exposed;
      return exposed;
    });
  } catch (error) {
    await dispose();
    await call("browser", "closeBrowser", { browserId, params: {} });
    throw error;
  }
}

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
Object.defineProperty(repl.context, "nodeRepl", {
  value: Object.freeze({
    write: nodeRepl.write,
    emitImage: nodeRepl.emitImage,
    cwd: nodeRepl.cwd,
    tmpDir: nodeRepl.tmpDir,
  }),
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
let serviceQueue = Promise.resolve();
if (workerData.service) parentPort.once("close", () => process.exit(0));
parentPort.on("message", (message) => {
  if (message.kind === "service") {
    serviceQueue = serviceQueue.then(() => runService(message));
    return;
  }
  if (message.kind === "reply") {
    const item = pending.get(message.id);
    if (!item) return;
    pending.delete(message.id);
    if (message.error) item.reject(new Error(message.error));
    else item.resolve(message.result);
  } else if (message.kind === "event") {
    services.get(message.browserId)?.worker.postMessage(message);
    for (const pipe of pipes)
      if (pipe.browserId === (message.browserId ?? "primary"))
        callScope.run(context, () =>
          pipe.send({
            jsonrpc: "2.0",
            method: "onCDPEvent",
            params: message.event,
          }),
        );
  } else if (message.kind === "browserClosed") {
    closedBrowsers.set(message.browserId, message.reason === "idle" ? "was released after idle timeout" : "was closed");
    if (closedBrowsers.size > 64) closedBrowsers.delete(closedBrowsers.keys().next().value);
    const service = services.get(message.browserId);
    if (service) void service.dispose().catch(error => { throw error; });
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
            scriptError: error.scriptError === true,
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
      const getBrowser = agent.browsers.get.bind(agent.browsers);
      const listBrowsers = agent.browsers.list.bind(agent.browsers);
      const browsers = Object.create(agent.browsers);
      browsers.get = async (id) => {
        const service = services.get(id);
        if (service) {
          requireCall();
          if (service.turn !== context.turn_id) throw new Error("Anonymous browser belongs to another Turn");
          return service.browser;
        }
        if (/^[0-9a-f]{8}-[0-9a-f-]{27}$/.test(id))
          throw new Error(`Anonymous browser ${closedBrowsers.get(id) ?? "is closed or unknown"}; list browsers before creating a replacement`);
        return getBrowser(id);
      };
      browsers.list = async () => [...await listBrowsers(), ...[...services].map(([id]) =>
        ({ id, name: `Anonymous ${id.slice(0, 4)}`, type: "cdp" }))];
      browsers.create = createBrowser;
      // 参考 SDK 缓存方法包装，扩展放在外层，不覆盖已缓存的方法。
      repl.context.agent = new Proxy(agent, { get(target, key) {
        return key === "browsers" ? browsers : Reflect.get(target, key, target);
      } });
      repl.context.browser = await getBrowser("cdp");
      await repl.context.browser.documentation();
      initialized = true;
    }
    if (message.endTurn) {
      for (const hook of turnHooks)
        await hook.run({
          session_id: context.session_id,
          turn_id: context.turn_id,
        });
      for (const service of [...services.values()])
        if (service.turn === context.turn_id) await service.dispose();
    } else {
      let scriptError;
      try {
        const value = await evaluate(message.code);
        if (value !== undefined) nodeRepl.write(value);
      } catch (error) { scriptError = error; }
      // 脚本失败也运行参考客户端的结算 hook；hook 失败不能保留绑定。
      for (const hook of hooks) await hook.run();
      if (scriptError) { scriptError.scriptError = true; throw scriptError; }
    }
    return output;
  } finally {
    try { for (const handle of handles) await handle.end(); }
    finally { context = undefined; }
  }
}
/** 顺序处理 service RPC，输出与 backend 调用始终绑定发起它的 Computer call。 */
async function runService(message) {
  context = message.context;
  output = [];
  await callScope.run(context, async () => {
    try {
      let result;
      if (message.rpc.method === "afterCode") {
        for (const hook of hooks) await hook.run();
      } else if (message.rpc.method === "endTurn") {
        for (const hook of turnHooks) await hook.run(message.rpc.params);
      } else result = await handleRpc(message.rpc);
      parentPort.postMessage({ kind: "serviceResult", id: message.id, result, content: output });
    } catch (error) {
      parentPort.postMessage({ kind: "serviceResult", id: message.id, error: error.message });
    } finally { context = undefined; }
  });
}

parentPort.postMessage({ kind: "ready" });

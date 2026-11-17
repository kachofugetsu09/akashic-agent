import vm from "node:vm";
import readline from "node:readline";

const logs = [];
let logChars = 0;
const MAX_LOG_CHARS = 160_000;
const MAX_RESULT_JSON_CHARS = 160_000;
let callsMade = 0;

function formatError(err) {
  if (!err) return "Unknown error";
  if (typeof err === "string") return err;
  return err.stack || err.message || String(err);
}

function send(msg) {
  process.stdout.write(JSON.stringify(msg) + "\n");
}

function safeFormatArg(v) {
  if (typeof v === "string") return v;
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}

function appendLog(prefix, args) {
  if (logChars >= MAX_LOG_CHARS) return;
  const line = prefix + args.map(safeFormatArg).join(" ");
  const remain = MAX_LOG_CHARS - logChars;
  if (line.length > remain) {
    logs.push(line.slice(0, remain) + "... [logs truncated]");
    logChars = MAX_LOG_CHARS;
  } else {
    logs.push(line);
    logChars += line.length;
  }
}

function safeResultForWire(val) {
  if (val === undefined) return null;
  try {
    const serialized = JSON.stringify(val);
    if (serialized === undefined) return null;
    if (serialized.length <= MAX_RESULT_JSON_CHARS) {
      return JSON.parse(serialized);
    }
    return serialized.slice(0, MAX_RESULT_JSON_CHARS) + "... [result truncated]";
  } catch {
    return String(val).slice(0, MAX_RESULT_JSON_CHARS);
  }
}

const rl = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
});

const pending = new Map();
let initialized = false;
let rlClosed = false;

rl.on("line", (line) => {
  if (!initialized) {
    if (!line.trim()) return;
    initialized = true;
    let init;
    try {
      init = JSON.parse(line);
    } catch (err) {
      send({ op: "done", ok: false, error: formatError(err), logs, callsMade, collectedArtifacts: [] });
      rl.close();
      return;
    }
    run(init).catch((err) => {
      send({ op: "done", ok: false, error: formatError(err), logs, callsMade, collectedArtifacts: [] });
      process.exitCode = 1;
      rl.close();
    });
    return;
  }
  if (!line.trim()) return;
  let msg;
  try {
    msg = JSON.parse(line);
  } catch {
    return;
  }
  const waiter = pending.get(msg.id);
  if (!waiter) return;
  pending.delete(msg.id);
  if (msg.ok) waiter.resolve(msg.result);
  else waiter.reject(new Error(msg.error || "Tool call failed"));
});

rl.on("close", () => {
  rlClosed = true;
  for (const waiter of pending.values()) {
    waiter.reject(new Error("Host RPC stream closed"));
  }
  pending.clear();
});

async function run(init) {
  const { code, toolsMeta, maxCalls } = init;
  let nextId = 1;
  const collectedArtifacts = [];
  const seenArtifacts = new Set();
  const activeTimers = new Set();

  function callHostTool(name, args) {
    if (rlClosed) {
      return Promise.reject(new Error("Host RPC stream closed"));
    }
    if (callsMade >= maxCalls) {
      return Promise.reject(new Error(`Exceeded max_calls limit (${maxCalls})`));
    }
    callsMade += 1;
    const id = nextId++;
    return new Promise((resolve, reject) => {
      pending.set(id, {
        resolve: (res) => {
          if (res && Array.isArray(res.artifacts)) {
            for (const aid of res.artifacts) {
              if (typeof aid === "string") seenArtifacts.add(aid);
            }
          }
          resolve(res);
        },
        reject,
      });
      send({
        op: "call",
        id,
        name: String(name),
        arguments: args && typeof args === "object" && !Array.isArray(args) ? args : {},
      });
    });
  }

  const toolsTarget = Object.assign(Object.create(null), {
    list() {
      return toolsMeta.map((t) => ({
        name: t.name,
        owner: t.owner,
        description: t.description,
      }));
    },
    describe(name) {
      return toolsMeta.find((t) => t.name === name) || null;
    },
  });

  const toolsProxy = new Proxy(toolsTarget, {
    get(target, prop) {
      if (prop === "then" || typeof prop !== "string") return undefined;
      if (Object.hasOwn(target, prop)) return target[prop];
      return (args = {}) => callHostTool(prop, args);
    },
  });

  const sandboxConsole = Object.assign(Object.create(null), {
    log: (...args) => appendLog("", args),
    info: (...args) => appendLog("", args),
    warn: (...args) => appendLog("[warn] ", args),
    error: (...args) => appendLog("[error] ", args),
  });

  // 不注入宿主 Object/Array/Promise/Function 等内建构造器，由 vm 自身 realm 提供独立内建对象。
  const sandbox = Object.assign(Object.create(null), {
    tools: toolsProxy,
    console: sandboxConsole,
    sleep: (ms) => new Promise((r) => {
      const clamped = Math.max(0, Math.min(Number(ms) || 0, 30_000));
      const timer = setTimeout(() => {
        activeTimers.delete(timer);
        r();
      }, clamped);
      activeTimers.add(timer);
    }),
    collectArtifact: (id) => {
      const s = String(id || "");
      if (seenArtifacts.has(s) && !collectedArtifacts.includes(s)) {
        collectedArtifacts.push(s);
        return true;
      }
      return false;
    },
  });

  const context = vm.createContext(sandbox, {
    codeGeneration: { strings: false, wasm: false },
  });

  const wrapped = `(async () => {\n${code}\n})()`;
  try {
    const script = new vm.Script(wrapped, { filename: "codemode-script.js" });
    const result = await script.runInContext(context);
    send({
      op: "done",
      ok: true,
      result: safeResultForWire(result),
      logs,
      callsMade,
      collectedArtifacts,
    });
  } catch (err) {
    send({
      op: "done",
      ok: false,
      error: formatError(err),
      logs,
      callsMade,
      collectedArtifacts,
    });
  } finally {
    for (const timer of activeTimers) {
      clearTimeout(timer);
    }
    activeTimers.clear();
    rl.close();
  }
}

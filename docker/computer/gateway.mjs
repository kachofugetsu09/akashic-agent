import { AsyncLocalStorage } from "node:async_hooks";
import { setTimeout as sleep } from "node:timers/promises";
import { ComputerDriver } from "./driver/runtime.mjs";
import { ComputerLifecycle, duration } from "./lifecycle.mjs";
import { execFile } from "node:child_process";
import { randomBytes } from "node:crypto";
import { createServer, request as httpRequest } from "node:http";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { connect as tcpConnect } from "node:net";
import { dirname } from "node:path";
import { promisify } from "node:util";

const exec = promisify(execFile);
// 流服务的主令牌只在容器里生成；Dashboard 经网关取得后签发观看令牌、切换输入权限和传输方式。
process.env.COMPUTER_STREAM_TOKEN ||= randomBytes(24).toString("base64url");
// WebRTC 的 ICE 服务器（远程用的 TURN）由 Dashboard 写入，流服务监视这个文件、新连接即生效。
process.env.COMPUTER_RTC_FILE ||= "/tmp/computer-stream/rtc.json";
let driver;
let driverReady;
const driverCalls = new Map();
/** 取消先于唤醒或请求到达时，也必须在准入前保留结果。 */
function driverCall(callId) {
  requiredString(callId, "call_id", 256);
  for (const [id, call] of driverCalls)
    if (call.expires < Date.now()) driverCalls.delete(id);
  let call = driverCalls.get(callId);
  if (!call) {
    if (driverCalls.size >= 4096) throw new Error("Too many pending Computer calls");
    call = { controller: new AbortController(), expires: Date.now() + 300000 };
    driverCalls.set(callId, call);
  }
  return call;
}
function forgetDriver() {
  driver = undefined;
  driverReady = undefined;
  publishCursor({ point: null });
}

async function readyDriver() {
  await computer.wake();
  return driverReady;
}
const computer = new ComputerLifecycle({
  idleMs: duration("COMPUTER_IDLE_MS", 600000),
  onChange: scheduleState,
  async start(exited) {
    // 1. 图形进程就绪后才创建 driver；控制服务健康检查不会走这里。
    const deadline = Date.now() + 20000;
    let failure;
    while (Date.now() < deadline) {
      if (exited()) throw new Error("Computer runtime exited during startup");
      try {
        await health();
        failure = null;
        break;
      } catch (error) {
        failure = error;
        await sleep(100);
      }
    }
    if (failure) throw failure;
    // 重启后旧一代可能还有迟到的事件；只转发当前这一代的。
    const owner = new ComputerDriver({ anonymousIdleMs: duration("COMPUTER_ANONYMOUS_IDLE_MS", 600000) });
    driver = owner;
    owner.on("cursor", (state) => { if (driver === owner) publishCursor(state); });
    owner.on("control", () => { if (driver === owner) scheduleState(); });
    owner.on("activity", () => {
      if (driver !== owner) return;
      // 一次调用里连续的几步之间只隔几毫秒：最后一步做完再等 BUSY_TAIL_MS 才放开，人的输入权限不来回闪。
      if (!owner.agentJobs.size) { busyUntil = Date.now() + BUSY_TAIL_MS; setTimeout(scheduleState, BUSY_TAIL_MS + 10); }
      scheduleState();
    });
    owner.on("targets", () => { if (driver === owner) scheduleState(); });
    driverReady = owner.start();
    await driverReady;
  },
  async stop() {
    // 2. 已确认正常输入释放后才允许关闭浏览器与重用 profile。
    const owner = driver;
    if (owner) await owner.close();
    // 重启超时后旧的 close 可能迟到完成，不能清掉新一代 driver。
    if (driver === owner) forgetDriver();
  },
  async abandon() {
    // 3. 只由人工重启调用：正常收尾失败或超时时强制结束 driver 的子进程。
    const owner = driver;
    try {
      await owner?.abandon();
    } finally {
      if (driver === owner) forgetDriver();
    }
  },
});
const activityPath = "/data/state/activity.json";
const maxBodyBytes = 256 * 1024;
let activity = { revision: 0, noticeId: 0, active: false, action: "", updatedAt: "" };
const activeOperations = new Set();
let activityWrites = Promise.resolve();
const legacyCall = new AsyncLocalStorage();
// 订阅者 → 它最近收到的状态快照；按订阅者去重，新订阅者也能补到最新状态。
const eventClients = new Map();
let cursor = { revision: 0, point: null, error: "" };
const BUSY_TAIL_MS = 300;
let busyUntil = 0;
let stateTimer = null;
let stateDirty = false;

function cursorEvent() {
  return `data: ${JSON.stringify({ type: "cursor", ...cursor })}\n\n`;
}

/** 慢消费者直接断开，不能让操作等待显示或积压事件。 */
function broadcast(event) {
  for (const client of eventClients.keys()) {
    if (!client.write(event)) client.destroy();
  }
}

/** 坐标只驻留内存；最后的落点一直保留，直到 Agent 下一次操作。 */
function publishCursor({ point, error = "" }) {
  cursor = { revision: cursor.revision + 1, point: point ?? null, error };
  broadcast(cursorEvent());
}

async function stateEvent() {
  // 先等目录，再同步读取其余字段，避免拿等待之前的旧值。
  const targets = driver && computer.state === "ready" ? await driver.targets() : [];
  const state = {
    type: "state",
    noticeId: activity.noticeId,
    active: activity.active || Boolean(driver?.agentJobs.size),
    // Agent 正在执行单个浏览器操作；只有这时才挡住人的键鼠。
    busy: Boolean(driver?.agentJobs.size) || Date.now() < busyUntil,
    turns: [...new Set([...computer.turns, ...[...(driver?.actives.values() ?? [])]
      .map(call => JSON.stringify([call.context.session_id, call.context.turn_id]))])],
    browser: { state: computer.state, error: computer.error },
    targets,
    control: Boolean(driver?.control),
  };
  return `data: ${JSON.stringify(state)}\n\n`;
}

/** 运行时、活动、控制权和目录变化时触发；合并 50ms 内的变化，内容不变就不推送。 */
function scheduleState() {
  stateDirty = true;
  if (!stateTimer) stateTimer = setTimeout(publishState, 50);
}

async function publishState() {
  // 读取期间到达的变化只记为 dirty，读完后补发一次最新快照，不能丢。
  stateDirty = false;
  try {
    const event = await stateEvent();
    for (const [client, sent] of eventClients) {
      if (sent === event) continue;
      eventClients.set(client, event);
      if (!client.write(event)) client.destroy();
    }
  } catch (error) {
    // 目录读取可能撞上浏览器正在关闭；稍后重读一次，不推送半截状态。
    console.error("Computer state:", error.message);
    setTimeout(scheduleState, 1000);
  } finally {
    stateTimer = null;
    if (stateDirty) scheduleState();
  }
}

class InputError extends Error {}

// 人在画面上操作就持有短租约：Agent 做完当前这一步后停下，人停手 LEASE_MS 后自动交还。
// 画面同一时刻只给一个窗口，租约不区分窗口。
const LEASE_ID = "viewer";
const LEASE_MS = 5000;
let leaseTimer;
let leasing = null;

async function holdLease() {
  if (!driver || computer.state !== "ready") throw new InputError("Computer is not awake");
  const owner = driver;
  // 1. 正在交还时等它结束再重新拿；正在拿时共用同一次。
  while (owner.control?.release) await owner.control.release.catch(() => undefined);
  if (!owner.control) {
    leasing ??= (async () => {
      const releaseWorkload = await computer.acquire();
      try {
        const front = (await owner.targets()).find(target => target.front)?.id ?? "";
        await owner.takeControl(LEASE_ID, front);
        owner.control.releaseWorkload = releaseWorkload;
      } catch (error) { releaseWorkload(); throw error; }
    })().finally(() => { leasing = null; });
  }
  if (leasing) await leasing;
  if (owner.control?.id !== LEASE_ID || !owner.control.ready) throw new Error("Computer viewer lease has ended");
  // 2. 每次人工输入都从头计时。
  clearTimeout(leaseTimer);
  leaseTimer = setTimeout(() => owner.releaseControl(LEASE_ID).catch(error => {
    owner.closed = true;
    owner.rejectAll(error);
    console.error("Computer viewer lease release failed:", error.message);
  }), LEASE_MS);
  return owner.control;
}

// 地址栏、标签条等人工操作同样持有租约；交还前等它做完。
async function humanAction(task) {
  const control = await holdLease();
  const job = computer.use(task);
  control.jobs.add(job);
  try { return await job; } finally { control.jobs.delete(job); }
}

// 地址栏只打开网页和空白页，不执行 javascript: 等会在页面里运行代码的地址。
function pageUrl(value) {
  const text = requiredString(value, "url", 8192);
  let url;
  try { url = new URL(text); }
  catch (error) {
    if (error instanceof TypeError) throw new InputError("Invalid page URL");
    throw error;
  }
  if (!["http:", "https:"].includes(url.protocol) && url.href !== "about:blank")
    throw new InputError("Only http, https and about:blank pages can be opened");
  return url.href;
}

function json(response, status, value) {
  const body = Buffer.from(JSON.stringify(value));
  response.writeHead(status, {
    "content-type": "application/json",
    "content-length": String(body.length),
    "cache-control": "no-store",
  });
  response.end(body);
}

async function body(request) {
  const chunks = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > maxBodyBytes) throw new InputError("request body is too large");
    chunks.push(chunk);
  }
  const value = JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new InputError("request body must be an object");
  }
  return value;
}

function saveActivity(action, started) {
  activity = {
    revision: activity.revision + 1,
    noticeId: started ? activity.noticeId + 1 : activity.noticeId,
    active: activeOperations.size > 0,
    action,
    updatedAt: new Date().toISOString(),
  };
  // 状态事件立即排队，不等文件写完：面板和代理要尽早收回人的输入。
  scheduleState();
  const saved = JSON.stringify(activity);
  // One writer owns the shared temporary file, even when Sessions run in parallel.
  activityWrites = activityWrites.then(async () => {
    await mkdir("/data/state", { recursive: true });
    const temporary = `${activityPath}.new`;
    await writeFile(temporary, saved, { mode: 0o600 });
    await rename(temporary, activityPath);
    scheduleState();
  });
  return activityWrites;
}

async function tracked(action, task) {
  const operation = {};
  activeOperations.add(operation);
  try {
    await saveActivity(action, true);
    return await task();
  } finally {
    activeOperations.delete(operation);
    await saveActivity(action, false);
  }
}

async function daemonStatus() {
  const response = await fetch("http://127.0.0.1:19825/status", {
    headers: { "X-OpenCLI": "1" },
    signal: AbortSignal.timeout(1500),
  });
  if (!response.ok) throw new Error(`OpenCLI daemon returned ${response.status}`);
  return await response.json();
}

async function rfbReady(port) {
  await new Promise((resolve, reject) => {
    const socket = tcpConnect({ host: "127.0.0.1", port });
    let buffer = Buffer.alloc(0);
    let stage = "version";
    let settled = false;
    const timer = setTimeout(() => {
      finish(() => reject(new Error(`RFB port ${port} timed out`)));
    }, 1500);

    function finish(callback) {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      socket.end();
      callback();
    }

    function fail(message) {
      finish(() => reject(new Error(message)));
    }

    function take(size) {
      if (buffer.length < size) return null;
      const value = buffer.subarray(0, size);
      buffer = buffer.subarray(size);
      return value;
    }

    function advance() {
      while (!settled) {
        if (stage === "version") {
          const version = take(12);
          if (version === null) return;
          if (!/^RFB 003\.\d{3}\n$/.test(version.toString("ascii"))) {
            fail(`port ${port} did not speak RFB`);
            return;
          }
          socket.write("RFB 003.008\n", "ascii");
          stage = "security";
          continue;
        }
        if (stage === "security") {
          if (buffer.length < 1) return;
          const count = buffer[0];
          if (count === 0) {
            fail("RFB server did not offer a security type");
            return;
          }
          const offered = take(1 + count);
          if (offered === null) return;
          if (!offered.subarray(1).includes(1)) {
            fail("RFB server did not offer None security");
            return;
          }
          socket.write(Buffer.from([1]));
          stage = "security-result";
          continue;
        }
        if (stage === "security-result") {
          const result = take(4);
          if (result === null) return;
          if (result.readUInt32BE(0) !== 0) {
            fail("RFB server rejected the health probe");
            return;
          }
          socket.write(Buffer.from([1]));
          stage = "server-init";
          continue;
        }
        if (buffer.length < 24) return;
        const nameLength = buffer.readUInt32BE(20);
        if (nameLength > 4096) {
          fail("RFB server name is too large");
          return;
        }
        const serverInit = take(24 + nameLength);
        if (serverInit === null) return;
        if (serverInit.readUInt16BE(0) === 0 || serverInit.readUInt16BE(2) === 0) {
          fail("RFB server reported an empty display");
          return;
        }
        finish(resolve);
      }
    }

    socket.once("error", (error) => finish(() => reject(error)));
    socket.on("data", (chunk) => {
      buffer = Buffer.concat([buffer, chunk]);
      advance();
    });
  });
}

async function health() {
  const [daemon, cdp] = await Promise.all([
    daemonStatus(),
    fetch("http://127.0.0.1:9222/json/version", {
      signal: AbortSignal.timeout(1500),
    }),
    rfbReady(5999),
    fetch("http://127.0.0.1:6081/api/health", {
      signal: AbortSignal.timeout(1500),
    }).then((response) => {
      if (!response.ok) throw new Error(`Computer stream returned ${response.status}`);
    }),
  ]);
  if (!cdp.ok) throw new Error(`Chromium CDP returned ${cdp.status}`);
  if (!daemon.extensionConnected) throw new Error("OpenCLI extension is not connected");
  return {
    status: "ready",
    display: "ready",
    browser: "ready",
    opencli: "ready",
  };
}

// Resolve the visible page through its own Context, including anonymous pages.
async function pageTarget() {
  const targets = await driver.targets();
  const target = targets.find(item => item.front);
  if (!target) throw new InputError("No browser page is visible; select a tab first");
  return driver.viewTarget(target.id);
}

async function cdp(target, method, params = {}) {
  await readyDriver();
  const call = legacyCall.getStore();
  call?.signal.throwIfAborted();
  return target.backend.execute({target: {tabId: target.tabId}, method, commandParams: params}, call?.context);
}

function requiredString(value, name, maxLength) {
  if (typeof value !== "string" || value.length === 0 || value.length > maxLength) {
    throw new InputError(`${name} must be a non-empty string of at most ${maxLength} characters`);
  }
  return value;
}

function boundedString(value, name, maxLength) {
  if (typeof value !== "string" || value.length > maxLength) {
    throw new InputError(`${name} must be a string of at most ${maxLength} characters`);
  }
  return value;
}

const keyCodes = {
  Enter: ["Enter", 13, "\r"], Return: ["Enter", 13, "\r"], Tab: ["Tab", 9, "\t"],
  Escape: ["Escape", 27, ""], Esc: ["Escape", 27, ""], Backspace: ["Backspace", 8, ""],
  BackSpace: ["Backspace", 8, ""], Delete: ["Delete", 46, ""], Home: ["Home", 36, ""],
  End: ["End", 35, ""], PageUp: ["PageUp", 33, ""], Page_Up: ["PageUp", 33, ""],
  PageDown: ["PageDown", 34, ""], Page_Down: ["PageDown", 34, ""], Up: ["ArrowUp", 38, ""],
  Down: ["ArrowDown", 40, ""], Left: ["ArrowLeft", 37, ""], Right: ["ArrowRight", 39, ""],
  ArrowUp: ["ArrowUp", 38, ""], ArrowDown: ["ArrowDown", 40, ""], ArrowLeft: ["ArrowLeft", 37, ""],
  ArrowRight: ["ArrowRight", 39, ""], space: [" ", 32, " "], Space: [" ", 32, " "],
};
const modifierBits = { alt: 1, ctrl: 2, control: 2, meta: 4, super: 4, cmd: 4, shift: 8 };

// "ctrl+shift+a" 这类组合键换成 CDP 键盘事件；未知键名直接报错，不猜。
function keySteps(combo) {
  const parts = combo.split("+").filter(Boolean);
  const name = parts.pop() ?? "";
  let modifiers = 0;
  for (const part of parts) {
    const bit = modifierBits[part.toLowerCase()];
    if (!bit) throw new InputError(`unsupported modifier: ${part}`);
    modifiers |= bit;
  }
  let key, keyCode, text, code;
  if (keyCodes[name]) [key, keyCode, text] = keyCodes[name];
  else if ([...name].length === 1) {
    key = name;
    keyCode = name.toUpperCase().charCodeAt(0);
    text = name;
    code = /[a-z]/i.test(name) ? `Key${name.toUpperCase()}` : /\d/.test(name) ? `Digit${name}` : undefined;
  } else throw new InputError(`unsupported key: ${name}`);
  if (modifiers & 7) text = "";
  const base = { key, code: code ?? key, modifiers, windowsVirtualKeyCode: keyCode };
  return [
    ["Input.dispatchKeyEvent", { ...base, type: text ? "keyDown" : "rawKeyDown", text }],
    ["Input.dispatchKeyEvent", { ...base, type: "keyUp" }],
  ];
}

function point(value, xName, yName) {
  const x = value[xName], y = value[yName];
  if (!Number.isInteger(x) || x < 0 || x > 16383 || !Number.isInteger(y) || y < 0 || y > 16383)
    throw new InputError(`${xName} and ${yName} must be non-negative integers`);
  return { x, y };
}

function pressSteps({ x, y }, button, clickCount) {
  return [
    ["Input.dispatchMouseEvent", { type: "mousePressed", x, y, button, clickCount }],
    ["Input.dispatchMouseEvent", { type: "mouseReleased", x, y, button, clickCount }],
  ];
}

// 坐标动作换成发给最前面页面的 CDP 输入；kiosk 窗口铺满屏幕，页面坐标就是屏幕坐标。
function inputSteps(value) {
  const action = value.action;
  if (["click", "double_click", "move"].includes(action)) {
    const at = point(value, "x", "y");
    const button = value.mouse_button ?? "left";
    if (!["left", "middle", "right"].includes(button)) throw new InputError("mouse_button is invalid");
    const steps = [["Input.dispatchMouseEvent", { type: "mouseMoved", ...at }]];
    if (action === "click") steps.push(...pressSteps(at, button, 1));
    if (action === "double_click") steps.push(...pressSteps(at, button, 1), ...pressSteps(at, button, 2));
    return steps;
  }
  if (action === "drag") {
    const from = point(value, "x", "y"), to = point(value, "to_x", "to_y");
    const steps = [["Input.dispatchMouseEvent", { type: "mouseMoved", ...from }],
      ["Input.dispatchMouseEvent", { type: "mousePressed", ...from, button: "left", clickCount: 1 }]];
    for (let index = 1; index <= 8; index++)
      steps.push(["Input.dispatchMouseEvent", { type: "mouseMoved", button: "left", buttons: 1,
        x: from.x + ((to.x - from.x) * index) / 8, y: from.y + ((to.y - from.y) * index) / 8 }]);
    steps.push(["Input.dispatchMouseEvent", { type: "mouseReleased", ...to, button: "left", clickCount: 1 }]);
    return steps;
  }
  if (action === "type") return [["Input.insertText", { text: boundedString(value.text, "text", 16384) }]];
  if (action === "key") return keySteps(requiredString(value.key, "key", 80));
  if (action === "scroll") {
    if (!Number.isInteger(value.amount) || Math.abs(value.amount) > 100)
      throw new InputError("amount must be an integer between -100 and 100");
    return [["Input.dispatchMouseEvent", { type: "mouseWheel", x: 640, y: 400, deltaX: 0, deltaY: value.amount * 100 }]];
  }
  throw new InputError("unsupported input action");
}

// 页面级坐标操作：作为一次 legacy 调用执行，经过同一个人工接管与输入释放边界。
async function pageOperation(task, signal) {
  await readyDriver();
  return driver.perform(
    (inner, context) => legacyCall.run({ signal: inner, context }, async () => {
      const target = await pageTarget();
      try { return await task(target); }
      finally { await target.backend.releaseInputs(context.session_id); }
    }),
    signal,
  );
}

async function runInput(value, signal) {
  if (value.action === "wait") {
    if (!Number.isInteger(value.ms) || value.ms < 0 || value.ms > 30000)
      throw new InputError("ms must be 0..30000");
    await tracked("wait", () => sleep(value.ms, undefined, { signal }));
    return { ok: true };
  }
  const steps = inputSteps(value);
  await tracked(String(value.action), () => pageOperation(async (target) => {
    for (const [method, params] of steps) await cdp(target, method, params);
  }, signal));
  return { ok: true };
}

async function screenshot(response, quiet) {
  const capture = () => pageOperation((target) =>
    cdp(target, "Page.captureScreenshot", { format: "png", captureBeyondViewport: false }));
  const result = quiet ? await capture() : await tracked("screenshot", capture);
  const bytes = Buffer.from(result.data, "base64");
  response.writeHead(200, {
    "content-type": "image/png",
    "content-length": String(bytes.length),
    "cache-control": "no-store",
  });
  response.end(bytes);
}

async function proxyOpenCli(request, response) {
  const url = new URL(request.url ?? "/", "http://opencli.local");
  if (request.headers["x-opencli"] !== "1" || request.headers.upgrade) {
    json(response, 403, { error: "Only OpenCLI client requests may wake Computer" });
    return;
  }
  if (url.pathname === "/shutdown") {
    json(response, 403, {
      error: "the Computer plugin owns the OpenCLI daemon lifecycle",
    });
    return;
  }
  const controller = new AbortController();
  response.once("close", () => {
    if (!response.writableFinished) controller.abort(new Error("OpenCLI caller disconnected"));
  });
  try {
    await computer.use(async () => {
      await readyDriver();
      return driver.agentOperation(() => new Promise((resolve, reject) => {
          if (response.destroyed) {
            resolve();
            return;
          }
          response.once("finish", resolve);
          response.once("close", resolve);
          const upstream = httpRequest(
            {
              hostname: "127.0.0.1",
              port: 19825,
              method: request.method,
              path: request.url,
              headers: { ...request.headers, host: "127.0.0.1:19825" },
            },
            (upstreamResponse) => {
              response.writeHead(
                upstreamResponse.statusCode ?? 502,
                upstreamResponse.headers,
              );
              upstreamResponse.pipe(response);
            },
          );
          upstream.setTimeout(125_000, () =>
            upstream.destroy(new Error("OpenCLI daemon timed out")),
          );
          upstream.on("error", (error) => {
            if (!response.headersSent)
              json(response, 502, { error: error.message });
            else response.destroy(error);
            reject(error);
          });
          response.once("close", () => upstream.destroy());
          request.pipe(upstream);
        }), controller.signal);
    });
  } catch (error) {
    if (!response.headersSent && !response.destroyed)
      json(response, 502, { error: error.message });
    else console.error("OpenCLI proxy:", error.message);
  }
}

try {
  activity = JSON.parse(await readFile(activityPath, "utf8"));
  if (!Number.isInteger(activity.revision) || !Number.isInteger(activity.noticeId)) {
    throw new Error("stored activity has an old shape");
  }
} catch {
  // A missing activity file is normal on the first boot.
}
if (activity.active) await saveActivity(activity.action, false);

const server = createServer(async (request, response) => {
  let release;
  let call;
  let callId;
  if (stopping) {
    json(response, 503, { error: "Computer is stopping" });
    return;
  }
  try {
    const url = new URL(request.url ?? "/", "http://computer.local");
    const driverPayload =
      request.method === "POST" && url.pathname === "/driver/run"
        ? await body(request)
        : null;
    if (driverPayload) {
      callId = driverPayload.context?.call_id;
      if (driverCalls.get(callId)?.done)
        throw new InputError("Computer call_id is already running");
      call = driverCall(callId);
      call.expires = Infinity;
      call.done = new Promise((resolve) => { call.finish = resolve; });
      response.once("close", () => {
        if (!response.writableFinished)
          call.controller.abort(new Error("Computer caller disconnected"));
      });
      call.controller.signal.throwIfAborted();
    }
    // 操作持有占用；活动轮询、握手与取消均不得唤醒主浏览器。
    const operation =
      [
        "/screenshot",
        "/input",
        "/wake",
      ].includes(url.pathname) ||
      (driverPayload && (!driverPayload.endTurn || driver));
    if (operation) release = await computer.acquire();
    if (request.method === "POST" && url.pathname === "/driver/run") {
      const payload = driverPayload;
      if (payload.endTurn && !driver) {
        json(response, 200, {
          content: [],
          call_id: payload.context?.call_id,
        });
        return;
      }
      await readyDriver();
      call.controller.signal.throwIfAborted();
      const turn = JSON.stringify([
        payload.context?.session_id,
        payload.context?.turn_id,
      ]);
      try {
        const result = await tracked("computer", () =>
          driver.run(
            {
              context: payload.context,
              code: payload.code,
              endTurn: payload.endTurn,
              timeoutMs: payload.timeoutMs,
            },
            call.controller.signal,
          ),
        );
        if (!payload.endTurn) computer.turns.add(turn);
        json(response, 200, result);
      } finally {
        if (payload.endTurn) computer.turns.delete(turn);
      }
    } else if (request.method === "POST" && url.pathname === "/driver/cancel") {
      const payload = await body(request);
      const cancelled = driverCall(payload.call_id);
      cancelled.controller.abort(new Error("Computer call cancelled before admission; earlier effects may remain"));
      if (driver?.findActive(payload.call_id))
        await driver.cancel(payload.call_id);
      if (cancelled.done) await cancelled.done;
      json(response, 200, { released: true });
    } else if (request.method === "POST" && url.pathname === "/driver/reset") {
      const payload = await body(request);
      requiredString(payload.session_id, "session_id", 256);
      if (driver) await driver.reset(payload.session_id);
      for (const turn of computer.turns)
        if (JSON.parse(turn)[0] === payload.session_id) computer.turns.delete(turn);
      json(response, 200, { reset: true });
    } else if (
      request.method === "GET" &&
      url.pathname === "/driver/status"
    ) {
      json(response, 200, {
        version: 3,
        source: true,
        ready: computer.state !== "failed",
        browser: computer.status(),
      });
    } else if (request.method === "GET" && url.pathname === "/health") {
      await daemonStatus();
      if (computer.state === "failed") throw new Error(computer.error);
      json(response, 200, {
        status: "ready",
        control: "ready",
        browser: computer.state,
      });
    } else if (request.method === "GET" && url.pathname === "/events") {
      // 只读订阅不唤醒，也不写 activity 或 profile；先发完整快照，之后只发变化。
      if (eventClients.size >= 32) {
        json(response, 503, { error: "Too many Computer viewers" });
        return;
      }
      const snapshot = await stateEvent();
      response.writeHead(200, {"content-type": "text/event-stream", "cache-control": "no-store"});
      response.write(snapshot + cursorEvent());
      eventClients.set(response, snapshot);
      response.once("close", () => eventClients.delete(response));
      // 快照读取期间可能有变化已经广播过，补一次核对。
      scheduleState();
    } else if (request.method === "GET" && url.pathname === "/targets") {
      json(response, 200, { targets: driver && computer.state === "ready" ? await driver.targets() : [],
        control: Boolean(driver?.control) });
    } else if (request.method === "POST" && url.pathname === "/view/show") {
      const payload = await body(request);
      if (!driver || computer.state !== "ready") throw new InputError("Computer is not awake");
      await driver.show(requiredString(payload.target, "target", 128));
      json(response, 200, { shown: true });
    } else if (request.method === "POST" && ["/view/navigate", "/view/open", "/view/close"].includes(url.pathname)) {
      const payload = await body(request);
      if (!driver || computer.state !== "ready") throw new InputError("Computer is not awake");
      if (url.pathname === "/view/open") {
        const address = payload.url === undefined ? "about:blank" : pageUrl(payload.url);
        json(response, 200, { target: await humanAction(() => driver.openPage(address)) });
        return;
      }
      const target = requiredString(payload.target, "target", 128);
      if (url.pathname === "/view/close") await humanAction(() => driver.closePage(target));
      else if (!["go", "back", "forward", "reload"].includes(payload.action))
        throw new InputError("action must be go, back, forward or reload");
      else {
        const address = payload.action === "go" ? pageUrl(payload.url) : "";
        await humanAction(() => driver.navigate(target, payload.action, address));
      }
      json(response, 200, { done: true });
    } else if (request.method === "POST" && url.pathname === "/view/touch") {
      // 观看者直接操作页面不经过网关：续期防止休眠；input 表示人刚按下键鼠，持有租约让 Agent 停下。
      const payload = await body(request);
      if (computer.state === "ready") computer.touch();
      // 拿不到租约（浏览器正在休眠或重启）不影响人的输入，也不能断开画面。
      if (payload.input === true && computer.state === "ready")
        await holdLease().catch(error => console.error("Computer viewer lease:", error.message));
      json(response, 200, { touched: computer.state === "ready" });
    } else if (request.method === "GET" && url.pathname === "/activity") {
      json(response, 200, { ...activity, browser: computer.status(), control: driver?.control ? { active: true } : { active: false } });
    } else if (request.method === "POST" && url.pathname === "/restart") {
      await computer.restart();
      json(response, 200, computer.status());
    } else if (request.method === "POST" && url.pathname === "/wake") {
      json(response, 200, computer.status());
    } else if (request.method === "GET" && url.pathname === "/stream/token") {
      json(response, 200, { token: process.env.COMPUTER_STREAM_TOKEN });
    } else if (request.method === "PUT" && url.pathname === "/stream/rtc") {
      const { iceServers } = await body(request);
      if (!Array.isArray(iceServers) || !iceServers.every(item => item && typeof item === "object" && item.urls)) {
        throw new InputError("iceServers must be a list of servers with urls");
      }
      // 流服务只信任属主自己、不可被他人写的文件；整文件换名，监视方不会读到半截。
      const target = process.env.COMPUTER_RTC_FILE;
      await mkdir(dirname(target), { recursive: true, mode: 0o700 });
      // 每次写自己的临时文件，并发请求不会互相换走对方写到一半的内容。
      const next = `${target}.${randomBytes(6).toString("hex")}`;
      await writeFile(next, JSON.stringify({ iceServers }), { mode: 0o600 });
      await rename(next, target);
      json(response, 200, { ok: true });
    } else if (request.method === "GET" && url.pathname === "/screenshot") {
      await screenshot(response, url.searchParams.get("quiet") === "1");
    } else if (request.method === "POST" && url.pathname === "/input") {
      const payload = await body(request);
      const controller = new AbortController();
      response.once("close", () => {
        if (!response.writableFinished)
          controller.abort(new Error("Computer caller disconnected"));
      });
      json(response, 200, await runInput(payload, controller.signal));
    } else {
      json(response, 404, { error: "not found" });
    }
  } catch (error) {
    if (driver?.closed && computer.state === "ready") {
      computer.fail(error);
    }
    const status = error instanceof InputError || error instanceof SyntaxError ? 400 : 500;
    json(response, status, {
      error: error instanceof Error ? error.message : String(error),
    });
  } finally {
    release?.();
    if (call?.finish) {
      driverCalls.delete(callId);
      call.finish();
    }
  }
}).listen(8080, "0.0.0.0");

const openCliServer = createServer(proxyOpenCli).listen(19826, "0.0.0.0");
let stopping = false;
let refreshTimer;
/** 一次刷新持有完整占用；失败回执与下一次计划均保留在控制服务。 */
async function refreshIdentity() {
  try {
    const { stdout } = await computer.use(async () => {
      await readyDriver();
      // Keep existing pages, Agent pages and popups. Any human takeover during
      // refresh makes ownership uncertain, so that refresh does not close pages.
      const pages = async () => (await driver.browser.browser.send("Target.getTargets")).targetInfos
        .filter(target => target.type === "page" && !driver.anonymous.contexts.has(target.browserContextId));
      let before = null;
      let controlVersion;
      try {
        return await driver.agentOperation(async () => {
          controlVersion = driver.controlVersion;
          before = new Set((await pages()).map(page => page.targetId));
          return exec(
          "opencli",
          [
            "auth",
            "refresh",
            "--all",
            "--site",
            process.env.OPENCLI_AUTH_REFRESH_SITES ?? "",
            "--concurrency",
            "2",
            "--timeout",
            "45",
            "--format",
            "json",
          ],
          { timeout: 120000 },
        ); });
      } finally {
        if (before && driver.controlVersion === controlVersion && !driver.control) {
          await driver.agentOperation(async () => {
            await driver.browser.listTabs();
            const agent = new Set([...driver.browser.tabs.values()]
              .filter(tab => tab.created || tab.claimed).map(tab => tab.targetId));
            const opened = (await pages()).filter(page => !before.has(page.targetId)
              && !agent.has(page.targetId) && !page.openerId);
            let closed = 0;
            for (const { targetId } of opened) {
              // A live Agent may still be assigning ownership to a new target.
              if (driver.actives.size || driver.agentJobs.size > 1 || driver.controlVersion !== controlVersion) break;
              const result = await driver.browser.browser.send("Target.closeTarget", { targetId });
              if (!result.success) throw new Error(`OpenCLI refresh could not close page ${targetId}`);
              closed++;
            }
            console.log(`OpenCLI refresh closed ${closed} pages; kept ${before.size}`);
          });
        }
      }
    });
    const sites = JSON.parse(stdout);
    if (!Array.isArray(sites) || sites.length === 0) throw new Error("OpenCLI returned no refresh results");
    const needsLogin = sites.filter(site => site.status === "not_logged_in");
    if (needsLogin.length) {
      console.warn(`OpenCLI login required: ${needsLogin.map(site => site.site).join(", ")}`);
    }
    const failed = sites.filter(site => !["refreshed", "touched", "not_logged_in"].includes(site.status));
    if (failed.length) throw new Error(`OpenCLI refresh incomplete: ${failed.map(site => `${site.site}:${site.status}`).join(", ")}`);
    // 未登录需要人工处理；按正常周期再检查，但不能写成整批刷新成功。
    if (!needsLogin.length) {
      await writeFile("/data/state/auth-refresh.ok", new Date().toISOString(), {
        mode: 0o600,
      });
    }
    refreshTimer = setTimeout(refreshIdentity, 43200000);
  } catch (error) {
    console.error("OpenCLI login refresh failed:", error.message);
    refreshTimer = setTimeout(refreshIdentity, 900000);
  }
}
refreshTimer = setTimeout(refreshIdentity, 900000);
async function stop() {
  if (stopping) return;
  stopping = true;
  clearTimeout(refreshTimer);
  server.close();
  for (const client of eventClients.keys()) client.end();
  openCliServer.close();
  const timer = setTimeout(() => {
    console.error("Computer shutdown timed out");
    process.exit(1);
  }, 30000);
  try {
    await computer.close();
    console.error("Computer driver input release confirmed");
    clearTimeout(timer);
    process.exit(0);
  } catch (error) {
    console.error(error);
    clearTimeout(timer);
    process.exit(1);
  }
}
process.on("SIGTERM", stop);
process.on("SIGINT", stop);

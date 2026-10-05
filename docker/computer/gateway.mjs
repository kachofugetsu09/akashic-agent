import { AsyncLocalStorage } from "node:async_hooks";
import { setTimeout as sleep } from "node:timers/promises";
import { ComputerDriver } from "./driver/runtime.mjs";
import { ComputerLifecycle, duration } from "./lifecycle.mjs";
import { execFile } from "node:child_process";
import { createServer, request as httpRequest } from "node:http";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { connect as tcpConnect } from "node:net";
import { promisify } from "node:util";

const exec = promisify(execFile);
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
async function readyDriver() {
  await computer.wake();
  return driverReady;
}
const computer = new ComputerLifecycle({
  idleMs: duration("COMPUTER_IDLE_MS", 600000),
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
    driver = new ComputerDriver({ anonymousIdleMs: duration("COMPUTER_ANONYMOUS_IDLE_MS", 600000) });
    driver.on("cursor", publishCursor);
    driverReady = driver.start();
    await driverReady;
  },
  async stop() {
    // 2. 已确认正常输入释放后才允许关闭浏览器与重用 profile。
    if (driver) await driver.close();
    driver = undefined;
    driverReady = undefined;
    activeTargetId = "";
    browserSnapshots.clear();
    publishCursor({ point: null });
  },
});
const activityPath = "/data/state/activity.json";
const maxBodyBytes = 256 * 1024;
let activity = { revision: 0, noticeId: 0, active: false, action: "", updatedAt: "" };
let activeTargetId = "";
const legacyCall = new AsyncLocalStorage();
const browserSnapshots = new Map();
const cursorClients = new Set();
let cursor = { revision: 0, point: null, error: "" };

function cursorEvent() {
  const ttlMs = cursor.point ? Math.max(0, cursor.point.expiresAt - Date.now()) : 0;
  return `data: ${JSON.stringify({ ...cursor, ttlMs })}\n\n`;
}

/** 坐标只驻留内存；慢消费者断开，不能让操作等待显示或积压帧。 */
function publishCursor({ point, error = "" }) {
  cursor = { revision: cursor.revision + 1,
    point: point ? { ...point, expiresAt: Date.now() + 5000 } : null, error };
  const event = cursorEvent();
  for (const client of cursorClients) {
    if (!client.write(event)) client.destroy();
  }
}

class InputError extends Error {}

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

async function saveActivity(action, active) {
  activity = {
    revision: activity.revision + 1,
    noticeId: active ? activity.noticeId + 1 : activity.noticeId,
    active,
    action,
    updatedAt: new Date().toISOString(),
  };
  await mkdir("/data/state", { recursive: true });
  const temporary = `${activityPath}.new`;
  await writeFile(temporary, JSON.stringify(activity), { mode: 0o600 });
  await rename(temporary, activityPath);
}

async function tracked(action, task) {
  await saveActivity(action, true);
  try {
    return await task();
  } finally {
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

async function tcpReady(port) {
  await new Promise((resolve, reject) => {
    const socket = tcpConnect({ host: "127.0.0.1", port });
    const timer = setTimeout(() => {
      socket.destroy();
      reject(new Error(`port ${port} timed out`));
    }, 1500);
    const finish = (callback) => {
      clearTimeout(timer);
      socket.destroy();
      callback();
    };
    socket.once("error", (error) => finish(() => reject(error)));
    socket.once("connect", () => finish(resolve));
  });
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
    tcpReady(6080),
    fetch("http://127.0.0.1:6081/api/health", {
      signal: AbortSignal.timeout(1500),
    }).then((response) => {
      if (!response.ok) throw new Error(`Computer stream returned ${response.status}`);
    }),
    exec("pgrep", ["-x", "xfce4-session"], { timeout: 1500 }),
  ]);
  if (!cdp.ok) throw new Error(`Chromium CDP returned ${cdp.status}`);
  if (!daemon.extensionConnected) throw new Error("OpenCLI extension is not connected");
  return {
    status: "ready",
    desktop: "ready",
    display: "ready",
    browser: "ready",
    opencli: "ready",
  };
}

async function pageTargets() {
  const response = await fetch("http://127.0.0.1:9222/json/list", {
    signal: AbortSignal.timeout(3000),
  });
  if (!response.ok) throw new Error(`Chromium targets returned ${response.status}`);
  const targets = (await response.json()).filter(
    (target) => target.type === "page" && typeof target.webSocketDebuggerUrl === "string",
  );
  if (targets.length === 0) throw new Error("Chromium has no page target");
  return targets;
}

async function pageTarget(requestedId, select = false) {
  const targets = await pageTargets();
  const targetId = requestedId || activeTargetId;
  const target = targets.find((item) => item.id === targetId);
  if (requestedId && !target) throw new InputError(`unknown browser target: ${requestedId}`);
  const selected = target ?? targets[0];
  if (select) activeTargetId = selected.id;
  return selected;
}

async function cdp(target, method, params = {}) {
  await readyDriver();
  const call = legacyCall.getStore();
  call?.signal.throwIfAborted();
  let tab = driver.browser.tabs.get(target.id);
  if (!tab) { await driver.browser.listTabs(); tab = driver.browser.tabs.get(target.id); }
  if (!tab) throw new InputError("Browser target closed; observe again");
  call?.signal.throwIfAborted();
  return driver.browser.execute({target: {tabId: tab.id}, method, commandParams: params}, call?.context);
}

async function evaluate(target, expression) {
  const result = await cdp(target, "Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
    userGesture: true,
  });
  if (result.exceptionDetails) {
    throw new Error(result.exceptionDetails.text || "browser evaluation failed");
  }
  return result.result?.value;
}

const interactiveRoles = new Set([
  "button", "checkbox", "combobox", "link", "listbox", "menuitem",
  "menuitemcheckbox", "menuitemradio", "option", "radio", "searchbox",
  "slider", "spinbutton", "switch", "tab", "textbox", "treeitem",
]);

function axValue(value) {
  return value && typeof value.value !== "undefined" ? String(value.value) : "";
}

async function pageLoaderId(target) {
  const tree = await cdp(target, "Page.getFrameTree");
  const loaderId = tree.frameTree?.frame?.loaderId;
  if (typeof loaderId !== "string" || loaderId.length === 0) {
    throw new Error("browser document identity is unavailable");
  }
  return loaderId;
}

async function browserSnapshot(target) {
  const [tree, ax] = await Promise.all([
    cdp(target, "Page.getFrameTree"),
    cdp(target, "Accessibility.getFullAXTree", { depth: 32 }),
  ]);
  const loaderId = tree.frameTree?.frame?.loaderId;
  if (typeof loaderId !== "string" || !Array.isArray(ax.nodes)) {
    throw new Error("browser snapshot is invalid");
  }
  const refs = new Map();
  const items = [];
  for (const node of ax.nodes) {
    const role = axValue(node.role);
    if (node.ignored || !interactiveRoles.has(role) || !Number.isInteger(node.backendDOMNodeId)) {
      continue;
    }
    const ref = `e${items.length + 1}`;
    refs.set(ref, { backendNodeId: node.backendDOMNodeId, role });
    items.push({
      ref,
      role,
      name: axValue(node.name).trim().replace(/\s+/g, " ").slice(0, 240),
      value: axValue(node.value).slice(0, 500),
      disabled: node.properties?.some(
        (property) => property.name === "disabled" && property.value?.value === true,
      ) ?? false,
    });
    if (items.length === 200) break;
  }
  const snapshotId = crypto.randomUUID();
  browserSnapshots.set(snapshotId, { targetId: target.id, loaderId, refs });
  while (browserSnapshots.size > 32) {
    browserSnapshots.delete(browserSnapshots.keys().next().value);
  }
  return { snapshot_id: snapshotId, target_id: target.id, url: target.url, title: target.title, items };
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

async function browserObserve(value) {
  const observe = value.observe;
  if (observe === "tab_list") {
    const targets = await pageTargets();
    const active = targets.some((item) => item.id === activeTargetId)
      ? activeTargetId
      : targets[0].id;
    return {
      active_target_id: active,
      tabs: targets.map((item) => ({ target_id: item.id, title: item.title, url: item.url })),
    };
  }
  const target = await pageTarget(value.target_id);
  if (observe === "snapshot") {
    return await browserSnapshot(target);
  }
  if (observe === "get_content") {
    return {
      url: target.url,
      title: target.title,
      content: await evaluate(target, "document.body?.innerText.slice(0, 200000) ?? ''"),
    };
  }
  if (observe === "get_url") return { target_id: target.id, url: target.url };
  if (observe === "get_title") return { target_id: target.id, title: target.title };
  if (observe === "screenshot") {
    const result = await cdp(target, "Page.captureScreenshot", { format: "png", fromSurface: true });
    return { target_id: target.id, mimeType: "image/png", data: result.data };
  }
  throw new InputError("unsupported browser observation");
}

async function refTarget(value) {
  const snapshotId = requiredString(value.snapshot_id, "snapshot_id", 64);
  const checkedRef = requiredString(value.ref, "ref", 16);
  const snapshot = browserSnapshots.get(snapshotId);
  if (!snapshot) throw new InputError(`stale browser snapshot: ${snapshotId}`);
  if (value.target_id && value.target_id !== snapshot.targetId) {
    throw new InputError("target_id does not match snapshot_id");
  }
  const refNode = snapshot.refs.get(checkedRef);
  if (!refNode) throw new InputError(`unknown browser ref: ${checkedRef}`);
  const target = await pageTarget(snapshot.targetId);
  if (await pageLoaderId(target) !== snapshot.loaderId) {
    browserSnapshots.delete(snapshotId);
    throw new InputError(`stale browser snapshot: ${snapshotId}`);
  }
  return { target, ...refNode, ref: checkedRef };
}

async function focusNode(target, backendNodeId) {
  try {
    await cdp(target, "DOM.scrollIntoViewIfNeeded", { backendNodeId });
    await cdp(target, "DOM.focus", { backendNodeId });
  } catch (error) {
    throw new InputError("browser ref is no longer attached", { cause: error });
  }
}

async function clickNode(target, backendNodeId) {
  let model;
  try {
    await cdp(target, "DOM.scrollIntoViewIfNeeded", { backendNodeId });
    model = await cdp(target, "DOM.getBoxModel", { backendNodeId });
  } catch (error) {
    throw new InputError("browser ref is no longer visible", { cause: error });
  }
  const quad = model.model?.border;
  if (!Array.isArray(quad) || quad.length !== 8) throw new InputError("browser ref has no click area");
  const x = (quad[0] + quad[2] + quad[4] + quad[6]) / 4;
  const y = (quad[1] + quad[3] + quad[5] + quad[7]) / 4;
  await cdp(target, "Input.dispatchMouseEvent", { type: "mouseMoved", x, y });
  await cdp(target, "Input.dispatchMouseEvent", { type: "mousePressed", x, y, button: "left", clickCount: 1 });
  await cdp(target, "Input.dispatchMouseEvent", { type: "mouseReleased", x, y, button: "left", clickCount: 1 });
}

async function openTarget(url) {
  const call = legacyCall.getStore();
  call.signal.throwIfAborted();
  const tab = await driver.browser.call("createTab", {}, call.context);
  const target = { id: driver.browser.tab(tab.id).targetId, url };
  await cdp(target, "Page.navigate", {url});
  await driver.browser.call("markTab", {tabId: tab.id, status: "deliverable"}, call.context);
  activeTargetId = target.id;
  return target;
}

async function browserAction(value) {
  const action = value.action;
  if (action === "wait") {
    const timeout = value.timeout ?? 1000;
    if (!Number.isInteger(timeout) || timeout < 0 || timeout > 30_000) {
      throw new InputError("timeout must be an integer between 0 and 30000");
    }
    await tracked("browser wait", () => sleep(timeout, undefined, {signal: legacyCall.getStore().signal}));
    return { ok: true, waited_ms: timeout };
  }
  if (action === "tab_new") {
    const url = checkedUrl(value.url ?? "about:blank");
    const target = await tracked("browser tab new", () => openTarget(url));
    return { ok: true, target_id: target.id, url: target.url };
  }
  const target = ["click", "fill", "type"].includes(action)
    ? null
    : await pageTarget(value.target_id);
  if (action === "tab_select") {
    requiredString(value.target_id, "target_id", 128);
    await cdp(target, "Page.bringToFront");
    activeTargetId = target.id;
    return { ok: true, target_id: target.id };
  }
  if (action === "tab_close") {
    requiredString(value.target_id, "target_id", 128);
    const call = legacyCall.getStore();
    await driver.browser.listTabs();
    const tab = driver.browser.tabs.get(target.id);
    await driver.browser.call("claimUserTab", {tabId: tab.id}, call.context);
    await cdp(target, "Page.close");
    if (activeTargetId === target.id) activeTargetId = "";
    return { ok: true, target_id: target.id };
  }
  return await tracked(`browser ${action}`, async () => {
    if (action === "navigate") {
      const url = checkedUrl(value.url);
      await cdp(target, "Page.navigate", { url });
      return { ok: true, target_id: target.id, url };
    }
    if (action === "click") {
      const selected = await refTarget(value);
      await clickNode(selected.target, selected.backendNodeId);
      return { ok: true, ref: selected.ref, snapshot_id: value.snapshot_id };
    }
    if (action === "fill" || action === "type") {
      const text = boundedString(value.text, "text", 16_384);
      const selected = await refTarget(value);
      if (!["combobox", "searchbox", "spinbutton", "textbox"].includes(selected.role)) {
        throw new InputError(`browser ref is not editable: ${selected.ref}`);
      }
      await focusNode(selected.target, selected.backendNodeId);
      if (action === "fill") {
        await cdp(selected.target, "Input.dispatchKeyEvent", {
          type: "rawKeyDown",
          key: "a",
          code: "KeyA",
          modifiers: 2,
          windowsVirtualKeyCode: 65,
          commands: ["SelectAll"],
        });
        await cdp(selected.target, "Input.dispatchKeyEvent", {
          type: "keyUp", key: "a", code: "KeyA", modifiers: 2, windowsVirtualKeyCode: 65,
        });
        await cdp(selected.target, "Input.dispatchKeyEvent", {
          type: "rawKeyDown", key: "Backspace", code: "Backspace", windowsVirtualKeyCode: 8,
        });
        await cdp(selected.target, "Input.dispatchKeyEvent", {
          type: "keyUp", key: "Backspace", code: "Backspace", windowsVirtualKeyCode: 8,
        });
      }
      await cdp(selected.target, "Input.insertText", { text });
      return { ok: true, ref: selected.ref, snapshot_id: value.snapshot_id, action };
    }
    if (action === "press") {
      const key = requiredString(value.key, "key", 80);
      await cdp(target, "Input.dispatchKeyEvent", { type: "keyDown", key });
      await cdp(target, "Input.dispatchKeyEvent", { type: "keyUp", key });
      return { ok: true, key };
    }
    if (action === "scroll") {
      const direction = value.direction ?? "down";
      const amount = value.amount ?? 500;
      if (!["up", "down", "left", "right"].includes(direction) ||
          !Number.isInteger(amount) || amount < 1 || amount > 5000) {
        throw new InputError("scroll direction or amount is invalid");
      }
      const x = direction === "left" ? -amount : direction === "right" ? amount : 0;
      const y = direction === "up" ? -amount : direction === "down" ? amount : 0;
      await cdp(target, "Input.dispatchMouseEvent", {
        type: "mouseWheel", x: 640, y: 400, deltaX: x, deltaY: y,
      });
      return { ok: true, direction, amount };
    }
    if (action === "reload") {
      await cdp(target, "Page.reload", {});
      return { ok: true, target_id: target.id };
    }
    if (action === "go_back" || action === "go_forward") {
      await evaluate(target, action === "go_back" ? "history.back(); true" : "history.forward(); true");
      return { ok: true, target_id: target.id };
    }
    throw new InputError("unsupported browser action");
  });
}

function checkedUrl(value) {
  const url = requiredString(value, "url", 8192);
  if (url === "about:blank") return url;
  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    throw new InputError("url must be an absolute HTTP or HTTPS URL");
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new InputError("url must use HTTP or HTTPS");
  }
  return parsed.href;
}

async function runInput(value, signal) {
  const action = value.action;
  let method, input;
  if (["click", "double_click", "move", "drag"].includes(action)) {
    if (
      !Number.isInteger(value.x) ||
      value.x < 0 ||
      value.x > 1279 ||
      !Number.isInteger(value.y) ||
      value.y < 0 ||
      value.y > 799
    ) {
      throw new InputError(
        "x and y must be integers inside the 1280 by 800 screen",
      );
    }
    method = action === "double_click" ? "click" : action;
    input = { x: value.x, y: value.y };
    if (action === "double_click") input.click_count = 2;
    if (action === "click" && value.mouse_button != null)
      input.mouse_button = value.mouse_button;
    if (action === "drag") {
      if (
        !Number.isInteger(value.to_x) ||
        value.to_x < 0 ||
        value.to_x > 1279 ||
        !Number.isInteger(value.to_y) ||
        value.to_y < 0 ||
        value.to_y > 799
      ) {
        throw new InputError(
          "to_x and to_y must be integers inside the 1280 by 800 screen",
        );
      }
      input = { path: [input, { x: value.to_x, y: value.to_y }] };
    }
  } else if (action === "type") {
    method = "type_text";
    input = { text: boundedString(value.text, "text", 16384) };
  } else if (action === "key") {
    method = "press_key";
    input = { key: requiredString(value.key, "key", 80) };
  } else if (action === "scroll") {
    if (!Number.isInteger(value.amount) || Math.abs(value.amount) > 100)
      throw new InputError("amount must be an integer between -100 and 100");
    method = "scroll";
    input = {
      direction: value.amount < 0 ? "up" : "down",
      pixels: Math.abs(value.amount) * 100,
    };
  } else if (action === "wait") {
    if (!Number.isInteger(value.ms) || value.ms < 0 || value.ms > 30000)
      throw new InputError("ms must be 0..30000");
    await tracked(
      "wait",
      () => sleep(value.ms, undefined, {signal}),
    );
    return { ok: true };
  } else throw new InputError("unsupported input action");
  await readyDriver();
  await tracked(String(action), () => driver.input(method, input, signal));
  return { ok: true };
}

async function screenshot(response, quiet) {
  await readyDriver();
  const capture = () => driver.desktop.call("get_screenshot");
  const result = quiet ? await capture() : await tracked("screenshot", capture);
  const bytes = Buffer.from(result.data, "base64");
  response.writeHead(200, {
    "content-type": result.mimeType,
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
        "/browser/observe",
        "/browser/action",
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
      if (driver?.active?.context.call_id === payload.call_id)
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
    } else if (request.method === "GET" && url.pathname === "/cursor") {
      // 只读订阅不唤醒、不 touch，也不写 activity 或 profile。
      if (cursorClients.size >= 32) {
        json(response, 503, { error: "Too many cursor viewers" });
        return;
      }
      response.writeHead(200, {"content-type": "text/event-stream", "cache-control": "no-store"});
      response.write(cursorEvent());
      cursorClients.add(response);
      response.once("close", () => cursorClients.delete(response));
    } else if (request.method === "GET" && url.pathname === "/targets") {
      json(response, 200, { targets: driver && computer.state === "ready" ? await driver.targets() : [],
        control: Boolean(driver?.control) });
    } else if (request.method === "GET" && url.pathname === "/view/frame") {
      if (!driver || computer.state !== "ready") throw new InputError("Computer is not awake");
      const frame = await driver.viewFrame(url.searchParams.get("target"));
      response.writeHead(200, { "content-type": "image/jpeg", "cache-control": "no-store" });
      response.end(frame);
    } else if (request.method === "POST" && url.pathname === "/view/input") {
      const payload = await body(request);
      if (!driver) throw new InputError("Computer is not awake");
      await driver.viewInput(payload.target, payload.owner, payload.input);
      computer.touch();
      json(response, 200, { sent: true });
    } else if (request.method === "GET" && url.pathname === "/activity") {
      json(response, 200, { ...activity, browser: computer.status(), control: driver?.control ? { active: true } : { active: false } });
    } else if (request.method === "POST" && url.pathname.startsWith("/control/")) {
      const payload = await body(request);
      requiredString(payload.id, "id", 128);
      if (url.pathname === "/control/take") {
        if (!driver || computer.state !== "ready") throw new InputError("Computer is not awake");
        const target = payload.target ?? "desktop";
        if (target !== "desktop") await driver.viewTarget(target);
        const controller = new AbortController();
        const disconnected = () => {
          if (!response.writableFinished) controller.abort(new Error("Control caller disconnected"));
        };
        response.once("close", disconnected);
        if (response.destroyed) disconnected();
        const releaseWorkload = await computer.acquire();
        try {
          await driver.takeControl(payload.id, target, controller.signal);
          if (driver.control?.id !== payload.id) throw new Error("Control connection has ended");
          driver.control.releaseWorkload = releaseWorkload;
        } catch (error) { releaseWorkload(); throw error; }
        finally { response.removeListener("close", disconnected); }
      } else if (url.pathname === "/control/release") {
        if (driver) await driver.releaseControl(payload.id);
      } else if (url.pathname !== "/control/renew") {
        throw new InputError("Unknown control operation");
      }
      if (url.pathname !== "/control/release") {
        if (driver?.control?.id !== payload.id || !driver.control.ready)
          throw new InputError("Control connection has ended");
        clearTimeout(controlTimer);
        const owner = driver;
        controlTimer = setTimeout(() => owner.releaseControl(payload.id).catch(error => {
          owner.closed = true;
          owner.active?.reject(error);
          console.error("Computer control release failed:", error.message);
        }), 30000);
      }
      json(response, 200, { controlled: url.pathname !== "/control/release" });
    } else if (request.method === "POST" && url.pathname === "/wake") {
      json(response, 200, computer.status());
    } else if (request.method === "POST" && url.pathname === "/touch") {
      computer.touch();
      json(response, 200, computer.status());
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
    } else if (
      request.method === "POST" &&
      url.pathname === "/browser/observe"
    ) {
      json(response, 200, await browserObserve(await body(request)));
    } else if (
      request.method === "POST" &&
      url.pathname === "/browser/action"
    ) {
      const payload = await body(request);
      await readyDriver();
      const controller = new AbortController();
      response.once("close", () => {
        if (!response.writableFinished)
          controller.abort(new Error("Computer caller disconnected"));
      });
      const result = await driver.perform(
        (signal) =>
          legacyCall.run(
            { signal, context: driver.active.context },
            async () => {
              try {
                return await browserAction(payload);
              } finally {
                await driver.browser.endTurn(driver.active.context);
              }
            },
          ),
        controller.signal,
      );
      json(response, 200, result);
    } else {
      json(response, 404, { error: "not found" });
    }
  } catch (error) {
    if (driver?.closed && computer.state === "ready") {
      computer.fail(error);
    }
    const status =
      error instanceof InputError || error instanceof SyntaxError ? 400 : 500;
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
let controlTimer;
/** 一次刷新持有完整占用；失败回执与下一次计划均保留在控制服务。 */
async function refreshIdentity() {
  try {
    const { stdout } = await computer.use(async () => {
      await readyDriver();
      return driver.agentOperation(() => exec(
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
      ));
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
  for (const client of cursorClients) client.end();
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

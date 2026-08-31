import { execFile } from "node:child_process";
import { createServer, request as httpRequest } from "node:http";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { connect as tcpConnect } from "node:net";
import { promisify } from "node:util";

const exec = promisify(execFile);
const activityPath = "/data/state/activity.json";
const maxBodyBytes = 256 * 1024;
let activity = { revision: 0, noticeId: 0, active: false, action: "", updatedAt: "" };
let activeTargetId = "";
let cdpRequestId = 0;
const browserSnapshots = new Map();

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
  const id = ++cdpRequestId;
  return await new Promise((resolve, reject) => {
    const socket = new WebSocket(target.webSocketDebuggerUrl);
    const timer = setTimeout(() => {
      socket.close();
      reject(new Error(`CDP ${method} timed out`));
    }, 15_000);
    const finish = (callback) => {
      clearTimeout(timer);
      socket.close();
      callback();
    };
    socket.addEventListener("open", () => {
      socket.send(JSON.stringify({ id, method, params }));
    });
    socket.addEventListener("message", (event) => {
      let message;
      try {
        message = JSON.parse(String(event.data));
      } catch {
        return;
      }
      if (message.id !== id) return;
      if (message.error) {
        finish(() => reject(new Error(`CDP ${method}: ${message.error.message}`)));
        return;
      }
      finish(() => resolve(message.result ?? {}));
    });
    socket.addEventListener("error", () => {
      finish(() => reject(new Error(`CDP ${method} connection failed`)));
    });
  });
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
  const response = await fetch(`http://127.0.0.1:9222/json/new?${encodeURIComponent(url)}`, {
    method: "PUT",
    signal: AbortSignal.timeout(3000),
  });
  if (!response.ok) throw new Error(`Chromium could not open tab: ${response.status}`);
  const target = await response.json();
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
    await tracked("browser wait", () => new Promise((resolve) => setTimeout(resolve, timeout)));
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
    const response = await fetch(`http://127.0.0.1:9222/json/activate/${target.id}`, {
      signal: AbortSignal.timeout(3000),
    });
    if (!response.ok) throw new Error(`Chromium could not activate tab: ${response.status}`);
    activeTargetId = target.id;
    return { ok: true, target_id: target.id };
  }
  if (action === "tab_close") {
    requiredString(value.target_id, "target_id", 128);
    const response = await fetch(`http://127.0.0.1:9222/json/close/${target.id}`, {
      signal: AbortSignal.timeout(3000),
    });
    if (!response.ok) throw new Error(`Chromium could not close tab: ${response.status}`);
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

async function runInput(value) {
  const action = value.action;
  const args = [];
  if (action === "click" || action === "double_click" || action === "move" || action === "drag") {
    if (!Number.isInteger(value.x) || value.x < 0 || value.x > 1279 ||
        !Number.isInteger(value.y) || value.y < 0 || value.y > 799) {
      throw new InputError("x and y must be integers inside the 1280 by 800 screen");
    }
    args.push("mousemove", String(value.x), String(value.y));
    if (action === "click") args.push("click", "1");
    if (action === "double_click") args.push("click", "--repeat", "2", "--delay", "120", "1");
    if (action === "drag") {
      if (!Number.isInteger(value.to_x) || value.to_x < 0 || value.to_x > 1279 ||
          !Number.isInteger(value.to_y) || value.to_y < 0 || value.to_y > 799) {
        throw new InputError("to_x and to_y must be integers inside the 1280 by 800 screen");
      }
      const dragSteps = 8;
      args.push("mousedown", "1", "sleep", "0.10");
      for (let step = 1; step <= dragSteps; step += 1) {
        const x = Math.round(value.x + ((value.to_x - value.x) * step) / dragSteps);
        const y = Math.round(value.y + ((value.to_y - value.y) * step) / dragSteps);
        args.push(
          "mousemove", "--sync", String(x), String(y), "sleep",
          step === dragSteps ? "0.15" : "0.04",
        );
      }
      args.push("mouseup", "1");
    }
  } else if (action === "type") {
    if (typeof value.text !== "string" || value.text.length > 16_384) {
      throw new InputError("text must be a string of at most 16384 characters");
    }
    args.push("type", "--clearmodifiers", "--delay", "1", value.text);
  } else if (action === "key") {
    if (typeof value.key !== "string" || !/^[A-Za-z0-9_+ -]{1,80}$/.test(value.key)) {
      throw new InputError("key is invalid");
    }
    args.push("key", value.key.replaceAll(" ", ""));
  } else if (action === "scroll") {
    if (!Number.isInteger(value.amount) || Math.abs(value.amount) > 100) {
      throw new InputError("amount must be an integer between -100 and 100");
    }
    args.push("click", "--repeat", String(Math.abs(value.amount)), value.amount < 0 ? "4" : "5");
  } else if (action === "wait") {
    if (!Number.isInteger(value.ms) || value.ms < 0 || value.ms > 30_000) {
      throw new InputError("ms must be an integer between 0 and 30000");
    }
    await tracked("wait", () => new Promise((resolve) => setTimeout(resolve, value.ms)));
    return { ok: true };
  } else {
    throw new InputError("unsupported input action");
  }
  await tracked(String(action), () => exec("xdotool", args, { timeout: 30_000 }));
  return { ok: true };
}

async function screenshot(response, quiet) {
  const capture = async () => {
    const result = await exec(
      "import",
      ["-display", ":99", "-window", "root", "png:-"],
      { timeout: 30_000, maxBuffer: 16 * 1024 * 1024, encoding: "buffer" },
    );
    return Buffer.isBuffer(result.stdout) ? result.stdout : Buffer.from(result.stdout);
  };
  const image = quiet ? await capture() : await tracked("screenshot", capture);
  response.writeHead(200, {
    "content-type": "image/png",
    "content-length": String(image.length),
    "cache-control": "no-store",
  });
  response.end(image);
}

function proxyOpenCli(request, response) {
  const url = new URL(request.url ?? "/", "http://opencli.local");
  if (url.pathname === "/shutdown") {
    json(response, 403, { error: "the Computer plugin owns the OpenCLI daemon lifecycle" });
    return;
  }
  const upstream = httpRequest(
    {
      hostname: "127.0.0.1",
      port: 19825,
      method: request.method,
      path: request.url,
      headers: { ...request.headers, host: "127.0.0.1:19825" },
    },
    (upstreamResponse) => {
      response.writeHead(upstreamResponse.statusCode ?? 502, upstreamResponse.headers);
      upstreamResponse.pipe(response);
    },
  );
  upstream.setTimeout(125_000, () => upstream.destroy(new Error("OpenCLI daemon timed out")));
  upstream.on("error", (error) => {
    if (!response.headersSent) json(response, 502, { error: error.message });
    else response.destroy(error);
  });
  request.pipe(upstream);
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

createServer(async (request, response) => {
  try {
    const url = new URL(request.url ?? "/", "http://computer.local");
    if (request.method === "GET" && url.pathname === "/health") {
      json(response, 200, await health());
    } else if (request.method === "GET" && url.pathname === "/activity") {
      json(response, 200, activity);
    } else if (request.method === "GET" && url.pathname === "/screenshot") {
      await screenshot(response, url.searchParams.get("quiet") === "1");
    } else if (request.method === "POST" && url.pathname === "/input") {
      json(response, 200, await runInput(await body(request)));
    } else if (request.method === "POST" && url.pathname === "/browser/observe") {
      json(response, 200, await browserObserve(await body(request)));
    } else if (request.method === "POST" && url.pathname === "/browser/action") {
      json(response, 200, await browserAction(await body(request)));
    } else {
      json(response, 404, { error: "not found" });
    }
  } catch (error) {
    const status = error instanceof InputError || error instanceof SyntaxError ? 400 : 500;
    json(response, status, { error: error instanceof Error ? error.message : String(error) });
  }
}).listen(8080, "0.0.0.0");

createServer(proxyOpenCli).listen(19826, "0.0.0.0");

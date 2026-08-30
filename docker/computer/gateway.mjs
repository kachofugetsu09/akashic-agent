import { execFile } from "node:child_process";
import { createServer } from "node:http";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { promisify } from "node:util";

const exec = promisify(execFile);
const activityPath = "/data/state/activity.json";
const maxBodyBytes = 256 * 1024;
let activity = { revision: 0, noticeId: 0, active: false, action: "", updatedAt: "" };

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
    if (size > maxBodyBytes) throw new Error("request body is too large");
    chunks.push(chunk);
  }
  const value = JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("request body must be an object");
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

async function health() {
  const [daemon, cdp] = await Promise.all([
    daemonStatus(),
    fetch("http://127.0.0.1:9222/json/version", {
      signal: AbortSignal.timeout(1500),
    }),
  ]);
  if (!cdp.ok) throw new Error(`Chromium CDP returned ${cdp.status}`);
  if (!daemon.extensionConnected) throw new Error("OpenCLI extension is not connected");
  return { status: "ready", browser: "ready", opencli: "ready" };
}

async function runOpenCli(args) {
  if (!Array.isArray(args) || args.length === 0 || args.length > 64) {
    throw new Error("args must contain 1 to 64 strings");
  }
  if (args.some((value) => typeof value !== "string" || value.length > 4096)) {
    throw new Error("args contains an invalid value");
  }
  return await tracked(`opencli ${args.slice(0, 3).join(" ")}`, async () => {
    const result = await exec("opencli", args, {
      timeout: 120_000,
      maxBuffer: 8 * 1024 * 1024,
      env: process.env,
    });
    return { stdout: result.stdout, stderr: result.stderr };
  });
}

async function runInput(value) {
  const action = value.action;
  const args = [];
  if (action === "click" || action === "double_click" || action === "move") {
    if (!Number.isInteger(value.x) || !Number.isInteger(value.y)) {
      throw new Error("x and y must be integers");
    }
    args.push("mousemove", String(value.x), String(value.y));
    if (action === "click") args.push("click", "1");
    if (action === "double_click") args.push("click", "--repeat", "2", "--delay", "120", "1");
  } else if (action === "type") {
    if (typeof value.text !== "string" || value.text.length > 16_384) {
      throw new Error("text must be a string of at most 16384 characters");
    }
    args.push("type", "--clearmodifiers", "--delay", "1", value.text);
  } else if (action === "key") {
    if (typeof value.key !== "string" || !/^[A-Za-z0-9_+ -]{1,80}$/.test(value.key)) {
      throw new Error("key is invalid");
    }
    args.push("key", value.key.replaceAll(" ", ""));
  } else if (action === "scroll") {
    if (!Number.isInteger(value.amount) || Math.abs(value.amount) > 100) {
      throw new Error("amount must be an integer between -100 and 100");
    }
    args.push("click", "--repeat", String(Math.abs(value.amount)), value.amount < 0 ? "4" : "5");
  } else if (action === "wait") {
    if (!Number.isInteger(value.ms) || value.ms < 0 || value.ms > 30_000) {
      throw new Error("ms must be an integer between 0 and 30000");
    }
    await tracked("wait", () => new Promise((resolve) => setTimeout(resolve, value.ms)));
    return { ok: true };
  } else {
    throw new Error("unsupported input action");
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
    } else if (request.method === "POST" && url.pathname === "/opencli") {
      const value = await body(request);
      json(response, 200, await runOpenCli(value.args));
    } else {
      json(response, 404, { error: "not found" });
    }
  } catch (error) {
    json(response, 500, { error: error instanceof Error ? error.message : String(error) });
  }
}).listen(8080, "0.0.0.0");

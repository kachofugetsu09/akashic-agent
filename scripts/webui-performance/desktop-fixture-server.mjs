import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, resolve, sep } from "node:path";

import { WebSocketServer } from "ws";

import {
  desktopMessagesForSession,
  desktopModels,
  desktopRuntimeDetail,
  desktopRuntimeOverview,
  desktopSessions,
  fixtureSessionId,
} from "./fixtures.mjs";

/** Serve the production desktop bundle with deterministic HTTP and real WebSocket fixtures. */
export async function startDesktopFixtureServer(root, { port = 0 } = {}) {
  const sockets = new Set();
  const receivedFrames = [];
  const receivedRequests = [];
  let pairingCreateCount = 0;
  let historyDelayMs = 0;
  const websocketServer = new WebSocketServer({ noServer: true });
  websocketServer.on("connection", (socket) => {
    sockets.add(socket);
    socket.on("message", (data) => receivedFrames.push(JSON.parse(String(data))));
    socket.once("close", () => sockets.delete(socket));
  });

  const server = createServer(async (request, response) => {
    const url = new URL(request.url, "http://127.0.0.1");
    if (request.method === "POST" && url.pathname === "/__fixture/reset") {
      receivedFrames.length = 0;
      receivedRequests.length = 0;
      historyDelayMs = 0;
      return sendJson(response, { ok: true });
    }
    if (request.method === "GET" && url.pathname === "/__fixture/received") {
      return sendJson(response, { items: receivedFrames, requests: receivedRequests });
    }
    if (request.method === "POST" && url.pathname === "/__fixture/history-delay") {
      historyDelayMs = boundedNumber(url.searchParams.get("ms"), 0, 0, 10_000);
      return sendJson(response, { historyDelayMs });
    }
    if (request.method === "POST" && url.pathname === "/api/chat/uploads") {
      const filename = url.searchParams.get("filename") || "upload.bin";
      return sendJson(response, { filename, upload_path: `uploads/${filename}`, upload_url: `/media/uploads/${filename}` });
    }
    if (request.method === "POST" && url.pathname === "/api/chat/mobile-pairing") {
      pairingCreateCount += 1;
      await delay(300);
      return sendJson(response, desktopPairingOffer(pairingCreateCount));
    }
    if (request.method === "GET" && /^\/api\/chat\/mobile-pairing\/fixture-pairing-/u.test(url.pathname)) {
      return sendJson(response, {
        pairing_id: url.pathname.split("/").at(-1), status: "waiting_for_desktop_confirmation",
        device_name: "Pixel 7", confirmation_code: "358864", capabilities: ["chat"],
      });
    }
    if (request.method === "POST" && /\/approve$/u.test(url.pathname)) {
      return sendJson(response, { device_id: "pixel-7", display_name: "Pixel 7" });
    }
    const settings = await settingsFixtureResponse(request, url, receivedRequests);
    if (settings !== undefined) return sendJson(response, settings);
    const historyMatch = url.pathname.match(/^\/api\/chat\/sessions\/([^/]+)\/messages$/u);
    if (request.method === "GET" && historyMatch) {
      receivedRequests.push(`${request.method} ${url.pathname}`);
      if (historyDelayMs > 0) await delay(historyDelayMs);
      return sendJson(response, desktopMessagesForSession(decodeURIComponent(historyMatch[1])));
    }
    if (request.method === "POST" && url.pathname === "/__fixture/stream") {
      if (sockets.size === 0) return sendJson(response, { error: "no_websocket_client" }, 409);
      const sessionId = url.searchParams.get("session_id") || fixtureSessionId;
      let count;
      let intervalMs;
      let terminal;
      try {
        count = boundedInteger(url.searchParams.get("count"), 600, 1, 10_000);
        intervalMs = boundedNumber(url.searchParams.get("interval_ms"), 2.5, 0, 1_000);
        terminal = boundedInteger(url.searchParams.get("terminal"), 1, 0, 1);
      } catch (error) {
        if (!(error instanceof RangeError)) throw error;
        return sendJson(response, { error: error.message }, 400);
      }
      const delta = url.searchParams.get("delta") || "片";
      const turnId = `fixture-${Date.now()}`;
      broadcast(sockets, { type: "turn.started", session_id: sessionId, turn_id: turnId, content: "" });
      for (let index = 0; index < count; index += 1) {
        broadcast(sockets, { type: "answer.delta", session_id: sessionId, turn_id: turnId, delta });
        if (intervalMs > 0) await delay(intervalMs);
      }
      if (terminal === 1) {
        broadcast(sockets, {
          type: "message.final",
          session_id: sessionId,
          turn_id: turnId,
          content: delta.repeat(count),
          duration_ms: 1,
        });
      }
      return sendJson(response, { sessionId, turnId, count, delta, intervalMs, terminal: terminal === 1 });
    }

    const api = fixtureApiResponse(url);
    if (api !== undefined) return sendJson(response, api);
    let requested = url.pathname === "/" || url.pathname === "/settings" ? "index.html" : url.pathname.replace(/^\//u, "");
    requested = requested.replace(/^assets\//u, "");
    const file = resolve(root, requested);
    if (!file.startsWith(`${root}${sep}`) || !existsSync(file) || !statSync(file).isFile()) {
      response.writeHead(404).end("not found");
      return;
    }
    response.writeHead(200, { "content-type": contentType(file), "cache-control": "no-store" });
    createReadStream(file).pipe(response);
  });

  server.on("upgrade", (request, socket, head) => {
    const url = new URL(request.url, "http://127.0.0.1");
    if (url.pathname !== "/ws") {
      socket.destroy();
      return;
    }
    websocketServer.handleUpgrade(request, socket, head, (websocket) => {
      websocketServer.emit("connection", websocket, request);
    });
  });

  await new Promise((resolveListen, reject) => {
    server.once("error", reject);
    server.listen(port, "127.0.0.1", resolveListen);
  });
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("fixture server did not bind TCP");
  return {
    origin: `http://127.0.0.1:${address.port}`,
    port: address.port,
    close: async () => {
      for (const socket of sockets) socket.close();
      await new Promise((resolveClose, reject) => server.close((error) => error ? reject(error) : resolveClose()));
      websocketServer.close();
    },
  };
}

async function settingsFixtureResponse(request, url, receivedRequests) {
  if (!url.pathname.startsWith("/api/settings/")) return undefined;
  receivedRequests.push(`${request.method} ${url.pathname}`);
  if (request.method === "GET" && url.pathname === "/api/settings/state") return desktopSettingsState();
  if (request.method === "POST" && url.pathname === "/api/settings/models") {
    await delay(150);
    return { models: [{ id: "fixture-discovered", contextWindow: 131_072, maxOutputTokens: 8_192, inputModalities: ["text"], supportedReasoningEfforts: ["medium"], defaultReasoningEffort: "medium" }] };
  }
  if (request.method === "POST" && url.pathname === "/api/settings/apply") return { ok: true };
  if (request.method === "POST" && url.pathname === "/api/settings/roles") return { ok: true };
  if (request.method === "POST" && url.pathname === "/api/settings/embedding-models") {
    await delay(150);
    return { status: "applied", revision: "fixture-embedding-revision", model: {
      id: "fixture-embedding", sourceId: "fixture-embedding-source", sourceName: "向量服务",
      provider: "openai", baseUrl: "https://embedding.example.com/v1", model: "fixture-embedding-model",
      dimensions: 1_024, credential: { id: "fixture-embedding-credential", configured: true },
    } };
  }
  if (request.method === "POST" && url.pathname === "/api/settings/memory") {
    await delay(150);
    return { status: "applied", operationId: "fixture-memory-operation" };
  }
  if (request.method === "POST" && url.pathname === "/api/settings/codex-login") {
    return { loginId: "fixture-login", status: "waiting", userCode: "ABCD-EFGH", verificationUri: "https://example.com/device", interval: 0, error: "" };
  }
  if (request.method === "GET" && url.pathname === "/api/settings/codex-login/fixture-login") {
    return { loginId: "fixture-login", status: "completed", userCode: "ABCD-EFGH", verificationUri: "https://example.com/device", interval: 0, error: "" };
  }
  return undefined;
}

function desktopSettingsState() {
  const runtimes = Array.from({ length: 48 }, (_, index) => ({
    id: `settings-runtime-${index}`,
    provider: index % 2 === 0 ? "deepseek" : "openai",
    model: `fixture-model-${index}`,
    sourceId: `settings-source-${index}`,
    sourceName: `设置连接 ${index + 1}`,
    catalogProvider: "fixture",
    baseUrl: "https://api.example.com/v1",
    contextWindow: 131_072,
    maxOutputTokens: 8_192,
    inputModalities: ["text"],
    reasoningEffort: "medium",
    supportedReasoningEfforts: ["medium"],
    credential: { id: `credential-${index}`, configured: true, source: "workspace" },
  }));
  return {
    mode: "ready",
    workspace: "fixture",
    activeRuntime: runtimes[0].id,
    runtimes,
    roleBindings: {},
    modelRevision: 7,
    codexConfigured: false,
    localOpenCodeConfigured: true,
    configRevision: "fixture-revision",
    memory: {
      configured: true, enabled: false, engine: "akasha", embeddingModelId: "",
      embeddingModels: [], changeLocked: false, revision: "fixture-memory-revision",
    },
  };
}

function desktopPairingOffer(sequence) {
  return {
    protocol_version: 1,
    server_id: "fixture-server",
    server_application_key_fingerprint: "fixture-fingerprint",
    server_application_public_key: "fixture-public-key",
    lan_endpoints: ["wss://192.0.2.1/ws"],
    tunnel_endpoints: [],
    tls_spki_pins: ["fixture-pin"],
    pairing_id: `fixture-pairing-${sequence}`,
    one_time_secret: `fixture-secret-${sequence}`,
    expires_at: new Date(Date.now() + 60_000).toISOString(),
  };
}

function fixtureApiResponse(url) {
  const { pathname } = url;
  if (pathname === "/api/shell/state") return { status: "ready", configured: true, chatReady: true, settingsPath: "/settings" };
  if (pathname === "/api/chat/sessions") return desktopSessions();
  if (pathname === "/api/chat/models") return desktopModels();
  if (pathname === "/api/chat/plugin-ui/catalog") return { catalog_revision: "0".repeat(64), items: [] };
  const runtimeOverview = desktopRuntimeOverview(pathname);
  if (runtimeOverview !== undefined) return runtimeOverview;
  const runtimeDetail = desktopRuntimeDetail(url);
  if (runtimeDetail !== undefined) return runtimeDetail;
  return undefined;
}

function broadcast(sockets, frame) {
  const encoded = JSON.stringify(frame);
  for (const socket of sockets) socket.send(encoded);
}

function boundedInteger(raw, fallback, minimum, maximum) {
  if (raw === null) return fallback;
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new RangeError(`fixture integer must be in ${minimum}..${maximum}`);
  }
  return value;
}

function boundedNumber(raw, fallback, minimum, maximum) {
  if (raw === null) return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw new RangeError(`fixture number must be in ${minimum}..${maximum}`);
  }
  return value;
}

function delay(durationMs) {
  return new Promise((resolveDelay) => setTimeout(resolveDelay, durationMs));
}

function sendJson(response, payload, status = 200) {
  if (payload === undefined) {
    response.writeHead(404).end("not found");
    return;
  }
  response.writeHead(status, { "content-type": "application/json", "cache-control": "no-store" });
  response.end(JSON.stringify(payload));
}

function contentType(file) {
  return ({
    ".css": "text/css",
    ".html": "text/html",
    ".js": "text/javascript",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".woff2": "font/woff2",
  })[extname(file)] ?? "application/octet-stream";
}

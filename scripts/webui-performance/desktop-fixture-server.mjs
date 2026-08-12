import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, resolve, sep } from "node:path";

import { WebSocketServer } from "ws";

import {
  desktopMessagesForSession,
  desktopModels,
  desktopSessions,
  fixtureSessionId,
} from "./fixtures.mjs";

/** Serve the production desktop bundle with deterministic HTTP and real WebSocket fixtures. */
export async function startDesktopFixtureServer(root, { port = 0 } = {}) {
  const sockets = new Set();
  const websocketServer = new WebSocketServer({ noServer: true });
  websocketServer.on("connection", (socket) => {
    sockets.add(socket);
    socket.once("close", () => sockets.delete(socket));
  });

  const server = createServer(async (request, response) => {
    const url = new URL(request.url, "http://127.0.0.1");
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

    const api = fixtureApiResponse(url.pathname);
    if (api !== undefined) return sendJson(response, api);
    let requested = url.pathname === "/" ? "index.html" : url.pathname.replace(/^\//u, "");
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

function fixtureApiResponse(pathname) {
  if (pathname === "/api/shell/state") return { status: "ready", configured: true, chatReady: true, settingsPath: "/settings" };
  if (pathname === "/api/chat/sessions") return desktopSessions();
  const match = pathname.match(/^\/api\/chat\/sessions\/([^/]+)\/messages$/u);
  if (match) return desktopMessagesForSession(decodeURIComponent(match[1]));
  if (pathname === "/api/chat/models") return desktopModels();
  if (pathname === "/api/chat/plugin-ui/catalog") return { catalog_revision: "0".repeat(64), items: [] };
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

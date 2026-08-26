import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");

export async function startMobileWebLabServer({
  root = resolve(repoRoot, "dist", "mobile-web-lab"),
  port = 0,
} = {}) {
  const outputRoot = resolve(root);
  if (!existsSync(resolve(outputRoot, "mobile-lab.html"))) {
    throw new Error(`Mobile Web Lab 尚未构建: ${outputRoot}`);
  }
  const server = createServer((request, response) => {
    if (request.method !== "GET" && request.method !== "HEAD") {
      response.writeHead(405, { allow: "GET, HEAD" });
      response.end();
      return;
    }
    let pathname;
    try {
      pathname = decodeURIComponent(new URL(request.url ?? "/", "http://127.0.0.1").pathname);
    } catch {
      response.writeHead(400);
      response.end("Bad request");
      return;
    }
    const relativePath = pathname === "/" ? "mobile-lab.html" : pathname.replace(/^\/+/, "");
    const filePath = resolve(outputRoot, relativePath);
    if (!filePath.startsWith(`${outputRoot}${sep}`) || !existsSync(filePath) || !statSync(filePath).isFile()) {
      response.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
      response.end("Not found");
      return;
    }
    response.writeHead(200, {
      "content-type": contentType(filePath),
      "cache-control": relativePath.endsWith(".html") ? "no-store" : "public, max-age=31536000, immutable",
      "x-content-type-options": "nosniff",
      "referrer-policy": "no-referrer",
    });
    if (request.method === "HEAD") response.end();
    else createReadStream(filePath).pipe(response);
  });
  await new Promise((resolveListen, reject) => {
    server.once("error", reject);
    server.listen(port, "127.0.0.1", resolveListen);
  });
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("无法读取 Mobile Web Lab listener");
  return {
    origin: `http://127.0.0.1:${address.port}`,
    close: () => new Promise((resolveClose, reject) => {
      server.close((error) => error ? reject(error) : resolveClose());
    }),
  };
}

function contentType(filePath) {
  return ({
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".woff2": "font/woff2",
  })[extname(filePath)] ?? "application/octet-stream";
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const configuredPort = Number(process.env.AKASHIC_MOBILE_WEB_LAB_PORT ?? "4174");
  if (!Number.isSafeInteger(configuredPort) || configuredPort < 1 || configuredPort > 65_535) {
    throw new Error("AKASHIC_MOBILE_WEB_LAB_PORT 必须是 1..65535 的整数");
  }
  const lab = await startMobileWebLabServer({ port: configuredPort });
  console.log(`Akashic Mobile Web Lab: ${lab.origin}`);
  const shutdown = async () => {
    await lab.close();
    process.exit(0);
  };
  process.once("SIGINT", shutdown);
  process.once("SIGTERM", shutdown);
}

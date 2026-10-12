// 仅回放 visuals_scenario.py 导出的实际消息、附件和安装资产。
import { createServer } from "node:http";
import { readFile, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve, join, extname } from "node:path";
import { build } from "vite";
import react from "@vitejs/plugin-react";

const repo = process.cwd();
const evidence = resolve(process.argv[2]);
const port = Number(process.argv[3] ?? 4330);
const output = await mkdtemp(join(tmpdir(), "akashic-visuals-browser-"));
await build({ configFile: false, root: repo, plugins: [react()],
  resolve: { alias: { "@": resolve("frontend/chat/src"), mermaid: resolve("frontend/chat/src/mermaid-stub.ts") } },
  build: { outDir: output, emptyOutDir: false, target: "es2022",
    rollupOptions: { input: resolve("scripts/visuals/browser-entry.tsx"), output: { entryFileNames: "app.js", assetFileNames: "[name][extname]" } } } });
const catalog = await readFile(join(evidence, "catalog.json"));
const html = `<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Akashic · 可视化插件验收</title><link rel="stylesheet" href="/browser-entry.css"><style>nav{display:flex;flex-wrap:wrap;gap:8px;padding:12px;border-bottom:1px solid var(--ak-color-border-default)}nav button{min-height:44px;padding:8px;border:1px solid var(--ak-color-border-default)}main{max-width:760px;margin:20px auto;padding:0 8px}</style><div id="root"></div><script type="module" src="/app.js"></script></html>`;
const server = createServer(async (req, res) => {
  const url = new URL(req.url, "http://127.0.0.1");
  try {
    if (url.pathname === "/favicon.ico") { res.writeHead(204).end(); return; }
    if (url.pathname === "/") { res.setHeader("content-type", "text/html; charset=utf-8"); res.end(html); return; }
    if (url.pathname === "/api/chat/plugin-ui/catalog") { res.setHeader("content-type", "application/json"); res.end(catalog); return; }
    const route = url.pathname.startsWith("/api/chat/plugin-ui/script/") ? "frame_runtime.js"
      : url.pathname === "/api/chat/plugin-ui/asset" ? "module.js"
      : url.pathname.startsWith("/api/chat/artifacts/") ? `${url.pathname.split("/").at(-1)}.json`
      : ["/messages.json", "/showcase-messages.json"].includes(url.pathname) ? url.pathname.slice(1) : null;
    const path = route ? join(evidence, route) : join(output, url.pathname);
    if (!path.startsWith(route ? evidence + "/" : output + "/")) { res.writeHead(403).end(); return; }
    res.setHeader("content-type", ({ ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".woff2": "font/woff2" })[extname(path)] ?? "application/octet-stream");
    res.end(await readFile(path));
  } catch (error) { console.error(error); res.writeHead(500).end(String(error)); }
});
server.listen(port, "127.0.0.1", () => console.log(`Visuals browser acceptance: http://127.0.0.1:${port}`));
for (const signal of ["SIGINT", "SIGTERM"]) process.on(signal, () => server.close(async () => { await rm(output, { recursive: true }); process.exit(); }));

// 用真实 React 组件验证故障、恢复和未知状态，HTTP 响应由实验控制。
import { build } from "esbuild";
import { chromium } from "playwright-core";
import { createServer } from "node:http";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

const output = await mkdtemp(resolve(tmpdir(), "bridge-notice-"));
let state = "degraded";
let httpFailure = false;
let browser;
const server = createServer(async (request, response) => {
  if (request.url === "/api/runtime/host-bridge") {
    response.writeHead(httpFailure ? 503 : 200, { "Content-Type": "application/json" });
    response.end(JSON.stringify({ state }));
  } else if (request.url === "/app.js") {
    response.writeHead(200, { "Content-Type": "text/javascript" });
    response.end(await readFile(resolve(output, "app.js")));
  } else {
    response.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
    response.end('<html lang="zh"><div id="root"></div><script src="/app.js"></script></html>');
  }
});
try {
  await build({
    stdin: {
      contents: 'import {createRoot} from "react-dom/client"; import {HostBridgeNotice} from "./frontend/chat/src/host-bridge-notice"; createRoot(document.getElementById("root")).render(<HostBridgeNotice/>);',
      resolveDir: process.cwd(), loader: "tsx",
    },
    bundle: true, jsx: "automatic", outfile: resolve(output, "app.js"),
  });
  await new Promise(resolveReady => server.listen(0, "127.0.0.1", resolveReady));
  browser = await chromium.launch({ executablePath: "/usr/bin/chromium", headless: true });
  const page = await browser.newPage();
  await page.clock.install();
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await page.getByRole("status").filter({ hasText: "宿主执行暂时不可用" }).waitFor();
  state = "healthy";
  await page.clock.fastForward(5100);
  await page.getByRole("status").waitFor({ state: "detached" });
  httpFailure = true;
  await page.clock.fastForward(5100);
  await page.getByRole("status").filter({ hasText: "暂时无法确认" }).waitFor();
  httpFailure = false;
  state = "disabled";
  await page.clock.fastForward(5100);
  await page.getByRole("status").waitFor({ state: "detached" });
  console.log(JSON.stringify({ result: "passed", experiments: ["degraded_visible", "recovery_clears_notice", "unknown_is_not_healthy", "local_mode_no_notice"] }));
} finally {
  await browser?.close();
  await new Promise(resolveClosed => server.close(resolveClosed));
  await rm(output, { recursive: true });
}

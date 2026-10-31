// 用真实 HTTP 和 Chromium 校准点击测量器；不代表产品性能验收。
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import http from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const output = await mkdtemp(join(tmpdir(), "akashic-click-check-"));
const server = http.createServer((request, response) => {
  const [mode, endpoint] = request.url.split("/").filter(Boolean);
  if (endpoint === "data") {
    if (mode === "hung") return;
    if (mode === "disconnect") { response.destroy(); return; }
    setTimeout(() => {
      response.writeHead(mode === "error" ? 503 : 200, { "content-type": "text/plain" });
      response.end("loaded");
    }, mode === "slow" ? 1800 : 20);
    return;
  }
  response.writeHead(200, { "content-type": "text/html; charset=utf-8" });
  response.end(`<button>${mode === "missing" ? "不存在" : "功能设置"}</button><section></section><script>
    document.querySelector('button').onclick = async () => {
      const response = await fetch('/${mode}/data');
      document.querySelector('section').textContent = await response.text();
    };
    document.onkeydown = event => {
      if (event.key === 'Escape') document.querySelector('section').textContent = '';
    };
  </script>`);
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
try {
  // 1. 每个场景通过完整 CLI 启动浏览器，检查报告和进程退出码。
  const storage = join(output, "storage.json");
  await writeFile(storage, JSON.stringify({ cookies: [], origins: [] }));
  for (const mode of ["slow", "error", "missing", "disconnect", "hung"]) {
    const reportPath = join(output, `${mode}.json`);
    const child = spawn(process.execPath, [fileURLToPath(new URL("./measure-clicks.mjs", import.meta.url)),
      "--origin", `http://127.0.0.1:${server.address().port}/${mode}`, "--storage", storage,
      "--runs", "1", "--only", "^settings.open$", "--out", reportPath], { stdio: "inherit" });
    const code = await new Promise((resolve, reject) => { child.on("error", reject); child.on("exit", resolve); });
    assert.equal(code, mode === "slow" ? 0 : 1, mode);
    const report = JSON.parse(await readFile(reportPath, "utf8")).scenarios["settings.open"];
    assert.equal(report.plannedRuns, 1);
    assert.equal(report.raw.length, 1);
    assert.equal(report.failedRuns, mode === "slow" ? 0 : 1);
    if (mode === "missing") { assert.match(report.raw[0].error, /找不到/); continue; }
    for (const phase of ["first", "again"]) {
      const sample = report.raw[0][phase];
      assert.ok(sample.waterfall.some(entry => entry.key === `/${mode}/data`));
      if (mode === "slow") assert.ok(sample.doneMs >= 1800, JSON.stringify(sample));
      if (mode === "error") assert.equal(sample.waterfall[0].status, 503);
      if (mode === "disconnect") assert.ok(sample.waterfall[0].error);
      if (mode === "hung") {
        assert.equal(sample.observationComplete, false);
        assert.equal(sample.doneMs, null);
        assert.equal(sample.pending, 1);
      }
    }
  }
  console.log("PASS: 慢响应、HTTP 错误、缺失按钮、断连、挂起均未伪装为快速成功");
} finally {
  // 2. 仅清理本脚本创建的隔离临时目录和连接。
  server.closeAllConnections();
  await new Promise(resolve => server.close(resolve));
  await rm(output, { recursive: true });
}

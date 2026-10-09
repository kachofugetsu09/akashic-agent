// 真实浏览器 → 实际设置模块 → Models HTTP/SQLite → 驱动 → 本地认证服务。
// node scripts/model_auth_ui_scenario.mjs [--baseline /path/to/original/frontend]
import assert from "node:assert/strict";
import {spawn} from "node:child_process";
import {once} from "node:events";
import {mkdtemp, writeFile} from "node:fs/promises";
import {createServer} from "node:net";
import {tmpdir} from "node:os";
import {join, resolve} from "node:path";
import {fileURLToPath} from "node:url";
import {chromium} from "playwright-core";

const root = fileURLToPath(new URL("../", import.meta.url));
const baseline = process.argv.includes("--baseline");
const frontend = baseline ? resolve(process.argv[process.argv.indexOf("--baseline") + 1]) : join(root, "frontend");
const output = await mkdtemp(join(tmpdir(), "akashic-model-auth-ui-"));
const listener = createServer();
listener.listen(0, "127.0.0.1");
await once(listener, "listening");
const port = listener.address().port;
await new Promise(resolve => listener.close(resolve));
const base = `http://127.0.0.1:${port}`;
const service = spawn(process.env.PYTHON ?? join(root, ".venv/bin/python"), [
  join(root, "scripts/model_auth_ui_fixture.py"), "--port", String(port), "--frontend", frontend,
], {stdio: ["ignore", "ignore", "pipe"]});
let log = "", browser;
const checks = [];

try {
  // 1. 启动日志确认监听完成；所有状态和上游服务都在本次隔离进程中。
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`登录夹具启动超时: ${log}`)), 15000);
    service.once("error", reject);
    service.once("exit", code => {clearTimeout(timer); reject(new Error(`登录夹具退出 ${code}: ${log}`));});
    service.stderr.on("data", data => {
      log += data;
      if (log.includes("Uvicorn running on")) {clearTimeout(timer); resolve();}
    });
  });
  browser = await chromium.launch({headless: true,
    ...(process.env.CHROMIUM_PATH ? {executablePath: process.env.CHROMIUM_PATH} : {}),
    args: ["--disable-gpu"],
  });
  const page = await browser.newPage({viewport: {width: 1280, height: 1000}});
  page.setDefaultTimeout(10000);
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("dialog", dialog => dialog.accept());
  const get = async path => {
    const response = await fetch(base + path);
    assert.equal(response.status, 200);
    return response.json();
  };
  await page.goto(base);
  await page.getByRole("button", {name: /^Codex/}).click();
  await page.getByRole("button", {name: "开始登录"}).click();
  if (baseline) {
    // 2. 原始源码必须真实触发严格驱动拒绝，证明场景覆盖报告的回归。
    await page.getByRole("alert").filter({hasText: "unsupported Codex auth input: auth_identity"}).waitFor();
    assert.equal((await get("/api/dashboard/models/catalog")).connections.length, 0);
    const receipts = await get("/receipts");
    assert.equal(receipts[0].status, 422);
    assert.ok(receipts[0].input_keys.includes("auth_identity"));
    checks.push("原始表单复现 auth_identity 拒绝，未创建连接");
  } else {
    // 3. 新连接走设备码、token 交换、目录校验和实际 SQLite 提交。
    await page.getByText("LOCAL-CODE", {exact: true}).waitFor();
    await page.screenshot({path: join(output, "device-code.png"), fullPage: true});
    await page.getByRole("dialog").getByRole("button", {name: "取消", exact: true}).click();
    let catalog = await get("/api/dashboard/models/catalog");
    assert.equal(catalog.connections.length, 1);
    assert.equal(catalog.connections[0].authIdentity, "local-account");
    const connectionId = catalog.connections[0].id;
    checks.push("Codex 新连接显示验证码并完成认证提交");

    // 4. 重新登录保留连接身份；取消下一次登录不改变已保存连接。
    await page.getByRole("button", {name: "编辑连接 Codex", exact: true}).click();
    await page.getByRole("button", {name: "重新登录"}).click();
    await page.getByText("LOCAL-CODE", {exact: true}).waitFor();
    await page.getByRole("dialog").getByRole("button", {name: "取消", exact: true}).click();
    catalog = await get("/api/dashboard/models/catalog");
    assert.equal(catalog.connections.length, 1);
    assert.equal(catalog.connections[0].id, connectionId);
    checks.push("已有 Codex 连接重新登录成功且身份不变");
    await page.getByRole("button", {name: "编辑连接 Codex", exact: true}).click();
    await page.getByRole("button", {name: "重新登录"}).click();
    await page.getByText("LOCAL-CODE", {exact: true}).waitFor();
    const cancelled = page.waitForResponse(response => response.url().endsWith("/command")
      && response.request().postDataJSON().type === "cancel_auth");
    await page.locator(".settings-dialog--inline").getByRole("button", {name: "关闭", exact: true}).click();
    assert.equal((await cancelled).status(), 200);
    assert.deepEqual((await get("/api/dashboard/models/catalog")).connections, catalog.connections);
    checks.push("取消登录不改变已有连接");

    // 5. OpenCode 使用驱动自身默认身份，API Key 路径不依赖表单注入。
    await page.getByRole("button", {name: "添加连接", exact: true}).click();
    await page.getByRole("button", {name: /^OpenCode Go/}).click();
    await page.locator('input[name="endpoint"]').fill(base + "/upstream/opencode/v1");
    await page.getByLabel("API Key", {exact: true}).fill("local-fixture-key");
    await page.getByRole("button", {name: "连接并选择模型", exact: true}).click();
    await page.getByRole("dialog").getByRole("button", {name: "取消", exact: true}).click();
    catalog = await get("/api/dashboard/models/catalog");
    assert.equal(catalog.connections.length, 2);
    assert.equal(catalog.connections.find(item => item.driverId === "opencode-go").authIdentity, "opencode-go");
    const starts = (await get("/receipts")).filter(item => item.type === "start_auth");
    assert.equal(starts.filter(item => item.driver_id === "codex").length, 3);
    assert.ok(starts.every(item => !item.input_keys.includes("auth_identity") && item.status === 200));
    assert.deepEqual(errors, []);
    checks.push("OpenCode API Key 连接成功；所有登录请求只传递驱动参数");
  }
  await page.screenshot({path: join(output, "result.png"), fullPage: true});
  const receipt = {baseline, checks, output};
  await writeFile(join(output, "result.json"), JSON.stringify(receipt, null, 2));
  console.log(JSON.stringify(receipt));
} finally {
  await browser?.close();
  if (service.exitCode === null) {service.kill("SIGTERM"); await once(service, "exit");}
}

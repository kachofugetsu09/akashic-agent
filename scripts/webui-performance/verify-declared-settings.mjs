/** 只在已核对的一次性 Runtime 中验证声明式表单的草稿、失败和真实保存。 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { copyFileSync, mkdirSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { resolve, sep } from "node:path";
import { parseArgs } from "node:util";
import { chromium } from "playwright-core";
const { values } = parseArgs({ options: { fixture: { type: "string" }, output: { type: "string" }, browser: { type: "string" } } });
for (const key of ["fixture", "output", "browser"]) assert.ok(values[key], `需要 --${key}`);
// 1. 标记、活跃进程及监听端点共同证明写入目标；localhost 本身不是隔离证明。
const state = realpathSync(values.fixture);
const marker = JSON.parse(readFileSync(resolve(state, ".akashic-oneoff-fixture.json"), "utf8"));
assert.equal(marker.kind, "akashic-webui-oneoff");
assert.equal(realpathSync(marker.state), state);
const workspace = realpathSync(marker.workspace);
assert.ok(workspace.startsWith(state + sep));
const command = readFileSync(`/proc/${marker.pid}/cmdline`, "utf8").split("\0");
assert.ok(command.includes(state));
assert.ok(command.some(part => /(?:serve-startup-runtime|serve-declared-settings)\.py$/.test(part)));
const listeners = execFileSync("ss", ["-ltnpH", `sport = :${marker.port}`], { encoding: "utf8" }).trim().split("\n");
assert.equal(listeners.length, 1);
assert.ok(listeners[0].includes(`127.0.0.1:${marker.port}`) && listeners[0].includes(`pid=${marker.pid},`));
const origin = `http://127.0.0.1:${marker.port}`;
const output = resolve(values.output);
mkdirSync(output, { recursive: false, mode: 0o700 });
copyFileSync(resolve(workspace, "bundle.patch.toml"), resolve(output, "bundle.patch.before.toml"));
const report = {};
const browser = await chromium.launch({ executablePath: values.browser });
try {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, serviceWorkers: "block" });
  await context.addInitScript(() => localStorage.setItem("akashic.onboarding.seen", "1"));
  const page = await context.newPage();
  const nextFrame = () => page.evaluate(() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done))));
  const section = name => page.locator(".declared-form").filter({ has: page.getByRole("heading", { name, exact: true }) });
  const note = () => section("Settings a").locator('input[name="note"]');
  const plugins = () => page.locator(".shell-settings-nav").getByRole("button", { name: "插件", exact: true });
  const card = () => page.locator(".shell-plugin-card").filter({ has: page.locator("strong", { hasText: /^插件设置$/ }) });
  const enter = async () => {
    await page.goto(origin);
    await page.getByRole("button", { name: "功能设置", exact: true }).click();
    await plugins().click(); await page.getByRole("tab", { name: "插件配置", exact: true }).click();
    await card().click();
    await page.waitForFunction(() => document.querySelectorAll(".declared-form .config-form > form").length === 2);
    assert.deepEqual(await page.locator(".declared-form h2").allTextContents(), ["Settings a", "Settings b"]);
  };
  await enter();
  const original = await note().inputValue();
  await plugins().click();
  // 2. 留住一次真实 GET；用户编辑后再放行，旧值不能覆盖草稿。
  let release;
  const gate = new Promise(done => { release = done; });
  const seen = page.waitForRequest(request => request.url().endsWith("/settings_probe_a/config") && request.method() === "GET");
  const configA = "**/api/dashboard/plugin_settings/settings_probe_a/config";
  await page.route(configA, async route => { await gate; await route.continue(); }, { times: 1 });
  await card().click(); await seen;
  assert.equal(await note().inputValue(), original);
  await note().fill("unsaved during refresh");
  const read = page.waitForResponse(response => response.url().endsWith("/settings_probe_a/config") && response.request().method() === "GET");
  release(); await (await read).finished(); await nextFrame();
  assert.equal(await note().inputValue(), "unsaved during refresh"); report.lateReadPreservesDraft = true;
  await plugins().click();
  await page.getByRole("button", { name: "继续编辑", exact: true }).click();
  assert.equal(await note().inputValue(), "unsaved during refresh");
  await plugins().click(); await page.getByRole("button", { name: "放弃并离开", exact: true }).click();
  await card().click(); assert.equal(await note().inputValue(), original); report.discardDoesNotCacheDraft = true;
  // 3. 缓存内容仍可见，但真实 503 必须报错；重试走未改写的真实端点。
  await plugins().click();
  await page.route("**/api/dashboard/plugin_settings/settings_probe_b/config", route => route.fulfill({
    status: 503, contentType: "application/json", body: JSON.stringify({ detail: "isolated read failure" }),
  }), { times: 1 });
  await card().click();
  await section("Settings b").getByText("isolated read failure", { exact: true }).waitFor();
  assert.equal(await section("Settings b").locator('input[name="note"]').getAttribute("placeholder"), "默认：initial-b");
  await section("Settings b").getByRole("button", { name: "重新加载", exact: true }).click();
  await page.waitForFunction(() => !document.querySelector(".declared-form .config-error")); report.failureAndRetryVisible = true;
  // 4. 唯一写入：测试 owner 的 note 字段；核对 active 回执，再用新页面读取持久配置。
  const savedValue = `saved through real settings ${Date.now()}`;
  await note().fill(savedValue);
  const active = page.waitForResponse(async response => response.url().includes("/settings_probe_a/config/receipts/")
    && response.ok() && (await response.json()).state === "active", { timeout: 30000 });
  const saved = page.waitForResponse(response => response.url().endsWith("/settings_probe_a/config") && response.request().method() === "POST");
  await section("Settings a").getByRole("button", { name: "保存配置", exact: true }).click();
  const response = await saved; assert.ok(response.ok());
  const receipt = await response.json(); assert.ok(receipt.request_id);
  await active;
  await page.waitForFunction(() => !document.querySelector('.declared-form .config-form[aria-busy="true"]'), {}, { timeout: 30000 });
  await enter();
  assert.equal(await note().inputValue(), savedValue); report.savedValueSurvivesReload = true;
  await page.screenshot({ path: resolve(output, "saved.png") });
} catch (error) { report.error = String(error); process.exitCode = 1; }
finally { writeFileSync(resolve(output, "report.json"), JSON.stringify(report, null, 2)); console.log(JSON.stringify(report)); await browser.close(); }

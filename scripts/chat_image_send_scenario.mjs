/** 手工验收：真实 Chromium 和生产构建；控制 HTTP 完成顺序，不访问正式数据。
 * node scripts/chat_image_send_scenario.mjs --output /tmp/akashic-image-send
 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { parseArgs } from "node:util";
import { chromium } from "playwright-core";
import { startDesktopFixtureServer } from "./webui-performance/desktop-fixture-server.mjs";
import { desktopModels } from "./webui-performance/fixtures.mjs";

const repo = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const { values } = parseArgs({ options: {
  output: { type: "string", default: "/tmp/akashic-image-send" },
  browser: { type: "string", default: process.env.CHROME_BIN || "/opt/google/chrome/chrome" },
} });
const output = resolve(values.output);
const temporary = await mkdtemp(resolve(tmpdir(), "akashic-image-send-"));
let fixture;
let browser;
const checks = [];
try {
  await mkdir(output, { recursive: true, mode: 0o700 });
  const bundle = resolve(temporary, "bundle");
  const built = spawnSync(process.execPath, [resolve(repo, "node_modules/vite/bin/vite.js"),
    "build", "--config", "frontend/chat/vite.config.ts", "--outDir", bundle, "--emptyOutDir"],
  { cwd: repo, encoding: "utf8" });
  assert.equal(built.status, 0, built.stderr);
  fixture = await startDesktopFixtureServer(bundle, { historyCount: 20 });
  browser = await chromium.launch({ executablePath: values.browser, headless: true });
  const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
  const errors = [];
  page.on("pageerror", (error) => { errors.push(error.message); console.error("page-error", error.message); });
  const models = [];
  const uploads = [];
  const source = Buffer.concat([Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aRZkAAAAASUVORK5CYII=", "base64"), Buffer.alloc(2_978_300)]);
  const file = { name: "photo.png", mimeType: "image/png", buffer: source };
  await page.route("**/api/chat/models*", (route) => { models.push(route); });
  await page.route("**/api/chat/uploads*", (route) => { uploads.push(route); });
  await page.route("**/api/chat/artifacts/*", (route) => route.fulfill({ contentType: "image/png", body: source }));
  const editor = page.getByRole("textbox", { name: "消息", exact: true });
  const selectedFiles = page.locator('.composer-attachments .attachment-chip');
  const input = page.locator('input[type="file"]');
  const received = async () => (await (await fetch(`${fixture.origin}/__fixture/received`)).json()).items;
  const waitFor = async (predicate) => {
    for (let attempts = 0; !predicate(); attempts += 1) {
      assert(attempts < 100, "请求未到达");
      await page.evaluate(() => new Promise(requestAnimationFrame));
    }
  };
  await page.goto(fixture.origin, { waitUntil: "domcontentloaded" });
  await editor.fill("解释这张图");
  assert.equal(await page.locator(".chat-model-notice").count(), 0);
  assert.equal(await page.getByRole("button", { name: "发送消息", exact: true }).isEnabled(), true);
  checks.push("首次模型读取尚未结束，已就绪的聊天仍可发送，没有核对提示");

  await input.setInputFiles(file);
  await page.getByRole("button", { name: "发送消息", exact: true }).click();
  await waitFor(() => uploads.length === 1);
  assert.equal(await selectedFiles.count(), 0);
  assert.equal(await editor.inputValue(), "");
  assert.equal(await page.getByText("正在上传附件…", { exact: true }).isVisible(), true);
  assert.equal(await page.getByText("解释这张图", { exact: true }).isVisible(), true);
  assert.equal(await page.locator('.conversation img[src^="blob:"]').count(), 1);
  const raw = uploads[0].request().postDataBuffer();
  assert.equal(raw.length, source.length);
  assert.equal(createHash("sha256").update(raw).digest("hex"), createHash("sha256").update(source).digest("hex"));
  await page.screenshot({ path: resolve(output, "uploading.png") });
  await editor.fill("上传期间的下一份草稿");
  await editor.press("Enter");
  assert.equal(uploads.length, 1);
  assert.equal((await received()).filter((item) => item.type === "message.send").length, 0);
  const newFile = { ...file, name: "next.png" };
  await input.setInputFiles(newFile);
  assert.equal(await selectedFiles.count(), 1);
  checks.push("慢上传时立即展示原图预览、清空编辑器并允许继续写；原始字节不变，Enter 不重复提交");

  await uploads[0].fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({
    filename: file.name, artifact_id: "fixture-photo", kind: "image", media_type: "image/png",
    size_bytes: raw.length, sha256: createHash("sha256").update(raw).digest("hex"),
    upload_url: "/api/chat/artifacts/fixture-photo",
  }) });
  await page.locator('.conversation img[src="/api/chat/artifacts/fixture-photo"]').waitFor();
  assert.equal(await editor.inputValue(), "上传期间的下一份草稿");
  assert.equal(await selectedFiles.count(), 1);
  const sent = (await received()).filter((item) => item.type === "message.send");
  assert.equal(sent.length, 1);
  assert.deepEqual(sent[0].media, ["fixture-photo"]);
  assert.equal("model_runtime_id" in sent[0], false);
  await page.getByRole("button", { name: "中止回答", exact: true }).waitFor();
  checks.push("上传完成才进入等待回答；成功只清除本次附件，不清掉后来草稿，不带未经确认的模型覆盖");

  // 刷新回到独立新草稿，验证失败与取消；fixture 没有正式消息或附件写入。
  await page.evaluate(() => sessionStorage.clear());
  await page.reload({ waitUntil: "domcontentloaded" });
  await editor.fill("失败后可重试的文字");
  await input.setInputFiles(file);
  await page.getByRole("button", { name: "发送消息", exact: true }).click();
  await waitFor(() => uploads.length === 2);
  await editor.fill("上传期间后写的草稿");
  await uploads[1].fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ detail: "上传暂不可用" }) });
  await page.getByRole("button", { name: "发送消息", exact: true }).waitFor();
  assert.equal(await editor.inputValue(), "失败后可重试的文字\n\n上传期间后写的草稿");
  assert.equal(await selectedFiles.count(), 1);
  assert.equal(await page.getByText("上传暂不可用", { exact: true }).isVisible(), true);
  assert.equal(await page.locator('.conversation img[src^="blob:"]').count(), 0);
  await page.screenshot({ path: resolve(output, "failed.png") });
  checks.push("上传失败移除未发送气泡，并恢复文字与原附件，失败原因可见");

  await page.getByRole("button", { name: "发送消息", exact: true }).click();
  await waitFor(() => uploads.length === 3);
  await page.getByRole("button", { name: "取消上传", exact: true }).click();
  await page.getByRole("button", { name: "发送消息", exact: true }).waitFor();
  assert.equal(await selectedFiles.count(), 1);
  assert.equal(await editor.inputValue(), "失败后可重试的文字\n\n上传期间后写的草稿");
  assert.equal(uploads.length, 3);
  assert.equal((await received()).filter((item) => item.type === "message.send").length, 1);
  checks.push("取消上传只取消本次发送，原草稿可重试，也不误发 /stop");
  assert.deepEqual(errors, []);
  const report = { boundary: "生产构建 + 真实 Chromium/Blob/HTTP/WebSocket；受控服务，不代表线上或真实模型验收", passed: checks.length, checks };
  await writeFile(resolve(output, "report.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
} catch (error) {
  console.error(error);
  throw error;
} finally {
  await browser?.close();
  await fixture?.close();
  await rm(temporary, { recursive: true, force: true });
}

import { image, sourceMounts } from "./image.mjs";
import assert from "node:assert/strict";
import { spawn, execFileSync } from "node:child_process";
import { mkdtemp, mkdir, chmod, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { createServer } from "node:net";
import { randomUUID } from "node:crypto";
import { build } from "esbuild";
import { chromium } from "playwright-core";

const root = resolve(import.meta.dirname, "../..");
const artifacts = await mkdtemp(join(tmpdir(), "akashic-computer-e2e-"));
const container = `akashic-computer-e2e-${randomUUID().slice(0, 8)}`;

const executablePath = process.env.COMPUTER_E2E_CHROMIUM;
if (!executablePath) throw new Error("Set COMPUTER_E2E_CHROMIUM to an isolated test browser executable");
const docker = (...args) => execFileSync("docker", args, { encoding: "utf8", timeout: 60000, stdio: ["ignore", "pipe", "pipe"] }).trim();
let service, browser, started = false;
const checks = [];
const errors = [];

async function until(read, predicate, label, timeout = 20000) {
  const end = Date.now() + timeout;
  let value;
  while (Date.now() < end) {
    value = await read();
    if (predicate(value)) return value;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error(`${label}: ${JSON.stringify(value)}`);
}
function pass(name) { checks.push(name); console.log(`PASS ${name}`); }
async function freePort() {
  const server = createServer();
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  const port = server.address().port;
  await new Promise(resolve => server.close(resolve));
  return port;
}

try {
  // 1. 独立数据目录和 loopback 端口，生产 workspace 与用户浏览器均不参与。
  const data = join(artifacts, "data");
  await mkdir(data); await chmod(data, 0o777);
  docker("run", "-d", "--name", container, "--security-opt", "seccomp=unconfined", "--shm-size=512m",
    "-p", "127.0.0.1::8080", "-p", "127.0.0.1::6081", "-p", "127.0.0.1::19826", "-v", `${data}:/data`,
    ...sourceMounts, image);
  started = true;
  const ports = JSON.parse(docker("inspect", container, "--format", "{{json .NetworkSettings.Ports}}"));
  const gateway = `http://127.0.0.1:${ports["8080/tcp"][0].HostPort}`;
  const openCli = `http://127.0.0.1:${ports["19826/tcp"][0].HostPort}`;
  const stream = `http://127.0.0.1:${ports["6081/tcp"][0].HostPort}`;
  await build({entryPoints:[join(root,"plugins/shell_ui/web_module.js")],bundle:true,format:"esm",platform:"browser",
    outfile:join(artifacts,"shell.js"),define:{"process.env.NODE_ENV":'"production"'},logLevel:"silent"});
  const port = await freePort(), origin = `http://127.0.0.1:${port}`;
  service = spawn(join(root, ".venv/bin/python"), ["scripts/computer-e2e/serve.py"], {
    cwd: root, env: { ...process.env, PYTHONPATH: root, COMPUTER_E2E_ARTIFACTS: artifacts,
      COMPUTER_E2E_GATEWAY: gateway, COMPUTER_E2E_STREAM: stream, COMPUTER_E2E_PORT: String(port) },
    stdio: ["ignore", "ignore", "pipe"],
  });
  let serverErrors = "";
  service.stderr.on("data", chunk => { serverErrors = (serverErrors + chunk).slice(-16000); });
  await until(async () => {
    if (service.exitCode !== null) throw new Error(serverErrors);
    try { return (await fetch(origin)).ok; } catch (error) { if (error.cause?.code !== "ECONNREFUSED") throw error; return false; }
  }, Boolean, "Dashboard startup");
  const post = async (path, body) => {
    const response = await fetch(gateway + path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const value = await response.json();
    assert.equal(response.ok, true, JSON.stringify(value));
    return value;
  };
  const action = (code, timeoutMs = 60000, extra = {}) => post("/driver/run", {
    context: { session_id: "computer-e2e", turn_id: "turn", call_id: randomUUID() }, code, timeoutMs, ...extra });
  const textResult = result => result.content.filter(item => item.type === "text").map(item => item.text).join("\n");
  await post("/wake", {});
  // 真实 HTTP 页面触发自己的存储、键盘和鼠标行为。
  docker("cp", join(root, "scripts/computer-e2e/remote-page.mjs"), `${container}:/tmp/remote-page.mjs`);
  docker("exec", "-d", container, "node", "/tmp/remote-page.mjs");
  await until(async () => docker("exec", container, "node", "-e",
    "fetch('http://127.0.0.1:8989/main').then(r=>console.log(r.status)).catch(()=>console.log('starting'))"),
    value => value === "200", "Remote HTTP page startup");
  await action(`var main = await browser.tabs.new(); await main.goto('http://127.0.0.1:8989/main');
    var anonA = await agent.browsers.create(); var pageA = await anonA.tabs.new(); await pageA.goto('http://127.0.0.1:8989/a');
    var anonB = await agent.browsers.create(); var pageB = await anonB.tabs.new(); await pageB.goto('http://127.0.0.1:8989/b');
    console.log(await pageB.playwright.evaluate(() => document.querySelector('#identity').textContent));`);
  const identity = textResult(await action(`console.log(await pageA.playwright.evaluate(() => document.querySelector('#identity').textContent));
    console.log(await pageB.playwright.evaluate(() => document.querySelector('#identity').textContent));
    console.log(await main.playwright.evaluate(() => document.querySelector('#identity').textContent));`));
  assert.match(identity, /A/); assert.equal((identity.match(/null/g) ?? []).length, 2);
  pass("anonymous contexts and main profile keep separate identities");
  const ids = JSON.parse(textResult(await action(`console.log(JSON.stringify([anonA.browserId, anonB.browserId]));`)));
  assert.notEqual(ids[0], ids[1]);
  assert.match(textResult(await action(`console.log(await agent.browsers.get(anonA.browserId) === anonA);
    console.log(JSON.stringify(await agent.browsers.list()));`)), /true/);
  const failAction = async code => {
    const response = await fetch(gateway + "/driver/run", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ context: { session_id: "computer-e2e", turn_id: "turn", call_id: randomUUID() }, code }) });
    assert.equal(response.status, 500);
    const value = await response.json();
    assert.match(value.error, /bindings and browser pages were kept/);
  };
  await action("const e2eMarker = 1;");
  await failAction("const e2eMarker = 2;");
  await failAction("var progress = 'saved'; throw new Error('E2E script failed');");
  await failAction("await pageA.playwright.getByRole('button', {name:'missing button'}).click();");
  assert.match(textResult(await action(`console.log(progress); console.log(await agent.browsers.get(anonA.browserId) === anonA);
    console.log(await pageA.playwright.evaluate(() => document.querySelector('#identity').textContent));`)), /saved[\s\S]*true[\s\S]*A/);
  pass("stable anonymous IDs reselect the same instance and script errors preserve progress");


  const navigating = action("await main.goto('http://127.0.0.1:8989/slow');");
  await until(async () => JSON.parse(docker("exec", container, "node", "-e",
    "fetch('http://127.0.0.1:8989/status').then(r=>r.text()).then(console.log)")), v => v.slowPending > 0, "navigation was admitted");
  const abandoned = new AbortController(), abandonedId = randomUUID();
  const taking = fetch(gateway + "/control/take", {method:"POST", headers:{"Content-Type":"application/json"},
    body:JSON.stringify({id:abandonedId}), signal:abandoned.signal}).catch(error => { assert.equal(error.name, "AbortError"); });
  await until(async () => (await (await fetch(gateway + "/targets")).json()).control, Boolean, "pending takeover");
  abandoned.abort(); await taking;
  await until(async () => (await (await fetch(gateway + "/targets")).json()).control, v => !v, "abandoned takeover releases owner");
  await navigating;
  pass("disconnecting a pending takeover releases owner without repeating admitted navigation");

  // 2. 两份独立用户浏览器 Context 观看同一桌面，真实帧持续到达。
  browser = await chromium.launch({ executablePath, headless: true, args: ["--no-sandbox"] });
  const createPage = async (options = {}) => {
    const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, ...options });
    const page = await context.newPage();
    page.on("pageerror", error => errors.push(error.message));
    await page.goto(origin); await page.waitForFunction(() => window.e2eReady);
    return page;
  };
  const first = await createPage(), second = await createPage();
  const open = page => page.getByRole("button", { name: /^打开 Computer/ }).click();
  const connected = page => page.waitForFunction(() => document.querySelector(".computer-status")?.textContent?.startsWith("只读观看"));
  const videoReady = page => page.waitForFunction(() => {
    const frame = document.querySelector(".computer-stream");
    return frame?.contentWindow?.fps > 0 && frame.contentWindow.selkiesTransport.readyState === 1;
  });
  await open(first); await connected(first); await videoReady(first);
  await open(second); await connected(second); await videoReady(second); await videoReady(first);
  assert.equal((await (await fetch(gateway + "/targets")).json()).control, false);
  pass("two desktop viewers remain connected without taking input");
  await first.screenshot({ path: join(artifacts, "desktop-two-viewers.png") });
  const screen = first.locator(".computer-screen");
  const rect = await screen.boundingBox();
  await first.mouse.click(rect.x + rect.width / 2, rect.y + rect.height / 2);
  assert.equal((await (await fetch(gateway + "/targets")).json()).control, false);
  pass("viewing pointer input does not claim control");
  await first.getByRole("button", { name: "接管操作", exact: true }).click();
  await first.waitForFunction(() => document.querySelector(".computer-status").textContent.startsWith("你正在操作"));
  await videoReady(second);
  await second.getByRole("button", { name: "另一窗口正在操作" }).waitFor();
  assert.equal(await second.getByRole("button", { name: "另一窗口正在操作" }).isDisabled(), true);
  const conflicting = await fetch(gateway + "/control/take", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id:randomUUID()})});
  assert.equal(conflicting.status,500); assert.match((await conflicting.json()).error,/Another viewer/);
  let openCliFinished = false;
  const pendingOpenCli = fetch(openCli+"/health",{headers:{"x-opencli":"1"}}).then(response=>{
    openCliFinished=true; void response.body?.cancel(); return response.status;
  });
  let resumed = false;
  const paused = action("console.log('RESUMED');", 1000).then(result => { resumed = true; return result; });
  await until(async () => (await (await fetch(gateway + "/activity")).json()).active, Boolean, "Agent waiting for human");
  // 直接观察真实运行状态，等待超过 Agent 的执行期限，暂停不消耗执行时间。
  await first.waitForTimeout(1400);
  assert.equal(resumed, false);
  assert.equal(openCliFinished, false);
  await first.getByRole("button", { name: "释放操作" }).click();
  assert.match(textResult(await paused), /RESUMED/);
  assert.ok((await pendingOpenCli) < 500);
  pass("explicit takeover pauses Agent clock and OpenCLI, refuses another controller, and release resumes them");
  await connected(first); await videoReady(first); await videoReady(second);

  // 3. 标签展示真实匿名页面；只读、输入、键盘、关闭和释放均经生产路由。
  const select = (page, path) => page.getByRole("tab", { name: new RegExp(`匿名 .* · /${path}$`) }).click();
  await select(first, "a");
  await first.locator(".computer-browser-screen").waitFor();
  await connected(first);
  await select(second, "b"); await connected(second);
  const before = textResult(await action(`console.log(await pageA.playwright.evaluate(() => document.querySelector('#hit').textContent));`));
  assert.match(before, /CLICK/);
  await first.getByRole("button", { name: "接管操作", exact: true }).click();
  await first.waitForFunction(() => document.querySelector(".computer-status").textContent.startsWith("你正在操作"));
  const imageNode = first.locator(".computer-browser-screen");
  const size = await imageNode.evaluate(node => ({ width: node.naturalWidth, height: node.naturalHeight }));
  const bounds = await imageNode.boundingBox();
  const scale = Math.min(bounds.width / size.width, bounds.height / size.height);
  const point = (x, y) => ({ x: bounds.x + (bounds.width - size.width * scale) / 2 + x * scale,
    y: bounds.y + (bounds.height - size.height * scale) / 2 + y * scale });
  const input = point(70, 90);
  await first.mouse.click(input.x, input.y);
  await first.keyboard.type("Human");
  await first.keyboard.press("Backspace");
  await first.keyboard.type("n");
  // main 页面默认字体下 input 位于 h1 之后；用真实 DOM 回执判断坐标是否命中。
  const hit = point(230, 90);
  await first.mouse.click(hit.x, hit.y);
  await first.getByRole("button", { name: "剪贴板", exact: true }).click();
  await first.getByRole("textbox", { name: "剪贴板文字" }).fill(" Clipboard");
  // 粘贴前重新把焦点放回匿名输入框。
  await first.mouse.click(input.x, input.y);
  await first.getByRole("button", { name: "粘贴到 Computer" }).click();
  await first.getByText("已粘贴", { exact: true }).waitFor();
  await first.getByRole("button", { name: "释放操作" }).click();
  const changed = textResult(await action(`console.log(await pageA.playwright.evaluate(() => ({
    input: document.querySelector('#x').value, button: document.querySelector('#hit').textContent })));`));
  assert.match(changed, /Human Clipboard/); assert.match(changed, /CLICKED/);
  assert.match(textResult(await action(`console.log(await pageB.playwright.evaluate(() => document.querySelector('#x').value));`)), /''|""|^\s*$/);
  pass("anonymous pointer, keyboard, Backspace and acknowledged paste affect only selected page");
  await first.screenshot({ path: join(artifacts, "anonymous-tabs.png") });
  await action("await anonB.close();");
  await until(() => second.locator(".computer-status").textContent(), value => value.includes("目标已结束"), "closed context disappears");
  assert.equal(await second.getByRole("tab", { name: /匿名 .* · \/b$/ }).count(), 0);
  pass("closed anonymous contexts remove tabs and stop their display");

  // 4. 手机活动仅增加徽标；手动打开、关闭与聊天输入保持独立。
  const mobile = await createPage({ viewport: { width: 320, height: 740 }, isMobile: true, hasTouch: true });
  await action("console.log('MOBILE_NOTICE');");
  await mobile.locator(".conversation-tools-toggle.has-attention").waitFor();
  assert.equal(await mobile.locator(".conversation-tools").isVisible(), false);
  await mobile.frameLocator(".conversation-page-frame").getByRole("textbox", { name: "对话输入" }).fill("移动端继续输入");
  await mobile.screenshot({ path: join(artifacts, "mobile-activity-chat.png") });
  await open(mobile); await connected(mobile); await videoReady(mobile);
  await mobile.getByRole("button", { name: "关闭工具区" }).click();
  await action("console.log('AFTER_CLOSE');");
  await mobile.locator(".conversation-tools-toggle.has-attention").waitFor();
  assert.equal(await mobile.locator(".conversation-tools").isVisible(), false);
  pass("320px mobile activity and later activity preserve closed panel and chat input");

  await first.getByRole("button", { name: "剪贴板", exact: true }).click();
  await first.getByRole("button", { name: "接管操作", exact: true }).click();
  await first.waitForFunction(() => document.querySelector(".computer-status").textContent.startsWith("你正在操作"));
  await first.getByRole("button", { name: "关闭工具区" }).click();
  await until(async () => (await (await fetch(gateway + "/targets")).json()).control, value => !value, "closing panel releases control");
  pass("closing the panel releases its operation connection");
  await action("", 60000, { endTurn: true });
  const ended = await (await fetch(gateway + "/targets")).json();
  assert.deepEqual(ended.targets.map(target => target.id), ["desktop"]);
  pass("Turn end disposes anonymous targets without dropping the main profile");
  await action("var finalAnon = await agent.browsers.create(); var finalPage = await finalAnon.tabs.new();");
  const interrupted = await fetch(gateway + "/driver/run", {method:"POST", headers:{"Content-Type":"application/json"},
    body:JSON.stringify({context:{session_id:"computer-e2e",turn_id:"next",call_id:randomUUID()},
      code:"await new Promise(resolve => setTimeout(resolve, 10000));", timeoutMs:100})});
  assert.equal(interrupted.status,500);
  assert.match((await interrupted.json()).error,/JS bindings for this session were reset/);
  assert.deepEqual((await (await fetch(gateway+"/targets")).json()).targets.map(t=>t.id),["desktop"]);
  assert.match(textResult(await action("console.log(typeof progress);")),/undefined/);
  pass("call timeout resets bindings and disposes anonymous contexts after confirmed cleanup");
  for(const [label,viewport] of [["desktop",{width:1440,height:900}],["mobile",{width:320,height:740}]]) {
    const context=await browser.newContext({viewport}); const page=await context.newPage();
    page.on("pageerror",error=>errors.push(error.message)); await page.goto(origin+"/shell");
    const launcher=page.getByRole("button",{name:"功能设置",exact:true}); await launcher.waitFor();
    const bounds=await launcher.boundingBox(); assert.ok(bounds.width>=44 && bounds.height>=44 && bounds.x+bounds.width<=viewport.width+0.5, JSON.stringify(bounds));
    await launcher.click(); const dialog=page.getByRole("dialog",{name:"功能设置"}); await dialog.waitFor();
    await dialog.getByRole("button",{name:"模型",exact:true}).click();
    await dialog.getByRole("heading",{name:"模型设置入口"}).waitFor();
    assert.match(page.url(),/#settings\/models$/);
    await page.screenshot({path:join(artifacts,`settings-${label}.png`)});
    await page.keyboard.press("Escape"); await dialog.waitFor({state:"hidden"});
    await page.keyboard.press("Control+Comma"); await dialog.waitFor();
    await context.close();
  }
  pass("actual Shell exposes settings at desktop and 320px root with existing routes and shortcut");
  assert.deepEqual(errors, []);
  const logs = docker("logs", container);
  assert.doesNotMatch(logs, /Killing old client|UnhandledPromiseRejection/);
  await writeFile(join(artifacts, "computer.log"), logs);
  await writeFile(join(artifacts, "dashboard.log"), serverErrors);
  await writeFile(join(artifacts, "result.json"), JSON.stringify({ image, sourceMounted: sourceMounts.length > 0, checks, errors, artifacts }, null, 2));
  console.log(JSON.stringify({ checks: checks.length, artifacts }));
} finally {
  if (browser) for (const [index, context] of browser.contexts().entries())
    for (const page of context.pages())
      await page.screenshot({ path: join(artifacts, `final-${index}.png`) }).catch(error => console.log(`Screenshot: ${error.message}`));
  await browser?.close();
  service?.kill("SIGTERM");
  if (started) {
    try { await writeFile(join(artifacts, "final.log"), docker("logs", container)); }
    finally { docker("rm", "-f", container); }
  }
  console.log(`Artifacts: ${artifacts}`);
}

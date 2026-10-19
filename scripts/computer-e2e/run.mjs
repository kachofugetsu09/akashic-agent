import { image, sourceMounts } from "./image.mjs";
import assert from "node:assert/strict";
import { spawn, execFile, execFileSync } from "node:child_process";
import { promisify } from "node:util";
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
    "-e", "OPENCLI_AUTH_REFRESH_SITES=",
    "-p", `127.0.0.1:${await freePort()}:8080`, "-p", `127.0.0.1:${await freePort()}:6081`,
    "-p", `127.0.0.1:${await freePort()}:19826`, "-v", `${data}:/data`,
    ...sourceMounts, image, "/opt/computer/start.sh");
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
  // Navigation uses the same control owner, and reports real browser failures.
  const human = (await post("/view/open", {})).target;
  const owner = randomUUID();
  await post("/control/take", {id: owner, target: human});
  for (const [path, body] of [["navigate", {target: human, action: "go", url: "http://127.0.0.1:8989/main"}],
    ["open", {}], ["close", {target: human}]]) {
    const response = await fetch(origin + `/api/dashboard/computer/${path}`, {method: "POST",
      headers: {"content-type": "application/json"}, body: JSON.stringify(body)});
    assert.equal(response.status, 409, path); await response.body.cancel();
  }
  await post("/view/navigate", {control: owner, target: human, action: "go", url: "http://127.0.0.1:8989/main"});
  await post("/control/release", {id: owner});
  for (const [url, expected] of [["http://", 400], ["javascript:alert(1)", 400], ["http://127.0.0.1:1/", 502]]) {
    const response = await fetch(origin + "/api/dashboard/computer/navigate", {method: "POST",
      headers: {"content-type": "application/json"}, body: JSON.stringify({target: human, action: "go", url})});
    assert.equal(response.status, expected, url); await response.body.cancel();
  }
  pass("navigation checks control ownership and reports invalid URLs and failed loads");
  const humanNavigation = post("/view/navigate", {target: human, action: "go", url: "http://127.0.0.1:8989/held?human"});
  await until(async () => JSON.parse(docker("exec", container, "node", "-e",
    "fetch('http://127.0.0.1:8989/status').then(r=>r.text()).then(console.log)")), v => v.held === 1, "human navigation admitted");
  let agentFinished = false;
  const afterHuman = post("/driver/run", {context: {session_id: "human-navigation", turn_id: "one", call_id: randomUUID()},
    code: "console.log(await browser.tabs.list());"}).then(result => { agentFinished = true; return result; });
  await until(async () => (await (await fetch(gateway + "/activity")).json()).active, Boolean, "Agent waits for navigation");
  assert.equal(agentFinished, false);
  docker("exec", container, "node", "-e", "fetch('http://127.0.0.1:8989/release').then(r=>r.text()).then(console.log)");
  await humanNavigation; await afterHuman;
  await post("/driver/run", {context: {session_id: "human-navigation", turn_id: "one", call_id: randomUUID()}, endTurn: true});
  await post("/view/close", {target: human});
  pass("Agent work waits for already admitted human navigation");
  // Hold six real navigations at the HTTP server so every Session overlaps.
  const parallel = Promise.all(Array.from({length: 6}, (_, index) => post("/driver/run", {
    context: {session_id: `parallel-${index}`, turn_id: "one", call_id: randomUUID()},
    code: `var tab = await browser.tabs.new(); await tab.goto('http://127.0.0.1:8989/held?${index}');`,
  })));
  void parallel.catch(() => {});
  await until(async () => JSON.parse(docker("exec", container, "node", "-e",
    "fetch('http://127.0.0.1:8989/status').then(r=>r.text()).then(console.log)")),
    value => value.held === 6, "six admitted Sessions");
  assert.equal((await (await fetch(gateway + "/activity")).json()).active, true);
  docker("exec", container, "node", "-e", "fetch('http://127.0.0.1:8989/release').then(r=>r.text()).then(console.log)");
  await parallel;
  assert.equal((await (await fetch(gateway + "/activity")).json()).active, false);
  for (let index = 0; index < 6; index++) await post("/driver/run", {
    context: {session_id: `parallel-${index}`, turn_id: "one", call_id: randomUUID()}, endTurn: true,
  });
  pass("concurrent gateway calls settle without losing activity writes");
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

  // 2. 画面同一时刻只给一个窗口：后打开的接走画面，先打开的提示另一个窗口正在看。
  browser = await chromium.launch({ executablePath, headless: true, args: ["--no-sandbox"] });
  const createPage = async (options = {}) => {
    const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, ...options });
    const page = await context.newPage();
    page.on("pageerror", error => errors.push(error.message));
    await page.goto(origin); await page.waitForFunction(() => window.e2eReady);
    return page;
  };
  const first = await createPage(), second = await createPage();
  const open = async page => {
    await page.getByRole("button", { name: /^打开工作台/ }).click();
    if (!await page.locator(".computer-view").isVisible())
      await page.getByRole("button", { name: /^浏览器/ }).click();
  };
  // 连上且没人接管：Agent 空闲时不显示胶囊，Agent 回合进行中显示“Agent 正在使用浏览器”。
  const connected = page => page.waitForFunction(() => {
    const status = document.querySelector(".computer-status");
    return document.querySelector(".computer-connection-state")?.hidden && status
      && (status.hidden || status.textContent.startsWith("Agent 正在使用浏览器"));
  });
  const videoReady = page => page.waitForFunction(() => {
    const frame = document.querySelector(".computer-stream");
    return frame?.contentWindow?.fps > 0 && frame.contentWindow.selkiesTransport.readyState === 1;
  });
  await open(second); await connected(second); await videoReady(second);
  await open(first); await connected(first); await videoReady(first);
  await second.getByText(/另一个窗口正在看这个浏览器/).waitFor();
  assert.equal((await (await fetch(gateway + "/targets")).json()).control, false);
  pass("a second viewer takes the picture without taking input; the first explains why");
  await first.screenshot({ path: join(artifacts, "browser-viewer.png") });
  const screen = first.locator(".computer-screen");
  const rect = await screen.boundingBox();
  await first.mouse.click(rect.x + rect.width / 2, rect.y + rect.height / 2);
  assert.equal((await (await fetch(gateway + "/targets")).json()).control, false);
  pass("viewing pointer input does not claim control");
  await first.getByRole("button", { name: "接管", exact: true }).click();
  await first.waitForFunction(() => document.querySelector(".computer-status").textContent.startsWith("你正在操作"));
  await videoReady(first);
  await second.getByText(/另一个窗口正在看这个浏览器/).waitFor();
  const conflicting = await fetch(gateway + "/control/take", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id:randomUUID()})});
  assert.equal(conflicting.status,400); assert.match((await conflicting.json()).error,/Another viewer/);
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
  await first.getByRole("button", { name: "交还给 Agent" }).click();
  assert.match(textResult(await paused), /RESUMED/);
  assert.ok((await pendingOpenCli) < 500);
  pass("explicit takeover pauses Agent clock and OpenCLI, refuses another controller, and release resumes them");
  await connected(first); await videoReady(first);

  // 3. 标签展示真实匿名页面；Agent 回合进行中挡住输入，接管、键盘、关闭和释放均经生产路由。
  const options = (page, path) => page.locator(".computer-tab")
    .evaluateAll((nodes, suffix) => nodes.filter(node => node.textContent === suffix).map(node => node.dataset.id), `/${path}`);
  const select = async (page, path) => {
    const id = (await options(page, path))[0];
    await page.locator(`.computer-tab[data-id="${id}"]`).click();
    await until(async () => (await (await fetch(gateway + "/targets")).json()).targets.find(t => t.front)?.id,
      value => value === id, "selected page is visible");
  };
  await first.evaluate(() => { window.e2eStream = document.querySelector(".computer-stream").contentWindow.selkiesTransport; });
  await select(first, "a");
  await select(first, "b");
  await select(first, "a");
  assert.equal(await first.evaluate(() => window.e2eStream === document.querySelector(".computer-stream").contentWindow.selkiesTransport), true);
  const shot = await fetch(gateway + "/screenshot?quiet=1");
  assert.equal(shot.status, 200); assert.equal(shot.headers.get("content-type"), "image/png");
  await shot.arrayBuffer();
  pass("tab switches retain the video connection and legacy capture supports the visible anonymous page");
  await connected(first);
  const before = textResult(await action(`console.log(await pageA.playwright.evaluate(() => document.querySelector('#hit').textContent));`));
  assert.match(before, /CLICK/);
  const geometry = JSON.parse(textResult(await action(`console.log(JSON.stringify(await pageA.playwright.evaluate(() => {
    const center = id => { const r = document.querySelector(id).getBoundingClientRect(); return {x:r.x+r.width/2,y:r.y+r.height/2}; };
    return {width:innerWidth,height:innerHeight,input:center('#x'),hit:center('#hit')};
  })));`)));
  await action("await pageA.playwright.getByRole('textbox').click();");
  await action("await pageA.clipboard.writeText('BEFORE_READ_ONLY');");
  const clipboardBefore = textResult(await action("nodeRepl.write(await pageA.clipboard.readText());"));
  await first.evaluate(async () => {
    const socket = document.querySelector(".computer-stream").contentWindow.selkiesTransport;
    // Bypass the UI's disabled input handlers and challenge the actual proxy.
    socket.send("kd,120"); socket.send("ku,120"); socket.send("cw," + btoa("READ_ONLY"));
    socket.send(new Uint8Array([1, 2, 3]));
    await new Promise(resolve => {
      const ack = event => { if (event.data === "_gz,1") { socket.removeEventListener("message", ack); resolve(); } };
      socket.addEventListener("message", ack); socket.send("_gz,1");
    });
  });
  assert.equal(textResult(await action("nodeRepl.write(await pageA.playwright.getByRole('textbox').evaluate(el => el.value));")).trim(), "");
  assert.equal(textResult(await action("nodeRepl.write(await pageA.clipboard.readText());")), clipboardBefore);
  pass("while the Agent's turn is open the proxy drops raw keyboard, clipboard and binary messages");
  await first.getByRole("button", { name: "接管", exact: true }).click();
  await first.waitForFunction(() => document.querySelector(".computer-status").textContent.startsWith("你正在操作"));
  await videoReady(first);
  await first.screenshot({path: join(artifacts, "before-human-input.png")});
  const bounds = await first.locator(".computer-stream").boundingBox();
  const point = ({x, y}) => ({ x: bounds.x + x * bounds.width / geometry.width,
    y: bounds.y + y * bounds.height / geometry.height });
  const input = point(geometry.input);
  await first.mouse.click(input.x, input.y);
  await first.keyboard.type("Human");
  await first.keyboard.press("Backspace");
  await first.keyboard.type("n");
  // main 页面默认字体下 input 位于 h1 之后；用真实 DOM 回执判断坐标是否命中。
  const hit = point(geometry.hit);
  await first.mouse.click(hit.x, hit.y);
  await first.getByRole("button", { name: "剪贴板", exact: true }).click();
  await first.getByRole("textbox", { name: "剪贴板文字" }).fill(" Clipboard");
  // 粘贴前重新把焦点放回匿名输入框。
  await first.mouse.click(input.x, input.y);
  await first.getByRole("button", { name: "粘贴到浏览器" }).click();
  await first.getByText("已粘贴", { exact: true }).waitFor();
  await first.getByRole("button", {name: "关闭剪贴板", exact: true}).click();
  await first.getByRole("button", { name: "交还给 Agent" }).click();
  const changed = textResult(await action(`console.log(await pageA.playwright.evaluate(() => ({
    input: document.querySelector('#x').value, button: document.querySelector('#hit').textContent })));`));
  assert.match(changed, /Human Clipboard/); assert.match(changed, /CLICKED/);
  assert.match(textResult(await action(`console.log(await pageB.playwright.evaluate(() => document.querySelector('#x').value));`)), /''|""|^\s*$/);
  pass("anonymous pointer, keyboard, Backspace and acknowledged paste affect only selected page");
  await first.screenshot({ path: join(artifacts, "anonymous-tabs.png") });
  await select(first, "b");
  await action("await anonB.close();");
  await until(() => options(first, "b"), value => value.length === 0, "closed context disappears");
  assert.equal((await options(second, "b")).length, 0);
  assert.ok((await (await fetch(gateway + "/targets")).json()).targets.some(t => t.front && !t.anonymous));
  await videoReady(first);
  pass("closing the visible anonymous Context removes its tabs and shows a personal page on the same stream");

  await select(first, "a");
  const manual = (await (await fetch(gateway + "/targets")).json()).targets.find(t => t.front).id;
  await action("var later = await browser.tabs.new(); await later.goto('http://127.0.0.1:8989/later');");
  await first.locator(".computer-tab").filter({hasText: "/later"}).waitFor();
  assert.equal((await (await fetch(gateway + "/targets")).json()).targets.find(t => t.front).id, manual);
  const nextTurn = code => post("/driver/run", {context: {session_id: "follow", turn_id: "next", call_id: randomUUID()}, code});
  await nextTurn("var next = await browser.tabs.new(); await next.goto('http://127.0.0.1:8989/next');");
  await until(async () => (await (await fetch(gateway + "/targets")).json()).targets.find(t => t.front)?.title,
    title => title === "/next", "next Turn follows");
  await first.getByRole("button", {name: "收起工具区", exact: true}).click();
  const hiddenFront = (await (await fetch(gateway + "/targets")).json()).targets.find(t => t.front).id;
  await nextTurn("var hidden = await browser.tabs.new(); await hidden.goto('http://127.0.0.1:8989/hidden');");
  await second.locator(".computer-tab").filter({hasText: "/hidden"}).waitFor();
  assert.equal((await (await fetch(gateway + "/targets")).json()).targets.find(t => t.front).id, hiddenFront);
  pass("manual selection lasts through a Turn; hidden and superseded panels do not switch pages");
  await open(first); await connected(first); await select(first, "a");
  await action("await pageA.screenshot({});");
  const oldWidth = Number(textResult(await action("nodeRepl.write(await pageA.playwright.evaluate(() => innerWidth));")));
  await first.setViewportSize({width: 1000, height: 800});
  await until(async () => Number(textResult(await action("nodeRepl.write(await pageA.playwright.evaluate(() => innerWidth));"))),
    width => width !== oldWidth, "browser follows resized panel");
  const staleClick = await fetch(gateway + "/driver/run", {method: "POST", headers: {"content-type": "application/json"},
    body: JSON.stringify({context: {session_id: "computer-e2e", turn_id: "turn", call_id: randomUUID()},
      code: "await pageA.cua.click({x:20,y:20});"})});
  assert.equal(staleClick.status, 500); assert.match((await staleClick.json()).error, /take a new screenshot/);
  await action("await pageA.playwright.getByRole('button', {name:'CLICKED'}).click(); await pageA.screenshot({});");
  const clicks = Number(textResult(await action("nodeRepl.write(await pageA.playwright.evaluate(() => Number(document.querySelector('#hit').dataset.clicks)));")));
  const fresh = JSON.parse(textResult(await action("nodeRepl.write(JSON.stringify(await pageA.playwright.evaluate(() => {const r=document.querySelector('#hit').getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2};})));")));
  await action(`await pageA.cua.click(${JSON.stringify(fresh)});`);
  assert.equal(Number(textResult(await action("nodeRepl.write(await pageA.playwright.evaluate(() => Number(document.querySelector('#hit').dataset.clicks)));"))), clicks + 1);
  await first.screenshot({path: join(artifacts, "resized-browser.png")});
  pass("resize rejects stale coordinates; element actions and a fresh screenshot still work");

  // 4. 手机活动仅增加徽标；手动打开、关闭与聊天输入保持独立。
  const mobile = await createPage({ viewport: { width: 320, height: 740 }, isMobile: true, hasTouch: true });
  await action("console.log('MOBILE_NOTICE');");
  await mobile.locator(".conversation-tools-toggle.has-attention").waitFor();
  assert.equal(await mobile.locator(".conversation-tools").isVisible(), false);
  await mobile.locator(".conversation-page-chat").getByRole("textbox", { name: "对话输入" }).fill("移动端继续输入");
  await mobile.screenshot({ path: join(artifacts, "mobile-activity-chat.png") });
  await open(mobile); await connected(mobile); await videoReady(mobile);
  await mobile.getByRole("button", { name: "收起工具区", exact: true }).click();
  await action("console.log('AFTER_CLOSE');");
  await mobile.locator(".conversation-tools-toggle.has-attention").waitFor();
  assert.equal(await mobile.locator(".conversation-tools").isVisible(), false);
  pass("320px mobile activity and later activity preserve closed panel and chat input");

  // 未接管时剪贴板浮层直接给出接管入口。
  await first.getByRole("button", { name: "剪贴板", exact: true }).click();
  await first.getByRole("dialog", { name: "剪贴板" }).getByRole("button", { name: "接管操作", exact: true }).click();
  await first.waitForFunction(() => document.querySelector(".computer-status").textContent.startsWith("你正在操作"));
  await first.getByRole("button", { name: "收起工具区", exact: true }).click();
  await until(async () => (await (await fetch(gateway + "/targets")).json()).control, value => !value, "closing panel releases control");
  pass("closing the panel releases its operation connection");
  await action("", 60000, { endTurn: true });
  const ended = await (await fetch(gateway + "/targets")).json();
  assert.ok(ended.targets.length > 0 && ended.targets.every(target => !target.anonymous));
  pass("Turn end disposes anonymous targets without dropping the main profile");
  await action("var finalAnon = await agent.browsers.create(); var finalPage = await finalAnon.tabs.new();");
  const interrupted = await fetch(gateway + "/driver/run", {method:"POST", headers:{"Content-Type":"application/json"},
    body:JSON.stringify({context:{session_id:"computer-e2e",turn_id:"next",call_id:randomUUID()},
      code:"await new Promise(resolve => setTimeout(resolve, 10000));", timeoutMs:100})});
  assert.equal(interrupted.status,500);
  assert.match((await interrupted.json()).error,/JS bindings for this session were reset/);
  assert.ok((await (await fetch(gateway+"/targets")).json()).targets.every(t=>!t.anonymous));
  assert.match(textResult(await action("console.log(typeof progress);")),/undefined/);
  pass("call timeout resets bindings and disposes anonymous contexts after confirmed cleanup");
  await action("", 60000, { endTurn: true });
  await post("/driver/run", {context: {session_id: "follow", turn_id: "next", call_id: randomUUID()}, endTurn: true});
  // Idle users can start from a blank page without asking an Agent or taking control.
  for (const width of [1440, 320]) {
    const page = await createPage({viewport: {width, height: 900}, hasTouch: width === 320, isMobile: width === 320});
    await open(page); await connected(page); await videoReady(page);
    const address = page.getByRole("textbox", {name: "地址和搜索栏"});
    await page.getByRole("button", {name: "新建标签页", exact: true}).click();
    await page.waitForFunction(() => document.activeElement?.classList.contains("computer-address-input"));
    await page.locator(".computer-blank").waitFor();
    await page.screenshot({path: join(artifacts, `blank-${width}.png`)});
    for (const name of ["main", "second"]) {
      await address.fill(`http://127.0.0.1:8989/${name}`); await address.press("Enter");
      await until(() => page.locator('.computer-tab[aria-selected="true"]').textContent(), title => title === `/${name}`, "address navigation");
    }
    await page.getByRole("button", {name: "后退（Alt+←）"}).click();
    await until(() => page.locator('.computer-tab[aria-selected="true"]').textContent(), title => title === "/main", "back");
    await page.getByRole("button", {name: "前进（Alt+→）"}).click();
    await until(() => page.locator('.computer-tab[aria-selected="true"]').textContent(), title => title === "/second", "forward");
    await page.getByRole("button", {name: "刷新", exact: true}).click();
    await videoReady(page);
    const front = (await (await fetch(gateway + "/targets")).json()).targets.find(t => t.front);
    const point = JSON.parse(textResult(await action(`var direct = await browser.tabs.get(${JSON.stringify(front.id.slice(5))});
      nodeRepl.write(JSON.stringify(await direct.playwright.evaluate(() => {const r=document.querySelector('#hit').getBoundingClientRect(); return {x:r.x+r.width/2,y:r.y+r.height/2,width:innerWidth,height:innerHeight};})));`)));
    await action("", 60000, {endTurn: true});
    await page.waitForFunction(() => document.querySelector(".computer-status")?.hidden === true);
    const bounds = await page.locator(".computer-stream").boundingBox();
    await page.mouse.click(bounds.x + point.x * bounds.width / point.width, bounds.y + point.y * bounds.height / point.height);
    assert.match(textResult(await action(`nodeRepl.write(await direct.playwright.domSnapshot());`)), /CLICKED/);
    await action("", 60000, {endTurn: true});
    const geometry = await page.locator(".computer-view").evaluate(el => ({width: el.clientWidth, scroll: el.scrollWidth,
      controls: [...el.querySelectorAll(".computer-icon-button, .computer-address-input")].map(e => e.getBoundingClientRect()).filter(r => r.width > 0).map(r => [r.width, r.height])}));
    assert.ok(geometry.scroll <= geometry.width, JSON.stringify(geometry));
    if (width === 320) assert.ok(geometry.controls.every(([w, h]) => w >= 44 && h >= 44), JSON.stringify(geometry));
    await page.screenshot({path: join(artifacts, `navigation-${width}.png`)});
    await page.locator('.computer-tab[aria-selected="true"]').getByRole("button", {name: "关闭 /second"}).click();
    await until(async () => (await (await fetch(gateway + "/targets")).json()).targets, targets => !targets.some(t => t.id === front.id), "close tab");
    await page.context().close();
  }
  pass("desktop and 320px touch users navigate, browse history, click and close from a blank tab");
  await action("var saved = await browser.tabs.new(); await saved.goto('http://127.0.0.1:8989/remember');");
  const beforeRestart = textResult(await action("nodeRepl.write(await saved.playwright.domSnapshot());"));
  assert.match(beforeRestart, /PROFILE/); assert.match(beforeRestart, /review=PROFILE/);
  await promisify(execFile)("docker", ["restart", container], {timeout: 60000});
  await until(async () => docker("exec", container, "node", "-e",
    "fetch('http://127.0.0.1:8080/health').then(r=>console.log(r.status)).catch(()=>console.log('starting'))"),
    value => value === "200", "restarted Computer health");
  docker("exec", "-d", container, "node", "/tmp/remote-page.mjs");
  await until(async () => docker("exec", container, "node", "-e",
    "fetch('http://127.0.0.1:8989/main').then(r=>console.log(r.status)).catch(()=>console.log('starting'))"),
    value => value === "200", "restarted remote HTTP page");
  await action("var restored = await browser.tabs.new(); await restored.goto('http://127.0.0.1:8989/main');");
  const retained = textResult(await action("nodeRepl.write(await restored.playwright.domSnapshot());"));
  assert.match(retained, /PROFILE/); assert.match(retained, /review=PROFILE/);
  pass("runtime restart retains personal profile storage and cookies");
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
  assert.doesNotMatch(logs, /UnhandledPromiseRejection|ENOENT.*activity/);
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

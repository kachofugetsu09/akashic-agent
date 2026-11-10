// 真实浏览器与 Models HTTP/SQLite：取消失效型号后更新连接，失败原样保留。
import assert from "node:assert/strict";
import {spawn} from "node:child_process";
import {once} from "node:events";
import {mkdtemp} from "node:fs/promises";
import {createServer} from "node:net";
import {tmpdir} from "node:os";
import {join} from "node:path";
import {fileURLToPath} from "node:url";
import {chromium} from "playwright-core";

const root = fileURLToPath(new URL("../", import.meta.url));
const frontend = join(root, "frontend");
const output = await mkdtemp(join(tmpdir(), "akashic-model-auth-ui-"));
const listener = createServer();
listener.listen(0, "127.0.0.1");
await once(listener, "listening");
const port = listener.address().port;
await new Promise(resolve => listener.close(resolve));
const base = `http://127.0.0.1:${port}`;
const service = spawn(process.env.PYTHON ?? join(root, ".venv/bin/python"), [
  join(root, "scripts/model_auth_ui_fixture.py"), "--port", String(port), "--frontend", frontend, "--connection-edit",
], {stdio: ["ignore", "ignore", "pipe"]});
let log = "", browser;

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
  const catalog = () => get("/api/dashboard/models/catalog");
  // 2. 非本连接的取消清单在边界拒绝，不写入任何配置。
  const before=await catalog();
  const rejected = await fetch(base + "/api/dashboard/models/command", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({
    type: "update_connection", expected_revision: before.revision, connection_id: "magpie", name: "Magpie fixture", auth_identity: "api:magpie", remove_model_ids: ["unknown-model"],
  })});
  assert.equal(rejected.status, 422);
  assert.deepEqual(await catalog(), before);
  // 3. 原错误、保留模型认证失败、恢复后原子提交与重新打开。

  await page.goto(base);
  await page.getByRole('button',{name:/Magpie fixture/}).click();
  // 展开连接详情时，Base URL 必须自动填充已保存的完整地址，而不是清空
  const endpointInput = page.getByRole('textbox',{name:'Base URL',exact:true});
  assert.equal(await endpointInput.inputValue(), `${base}/upstream/v1`, 'saved Base URL is pre-populated on edit');
  assert.equal(await endpointInput.getAttribute('autocomplete'), 'url', 'Base URL input uses autocomplete="url"');
  await page.screenshot({path:join(output, 'editor-prefilled.png'),fullPage:true});
  await page.getByRole('textbox',{name:'API Key',exact:true}).fill('fixture-key-replacement');
  await page.getByRole('button',{name:'保存',exact:true}).click();
  await page.getByRole('alert').filter({hasText:'old-removed'}).waitFor();
  assert.deepEqual(await catalog(),before,'failed validation preserves original catalog');
  await page.getByRole('checkbox',{name:'使用 old-removed',exact:true}).uncheck();
  await page.getByRole('textbox',{name:'API Key',exact:true}).fill('invalid-key');
  await page.getByRole('button',{name:'保存',exact:true}).click();
  await page.getByRole('alert').filter({hasText:'invalid fixture key'}).waitFor();
  assert.deepEqual(await catalog(),before,'failed retained model check must not remove old model');
  await page.getByRole('textbox',{name:'API Key',exact:true}).fill('fixture-key-replacement');
  await page.getByRole('button',{name:'保存',exact:true}).click();
  await page.locator('[data-save]').waitFor({state:'hidden'});
  const after=await catalog();
  assert.equal(after.revision,before.revision+1,'one atomic revision');
  assert.deepEqual(after.models.map(x=>x.model),['still-valid']);
  // 重新打开编辑：Base URL 仍然自动回填，且仅修改连接名称（保持预填 Base URL 不变）保存时不触发模型调用
  await page.getByRole('button',{name:/Magpie fixture/}).click();
  assert.equal(await endpointInput.inputValue(), `${base}/upstream/v1`, 'reopened editor still pre-populates saved Base URL');
  assert.equal(await page.getByRole('checkbox',{name:'使用 old-removed',exact:true}).count(),0);
  assert.equal(await page.getByRole('checkbox',{name:'使用 still-valid',exact:true}).isChecked(),true);
  const callsBeforeRename = (await get('/completion_calls')).count;
  await page.getByRole('textbox',{name:'连接名称',exact:true}).fill('Magpie 内网');
  await page.getByRole('button',{name:'保存',exact:true}).click();
  await page.locator('[data-save]').waitFor({state:'hidden'});
  const callsAfterRename = (await get('/completion_calls')).count;
  assert.equal(callsAfterRename, callsBeforeRename, 'renaming without changing pre-populated Base URL must not probe models');
  const receipts = await get('/receipts');
  assert.equal(receipts.at(-1).endpoint, null, 'unchanged pre-populated Base URL sends endpoint: null');
  const renamed = await catalog();
  const magpieConn = renamed.connections.find(c => c.id === 'magpie');
  assert.equal(magpieConn.name, 'Magpie 内网');
  assert.equal(magpieConn.endpoint, `${base}/upstream/v1`);

  // 4. Gemini 原生连接与 OpenCode Go 展开编辑时同样自动填充上次保存的 Base URL
  await page.getByRole('button',{name:/Gemini 网关/}).click();
  assert.equal(await endpointInput.inputValue(), `${base}/upstream/gemini/v1beta`, 'Gemini saved Base URL is pre-populated on edit');
  await page.getByRole('button',{name:'取消',exact:true}).click();

  await page.getByRole('button',{name:/OpenCode Go/}).click();
  assert.equal(await endpointInput.inputValue(), `${base}/upstream/opencode/v1`, 'OpenCode Go saved Base URL is pre-populated on edit');
  await page.getByRole('button',{name:'取消',exact:true}).click();

  console.log('PASS: Base URL pre-populated across OpenAI-compatible/Gemini/OpenCode; invalid removal rejected; failed validation preserves catalog; retry commits once; rename with pre-populated URL skips model probe');
  await page.screenshot({path:join(output, 'after.png'),fullPage:true});
  assert.deepEqual(errors, []);
} finally {
  await browser?.close();
  service.kill("SIGTERM");
  await once(service, "exit");
}

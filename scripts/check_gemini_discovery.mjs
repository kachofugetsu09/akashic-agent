/** 在一次性 HTTP 夹具上操作真实 Models 页面，验证原生探测和保存边界。 */
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { parseArgs } from 'node:util';
import { chromium } from 'playwright-core';

const {values} = parseArgs({options: {
  url: {type: 'string', default: 'http://127.0.0.1:2317'},
  browser: {type: 'string', default: process.env.CHROME_BIN ?? '/opt/google/chrome/chrome'},
  output: {type: 'string', default: '/tmp/akashic-gemini-discovery'},
}});
const origin = new URL(values.url);
assert(['127.0.0.1', 'localhost'].includes(origin.hostname), '只允许一次性本地夹具');
await mkdir(values.output, {recursive: true, mode: 0o700});
const browser = await chromium.launch({executablePath: values.browser, headless: true});
const page = await browser.newPage({viewport: {width: 1280, height: 900}});
page.setDefaultTimeout(15000);
const report = {passed: [], errors: []};
page.on('pageerror', error => report.errors.push(error.message));
page.on('dialog', dialog => dialog.accept());
const form = () => page.locator('.settings-dialog-form');
const sheet = () => page.locator('.settings-sheet-scrim');
const button = name => page.getByRole('button', {name, exact: true});

/** 只控制外部服务；配置写入始终由页面发起。 */
async function control(action) {
  const response = await fetch(`${values.url}/fixture/${action}`, {method: 'POST'});
  assert(response.ok, `fixture ${action}`);
}
async function state() {
  const response = await fetch(`${values.url}/fixture/state`);
  assert(response.ok);
  return response.json();
}
async function open(name) {
  const template = page.getByRole('button', {name: /^Gemini 原生 API/});
  await page.getByRole('button', {name: /^(添加连接|选择连接方式)$/}).waitFor();
  if (!await template.isVisible()) await page.getByRole('button', {name: /^(添加连接|选择连接方式)$/}).click();
  await template.click();
  await form().getByLabel('连接名称', {exact: true}).fill(name);
  await form().getByLabel('Base URL', {exact: true}).fill(`${values.url}/provider`);
  await form().getByLabel('API Key', {exact: true}).fill('fixture-key');
}
async function pick(names) {
  await button('探测可用模型').click();
  await sheet().waitFor();
  assert.equal(await sheet().locator('.settings-sheet-row').count(), 3, '分页目录完整显示');
  for (const name of names) {
    const row = sheet().locator('.settings-sheet-row').filter({has: page.getByText(name, {exact: true})});
    await row.getByRole('checkbox').check();
  }
  await sheet().getByRole('button', {name: `开放所选 (${names.length})`, exact: true}).click();
}
async function save() {
  await button('保存连接').click();
  await form().waitFor({state: 'hidden'});
}
async function screenshot(name) {
  await page.screenshot({path: `${values.output}/${name}.png`, mask: [page.locator('input[type=password]')]});
}

try {
  // 1. 探测和取消不提交配置；改动连接使等待中的探测失效。
  await page.goto(values.url);
  await open('原生探测');
  const before = await state();
  await button('探测可用模型').click();
  await sheet().waitFor();
  await sheet().getByRole('button', {name: '取消', exact: true}).click();
  assert.equal((await state()).digest, before.digest);
  report.passed.push('分页探测与取消不保存');
  await control('hold');
  await button('探测可用模型').click();
  await control('entered');
  const aborted = page.waitForEvent('requestfailed', {predicate: request => request.url().endsWith('/discover')});
  await form().getByLabel('API Key', {exact: true}).fill('changed-key');
  await aborted;
  await control('release');
  assert.equal(await sheet().count(), 0);
  assert.equal(await form().locator('[data-footer]').isVisible(), false);
  assert.equal((await state()).digest, before.digest);
  report.passed.push('修改凭据取消探测并丢弃迟到结果');
  await form().getByLabel('API Key', {exact: true}).fill('fixture-key');
  await pick(['gemini-good', 'gemini-extra']);
  await screenshot('desktop-picked');

  // 2. 首项验证失败保留表单且不保存；重试成功后两项均存在。
  await control('fail');
  await button('保存连接').click();
  await form().locator('[data-error]').waitFor();
  assert.equal((await state()).digest, before.digest);
  assert.equal(await form().getByLabel('API Key', {exact: true}).inputValue(), 'fixture-key');
  report.passed.push('验证失败无配置提交且表单保留');
  await control('recover');
  await save();
  const saved = await state();
  assert.equal(saved.connections, 1);
  assert.deepEqual(saved.models.sort(), ['gemini-extra', 'gemini-good']);
  assert(saved.calls.every(path => path.startsWith('/provider/v1beta/')), '目录与生成共用默认版本');
  await page.reload();
  await page.getByText('原生探测', {exact: true}).waitFor();
  assert.equal((await state()).digest, saved.digest);
  report.passed.push('根 URL 原生验证、批量保存和重载');

  // 3. 后续型号验证失败准确报告部分完成，不撤销已保存项。
  await open('部分完成');
  await pick(['gemini-good', 'gemini-denied']);
  await save();
  await page.getByText(/已开放 1\/2 个模型。未开放：gemini-denied/).waitFor();
  const partial = await state();
  assert.equal(partial.connections, 2);
  assert.equal(partial.models.length, 3);
  assert(!partial.models.includes('gemini-denied'));
  report.passed.push('部分失败保留成功项并报告失败型号');

  // 4. 手动路径要求明确确认；窄屏目录控件可见，关闭页面取消工作。
  await open('手动路径');
  await button('手动填写型号').click();
  await form().getByLabel('模型名称', {exact: true}).fill('gemini-extra');
  await button('保存连接').click();
  assert.equal((await state()).digest, partial.digest);
  await form().getByRole('checkbox').check();
  await save();
  assert.equal((await state()).connections, 3);
  report.passed.push('手动型号需要确认和真实验证');
  await page.setViewportSize({width: 320, height: 850});
  await open('窄屏探测');
  await button('探测可用模型').click();
  await sheet().waitFor();
  const bounds = await sheet().locator('.settings-sheet').boundingBox();
  assert(bounds.x >= -1 && bounds.x + bounds.width <= 321, '窄屏选择层不横向溢出');
  await screenshot('mobile-directory');
  await sheet().getByRole('button', {name: '取消', exact: true}).click();
  await control('hold');
  await button('探测可用模型').click();
  await control('entered');
  const disposed = page.waitForEvent('requestfailed', {predicate: request => request.url().endsWith('/discover')});
  const finalState = await state();
  await page.evaluate(() => window.disposeModels());
  await disposed;
  await control('release');
  assert.equal(await page.locator('#host').innerText(), '');
  assert.equal((await state()).digest, finalState.digest);
  report.passed.push('320px 目录选择和卸载取消');
  assert.deepEqual(report.errors, []);
  console.log(JSON.stringify(report));
} finally {
  await control('release');
  await writeFile(`${values.output}/report.json`, JSON.stringify(report, null, 2));
  await browser.close();
}

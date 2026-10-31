/** 对照真实选择器的几何、焦点、选择、收藏和窗口重定位；只用于隔离 Runtime。 */
import assert from "node:assert/strict";
import { chromium } from "playwright-core";
import { mkdirSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";

const { values } = parseArgs({ options: { baseline: { type: "string" }, candidate: { type: "string" },
  browser: { type: "string" }, output: { type: "string" } } });
for (const name of ['baseline', 'candidate', 'browser', 'output']) if (!values[name]) throw new Error(`需要 --${name}`);
mkdirSync(values.output);
const browser = await chromium.launch({ executablePath: values.browser });
const report = { complete: false, browser: browser.version(), samples: [] };
let currentPage;

/** 等待窗口重定位后的现存动画完成；取消的旧动画不代表界面失败。 */
async function snapshot(page, panel, path) {
  await page.evaluate(() => document.fonts.ready);
  await page.evaluate(() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done))));
  await page.waitForFunction(el => el.getAnimations({ subtree: true }).every(animation => animation.playState !== 'running' && !animation.pending), await panel.elementHandle());
  const geometry = await panel.evaluate(el => {
    const rect = el.getBoundingClientRect();
    return { x: rect.x, y: rect.y, width: rect.width, height: rect.height,
      origin: getComputedStyle(el).transformOrigin,
      rows: [...el.querySelectorAll('.model-picker__row, .model-picker__group-title, .model-effort__option')].map(row => {
        const box = row.getBoundingClientRect();
        return { text: row.textContent, x: box.x - rect.x, y: box.y - rect.y, width: box.width, height: box.height };
      }) };
  });
  const viewport = page.viewportSize();
  assert.ok(geometry.x >= 0 && geometry.y >= 0 && geometry.x + geometry.width <= viewport.width + 1
    && geometry.y + geometry.height <= viewport.height + 1, '面板超出视口');
  return { geometry, pixels: await panel.screenshot({ path }) };
}

try {
  for (const [width, height] of [[1440, 900], [390, 900], [390, 380]]) {
    let baseline;
    for (const kind of ['baseline', 'candidate']) {
      const context = await browser.newContext({ viewport: { width, height }, hasTouch: width === 390,
        isMobile: width === 390, serviceWorkers: 'block' });
      await context.addInitScript(() => localStorage.setItem('akashic.onboarding.seen', '1'));
      const page = await context.newPage(), errors = []; currentPage = page;
      page.on('pageerror', error => errors.push(error.message));
      const response = page.waitForResponse(r => r.url().endsWith('/api/chat/plugin-ui/query')
        && r.request().postDataJSON()?.method === 'chat.models' && r.status() === 200);
      await page.goto(values[kind]); await page.getByRole('textbox', { name: '消息', exact: true }).click();
      const catalog = await (await response).json(), names = catalog.runtimes.map(model => model.model).sort();
      assert.ok(names.length >= 43, '请使用至少 43 个模型的隔离目录');
      const trigger = page.locator('.model-picker__trigger'), panel = page.getByRole('dialog', { name: '选择模型', exact: true });
      const search = panel.getByRole('textbox', { name: '筛选模型', exact: true });
      const row = { width, height, kind, success: false }; report.samples.push(row);
      // 1. 首次与重新打开都显示完整目录；桌面聚焦搜索，触屏聚焦当前选中行。
      await trigger.click(); await checkNames(names);
      await page.waitForFunction(coarse => coarse
        ? document.activeElement?.matches('.model-picker__option[aria-pressed="true"]')
        : document.activeElement?.matches('.model-picker__search input'), width === 390);
      const first = await snapshot(page, panel, `${values.output}/${width}-${height}-${kind}.png`);
      await search.fill('picker/model-0042'); await checkNames(['picker/model-0042']);
      await search.press('ArrowDown'); await page.locator(':focus').press('Enter');
      await panel.waitFor({ state: 'hidden' }); assert.ok((await trigger.getAttribute('title')).includes('picker/model-0042'));
      assert.ok(await trigger.evaluate(el => el === document.activeElement));
      await trigger.click(); await checkNames(names);
      // 2. 收藏只是本地列表投影；来源筛选和取消收藏恢复完整集合。
      await panel.getByRole('button', { name: '收藏 picker/model-0042', exact: true }).click();
      await checkNames([...names, 'picker/model-0042'].sort());
      await panel.getByRole('button', { name: '收藏', exact: true }).click(); await checkNames(['picker/model-0042']);
      await panel.getByRole('button', { name: '取消收藏 picker/model-0042', exact: true }).click(); await checkNames([]);
      await panel.getByText('还没有收藏；在模型行右侧点星标收藏', { exact: true }).waitFor();
      await panel.getByRole('button', { name: '模型来源 1', exact: true }).click();
      await checkNames(catalog.runtimes.filter(model => model.sourceName === '模型来源 1').map(model => model.model).sort());
      await panel.getByRole('button', { name: '全部来源', exact: true }).click(); await checkNames(names);
      await search.press('Escape'); await panel.waitFor({ state: 'hidden' }); assert.ok(await trigger.evaluate(el => el === document.activeElement));
      // 3. 独立的思考强度选择与焦点归还沿原路径执行。
      const effort = page.locator('.model-effort__trigger'), effortPanel = page.locator('.model-effort__panel');
      await effort.click(); await effortPanel.getByRole('option', { name: '高', exact: true }).click();
      await effortPanel.waitFor({ state: 'hidden' }); assert.ok((await effort.textContent()).includes('高'));
      assert.ok(await effort.evaluate(el => el === document.activeElement));
      await effort.click(); await effortPanel.waitFor();
      const strength = await snapshot(page, effortPanel, `${values.output}/${width}-${height}-${kind}-effort.png`);
      await effort.press('Escape'); await effortPanel.waitFor({ state: 'hidden' });
      // 4. 面板保持打开时改变窗口，检查重定位与窄屏遮罩关闭。
      await trigger.click(); await checkNames(names);
      await page.setViewportSize({ width: width === 1440 ? 1100 : 460, height: height === 380 ? 700 : height });
      const resized = await snapshot(page, panel, `${values.output}/${width}-${height}-${kind}-resized.png`);
      const scrim = page.locator('.model-picker .picker-sheet-scrim');
      if (await scrim.isVisible()) await scrim.click({ position: { x: 5, y: 5 } });
      else await search.press('Escape');
      await panel.waitFor({ state: 'hidden' });
      const shots = { first, strength, resized };
      row.geometry = Object.fromEntries(Object.entries(shots).map(([name, shot]) => [name, shot.geometry]));
      if (kind === 'baseline') baseline = shots;
      else {
        row.pixelsEqual = {};
        for (const name of Object.keys(shots)) {
          assert.deepEqual(shots[name].geometry, baseline[name].geometry, `${name} 几何改变`);
          // 圆角抗锯齿可能有单色阶差异；原图和相等性保留，交付前逐项检查。
          row.pixelsEqual[name] = shots[name].pixels.equals(baseline[name].pixels);
        }
      }
      assert.equal(errors.length, 0, errors.join('\n')); row.success = true;
      await context.close();
      async function checkNames(expected) {
        await panel.waitFor();
        await page.waitForFunction(names => JSON.stringify([...document.querySelectorAll('.model-picker__row .model-picker__copy strong')]
          .map(el => el.textContent).sort()) === JSON.stringify(names), expected);
      }
    }
  }
  report.complete = true;
} catch (error) {
  report.error = String(error); process.exitCode = 1;
  if (currentPage && !currentPage.isClosed()) await currentPage.screenshot({ path: `${values.output}/failure.png` });
} finally { writeFileSync(`${values.output}/report.json`, JSON.stringify(report, null, 2)); await browser.close(); }

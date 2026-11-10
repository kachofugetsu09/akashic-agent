/** 在真实局部卸载期间核对目录、草稿、焦点与窄屏读取。 */
import assert from 'node:assert/strict';
import { writeFile } from 'node:fs/promises';
import { chromium } from 'playwright-core';

const [url, output] = process.argv.slice(2);
const browser = await chromium.launch({executablePath: '/usr/bin/chromium', headless: true});
const errors = [];
let page;
try {
  page = await browser.newPage({viewport: {width: 1100, height: 800}});
  page.setDefaultTimeout(15000);
  page.on('console', message => { if (message.type() === 'error') console.error(message.text()); });
  page.on('pageerror', error => errors.push(String(error)));
  await page.goto(`${url}/dashboard/#e2e-notes`);
  const draft = page.getByRole('textbox', {name: 'E2E draft'});
  await draft.waitFor();
  await draft.fill('retained draft');
  await draft.focus();
  const catalog = await page.locator('#root').getAttribute('data-akashic-catalog');
  assert(catalog);
  const stateResponse = page.waitForResponse(response => response.url().endsWith('/api/chat/web-ui/state'));
  await page.evaluate(() => window.dispatchEvent(new CustomEvent('akashic:configuration-submitted')));
  const state = await (await stateResponse).json();
  assert.equal(state.updating, true, '真实卸载仍在等待目标 owner');
  assert.equal(state.catalogId, catalog);
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  assert.equal(await draft.inputValue(), 'retained draft');
  assert(await draft.evaluate(element => element === document.activeElement));
  assert.equal(await page.locator('.web-host-stale,.web-host-entry-error').count(), 0);
  await page.screenshot({path: `${output}/settings-uninstall-desktop.png`});
  await page.setViewportSize({width: 320, height: 800});
  await page.addStyleTag({content: 'html {font-size: 200%;}'});
  assert.equal(await draft.inputValue(), 'retained draft');
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await page.screenshot({path: `${output}/settings-uninstall-narrow-200.png`});
  await page.reload();
  await draft.waitFor();
  assert.equal(await page.locator('.web-host-entry-error').count(), 0);
  assert.equal(await page.locator('#root').getAttribute('data-akashic-catalog'), catalog);
  assert.deepEqual(errors, []);
  await writeFile(`${output}/browser-result.json`, JSON.stringify({status: 'passed', catalog,
    draftPreserved: true, focusPreserved: true, narrow200Percent: true, freshBootstrap: true}, null, 2));
} catch (error) {
  if (page) {
    await writeFile(`${output}/browser-failure.html`, await page.content());
    await page.screenshot({path: `${output}/browser-failure.png`});
  }
  throw error;
} finally {
  await browser.close();
}

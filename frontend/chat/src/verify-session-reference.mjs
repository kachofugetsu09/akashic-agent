/** 会话引用条的验收：chip 出现/删除、重复与超限提示、拖拽结束复位高亮。
 *
 * 用真实组件 + 真实 DOM 事件，不手拼 HTML。拖拽用构造的 DataTransfer 走真实
 * dragstart/dragover/drop 路径，与浏览器里用户操作走同一段代码。
 */
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, symlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'vite';
import { chromium } from 'playwright-core';

const repo = process.env.REVIEW_SOURCE_ROOT ?? resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
const root = mkdtempSync(resolve(process.env.EVIDENCE_DIR ?? tmpdir(), 'akashic-refs-'));
symlinkSync(resolve(repo, 'node_modules'), root + '/node_modules', 'dir');
writeFileSync(root + '/index.html', '<html><body><div id="root"></div><script type="module" src="./fixture.tsx"></script></body></html>');
writeFileSync(root + '/fixture.tsx', `
import { createRoot } from 'react-dom/client';
import { useRef } from 'react';
import { ComposerReferences, useReferenceDrop } from ${JSON.stringify(resolve(repo, 'frontend/chat/src/composer-references-ui.tsx'))};
import { addReference, clearReferences, composerReferences } from ${JSON.stringify(resolve(repo, 'frontend/chat/src/composer-references.ts'))};
import ${JSON.stringify(resolve(repo, 'frontend/chat/src/message-view.css'))};

function App() {
  const ref = useRef<HTMLFormElement | null>(null);
  const active = useReferenceDrop(ref, 'draft-1', true);
  return (
    <form ref={ref} className={'composer' + (active ? ' is-reference-target' : '')} id="drop">
      <ComposerReferences draftKey="draft-1" />
      <textarea id="text" />
    </form>
  );
}
createRoot(document.getElementById('root')!).render(<App />);
(window as any).refs = { addReference, clearReferences, composerReferences };
`);

await build({
  configFile: resolve(repo, 'frontend/chat/vite.harness.config.ts'), root, logLevel: 'warn',
  css: { postcss: resolve(repo, 'frontend/chat') }, build: { outDir: root + '/dist', emptyOutDir: true },
});

const { createServer } = await import('node:http');
const { readFileSync, existsSync } = await import('node:fs');
const { extname, join } = await import('node:path');
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' };
const server = createServer((req, res) => {
  const raw = (req.url ?? '/').split('?')[0];
  const rel = raw === '/' ? 'index.html' : raw.replace(/^\/assets\//, '');
  const path = join(root + '/dist', rel);
  if (!existsSync(path)) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'content-type': types[extname(path)] ?? 'application/octet-stream' });
  res.end(readFileSync(path));
});
await new Promise((done) => server.listen(0, '127.0.0.1', done));
const port = server.address().port;

const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH, headless: true });
const results = [];
try {
  const page = await browser.newPage({ viewport: { width: 900, height: 700 } });
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto(`http://127.0.0.1:${port}/`);
  await page.locator('form#drop').waitFor();

  // 1. 初始无 chip
  assert.equal(await page.locator('.composer-reference').count(), 0);
  results.push('初始无 chip: ok');

  // 2. 真实拖拽路径：dragstart → dragover → drop
  const drag = async (sessionId, title) => page.evaluate(([sid, t]) => {
    const row = document.createElement('div');
    row.dataset.row = sid;
    document.body.appendChild(row);
    const dt = new DataTransfer();
    row.dispatchEvent(new DragEvent('dragstart', { bubbles: true, dataTransfer: dt }));
    dt.setData('application/x-akashic-session', JSON.stringify({ sessionId: sid, title: t }));
    const form = document.getElementById('drop');
    form.dispatchEvent(new DragEvent('dragover', { bubbles: true, cancelable: true, dataTransfer: dt }));
    form.dispatchEvent(new DragEvent('drop', { bubbles: true, cancelable: true, dataTransfer: dt }));
    row.remove();
    return dt.types.length;
  }, [sessionId, title]);

  await drag('akashic:a', '会话 A');
  await page.locator('.composer-reference').first().waitFor();
  assert.equal(await page.locator('.composer-reference').count(), 1);
  assert.match(await page.locator('.composer-reference__title').first().innerText(), /会话 A/);
  results.push('拖入产生 chip: ok');

  // 3. 重复拖同一个 → 提示「已经引用过」，不出现上限文案
  await drag('akashic:a', '会话 A');
  await page.locator('.composer-reference__notice').waitFor();
  const duplicateNotice = await page.locator('.composer-reference__notice').innerText();
  assert.match(duplicateNotice, /已经引用过/);
  assert.equal(await page.locator('.composer-reference').count(), 1);
  results.push(`重复拖入提示: ok（${duplicateNotice}）`);

  // 4. 拖满 4 个后再拖 → 上限提示
  for (const n of ['b', 'c', 'd']) await drag(`akashic:${n}`, `会话 ${n.toUpperCase()}`);
  assert.equal(await page.locator('.composer-reference').count(), 4);
  await drag('akashic:e', '会话 E');
  await page.waitForFunction(() => document.querySelector('.composer-reference__notice')?.textContent?.includes('最多'));
  const limitNotice = await page.locator('.composer-reference__notice').innerText();
  assert.match(limitNotice, /最多引用 4 个/);
  assert.equal(await page.locator('.composer-reference').count(), 4);
  results.push(`超限提示: ok（${limitNotice}）`);

  // 5. 单个 chip 的删除按钮
  await page.locator('.composer-reference button').first().click();
  assert.equal(await page.locator('.composer-reference').count(), 3);
  results.push('删除单个 chip: ok');

  // 6. dragover 时输入卡描边；dragend 后复位（dragend 从被拖元素发出，不冒泡到 form）
  await page.evaluate(() => {
    const dt = new DataTransfer();
    dt.setData('application/x-akashic-session', JSON.stringify({ sessionId: 'akashic:z', title: 'Z' }));
    const form = document.getElementById('drop');
    form.dispatchEvent(new DragEvent('dragover', { bubbles: true, cancelable: true, dataTransfer: dt }));
  });
  // dragover 之后 React 状态更新是异步的，直接断言会偶发失败；等它真的点亮。
  await page.waitForFunction(() => document.getElementById('drop')?.classList.contains('is-reference-target'));
  results.push('dragover 点亮投放态: ok');
  // 在被拖元素上派发 dragend：只有 window 上的监听能收到
  await page.evaluate(() => {
    const row = document.createElement('div');
    document.body.appendChild(row);
    row.dispatchEvent(new DragEvent('dragend', { bubbles: true }));
    row.remove();
  });
  await page.waitForFunction(() => !document.getElementById('drop').classList.contains('is-reference-target'));
  results.push('dragend 复位高亮（window 监听）: ok');

  // 7. 提交的 parts 形状
  const parts = await page.evaluate(() => window.refs.composerReferences('draft-1'));
  assert.ok(Array.isArray(parts) && parts.length === 3);
  assert.ok(parts.every((p) => p.sessionId && typeof p.title === 'string'));
  results.push(`引用 store 形状: ok（${parts.length} 条）`);

  assert.deepEqual(errors, [], '不应有页面错误');
  await page.screenshot({ path: resolve(process.env.EVIDENCE_DIR ?? root, 'reference-chip.png') });
} finally {
  await browser.close();
  server.close();
}
console.log(results.join('\n'));

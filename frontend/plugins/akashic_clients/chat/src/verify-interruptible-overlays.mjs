/**
 * 可打断浮层与双向过渡的真实浏览器验收（PR 2）：
 *   1. 紧凑导航抽屉（.compact-navigation-dialog / overlay）：开合中间帧、关闭途中可打断回弹
 *   2. 纸气泡提示（.paper-tooltip）：出现与离开均有平滑透明度/位移过渡
 *   3. 回到底部按钮（.desktop-scroll-return）：出现与离开均有平滑过渡
 *   4. 对话内查找结果栏（.thread-find__meta-wrap）：0fr <-> 1fr 双向平滑展开/收起
 */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, extname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { WebSocketServer } from 'ws';
import { chromium } from 'playwright-core';
import { desktopModels } from '../../../../../scripts/webui-performance/fixtures.mjs';

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '../../../../..');
const root = process.env.EVIDENCE_DIR
  ? (mkdirSync(process.env.EVIDENCE_DIR, { recursive: true }), resolve(process.env.EVIDENCE_DIR))
  : mkdtempSync(resolve(tmpdir(), 'akashic-overlays-'));

const build = spawnSync(process.execPath, [
  resolve(repo, 'node_modules/vite/bin/vite.js'), 'build',
  '--config', 'frontend/plugins/akashic_clients/chat/vite.harness.config.ts',
  '--outDir', root + '/dist',
], { cwd: repo, encoding: 'utf8' });
writeFileSync(root + '/build.log', build.stdout + build.stderr);
assert.equal(build.status, 0, '前端构建失败，见 ' + root + '/build.log');

const base = Date.UTC(2026, 10, 17, 1, 0, 0);
const at = (seconds) => new Date(base + seconds * 1000).toISOString();
const SESSION = 'overlay-motion';
const messages = Array.from({ length: 16 }, (_, index) => ({
  id: `m-${index}`, seq: index, session_id: SESSION, timestamp: at(index),
  author: index % 2 === 0 ? 'user' : 'assistant',
  actor: index % 2 === 0 ? 'user' : 'model',
  source: 'akashic', attachments: [], metadata: {},
  body: {
    kind: index % 2 === 0 ? 'input' : 'output',
    ...(index % 2 === 0 ? {} : { finish: 'complete' }),
    parts: [{ kind: 'text', value: `第 ${index + 1} 条消息用于制造足够长的滚动列表，并包含查找关键字 smooth-target。` }],
  },
}));

const json = (res, data, status = 200) => {
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(JSON.stringify(data));
};

const sockets = new Set();
const server = createServer((req, res) => {
  const url = new URL(req.url, 'http://127.0.0.1');
  const path = url.pathname;
  if (path === '/api/shell/state') return json(res, { status: 'ready', configured: true, chatReady: true });
  if (path === '/api/chat/models') return json(res, desktopModels(1));
  if (path === '/api/chat/navigation/pins') return json(res, { pins: [], sessions: [] });
  if (path === '/api/chat/sessions') {
    return json(res, {
      items: [{ key: SESSION, first_message_content: '浮层过渡测试', updated_at: messages.at(-1).timestamp, message_count: messages.length }],
      next_cursor: null,
    });
  }
  if (path === `/api/chat/sessions/${SESSION}/messages`) {
    return json(res, { version: 2, items: messages, through_seq: messages.at(-1).seq, has_more: false, before_seq: null });
  }
  if (path === '/api/chat/plugin-ui/catalog') return json(res, { catalog_revision: '0'.repeat(64), items: [] });
  if (path === '/api/runtime/host-bridge') return json(res, { state: 'healthy', failures: 0 });
  if (path.startsWith('/api/')) return json(res, { items: [] });
  const file = resolve(root + '/dist', path === '/' || path === '/chat' ? 'index.html' : path.replace(/^\/assets\//, '').replace(/^\//, ''));
  if (!file.startsWith(root + '/dist/') || !existsSync(file)) return res.writeHead(404).end();
  res.writeHead(200, {
    'content-type': ({ '.js': 'text/javascript', '.css': 'text/css', '.html': 'text/html', '.woff2': 'font/woff2', '.svg': 'image/svg+xml' })[extname(file)] ?? 'application/octet-stream',
  });
  res.end(readFileSync(file));
});

const wss = new WebSocketServer({ noServer: true });
wss.on('connection', (socket) => {
  sockets.add(socket);
  socket.on('message', (raw) => {
    const frame = JSON.parse(String(raw));
    if (frame.type !== 'session.follow') return;
    socket.send(JSON.stringify({
      type: 'session.following', version: 2, session_id: frame.session_id,
      request_id: frame.request_id, through_seq: messages.at(-1).seq,
    }));
    socket.send(JSON.stringify({
      type: 'reply.status', version: 2, session_id: frame.session_id,
      snapshot_id: 'overlay-snap', available: true, items: [],
    }));
  });
  socket.on('close', () => sockets.delete(socket));
});
server.on('upgrade', (request, socket, head) => {
  const { pathname } = new URL(request.url, 'http://127.0.0.1');
  if (pathname !== '/ws') return socket.destroy();
  wss.handleUpgrade(request, socket, head, (ws) => wss.emit('connection', ws, request));
});

await new Promise((done) => server.listen(0, '127.0.0.1', done));
const origin = `http://127.0.0.1:${server.address().port}`;

const middles = (values, eps = 0.05) => {
  const valid = values.filter((value) => Number.isFinite(value));
  if (valid.length < 2) return 0;
  const low = Math.min(...valid);
  const high = Math.max(...valid);
  return valid.filter((value) => value > low + eps && value < high - eps).length;
};

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH ?? '/usr/bin/chromium',
  headless: true,
  args: ['--no-sandbox', '--disable-dev-shm-usage'],
});

try {
  // 1. 桌面端：PaperTooltipLayer、ReadingReturnButton、thread-find__meta-wrap
  {
    const context = await browser.newContext({ viewport: { width: 1200, height: 700 } });
    const page = await context.newPage();
    page.setDefaultTimeout(15_000);
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));

    await page.goto(`${origin}/?session=${SESSION}`);
    await page.locator('.web-message-anchor').first().waitFor();
    await page.waitForTimeout(200);

    // 1a. ReadingReturnButton 出现与离开（滚动离开底部再滚回）
    const returnMotion = await page.evaluate(async () => {
      const scroller = document.querySelector('.conversation-scroll');
      scroller.scrollTop = 0;
      const enterOpacities = [];
      for (let i = 0; i < 20; i += 1) {
        await new Promise((resolve) => requestAnimationFrame(resolve));
        const btn = document.querySelector('.desktop-scroll-return');
        enterOpacities.push(btn ? Number(getComputedStyle(btn).opacity) : null);
      }
      scroller.scrollTop = scroller.scrollHeight;
      const exitOpacities = [];
      for (let i = 0; i < 20; i += 1) {
        await new Promise((resolve) => requestAnimationFrame(resolve));
        const btn = document.querySelector('.desktop-scroll-return');
        exitOpacities.push(btn ? Number(getComputedStyle(btn).opacity) : null);
      }
      return { enterOpacities, exitOpacities };
    });
    console.log('[1a] ReadingReturnButton enter/exit:', JSON.stringify(returnMotion));
    assert.ok(middles(returnMotion.enterOpacities.filter((v) => v !== null)) >= 1, '回到底部按钮入场应有透明度过渡');
    assert.ok(middles(returnMotion.exitOpacities.filter((v) => v !== null)) >= 1, '回到底部按钮退场应有透明度过渡');
    assert.equal(returnMotion.exitOpacities.at(-1), null, '退场完成后应卸载');

    // 1b. PaperTooltipLayer 出现与离开（悬停新对话按钮等带 data-tip 控件）
    const tipButton = page.locator('[data-tip]').first();
    const box = await tipButton.boundingBox();
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.locator('.paper-tooltip[data-state="open"]').waitFor();
    await page.waitForTimeout(180);
    const tipExit = await page.evaluate(async () => {
      const target = document.querySelector('[data-tip]');
      target.dispatchEvent(new PointerEvent('pointerout', { bubbles: true, pointerType: 'mouse' }));
      const opacities = [];
      for (let i = 0; i < 16; i += 1) {
        await new Promise((resolve) => requestAnimationFrame(resolve));
        const tip = document.querySelector('.paper-tooltip');
        opacities.push(tip ? Number(getComputedStyle(tip).opacity) : null);
      }
      return opacities;
    });
    console.log('[1b] PaperTooltipLayer exit:', JSON.stringify(tipExit));
    assert.ok(middles(tipExit.filter((v) => v !== null)) >= 1, `Tooltip 退场应平滑淡出，实际：${tipExit}`);
    assert.equal(tipExit.at(-1), null, 'Tooltip 退场后应卸载');

    // 1c. thread-find__meta-wrap 双向 0fr <-> 1fr
    await page.keyboard.press('Control+f');
    await page.locator('.thread-find input').waitFor();
    const findInput = page.locator('.thread-find input');
    const findExpand = await page.evaluate(async () => {
      const wrap = document.querySelector('.thread-find__meta-wrap');
      const input = document.querySelector('.thread-find input');
      const proto = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
      proto.set.call(input, 'smooth-target');
      input.dispatchEvent(new Event('input', { bubbles: true }));
      const heights = [];
      for (let i = 0; i < 20; i += 1) {
        await new Promise((resolve) => requestAnimationFrame(resolve));
        heights.push(Math.round(wrap.getBoundingClientRect().height * 100) / 100);
      }
      return heights;
    });
    console.log('[1c] thread-find expand:', JSON.stringify(findExpand));
    assert.ok(middles(findExpand, 0.5) >= 1, `查找结果栏展开应有中间帧，实际：${findExpand}`);
    await findInput.fill('');
    await page.waitForTimeout(260);
    assert.equal(Math.round(await page.locator('.thread-find__meta-wrap').evaluate((n) => n.getBoundingClientRect().height)), 0);

    assert.deepEqual(errors, []);
    await context.close();
  }

  // 2. 窄屏紧凑导航抽屉：打开平滑 + 关闭途中可打断回弹 + 完整退场卸载
  {
    const context = await browser.newContext({ viewport: { width: 390, height: 800 } });
    const page = await context.newPage();
    page.setDefaultTimeout(15_000);
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));

    await page.goto(`${origin}/?session=${SESSION}`);
    await page.locator('.compact-navigation-trigger').waitFor();

    await page.locator('.compact-navigation-trigger').click();
    const drawer = page.locator('.compact-navigation-dialog');
    await drawer.waitFor();
    const enterXs = await page.evaluate(async () => {
      const el = document.querySelector('.compact-navigation-dialog');
      const xs = [];
      for (let i = 0; i < 24; i += 1) {
        await new Promise((resolve) => requestAnimationFrame(resolve));
        xs.push(Math.round(el.getBoundingClientRect().left * 100) / 100);
      }
      return xs;
    });
    console.log('[2a] Compact drawer enter lefts:', JSON.stringify(enterXs));
    assert.ok(middles(enterXs, 1) >= 1, `抽屉展开应平滑滑动，实际：${enterXs}`);
    assert.equal(Math.round(enterXs.at(-1)), 0, '展开完成后 left 应落在 0');

    // 按 Esc 关闭后 4 帧立即再点打开，验证可打断回弹；再彻底关闭等卸载
    const interrupt = await page.evaluate(async () => {
      const el = document.querySelector('.compact-navigation-dialog');
      const before = { state: el.dataset.state, left: el.getBoundingClientRect().left, tr: getComputedStyle(el).transition, anim: getComputedStyle(el).animationName };
      document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
      const xs = [];
      for (let i = 0; i < 4; i += 1) {
        await new Promise((resolve) => requestAnimationFrame(resolve));
        xs.push(Math.round(el.getBoundingClientRect().left * 100) / 100);
      }
      const mid = { state: el.dataset.state, tr: getComputedStyle(el).transition, anim: getComputedStyle(el).animationName, tf: getComputedStyle(el).transform };
      document.querySelector('.compact-navigation-trigger').click();
      for (let i = 0; i < 24; i += 1) {
        await new Promise((resolve) => requestAnimationFrame(resolve));
        xs.push(Math.round(el.getBoundingClientRect().left * 100) / 100);
      }
      return { xs, before, mid };
    });
    console.log('[2b] Compact drawer interrupt close -> open lefts:', JSON.stringify(interrupt));
    assert.ok(interrupt.xs[0] > -220, `关闭首帧不应直接跳到终点，实际：${interrupt.xs[0]}`);
    assert.ok(interrupt.xs[3] < -5, '关闭途中抽屉应正在向左退出');
    assert.equal(Math.round(interrupt.xs.at(-1)), 0, '打断重开后应回到 left=0');

    await page.keyboard.press('Escape');
    await drawer.waitFor({ state: 'detached' });

    assert.deepEqual(errors, []);
    await context.close();
  }

  console.log('PASS verify-interruptible-overlays');
} finally {
  await browser.close();
  for (const socket of sockets) socket.terminate();
  await new Promise((done) => server.close(done));
  wss.close();
  console.log('证据目录: ' + root);
}

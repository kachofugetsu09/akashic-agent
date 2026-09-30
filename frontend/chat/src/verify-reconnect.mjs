/** 真实 Chromium/HTTP/WebSocket 验证短连预算；虚拟时钟只推进等待。 */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdtempSync, readFileSync, existsSync, writeFileSync } from 'node:fs';
import { resolve, dirname, extname } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { WebSocketServer } from 'ws';
import { chromium } from 'playwright-core';
import { desktopModels } from '../../../scripts/webui-performance/fixtures.mjs';

const sourceIndex = process.argv.indexOf('--source');
const repo = sourceIndex < 0 ? resolve(dirname(fileURLToPath(import.meta.url)), '../../..')
  : resolve(process.argv[sourceIndex + 1]);
const baseline = process.argv.includes('--baseline');
const root = mkdtempSync(resolve(tmpdir(), 'akashic-reconnect-'));
console.info('artifact: ' + root);
const build = spawnSync(process.execPath, [resolve(repo, 'node_modules/vite/bin/vite.js'),
  'build', '--config', 'frontend/chat/vite.config.ts', '--outDir', root + '/dist'],
{ cwd: repo, encoding: 'utf8' });
writeFileSync(root + '/build.log', build.stdout + build.stderr);
assert.equal(build.status, 0, '前端构建失败：' + root + '/build.log');
const sockets = new Set();
const received = [];
const frameWaiters = new Set();
let connectionCount = 0;
let showSession = false;
let resolveStop;
const stopReceived = new Promise(done => { resolveStop = done; });
const session = 'akashic:reconnect';
const input = { id: 'reconnect-input', seq: 0, author: 'user', source: 'akashic',
  timestamp: '2026-09-30T00:00:00Z', session_id: session, attachments: [],
  body: { kind: 'input', parts: [{ kind: 'text', value: '重连后的停止验证' }] } };
const server = createServer((req, res) => {
  const path = new URL(req.url, 'http://localhost').pathname;
  const json = data => { res.writeHead(200, { 'content-type': 'application/json' }); res.end(JSON.stringify(data)); };
  if (path === '/api/shell/state') return json({ status: 'ready', configured: true, chatReady: true });
  if (path === '/api/chat/models') return json({ ...desktopModels(1), unavailableRuntimes: [] });
  if (path === '/api/chat/sessions') return json({ items: showSession ? [{ key: session,
    first_message_content: '重连后的停止验证', updated_at: input.timestamp, message_count: 1 }] : [], next_cursor: null });
  if (path === '/api/chat/sessions/' + encodeURIComponent(session) + '/messages') {
    return json({ version: 2, items: [input], through_seq: 0, has_more: false, before_seq: null });
  }
  if (path === '/api/chat/plugin-ui/catalog') return json({ catalog_revision: '0'.repeat(64), items: [] });
  if (path === '/api/runtime/host-bridge') return json({ state: 'healthy', failures: 0 });
  const file = resolve(root + '/dist', path === '/' || path === '/chat' ? 'index.html'
    : path.replace(/^\/assets\//, '').replace(/^\//, ''));
  if (!file.startsWith(root + '/dist/') || !existsSync(file)) return res.writeHead(404).end();
  res.writeHead(200, { 'content-type': ({ '.js': 'text/javascript', '.css': 'text/css', '.html': 'text/html',
    '.woff2': 'font/woff2', '.svg': 'image/svg+xml' })[extname(file)] ?? 'application/octet-stream' });
  res.end(readFileSync(file));
});
const ws = new WebSocketServer({ server, path: '/ws' });
ws.on('connection', socket => {
  const owner = ++connectionCount;
  sockets.add(socket);
  socket.on('message', raw => {
    const frame = JSON.parse(String(raw));
    const receipt = { ...frame, fixture_owner: owner };
    received.push(receipt);
    for (const waiter of frameWaiters) waiter(receipt);
    if (frame.type === 'message.send' && frame.text === '/stop') resolveStop(receipt);
  });
  socket.on('close', () => sockets.delete(socket));
});
await new Promise(done => server.listen(0, '127.0.0.1', done));
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH ?? '/usr/bin/chromium',
  headless: true, args: ['--no-sandbox'] });
const report = { source: repo, baseline, artifact: root, checks: [] };

/** 每次只关闭已观察到的当前连接，防止网络事件与虚拟等待竞态。 */
async function openPage() {
  const context = await browser.newContext();
  await context.addInitScript(() => {
    window.connectionEvents = [];
    const Native = window.WebSocket;
    window.WebSocket = class extends Native {
      constructor(...args) {
        super(...args);
        this.addEventListener('open', () => window.connectionEvents.push({ kind: 'open', at: performance.now() }));
        this.addEventListener('close', () => window.connectionEvents.push({ kind: 'close', at: performance.now() }));
      }
    };
  });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.clock.install();
  await page.goto('http://127.0.0.1:' + server.address().port);
  await page.waitForFunction(() => window.connectionEvents.some(e => e.kind === 'open'));
  await page.clock.pauseAt(await page.evaluate(() => Date.now() + 1000));
  return { context, page, errors };
}
const opened = page => page.evaluate(() => window.connectionEvents.filter(e => e.kind === 'open').length);
async function savePage(page, name) {
  writeFileSync(root + '/' + name + '.json', JSON.stringify(await page.evaluate(() => ({
    events: window.connectionEvents, text: document.body.innerText,
  })), null, 2));
  writeFileSync(root + '/received.json', JSON.stringify(received, null, 2));
}
async function waitForFrame(check) {
  const saved = received.find(check);
  if (saved) return saved;
  return new Promise((done, reject) => {
    const timer = setTimeout(() => {
      frameWaiters.delete(waiter);
      reject(new Error('未收到约定帧'));
    }, 10_000);
    const waiter = receipt => {
      if (!check(receipt)) return;
      clearTimeout(timer);
      frameWaiters.delete(waiter);
      done(receipt);
    };
    frameWaiters.add(waiter);
  });
}
async function closeAndAdvance(page) {
  assert.equal(sockets.size, 1, '出现多个活连接');
  const before = await page.evaluate(() => window.connectionEvents.filter(e => e.kind === 'close').length);
  [...sockets][0].close(1013, 'local short connection');
  await page.waitForFunction(count => window.connectionEvents.filter(e => e.kind === 'close').length > count, before);
  await page.clock.fastForward(30_001);
}
try {
  // 1. 初次连接加12次短连重试后必须停止，基线能越过此上限。
  const first = await openPage();
  try {
    for (let count = 1; count <= 13; count++) {
      assert.equal(await opened(first.page), count);
      await closeAndAdvance(first.page);
      if (count < 13 || baseline) await first.page.waitForFunction(count =>
        window.connectionEvents.filter(e => e.kind === 'open').length === count + 1, count);
    }
    if (baseline) {
      assert.equal(await opened(first.page), 14);
      report.checks.push({ case: 'short connection bypasses budget', connections: 14 });
    } else {
      await first.page.waitForFunction(() => document.body.innerText.includes('暂时无法连接，请重试'));
      await first.page.clock.fastForward(300_000);
      assert.equal(await opened(first.page), 13);
      report.checks.push({ case: 'short connections stop', connections: 13 });
      // 2. 用户手动重试必须取得新的预算，不复用已耗尽 schedule。
      await first.page.getByRole('button', { name: '重试', exact: true }).click();
      await first.page.waitForFunction(() => window.connectionEvents.filter(e => e.kind === 'open').length === 14);
      assert.equal(sockets.size, 1);
      report.checks.push({ case: 'manual retry reconnects', connections: 14 });
    }
    assert.deepEqual(first.errors, []);
  } finally { await savePage(first.page, 'short'); await first.context.close(); }
  if (!baseline) {
    // 3. 已消耗两次的连接稳定30秒后，下一次中断拥有完整12次预算。
    const stable = await openPage();
    try {
      for (let count = 1; count <= 2; count++) {
        await closeAndAdvance(stable.page);
        await stable.page.waitForFunction(count => window.connectionEvents.filter(e => e.kind === 'open').length === count + 1, count);
      }
      await stable.page.clock.fastForward(30_001);
      for (let count = 3; count <= 15; count++) {
        assert.equal(await opened(stable.page), count);
        await closeAndAdvance(stable.page);
        if (count < 15) await stable.page.waitForFunction(count => window.connectionEvents.filter(e => e.kind === 'open').length === count + 1, count);
      }
      await stable.page.waitForFunction(() => document.body.innerText.includes('暂时无法连接，请重试'));
      assert.equal(await opened(stable.page), 15);
      assert.deepEqual(stable.errors, []);
      report.checks.push({ case: 'stable connection refreshes budget', connections: 15 });
    } finally { await savePage(stable.page, 'stable'); await stable.context.close(); }
    // 4. 原页面实际重连后恢复同 Session 跟随，stop 走当前唯一 socket。
    showSession = true;
    const control = await openPage();
    try {
      await control.page.locator('.conversation-session').filter({ hasText: '重连后的停止验证' }).click();
      await control.page.waitForFunction(() => document.querySelector('.composer-action-button')?.dataset.mode === 'stop');
      const count = await opened(control.page);
      await closeAndAdvance(control.page);
      await control.page.waitForFunction(count => window.connectionEvents.filter(e => e.kind === 'open').length === count + 1, count);
      assert.equal(sockets.size, 1);
      await waitForFrame(frame => frame.type === 'session.follow' && frame.session_id === session
        && frame.fixture_owner === connectionCount);
      [...sockets][0].send(JSON.stringify({ type: 'reply.status', version: 2, session_id: session,
        snapshot_id: 'fixture', available: true, items: [{ session_id: session, handle: 'current', source: 'akashic', head: 0,
          active: true, admission_busy: false, pausing: false, restoring: false, preview: null }] }));
      await control.page.locator('.composer-action-button[data-mode="stop"]').click();
      const stop = await Promise.race([stopReceived, new Promise((_, reject) => {
        const timer = setTimeout(() => reject(new Error('未收到 stop')), 10_000); timer.unref();
      })]);
      assert.equal(stop.session_id, session);
      assert.equal(stop.fixture_owner, connectionCount);
      assert.deepEqual(control.errors, []);
      report.checks.push({ case: 'reconnect follows and sends stop', session_id: stop.session_id });
    } finally { await savePage(control.page, 'control'); await control.context.close(); }
  }
  assert.equal(received.filter(frame => frame.type === 'message.send').length, baseline ? 0 : 1);
  writeFileSync(root + '/report.json', JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
} finally {
  await browser.close();
  for (const socket of sockets) socket.terminate();
  await new Promise(done => ws.close(done));
  await new Promise(done => server.close(done));
}

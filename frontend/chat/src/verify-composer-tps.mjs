/** 真实 Chromium 验证输入框下方状态行的生成中解码速度：增量与解码时长由服务端事实给出，
 * 重试清空增量时读数撤下，结算后仍由 composer.status 插槽给终态摘要。
 * 用法：node frontend/chat/src/verify-composer-tps.mjs [--source <repo>] */
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
const root = mkdtempSync(resolve(tmpdir(), 'akashic-composer-tps-'));
console.info('artifact: ' + root);
const build = spawnSync(process.execPath, [resolve(repo, 'node_modules/vite/bin/vite.js'),
  'build', '--config', 'frontend/chat/vite.harness.config.ts', '--outDir', root + '/dist'],
{ cwd: repo, encoding: 'utf8' });
writeFileSync(root + '/build.log', build.stdout + build.stderr);
assert.equal(build.status, 0, '前端构建失败：' + root + '/build.log');

// 1. 夹具：一条已完成的问答，回合从 message.send 开始，由脚本按固定节奏推预览。
const sessionId = 'akashic:tps';
const items = [{
  id: 'm0', seq: 0, author: 'user', source: 'akashic', timestamp: new Date(Date.UTC(2026, 9, 19, 8, 0, 0)).toISOString(),
  session_id: sessionId, attachments: [], metadata: {}, body: { kind: 'input', parts: [{ kind: 'text', value: '算一下速度' }] },
}, {
  id: 'm1', seq: 1, author: 'assistant', source: 'akashic', timestamp: new Date(Date.UTC(2026, 9, 19, 8, 0, 1)).toISOString(),
  session_id: sessionId, attachments: [], metadata: {}, body: { kind: 'output', finish: 'complete',
    parts: [{ kind: 'text', value: '先问一句。' }] },
}];
const received = [];
const followers = new Set();
const json = (res, data) => { res.writeHead(200, { 'content-type': 'application/json' }); res.end(JSON.stringify(data)); };
const server = createServer((req, res) => {
  const url = new URL(req.url, 'http://localhost');
  const path = url.pathname;
  if (path === '/api/shell/state') return json(res, { status: 'ready', configured: true, chatReady: true });
  if (path === '/api/chat/models') return json(res, { ...desktopModels(1), unavailableRuntimes: [] });
  if (path === '/api/chat/sessions') return json(res, { items: [{ key: sessionId, title: '速度', first_message_content: '算一下速度',
    updated_at: items.at(-1).timestamp, message_count: items.length }], next_cursor: null });
  const history = path.match(/^\/api\/chat\/sessions\/([^/]+)\/messages$/u);
  if (history) return json(res, { version: 2, items, through_seq: items.length - 1, has_more: false, before_seq: null });
  if (path === '/api/chat/plugin-ui/catalog') return json(res, { catalog_revision: '0'.repeat(64), items: [] });
  if (path === '/api/runtime/host-bridge') return json(res, { state: 'healthy', failures: 0 });
  if (path.startsWith('/api/')) return json(res, { items: [], pins: [], sessions: [] });
  const file = resolve(root + '/dist', path === '/' ? 'index.html' : path.replace(/^\/assets\//, '').replace(/^\//, ''));
  if (!file.startsWith(root + '/dist/') || !existsSync(file)) return res.writeHead(404).end();
  res.writeHead(200, { 'content-type': ({ '.js': 'text/javascript', '.css': 'text/css', '.html': 'text/html',
    '.woff2': 'font/woff2', '.svg': 'image/svg+xml' })[extname(file)] ?? 'application/octet-stream' });
  res.end(readFileSync(file));
});
const send = (frame) => { sentFrames.push(frame.type); for (const socket of followers) socket.send(JSON.stringify(frame)); };
const status = (active, preview) => send({ type: 'reply.status', version: 2, session_id: sessionId,
  snapshot_id: `fixture-${Date.now()}-${Math.random()}`, available: true,
  items: active ? [{ session_id: sessionId, handle: 'fixture', source: 'akashic', active: true, preview }] : [] });
const append = (author, body) => {
  const item = { id: `${sessionId}-m${items.length}`, seq: items.length, author, source: 'akashic',
    timestamp: new Date().toISOString(), session_id: sessionId, attachments: [], metadata: {}, body };
  items.push(item);
  send({ type: 'messages.appended', version: 2, session_id: sessionId, after_seq: item.seq - 1,
    through_seq: item.seq, next_after_seq: item.seq, has_more: false, items: [item] });
};
// 5 个增量 / 1 秒的稳态；首段到达即起表，与服务端 decode_ms 同一口径。
const TICK_MS = 200;
const TICKS = 20;
function startStream() {
  const started = Date.now();
  status(true, { message_id: 'draft-1', text: '', thinking: '', call_record_id: 'call-1', first_token_ms: 200 });
  let step = 0;
  const timer = setInterval(() => {
    step += 1;
    const decodeMs = Date.now() - started;
    if (step <= TICKS) {
      status(true, { message_id: 'draft-1', text: '好'.repeat(step), thinking: '', call_record_id: 'call-1',
        first_token_ms: 200, output_deltas: step, decode_ms: decodeMs });
      return;
    }
    if (step <= TICKS + 6) {
      // 退避前撤下草稿增量：读数应随增量清零消失，只留“重试中”；停留 1 秒便于观察。
      status(true, { message_id: 'draft-1', text: '', thinking: '', call_record_id: 'call-1',
        retry_status: '退避 1s', first_token_ms: null, output_deltas: 0, decode_ms: null });
      return;
    }
    clearInterval(timer);
    append('assistant', { kind: 'output', finish: 'complete', parts: [{ kind: 'text', value: '本轮完成。' }] });
    status(false, null);
  }, TICK_MS);
  return () => clearInterval(timer);
}
const ws = new WebSocketServer({ server, path: '/ws' });
ws.on('connection', socket => {
  socket.on('message', raw => {
    const frame = JSON.parse(String(raw));
    received.push(frame);
    if (frame.type === 'ping') return socket.send(JSON.stringify({ type: 'pong', request_id: frame.request_id }));
    if (frame.type === 'session.follow') {
      followers.add(socket);
      socket.send(JSON.stringify({ type: 'session.following', version: 2, session_id: frame.session_id,
        through_seq: items.length - 1, request_id: frame.request_id }));
      return status(false, null);
    }
    if (frame.type !== 'message.send') return;
    socket.send(JSON.stringify({ type: 'message.accepted', request_id: frame.request_id, session_id: frame.session_id }));
    append('user', { kind: 'input', parts: [{ kind: 'text', value: frame.text }] });
    startStream();
  });
  socket.on('close', () => followers.delete(socket));
});
await new Promise(done => server.listen(0, '127.0.0.1', done));
const origin = 'http://127.0.0.1:' + server.address().port;
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH ?? '/usr/bin/chromium',
  headless: true, args: ['--no-sandbox'] });
const report = { source: repo, artifact: root, checks: [] };
const check = (ok, label, detail) => {
  report.checks.push({ label, ok: Boolean(ok), detail });
  assert.ok(ok, `${label}：${detail}`);
};
const pageText = () => page.locator('.composer-stats-line').innerText().catch(() => '');
const sentFrames = [];
const waitFor = async (read, label) => {
  for (let tries = 0; tries < 200; tries += 1) {
    const value = await read();
    if (value) return value;
    await new Promise(done => setTimeout(done, 50));
  }
  const seen = await pageText();
  throw new Error('等待超时：' + label + '；当前状态行：' + JSON.stringify(seen)
    + '；followers=' + followers.size + '；发送=' + JSON.stringify(sentFrames.slice(-8)) + '；收到=' + JSON.stringify(received.slice(-6))
    + '；输入区=' + JSON.stringify(await page.locator('.composer').innerText().catch(() => ''))
    + '；控制台=' + JSON.stringify(logs.slice(-6)) + '；页面错误=' + JSON.stringify(errors));
};

const context = await browser.newContext({ viewport: { width: 1365, height: 860 } });
const page = await context.newPage();
const errors = [];
const logs = [];
page.on('pageerror', error => errors.push(error.message));
page.on('console', message => logs.push(message.type() + ': ' + message.text()));
try {
  await page.goto(origin + '/?session=' + encodeURIComponent(sessionId));
  await page.waitForSelector('.composer textarea');
  await page.locator('.composer textarea').fill('算一下速度');
  await page.keyboard.press('Enter');
  const line = page.locator('.composer-stats-line');
  const text = () => line.innerText();
  const sampled = await waitFor(async () => {
    const value = await text().catch(() => '');
    return /≈\s*\d+(\.\d)?\s*tok\/s/u.test(value) ? value : '';
  }, '生成中读数');
  report.first = sampled;
  const rate = Number(sampled.match(/≈\s*(\d+(?:\.\d)?)/u)[1]);
  check(rate >= 4 && rate <= 6.5, '稳态约 5 tok/s', sampled);
  await new Promise(done => setTimeout(done, 800));
  const later = await text();
  report.later = later;
  check(/≈\s*\d+(\.\d)?\s*tok\/s/u.test(later) && /输出中/u.test(later), '读数随增量持续推进', later);
  const lineBox = await line.boundingBox();
  const composerBox = await page.locator('.composer').boundingBox();
  if (lineBox !== null && composerBox !== null) {
    const left = Math.max(0, Math.min(composerBox.x, lineBox.x) - 24);
    const top = Math.max(0, composerBox.y - 24);
    await page.screenshot({ path: root + '/composer-tps.png', clip: {
      x: left, y: top, width: Math.min(Math.max(composerBox.width, lineBox.width) + 48, 1365 - left),
      height: Math.min(lineBox.y + lineBox.height + 24 - top, 860 - top) } });
  } else {
    await page.screenshot({ path: root + '/composer-tps.png' });
  }
  const retry = await waitFor(async () => {
    const value = await text().catch(() => '');
    return /重试中/u.test(value) ? value : '';
  }, '重试阶段');
  report.retry = retry;
  check(!/tok\/s/u.test(retry), '重试清空增量后不保留旧读数', retry);
  check(errors.length === 0, '页面无未捕获错误', errors.join(' | '));
  report.screenshot = root + '/composer-tps.png';
} finally {
  await browser.close();
  await new Promise(done => server.close(done));
}
writeFileSync(root + '/report.json', JSON.stringify(report, null, 2));
console.info(JSON.stringify(report, null, 2));

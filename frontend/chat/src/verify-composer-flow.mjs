/** 真实 Chromium/HTTP/WebSocket 验证输入区键盘与阅读交互：Esc 停止并退回插话、↑ 取回、引用回复聚焦、切回会话回到原位。
 * 回合由脚本显式结束，不依赖定时器；
 * 用法：node frontend/chat/src/verify-composer-flow.mjs [--source <repo>] */
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
const root = mkdtempSync(resolve(tmpdir(), 'akashic-composer-flow-'));
console.info('artifact: ' + root);
const build = spawnSync(process.execPath, [resolve(repo, 'node_modules/vite/bin/vite.js'),
  'build', '--config', 'frontend/chat/vite.harness.config.ts', '--outDir', root + '/dist'],
{ cwd: repo, encoding: 'utf8' });
writeFileSync(root + '/build.log', build.stdout + build.stderr);
assert.equal(build.status, 0, '前端构建失败：' + root + '/build.log');

// 1. 夹具：一个长会话（够滚动）与一个短会话；回合在 message.send 时开始，由脚本或 /stop 结束。
const sessions = new Map();
const message = (sessionId, seq, author, body, id) => ({ id: id ?? `${sessionId}-m${seq}`, seq, author, source: 'akashic',
  timestamp: new Date(Date.UTC(2026, 9, 1, 8, 0, seq)).toISOString(), session_id: sessionId, attachments: [], metadata: {}, body });
function addSession(key, title, pairs) {
  const items = [];
  for (let index = 0; index < pairs; index += 1) {
    items.push(message(key, items.length, 'user', { kind: 'input', parts: [{ kind: 'text', value: `${title} 第 ${index} 问` }] }));
    items.push(message(key, items.length, 'assistant', { kind: 'output', finish: 'complete',
      parts: [{ kind: 'text', value: Array.from({ length: 4 }, (_, line) => `第 ${index} 条回复的第 ${line} 行。`).join('\n\n') }] }));
  }
  sessions.set(key, { title, items });
}
addSession('akashic:long', '长会话', 80);
addSession('akashic:short', '短会话', 2);
const received = [];
const followers = new Map();
const turns = new Map();
const json = (res, data) => { res.writeHead(200, { 'content-type': 'application/json' }); res.end(JSON.stringify(data)); };
const server = createServer((req, res) => {
  const url = new URL(req.url, 'http://localhost');
  const path = url.pathname;
  if (path === '/api/shell/state') return json(res, { status: 'ready', configured: true, chatReady: true });
  if (path === '/api/chat/models') return json(res, { ...desktopModels(1), unavailableRuntimes: [] });
  if (path === '/api/chat/sessions') return json(res, { items: [...sessions].map(([key, s]) => ({ key, title: s.title,
    first_message_content: s.title, updated_at: s.items.at(-1).timestamp, message_count: s.items.length })), next_cursor: null });
  const history = path.match(/^\/api\/chat\/sessions\/([^/]+)\/messages$/u);
  if (history) {
    const items = sessions.get(decodeURIComponent(history[1]))?.items ?? [];
    const before = url.searchParams.has('before_seq') ? Number(url.searchParams.get('before_seq')) : items.length;
    const start = Math.max(0, before - Number(url.searchParams.get('page_size') || 50));
    return json(res, { version: 2, items: items.slice(start, before), through_seq: items.length - 1,
      has_more: start > 0, before_seq: start > 0 ? start : null });
  }
  if (path === '/api/chat/plugin-ui/catalog') return json(res, { catalog_revision: '0'.repeat(64), items: [] });
  if (path === '/api/runtime/host-bridge') return json(res, { state: 'healthy', failures: 0 });
  if (path.startsWith('/api/')) return json(res, { items: [], pins: [], sessions: [] });
  const file = resolve(root + '/dist', path === '/' ? 'index.html' : path.replace(/^\/assets\//, '').replace(/^\//, ''));
  if (!file.startsWith(root + '/dist/') || !existsSync(file)) return res.writeHead(404).end();
  res.writeHead(200, { 'content-type': ({ '.js': 'text/javascript', '.css': 'text/css', '.html': 'text/html',
    '.woff2': 'font/woff2', '.svg': 'image/svg+xml' })[extname(file)] ?? 'application/octet-stream' });
  res.end(readFileSync(file));
});
const broadcast = (sessionId, frame) => {
  for (const [socket, followed] of followers) if (followed === sessionId) socket.send(JSON.stringify(frame));
};
function append(sessionId, author, body, id) {
  const items = sessions.get(sessionId).items;
  const item = message(sessionId, items.length, author, body, id);
  items.push(item);
  broadcast(sessionId, { type: 'messages.appended', version: 2, session_id: sessionId, after_seq: item.seq - 1,
    through_seq: item.seq, next_after_seq: item.seq, has_more: false, items: [item] });
}
const replyStatus = (sessionId, active) => broadcast(sessionId, { type: 'reply.status', version: 2, session_id: sessionId,
  snapshot_id: `fixture-${Date.now()}-${Math.random()}`, available: true,
  items: active ? [{ session_id: sessionId, handle: 'fixture', source: 'akashic', active: true,
    preview: { message_id: `draft-${sessionId}`, text: '', thinking: '' } }] : [] });
function endTurn(sessionId, stopped = false) {
  assert.ok(turns.delete(sessionId), '没有进行中的回合');
  if (stopped) append(sessionId, 'system', { kind: 'control', action: 'pause', through_seq: sessions.get(sessionId).items.length - 1, reason: null });
  else append(sessionId, 'assistant', { kind: 'output', finish: 'complete', parts: [{ kind: 'text', value: '本轮完成。' }] });
  replyStatus(sessionId, false);
}
const ws = new WebSocketServer({ server, path: '/ws' });
ws.on('connection', socket => {
  socket.on('message', raw => {
    const frame = JSON.parse(String(raw));
    received.push(frame);
    if (frame.type === 'ping') return socket.send(JSON.stringify({ type: 'pong', request_id: frame.request_id }));
    if (frame.type === 'session.follow') {
      followers.set(socket, frame.session_id);
      socket.send(JSON.stringify({ type: 'session.following', version: 2, session_id: frame.session_id,
        through_seq: sessions.get(frame.session_id).items.length - 1, request_id: frame.request_id }));
      return replyStatus(frame.session_id, turns.has(frame.session_id));
    }
    // 生成中的插话：夹具只暂存不写入，撤回总是成功。
    if (frame.type === 'message.steer') return socket.send(JSON.stringify({ type: 'message.steer.result', request_id: frame.request_id, staged: true }));
    if (frame.type === 'message.steer.withdraw') return socket.send(JSON.stringify({ type: 'message.steer.result', request_id: frame.request_id, withdrawn: true }));
    if (frame.type !== 'message.send') return;
    if (frame.text === '/stop') return turns.has(frame.session_id) && endTurn(frame.session_id, true);
    socket.send(JSON.stringify({ type: 'message.accepted', request_id: frame.request_id, session_id: frame.session_id }));
    append(frame.session_id, 'user', { kind: 'input', parts: [{ kind: 'text', value: frame.text }] }, frame.request_id);
    turns.set(frame.session_id, true);
    replyStatus(frame.session_id, true);
  });
  socket.on('close', () => followers.delete(socket));
});
await new Promise(done => server.listen(0, '127.0.0.1', done));
const origin = 'http://127.0.0.1:' + server.address().port;
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH ?? '/usr/bin/chromium',
  headless: true, args: ['--no-sandbox'] });
const report = { source: repo, artifact: root, checks: [] };
const sent = text => received.some(frame => frame.type === 'message.send' && frame.text === text);
const waitFor = async (check, label) => {
  for (let tries = 0; tries < 200; tries += 1) {
    if (check()) return;
    await new Promise(done => setTimeout(done, 50));
  }
  throw new Error('等待超时：' + label);
};

// 2. 发出一问并等到生成中（发送钮变成停止钮）。
async function startTurn(page, text) {
  await page.locator('.composer textarea').fill(text);
  await page.keyboard.press('Enter');
  await waitFor(() => sent(text), '发送 ' + text);
  await page.waitForSelector('.composer-action-button[data-mode="stop"]');
}
async function steer(page, text) {
  await page.locator('.composer textarea').fill(text);
  await page.keyboard.press('Enter');
  await page.locator('.steer-card').filter({ hasText: text }).waitFor();
}
const draft = page => page.locator('.composer textarea').inputValue();

const context = await browser.newContext({ viewport: { width: 1365, height: 860 } });
const page = await context.newPage();
const errors = [];
page.on('pageerror', error => errors.push(error.message));
try {
  await page.goto(origin + '/?session=' + encodeURIComponent('akashic:long'));
  await page.waitForSelector('.web-message-anchor');

  // 3. 生成中 Esc 停止：发出 /stop，尚未交给模型的插话退回输入框且不发送。
  await startTurn(page, '第一问');
  await steer(page, '退回的话');
  await page.keyboard.press('Escape');
  await waitFor(() => sent('/stop'), 'Esc 发出 /stop');
  await page.waitForSelector('.steer-card', { state: 'detached' });
  assert.equal(await draft(page), '退回的话');
  assert.ok(!sent('退回的话'), '停止后插话仍被发出');
  report.checks.push({ case: 'escape stops and restores steer' });

  // 4. 引用一条回复：焦点进输入框，引用条显示给人看的名字。
  const reply = page.locator('.web-message-anchor.timeline-output').last();
  await reply.hover();
  await reply.getByRole('button', { name: '引用此消息' }).click();
  await page.waitForFunction(() => document.activeElement === document.querySelector('.composer textarea'));
  assert.equal(await page.locator('.composer-reply strong').innerText(), '回复 Akashic');
  await page.getByRole('button', { name: '取消引用' }).click();
  await page.locator('.composer textarea').fill('');
  report.checks.push({ case: 'reply focuses composer with readable author' });

  // 5. 空输入框按 ↑ 取回最后一条提问；有内容时 ↑ 不改写。
  await page.locator('.composer textarea').fill('');
  await page.keyboard.press('ArrowUp');
  await page.waitForFunction(() => document.querySelector('.composer textarea')?.value === '第一问');
  // 取回后下一帧才把光标移到末尾；等它落定再改写，否则 fill 的全选会被折叠成追加。
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  await page.locator('.composer textarea').fill('已有内容');
  await page.keyboard.press('ArrowUp');
  assert.equal(await draft(page), '已有内容');
  await page.locator('.composer textarea').fill('');
  report.checks.push({ case: 'arrow up recalls last input' });

  // 6. 读到中间切走再切回：同一条消息回到原偏移。
  const position = () => page.evaluate(() => {
    const scroller = document.querySelector('.conversation-scroll');
    const top = scroller.getBoundingClientRect().top;
    const anchor = [...scroller.querySelectorAll('.web-message-anchor[data-message-id]')]
      .find(item => item.getBoundingClientRect().bottom > top + 80);
    return { id: anchor.dataset.messageId, offset: Math.round(anchor.getBoundingClientRect().top - top),
      gap: Math.round(scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight) };
  });
  await page.mouse.move(700, 400);
  for (let step = 0; step < 6; step += 1) { await page.mouse.wheel(0, -500); await page.waitForTimeout(80); }
  await page.waitForTimeout(500);
  const before = await position();
  assert.ok(before.gap > 80, '没有离开底部');
  await page.locator('.conversation-session').filter({ hasText: '短会话' }).click();
  await page.waitForFunction(() => document.querySelectorAll('.web-message-anchor').length > 0
    && document.querySelector('.conversation-heading')?.textContent.includes('短会话'));
  await page.locator('.conversation-session').filter({ hasText: '长会话' }).click();
  await page.waitForFunction(() => document.querySelector('.conversation-heading')?.textContent.includes('长会话'));
  await page.waitForTimeout(800);
  const after = await position();
  assert.equal(after.id, before.id, '切回后不是同一条消息');
  assert.ok(Math.abs(after.offset - before.offset) <= 2, `偏移 ${before.offset} → ${after.offset}`);
  report.checks.push({ case: 'reading position restored', id: after.id, offset: [before.offset, after.offset] });

  assert.deepEqual(errors, []);
  writeFileSync(root + '/report.json', JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
} finally {
  writeFileSync(root + '/received.json', JSON.stringify(received, null, 2));
  await browser.close();
  await new Promise(done => ws.close(done));
  await new Promise(done => server.close(done));
}

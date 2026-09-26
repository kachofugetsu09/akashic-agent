/** 完整聊天页面的等待交接实验；HTTP/WS 使用受控本地夹具，不调用模型。 */
import { createServer } from 'node:http';
import { readFileSync, existsSync, appendFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { resolve, extname, dirname } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import assert from 'node:assert/strict';
import { WebSocketServer } from 'ws';
import { chromium } from 'playwright-core';
import { desktopModels } from '../../../scripts/webui-performance/fixtures.mjs';
const repo = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
const root = mkdtempSync(resolve(tmpdir(), 'akashic-reply-waiting-'));
const build = spawnSync(process.execPath, [resolve(repo, 'node_modules/vite/bin/vite.js'), 'build', '--config', 'frontend/chat/vite.config.ts', '--outDir', root + '/dist'], { cwd: repo, encoding: 'utf8' });
writeFileSync(root + '/build.log', build.stdout + build.stderr);
assert.equal(build.status, 0, '前端构建失败，见 ' + root + '/build.log');
const sockets = new Set(); const events = []; const histories = new Map();
const log = (direction, data) => { const event = { at: new Date().toISOString(), direction, data }; events.push(event); appendFileSync(root + '/frames.jsonl', JSON.stringify(event) + '\n'); };
const json = (res, data, status = 200) => { res.writeHead(status, { 'content-type': 'application/json' }); res.end(JSON.stringify(data)); };
const server = createServer(async (req, res) => {
  const path = new URL(req.url, 'http://localhost').pathname;
  if (path === '/lab/events') return json(res, events);
  if (path === '/lab/frame' && req.method === 'POST') {
    const chunks = []; for await (const chunk of req) chunks.push(chunk);
    const frame = JSON.parse(Buffer.concat(chunks));
    if (frame.type === 'messages.appended') histories.set(frame.session_id, [...(histories.get(frame.session_id) ?? []), ...frame.items]);
    log('sent', frame); for (const socket of sockets) socket.send(JSON.stringify(frame));
    return json(res, { clients: sockets.size });
  }
  if (path === '/api/shell/state') return json(res, { status: 'ready', configured: true, chatReady: true });
  if (path === '/api/chat/models') return json(res, desktopModels(1));
  if (path === '/api/chat/sessions') return json(res, { items: [...histories].map(([key, items]) => ({ key, first_message_content: items[0].body.parts[0].value, updated_at: items.at(-1).timestamp, message_count: items.length })), next_cursor: null });
  const history = path.match(/^\/api\/chat\/sessions\/([^/]+)\/messages$/);
  if (history) { const items = histories.get(decodeURIComponent(history[1])) ?? []; return json(res, { version: 2, items, through_seq: items.at(-1)?.seq ?? -1, has_more: false, before_seq: null }); }
  if (path === '/api/chat/plugin-ui/catalog') return json(res, { catalog_revision: '0'.repeat(64), items: [] });
  if (path === '/api/runtime/host-bridge') return json(res, { state: 'healthy', failures: 0 });
  if (path.startsWith('/api/')) { log('unexpected_http', path); return json(res, { error: 'unexpected fixture request' }, 404); }
  const file = resolve(root + '/dist', path === '/' || path === '/chat' ? 'index.html' : path.replace(/^\/assets\//, '').replace(/^\//, ''));
  if (!file.startsWith(root + '/dist/') || !existsSync(file)) return res.writeHead(404).end();
  res.writeHead(200, { 'content-type': ({ '.js': 'text/javascript', '.css': 'text/css', '.html': 'text/html', '.woff2': 'font/woff2', '.svg': 'image/svg+xml' })[extname(file)] ?? 'application/octet-stream' }); res.end(readFileSync(file));
});
const ws = new WebSocketServer({ server, path: '/ws' });
ws.on('connection', socket => { sockets.add(socket); log('connection', 'open'); socket.on('message', raw => log('received', JSON.parse(String(raw)))); socket.on('close', () => { sockets.delete(socket); log('connection', 'close'); }); });
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const origin = 'http://127.0.0.1:' + server.address().port;
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH ?? '/usr/bin/chromium', headless: true, args: ['--no-sandbox'] });
const results = [];
/** 每个场景通过真实输入框发一次消息，逐帧检查 DOM 与连接计数。 */
async function run(name, sequence) {
  const ctx = await browser.newContext({ viewport: { width: 1200, height: 850 } });
  await ctx.tracing.start({ screenshots: true, snapshots: true, sources: true });
  const page = await ctx.newPage(); const errors = []; let wsOpen = 0, wsClose = 0, expectedOpens = 1; const sent = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('websocket', ws => { wsOpen++; ws.on('close', () => wsClose++); ws.on('framesent', f => sent.push(JSON.parse(f.payload))); });
  try {
    await page.goto(origin); await page.getByRole('textbox', { name: '消息', exact: true }).fill('本地等待状态实验 ' + name);
    await page.getByRole('textbox', { name: '消息', exact: true }).press('Enter');
    await page.waitForFunction(() => document.querySelectorAll('.thinking-placeholder').length === 1);
    const deadline = Date.now() + 10000;
    while (!sent.some(f => f.type === 'message.send')) { assert.ok(Date.now() < deadline, '未观察到 message.send'); await page.evaluate(() => new Promise(requestAnimationFrame)); }
    const message = sent.find(f => f.type === 'message.send'); const session = message.session_id; const handle = 'experiment-' + name;
    const activity = (preview = null, active = true) => ({ session_id: session, source: 'akashic', handle, active, preview });
    const reply = items => ({ type: 'reply.status', version: 2, session_id: session, snapshot_id: 'local-experiment', available: true, items });
    const row = (id, seq, kind, text) => ({ id, seq, session_id: session, timestamp: new Date().toISOString(), author: kind === 'input' ? 'user' : 'assistant', source: 'akashic', attachments: [], metadata: {}, body: { kind, parts: [{ kind: 'text', value: text }], ...(kind === 'output' ? { finish: 'complete' } : {}) } });
    const append = (item, after) => ({ type: 'messages.appended', version: 2, session_id: session, after_seq: after, through_seq: item.seq, next_after_seq: item.seq, has_more: false, items: [item] });
    const api = { reload: async () => { expectedOpens += 2; await page.reload(); await page.locator('.conversation-session').filter({ hasText: name }).first().click(); }, activity, reply, ack: append(row(message.request_id, 0, 'input', message.text), -1), draft: { message_id: 'answer-' + name, text: '', thinking: '' }, commit: (text, finish = 'complete', source = 'akashic', seq = 1) => { const r = row('answer-' + name + '-' + seq, seq, 'output', text); r.id = seq === 1 ? 'answer-' + name : r.id; r.source = source; r.body.finish = finish; return append(r, seq - 1); }, control: (action, seq, through) => append({ ...row('control-' + seq, seq, 'input', ''), body: { kind: 'control', action, through_seq: through, reason: null } }, seq - 1) };
    const records = [];
    async function step(label, frame, waiting, mode, contains) {
      if (frame) { const r = await fetch(origin + '/lab/frame', { method: 'POST', body: JSON.stringify(frame) }); assert.equal(r.status, 200); }
      await page.waitForFunction(({ waiting, mode, contains }) => document.querySelectorAll('.thinking-placeholder').length === waiting && document.querySelector('.composer-action-button')?.dataset.mode === mode && (!contains || document.body.innerText.includes(contains)), { waiting, mode, contains });
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      const state = await page.evaluate(() => ({ waiting: document.querySelectorAll('.thinking-placeholder').length, button: document.querySelector('.composer-action-button')?.dataset.mode, activities: [...document.querySelectorAll('.reply-activity')].map(e => ({ busy: e.getAttribute('aria-busy'), text: e.innerText })), disconnected: document.body.innerText.includes('连接已断开') }));
      const file = name + '-' + records.length + '.png'; await page.screenshot({ path: root + '/' + file }); records.push({ label, ...state, screenshot: file }); console.log(name, label, JSON.stringify(state));
    }
    await step('发送后', null, 1, 'stop'); await sequence(api, step);
    assert.equal(errors.length, 0); assert.equal(wsOpen, expectedOpens); if (expectedOpens === 1) assert.equal(wsClose, 0);
    assert.equal(sockets.size, 1); assert.equal(sent.filter(f => f.type === 'message.send').length, 1);
    results.push({ name, records, errors, wsOpen, wsClose, messageSendCount: 1 });
  } finally { await ctx.tracing.stop({ path: root + '/' + name + '-trace.zip' }); await ctx.close(); }
}
try {
  await run('A-ack-first', async (a, step) => {
    await step('用户消息先落库', a.ack, 1, 'stop');
    await step('回复活动稍后到达', a.reply([a.activity()]), 1, 'stop');
    await step('空草稿已创建', a.reply([a.activity(a.draft)]), 1, 'stop');
    await step('正文到达', a.reply([a.activity({ ...a.draft, text: '这是最终回答 A' })]), 0, 'stop', '这是最终回答 A');
    await step('正文落库', a.commit('这是最终回答 A'), 0, 'stop', '这是最终回答 A');
    await step('活动结束', a.reply([]), 0, 'send', '这是最终回答 A');
  });
  await run('B-activity-first', async (a, step) => {
    await step('回复活动先到达', a.reply([a.activity()]), 1, 'stop');
    await step('用户消息随后落库', a.ack, 1, 'stop');
    await step('空草稿', a.reply([a.activity(a.draft)]), 1, 'stop');
    await step('有思考内容', a.reply([a.activity({ ...a.draft, thinking: '正在分析本地实验' })]), 0, 'stop', '正在分析本地实验');
  });
  await run('C-retain-preview', async (a, step) => {
    await step('确认输入', a.ack, 1, 'stop');
    await step('直接收到正文草稿', a.reply([a.activity({ ...a.draft, text: '已有正文 C' })]), 0, 'stop', '已有正文 C');
    await step('预览 scope 退出但活动仍在', a.reply([a.activity()]), 0, 'stop', '已有正文 C');
    await step('正文落库', a.commit('已有正文 C'), 0, 'stop', '已有正文 C');
    await step('活动结束', a.reply([]), 0, 'send', '已有正文 C');
  });
  await run('D-inactive', async (a, step) => {
    await step('确认输入', a.ack, 1, 'stop');
    await step('活动开始', a.reply([a.activity()]), 1, 'stop');
    await step('活动已撤权等待排空', a.reply([a.activity(null, false)]), 0, 'send');
    await step('暂停记录到达', a.control('pause', 1, 0), 0, 'send');
    await step('排空结束', a.reply([]), 0, 'send');
  });
  await run('E-quiet', async (a, step) => {
    await step('接纳输入', a.ack, 1, 'stop');
    await step('静默完成', a.commit('', 'quiet'), 0, 'send');
  });
  await run('F-controls', async (a, step) => {
    await step('接纳输入', a.ack, 1, 'stop');
    await step('暂停', a.control('pause', 1, 0), 0, 'send');
    await step('恢复', a.control('resume', 2, 1), 1, 'stop');
    await step('失败', a.control('failure', 3, 2), 0, 'send');
    await step('再次恢复', a.control('resume', 4, 3), 1, 'stop');
    await step('放弃', a.control('abandon', 5, 4), 0, 'send');
  });
  await run('G-source-isolation', async (a, step) => {
    await step('接纳输入', a.ack, 1, 'stop');
    await step('其他来源完成不结束本次等待', a.commit('另一来源', 'complete', 'other'), 1, 'stop');
    await step('本来源完成', a.commit('本次完成', 'complete', 'akashic', 2), 0, 'send', '本次完成');
  });
  await run('H-availability', async (a, step) => {
    await step('接纳输入', a.ack, 1, 'stop');
    await step('回复能力不可用', { ...a.reply([]), snapshot_id: null, available: false }, 0, 'send');
    await step('回复能力恢复', a.reply([]), 1, 'stop');
    await step('完成', a.commit('恢复后完成'), 0, 'send', '恢复后完成');
  });
  await run('I-history-reload', async (a, step) => {
    await step('接纳输入', a.ack, 1, 'stop');
    await a.reload();
    await step('重新打开历史仍显示等待', null, 1, 'stop');
    await step('空活动快照不清掉已接纳输入', a.reply([]), 1, 'stop');
    await step('完成', a.commit('重新打开后完成'), 0, 'send', '重新打开后完成');
  });
  writeFileSync(root + '/results.json', JSON.stringify(results, null, 2));
} finally { await browser.close(); for (const socket of sockets) socket.terminate(); await new Promise(resolve => server.close(resolve)); ws.close(); }
console.log('证据目录: ' + root);

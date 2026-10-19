/** 通过真实聊天页面验证排队、停止和草稿恢复；HTTP/WS 由隔离夹具控制，不调用模型。 */
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
const root = mkdtempSync(resolve(tmpdir(), 'akashic-follow-up-review-'));
const build = spawnSync(process.execPath, [resolve(repo, 'node_modules/vite/bin/vite.js'), 'build', '--config', 'frontend/chat/vite.harness.config.ts', '--outDir', root + '/dist'], { cwd: repo, encoding: 'utf8' });
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
const frame = async (data) => {
  const response = await fetch(origin + '/lab/frame', { method: 'POST', body: JSON.stringify(data) });
  assert.equal(response.status, 200);
};
const paint = page => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
/** 等待协议事件并保留明确超时，不把缺失发送挂成无限循环。 */
async function waitFor(page, condition) {
  const deadline = Date.now() + 10000;
  while (!condition()) {
    assert.ok(Date.now() < deadline, '没有收到预期协议事件');
    await paint(page);
  }
}

/** 从真实聊天入口发送并确认输入，后端帧由场景显式推进。 */
async function run(name, scenario, options = {}, beforeAck = null) {
  const ctx = await browser.newContext({ viewport: { width: 1365, height: 860 }, ...options });
  const page = await ctx.newPage();
  page.setDefaultTimeout(10000);
  const sent = []; const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('websocket', socket => socket.on('framesent', event => sent.push(JSON.parse(event.payload))));
  try {
    await page.goto(origin);
    const editor = page.getByRole('textbox', { name: '消息', exact: true });
    await editor.fill('排队回归 ' + name);
    await page.getByRole('button', { name: '发送消息', exact: true }).click();
    await page.waitForFunction(() => document.querySelector('.composer-action-button')?.dataset.mode === 'stop');
    await waitFor(page, () => sent.some(item => item.type === 'message.send'));
    const first = sent.find(item => item.type === 'message.send');
    const session = first.session_id;
    let seq = -1;
    const append = async (body, author = 'assistant', id, source = 'akashic') => {
      const item = { id: id ?? name + '-' + (seq + 1), seq: seq + 1, session_id: session, timestamp: new Date().toISOString(),
        source, author, attachments: [], metadata: {}, body };
      await frame({ type: 'messages.appended', version: 2, session_id: session, after_seq: seq, through_seq: ++seq, next_after_seq: seq, has_more: false, items: [item] });
      await paint(page);
    };
    const reply = async (active, retry = '') => {
      await frame({ type: 'reply.status', version: 2, session_id: session, snapshot_id: name, available: true,
        items: active === null ? [] : [{ session_id: session, source: 'akashic', handle: name, active,
          preview: active ? { message_id: 'preview-' + name, text: '', thinking: '', retry_status: retry } : null }] });
      await paint(page);
    };
    const output = async () => append({ kind: 'output', parts: [{ kind: 'text', value: '本轮正常完成' }], finish: 'complete' });
    const control = async action => append({ kind: 'control', action, through_seq: seq, reason: '本地回归' });
    const sends = () => sent.filter(item => item.type === 'message.send' && item.text !== '/stop');
    const queue = async text => {
      await editor.fill(text); await editor.press('Enter');
      await page.waitForFunction(text => document.querySelector('.composer-queued')?.textContent.includes(text), text);
    };
    const api = { page, editor, session, reply, output, control, queue, sends, append };
    if (beforeAck) await beforeAck(api);
    await append({ kind: 'input', parts: [{ kind: 'text', value: first.text }] }, 'user', first.request_id);
    await reply(true);
    await page.waitForFunction(() => document.querySelector('.composer-action-button')?.dataset.mode === 'stop');
    await scenario(api);
    assert.deepEqual(errors, []);
    await page.screenshot({ path: root + '/' + name + '.png', fullPage: false });
    results.push({ name, sent: sends().map(item => item.text), errors });
    console.log('PASS', name);
  } finally { await ctx.close(); }
}
try {
  await run('normal-finalizing', async a => {
    await a.queue('先排队'); await a.reply(false);
    await a.queue('收尾时追加');
    assert.equal(a.sends().length, 1);
    await a.editor.fill('保留后写草稿'); await a.output(); await a.reply(null);
    await a.page.waitForFunction(() => !document.querySelector('.composer-queued'));
    await waitFor(a.page, () => a.sends().length >= 2);
    assert.equal(a.sends()[1].text, '先排队\n\n收尾时追加');
    assert.equal(await a.editor.inputValue(), '保留后写草稿');
  });
  await run('before-ack', async a => {
    assert.ok((await a.page.locator('.composer-queued').innerText()).includes('接纳前排队'));
    assert.equal(await a.page.evaluate(key => sessionStorage.getItem('akashic.chat.draft:' + key), a.session), '接纳前排队');
    await a.output(); await a.reply(null);
    await waitFor(a.page, () => a.sends().length >= 2);
    assert.equal(a.sends()[1].text, '接纳前排队');
  }, {}, a => a.queue('接纳前排队'));
  await run('new-input', async a => {
    await a.queue('不能跟随另一条输入');
    await a.append({ kind: 'input', parts: [{ kind: 'text', value: '另一个客户端输入' }] }, 'user', 'other-input', 'other');
    await a.page.waitForFunction(() => !document.querySelector('.composer-queued'));
    assert.equal(await a.editor.inputValue(), '不能跟随另一条输入'); assert.equal(a.sends().length, 1);
  });
  await run('send-rejected', async a => {
    await a.queue('失败后保留'); await a.output(); await a.reply(null);
    await waitFor(a.page, () => a.sends().length >= 2);
    await a.editor.fill('后写草稿');
    await frame({ type: 'error', request_id: a.sends()[1].request_id, message: '本地夹具拒绝发送' });
    await a.page.waitForFunction(() => document.querySelector('textarea')?.value === '失败后保留\n\n后写草稿');
  });
  for (const source of ['composer', 'retry']) await run('stop-' + source, async a => {
    await a.reply(true, '正在重试'); await a.queue('停止后不要发送');
    await (source === 'composer' ? a.page.getByRole('button', { name: '中止回答', exact: true }) : a.page.locator('.retry-status button')).click();
    await a.control('abandon'); await a.reply(null);
    await a.page.waitForFunction(() => !document.querySelector('.composer-queued'));
    assert.equal(await a.editor.inputValue(), '停止后不要发送'); assert.equal(a.sends().length, 1);
  });
  for (const action of ['failure', 'pause', 'abandon']) await run('server-' + action, async a => {
    await a.queue('异常后保留'); await a.control(action); await a.reply(null);
    await a.page.waitForFunction(() => !document.querySelector('.composer-queued'));
    assert.equal(await a.editor.inputValue(), '异常后保留'); assert.equal(a.sends().length, 1);
  });
  await run('withdraw', async a => {
    await a.queue('撤回文字'); await a.editor.fill('后写草稿');
    await a.page.getByRole('button', { name: '撤回到输入框' }).click();
    assert.equal(await a.editor.inputValue(), '撤回文字\n\n后写草稿'); assert.equal(a.sends().length, 1);
  });
  await run('refresh', async a => {
    await a.queue('刷新不丢'); await a.editor.fill('后写内容');
    assert.equal(await a.page.evaluate(key => sessionStorage.getItem('akashic.chat.draft:' + key), a.session), '刷新不丢\n\n后写内容');
    await a.page.reload(); await a.editor.waitFor();
    await a.page.waitForFunction(() => document.querySelector('textarea')?.value === '刷新不丢\n\n后写内容');
    assert.equal(await a.page.locator('.composer-queued').count(), 0); assert.equal(a.sends().length, 1);
  });
  await run('switch-session', async a => {
    await a.queue('切走不丢'); await a.editor.fill('原会话草稿');
    await a.page.locator('.chat-sidebar__new').click();
    await paint(a.page);
    assert.equal(await a.editor.inputValue(), '');
    assert.equal(await a.page.evaluate(key => sessionStorage.getItem('akashic.chat.draft:' + key), a.session), '切走不丢\n\n原会话草稿');
    assert.equal(a.sends().length, 1);
  });
  await run('attachment-refused', async a => {
    await a.page.locator('input[type=file]').setInputFiles({ name: 'note.txt', mimeType: 'text/plain', buffer: Buffer.from('附件') });
    await a.editor.fill('带附件的追问'); await a.editor.press('Enter');
    await a.page.getByText('带附件或引用的消息，本轮结束后再发送', { exact: true }).waitFor();
    assert.equal(await a.editor.inputValue(), '带附件的追问');
    assert.equal(await a.page.locator('.composer-queued').count(), 0); assert.equal(a.sends().length, 1);
  });
  await run('ime', async a => {
    await a.editor.fill('你好');
    await a.editor.dispatchEvent('keydown', { key: 'Enter', keyCode: 229, bubbles: true });
    assert.equal(await a.editor.inputValue(), '你好'); assert.equal(await a.page.locator('.composer-queued').count(), 0);
  });
  await run('touch-enter', async a => {
    await a.editor.fill('第一行'); await a.editor.press('Enter'); await a.editor.type('第二行');
    assert.equal(await a.editor.inputValue(), '第一行\n第二行'); assert.equal(a.sends().length, 1);
    assert.equal(await a.page.locator('.composer-queued').count(), 0);
    assert.ok(await a.page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  }, { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  writeFileSync(root + '/results.json', JSON.stringify(results, null, 2));
} finally {
  await browser.close(); for (const socket of sockets) socket.terminate();
  await new Promise(resolve => server.close(resolve)); ws.close();
}
console.log('证据目录: ' + root);

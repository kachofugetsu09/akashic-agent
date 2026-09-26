/** 用当前 Message v2 协议验证流式草稿不重复渲染历史，不访问正式服务器。 */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFileSync, existsSync, writeFileSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { resolve, extname, dirname } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { build } from 'vite';
import { WebSocketServer } from 'ws';
import { chromium } from 'playwright-core';
import { desktopModels } from '../../../scripts/webui-performance/fixtures.mjs';

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
const output = resolve(process.argv[2] ?? 'artifacts/timeline-performance');
const baseline = process.argv.includes('--baseline');
const root = mkdtempSync(resolve(tmpdir(), 'akashic-timeline-perf-'));
const session = 'akashic:timeline-performance';
const count = 200;
const chunks = 120;
const timestamp = '2026-09-26T00:00:00Z';
const row = (seq, body, source = 'akashic') => ({ id: `message-${seq}`, seq, session_id: session,
  timestamp, author: body.kind === 'input' ? 'user' : 'assistant', source, attachments: [], metadata: {}, body });
const textBody = (kind, text) => ({ kind, parts: [{ kind: 'text', value: text }], ...(kind === 'output' ? { finish: 'complete' } : {}) });
const history = Array.from({ length: count }, (_, seq) => row(seq,
  textBody(seq % 2 ? 'output' : 'input', `历史消息 ${seq}\n\n这段**已经完成的文字**不应随下一条回复反复渲染。\n\n- 项目一\n- 项目二`)));
let items = history;
let socket;
let browser;
let server;
let ws;
mkdirSync(output, { recursive: true });

try {
  // 1. 只在实验构建插入计数器；生产源码和正常 bundle 不包含该计数器。
  await build({ root: repo + '/frontend/chat', configFile: repo + '/frontend/chat/vite.config.ts', logLevel: 'warn',
    build: { outDir: root + '/dist', emptyOutDir: true }, plugins: [{
      name: 'count-timeline-renders', enforce: 'pre', transform(code, id) {
        if (!id.endsWith('/desktop-conversation.tsx')) return;
        const marker = '  const byId = useMemo(() => new Map(messages.map((message) => [message.id, message])), [messages]);';
        assert.ok(code.includes(marker), '历史组件计数位置已变化');
        return code.replace(marker, '  window.__timelineRenders = (window.__timelineRenders ?? 0) + 1;\n' + marker);
      },
    }] });
  const json = (res, data, status = 200) => { res.writeHead(status, { 'content-type': 'application/json' }); res.end(JSON.stringify(data)); };
  server = createServer((req, res) => {
    const url = new URL(req.url, 'http://localhost');
    const path = url.pathname;
    if (path === '/api/shell/state') return json(res, { status: 'ready', configured: true, chatReady: true });
    if (path === '/api/chat/models') return json(res, desktopModels(1));
    if (path === '/api/chat/sessions') return json(res, { items: [{ key: session, first_message_content: '性能实验', updated_at: timestamp, message_count: items.length }], next_cursor: null });
    if (path.startsWith('/api/chat/sessions/') && path.endsWith('/messages')) {
      const end = Number(url.searchParams.get('before_seq') ?? items.length);
      const start = Math.max(0, end - 50);
      return json(res, { version: 2, items: items.slice(start, end), through_seq: Number(url.searchParams.get('through_seq') ?? items.at(-1).seq), has_more: start > 0, before_seq: start || null });
    }
    if (path === '/api/chat/plugin-ui/catalog') return json(res, { catalog_revision: '0'.repeat(64), items: [] });
    if (path === '/api/runtime/host-bridge') return json(res, { state: 'healthy', failures: 0 });
    if (path.startsWith('/api/')) return json(res, { error: 'unexpected fixture request', path }, 404);
    const file = resolve(root + '/dist', path === '/' ? 'index.html' : path.replace(/^\/assets\//, '').replace(/^\//, ''));
    if (!file.startsWith(root + '/dist/') || !existsSync(file)) return res.writeHead(404).end();
    res.writeHead(200, { 'content-type': ({ '.js': 'text/javascript', '.css': 'text/css', '.html': 'text/html', '.woff2': 'font/woff2' })[extname(file)] ?? 'application/octet-stream' });
    res.end(readFileSync(file));
  });
  ws = new WebSocketServer({ server, path: '/ws' });
  ws.on('connection', current => { socket = current; });
  await new Promise(done => server.listen(0, '127.0.0.1', done));
  const origin = 'http://127.0.0.1:' + server.address().port;
  browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH ?? '/usr/bin/chromium', headless: true });
  const results = [];
  // 2. 固定协议、历史、视口和 CPU 限速；逐帧交付相同的 120 次真实 WebSocket 草稿。
  for (const profile of [{ name: 'narrow', width: 412, height: 915, cpu: 4 }, { name: 'desktop', width: 1440, height: 1000, cpu: 1 }]) {
    for (let run = 1; run <= 3; run++) {
      items = [...history];
      const context = await browser.newContext({ viewport: profile, permissions: ['clipboard-read', 'clipboard-write'] });
      const page = await context.newPage();
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      const cdp = await context.newCDPSession(page);
      await cdp.send('Emulation.setCPUThrottlingRate', { rate: profile.cpu });
      await cdp.send('Performance.enable');
      await page.goto(origin + '/?session=' + encodeURIComponent(session));
      await page.locator('[data-message-id="message-199"]').waitFor();
      while (await page.getByRole('button', { name: '加载更早消息' }).count()) {
        const before = await page.locator('[data-message-seq]').count();
        await page.getByRole('button', { name: '加载更早消息' }).click();
        await page.waitForFunction(n => document.querySelectorAll('[data-message-seq]').length > n, before);
      }
      assert.equal(await page.locator('[data-message-seq]').count(), count);
      const activity = (text, source = 'akashic', handle = 'reply-1') => ({ session_id: session, source, handle, active: true, preview: { message_id: 'message-200', text, thinking: '' } });
      const send = frame => socket.send(JSON.stringify(frame));
      const reply = activities => send({ type: 'reply.status', version: 2, session_id: session, snapshot_id: 'fixture', available: true, items: activities });
      const settle = () => page.evaluate(() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done))));
      reply([activity('开始')]);
      await page.locator('.reply-activity').getByText('开始', { exact: true }).waitFor();
      await settle();
      await page.evaluate(() => {
        window.__timelineRenders = 0;
        window.__longTasks = [];
        window.__taskObserver = new PerformanceObserver(list => window.__longTasks.push(...list.getEntries().map(e => e.duration)));
        window.__taskObserver.observe({ type: 'longtask' });
      });
      const metrics = async () => Object.fromEntries((await cdp.send('Performance.getMetrics')).metrics.map(m => [m.name, m.value]));
      const before = await metrics();
      let text = '开始';
      for (let chunk = 0; chunk < chunks; chunk++) {
        text += ` 片${chunk}`;
        reply([activity(text)]);
        await page.waitForFunction(expected => document.querySelector('.reply-activity')?.textContent.includes(expected), text);
        await settle();
      }
      const after = await metrics();
      const measured = await page.evaluate(() => { window.__taskObserver.disconnect(); return { historyRenders: window.__timelineRenders, longTasks: window.__longTasks }; });
      const result = { profile: profile.name, run, historyCount: count, chunks,
        historyRenders: measured.historyRenders, scriptMs: (after.ScriptDuration - before.ScriptDuration) * 1000,
        taskMs: (after.TaskDuration - before.TaskDuration) * 1000,
        layoutMs: (after.LayoutDuration - before.LayoutDuration) * 1000,
        longTaskMaxMs: Math.max(0, ...measured.longTasks) };
      results.push(result);
      console.log(JSON.stringify(result));
      if (!baseline) assert.equal(measured.historyRenders, 0, '纯草稿更新不应渲染已完成历史');

      // 3. memo 不能挡住来源/身份/顺序变化、新消息提交、复制反馈和回复操作。
      for (const activities of [[activity(text, 'scheduled')], [activity(text, 'scheduled', 'reply-2')],
        [activity(text, 'scheduled', 'reply-2'), { ...activity('', 'akashic', 'reply-3'), preview: null }],
        [{ ...activity('', 'akashic', 'reply-3'), preview: null }, activity(text, 'scheduled', 'reply-2')]]) {
        const renders = await page.evaluate(() => window.__timelineRenders);
        reply(activities);
        await page.waitForFunction(n => window.__timelineRenders > n, renders);
      }
      const committed = row(200, textBody('output', text));
      items.push(committed);
      send({ type: 'messages.appended', version: 2, session_id: session, after_seq: 199, through_seq: 200, next_after_seq: 200, has_more: false, items: [committed] });
      reply([]);
      const last = page.locator('[data-message-id="message-200"]');
      await last.getByRole('button', { name: '引用此消息' }).click();
      await page.getByRole('button', { name: '取消引用' }).waitFor();
      await page.getByRole('button', { name: '取消引用' }).click();
      await last.getByRole('button', { name: '复制消息', exact: true }).click();
      await last.getByRole('button', { name: '已复制', exact: true }).waitFor();
      assert.equal(await page.evaluate(() => navigator.clipboard.readText()), text);
      assert.deepEqual(errors, []);
      if (run === 1) await page.screenshot({ path: output + '/' + profile.name + '.png' });
      await context.close();
    }
  }
  writeFileSync(output + '/results.json', JSON.stringify({ sourceCommit: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: repo, encoding: 'utf8' }).trim(), browser: await browser.version(), baseline, results }, null, 2) + '\n');
} finally {
  await browser?.close();
  ws?.clients.forEach(client => client.terminate());
  if (ws) await new Promise(done => ws.close(done));
  if (server) await new Promise(done => server.close(done));
  rmSync(root, { recursive: true, force: true });
}

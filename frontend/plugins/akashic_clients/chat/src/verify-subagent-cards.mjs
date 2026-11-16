/** 子任务时间线卡片、并发委派折叠组、悬停浮卡与 tool.result 插件渲染的真实浏览器验收 + Monkey 点击压测。 */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { existsSync, mkdirSync, readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { resolve, dirname, extname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { WebSocketServer } from 'ws';
import { chromium } from 'playwright-core';
import { desktopModels } from '../../../../../scripts/webui-performance/fixtures.mjs';

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '../../../../..');
const root = process.env.EVIDENCE_DIR
  ? (mkdirSync(process.env.EVIDENCE_DIR, { recursive: true }), resolve(process.env.EVIDENCE_DIR))
  : mkdtempSync(resolve(tmpdir(), 'akashic-subagent-cards-'));

// 1. 构建聊天宿主 harness
const build = spawnSync(process.execPath, [
  resolve(repo, 'node_modules/vite/bin/vite.js'), 'build',
  '--config', 'frontend/plugins/akashic_clients/chat/vite.harness.config.ts',
  '--outDir', root + '/dist',
], { cwd: repo, encoding: 'utf8' });
writeFileSync(root + '/build.log', build.stdout + build.stderr);
assert.equal(build.status, 0, '前端构建失败，见 ' + root + '/build.log');

// 2. 夹具消息：包含运行中的并发委派批次 + 独立单任务
const base = Date.UTC(2026, 10, 17, 1, 30, 0);
const at = (seconds) => new Date(base + seconds * 1000).toISOString();

function msg(id, seq, author, actor, body, seconds) {
  return { id, seq, session_id: 'cards-live', timestamp: at(seconds), author, actor, source: 'akashic', attachments: [], metadata: {}, body };
}

const LIVE_ITEMS = [
  msg('u-1', 0, 'user', 'user', { kind: 'input', parts: [{ kind: 'text', value: '并行调研这三个方向' }] }, 0),
  msg('a-1', 1, 'assistant', 'model', {
    kind: 'output',
    finish: 'continue',
    parts: [
      { kind: 'tool_call', binding_id: 'b-spawn-1', name: 'spawn', owner: 'subagent', arguments: {
        label: '调研后端架构', task: '检查 plugins/subagent 状态流转', connection: 'Magpie', model: 'deepseek-chat', reasoning_effort: 'high', fork: true, cwd: '/srv/app',
      } },
      { kind: 'tool_call', binding_id: 'b-spawn-2', name: 'spawn', owner: 'subagent', arguments: {
        label: '修改隔离分支测试', task: '修复 worktree 用例', isolation: 'worktree', read_only: false,
      } },
      { kind: 'tool_call', binding_id: 'b-spawn-3', name: 'spawn', owner: 'subagent', arguments: {
        label: '检索前端组件树', task: '只读扫描 message-view.tsx', read_only: true,
      } },
    ],
  }, 1),
  // 第二个（index 1）失败，第三个（index 2）成功，第一个（index 0）仍在运行
  msg('r-2', 2, 'tool', 'tool', {
    kind: 'tool_result',
    call_ref: { message_id: 'a-1', part_index: 1 },
    outcome: 'error',
    parts: [{ kind: 'text', value: '[子任务「修改隔离分支测试」结果]\n退出原因: failed\n\nAssertionError: 端口已被占用' }],
  }, 4),
  msg('r-3', 3, 'tool', 'tool', {
    kind: 'tool_result',
    call_ref: { message_id: 'a-1', part_index: 2 },
    outcome: 'success',
    parts: [{ kind: 'text', value: '[子任务「检索前端组件树」结果]\n退出原因: completed\n\n已定位 TimelineProcess 与 CollapsedToolGroup 所在位置。' }],
  }, 6),
  // 再挂一个独立的单卡（已接纳后台任务）
  msg('a-2', 4, 'assistant', 'model', {
    kind: 'output',
    finish: 'continue',
    parts: [
      { kind: 'text', value: '并行调研已派发，下面单独挂载后台压测任务。' },
      { kind: 'tool_call', binding_id: 'b-spawn-4', name: 'spawn', owner: 'subagent', arguments: {
        label: '后台压测回归', task: '运行完整回归', run_in_background: true,
      } },
    ],
  }, 8),
  msg('r-4', 5, 'tool', 'tool', {
    kind: 'tool_result',
    call_ref: { message_id: 'a-2', part_index: 1 },
    outcome: 'success',
    parts: [{ kind: 'text', value: '已创建后台任务「后台压测回归」（job_id=deadbeef12345678）。完成后会回传当前会话。' }],
  }, 9),
];

const runnerJs = readFileSync(resolve(repo, 'plugins/subagent/runner_ui.js'), 'utf8');
const runnerCss = readFileSync(resolve(repo, 'plugins/subagent/runner_ui.css'), 'utf8');
const sha256 = (s) => createHash('sha256').update(s).digest('hex');
const runnerModuleHash = sha256(runnerJs);
const runnerCssHash = sha256(runnerCss);

const catalogPayload = {
  catalog_revision: 'c'.repeat(64),
  items: [
    {
      id: 'subagent',
      revision: '1',
      module_sha256: runnerModuleHash,
      stylesheet_sha256: runnerCssHash,
      slots: ['header.meta', 'tool.result'],
    },
  ],
};

const sockets = new Set();
const json = (res, data, status = 200) => {
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(JSON.stringify(data));
};

const server = createServer((req, res) => {
  const path = new URL(req.url, 'http://127.0.0.1').pathname;
  if (path === '/api/shell/state') return json(res, { status: 'ready', configured: true, chatReady: true });
  if (path === '/api/chat/models') return json(res, desktopModels(1));
  if (path === '/api/chat/navigation/pins') return json(res, { pins: [], sessions: [] });
  if (path === '/api/chat/sessions') {
    return json(res, {
      items: [{ key: 'cards-live', first_message_content: '并行调研这三个方向', updated_at: LIVE_ITEMS.at(-1).timestamp, message_count: LIVE_ITEMS.length }],
      next_cursor: null,
    });
  }
  if (path === '/api/chat/sessions/cards-live/messages') {
    return json(res, { version: 2, items: LIVE_ITEMS, through_seq: LIVE_ITEMS.at(-1).seq, has_more: false, before_seq: null });
  }
  if (path === '/api/chat/plugin-ui/catalog') return json(res, catalogPayload);
  if (path === '/api/chat/plugin-ui/asset') {
    const kind = new URL(req.url, 'http://127.0.0.1').searchParams.get('kind');
    if (kind === 'stylesheet') {
      res.writeHead(200, { 'content-type': 'text/css' });
      return res.end(runnerCss);
    }
    res.writeHead(200, { 'content-type': 'text/javascript' });
    return res.end(runnerJs);
  }
  if (path === '/api/runtime/host-bridge') return json(res, { state: 'healthy', failures: 0 });
  if (path.startsWith('/api/')) return json(res, { error: 'unexpected: ' + path }, 404);
  const file = resolve(root + '/dist', path === '/' || path === '/chat' ? 'index.html' : path.replace(/^\/assets\//, ''));
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
    if (frame.type === 'session.follow') {
      socket.send(JSON.stringify({
        type: 'session.following', version: 2, session_id: frame.session_id,
        request_id: frame.request_id, through_seq: LIVE_ITEMS.at(-1).seq,
      }));
      socket.send(JSON.stringify({
        type: 'reply.status', version: 2, session_id: frame.session_id,
        snapshot_id: 'live-snap', available: true,
        items: [{ session_id: frame.session_id, source: 'akashic', handle: 'h1', active: true, preview: null }],
      }));
    }
  });
  socket.on('close', () => sockets.delete(socket));
});
server.on('upgrade', (request, socket, head) => {
  const { pathname } = new URL(request.url, 'http://127.0.0.1');
  if (pathname !== '/ws') return socket.destroy();
  wss.handleUpgrade(request, socket, head, (ws) => wss.emit('connection', ws, request));
});
await new Promise((done) => server.listen(0, '127.0.0.1', done));
const origin = 'http://127.0.0.1:' + server.address().port;

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH ?? '/usr/bin/chromium',
  headless: true,
  args: ['--no-sandbox', '--disable-dev-shm-usage'],
});

try {
  // Desktop verification + hover tooltip + plugin tool.result + monkey click storm
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 820 } });
  const page = await ctx.newPage();
  page.setDefaultTimeout(15_000);
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));

  await page.goto(origin + '/?session=cards-live');
  await page.locator('.delegated-group').first().waitFor();

  const groupText = await page.locator('.delegated-group-summary').first().innerText();
  assert.match(groupText, /3 个委派任务/);
  assert.match(groupText, /1 运行中 · 1 失败 · 1 完成/);
  assert.equal(await page.locator('.delegated-group .delegated-card.is-member').count(), 3);
  assert.equal(await page.locator('.delegated-card.is-single').count(), 1);

  // Hover first member to trigger PaperTooltipLayer
  const firstMember = page.locator('.delegated-card.is-member .delegated-row-trigger').first();
  await firstMember.hover();
  await page.locator('.delegated-hover-card').waitFor({ timeout: 5000 });
  const tipText = await page.locator('.delegated-hover-card').innerText();
  assert.match(tipText, /调研后端架构/);
  assert.match(tipText, /deepseek-chat/);

  // Click 3rd member (completed) to expand and verify plugins/subagent tool.result card mounts
  await page.locator('.delegated-card.is-member .delegated-row-trigger').nth(2).click();
  await page.waitForFunction(() => document.querySelectorAll('.subagent-tool-result__badge').length >= 2);
  const badges = (await page.locator('.subagent-tool-result__badge').allInnerTexts()).map(t => t.trim());
  assert.deepEqual(badges, ['失败', '已完成']);

  await page.screenshot({ path: root + '/desktop-subagent-cards.png', animations: 'disabled' });

  // Monkey click storm on group toggle and single card toggle
  const groupToggle = page.locator('.delegated-group-summary').first();
  const singleToggle = page.locator('.delegated-card.is-single .delegated-row-trigger').first();
  for (let i = 0; i < 8; i += 1) {
    await groupToggle.click();
    await singleToggle.click();
  }
  // Ensure group toggle ended expanded (initial true, 8 toggles -> true)
  assert.equal(await groupToggle.getAttribute('aria-expanded'), 'true');
  assert.equal(await singleToggle.getAttribute('aria-expanded'), 'false');

  // Verify no horizontal overflow on desktop
  const desktopOverflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
  assert.equal(desktopOverflow, false, '桌面端不得横向溢出');
  assert.deepEqual(errors, [], '桌面端页面运行时错误: ' + errors.join('; '));
  await ctx.close();

  // Mobile viewport verification
  const mctx = await browser.newContext({ viewport: { width: 390, height: 844 } });
  const mpage = await mctx.newPage();
  mpage.setDefaultTimeout(15_000);
  const merrors = [];
  mpage.on('pageerror', (e) => merrors.push(e.message));
  await mpage.goto(origin + '/?session=cards-live');
  await mpage.locator('.delegated-group').first().waitFor();
  const mobileOverflow = await mpage.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
  assert.equal(mobileOverflow, false, '移动端不得横向溢出');
  await mpage.screenshot({ path: root + '/mobile-subagent-cards.png', animations: 'disabled' });
  assert.deepEqual(merrors, [], '移动端页面运行时错误: ' + merrors.join('; '));
  await mctx.close();

  console.log('PASS verify-subagent-cards (desktop + mobile + hover + tool.result + monkey storm)');
  console.log('EVIDENCE_DIR=' + root);
} finally {
  await browser.close();
  for (const s of sockets) s.terminate();
  await new Promise((done) => server.close(done));
  wss.close();
}

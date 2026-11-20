/** 工具图标插件化的真实浏览器验收：vite harness 构建真实聊天页，挂载 userinformation/plugin_ui.js，验证静态声明、自定义 SVG 矢量、动态 tool.icon 覆写、历史兜底 Wrench 与亮/暗主题截图。 */
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
  : mkdtempSync(resolve(tmpdir(), 'akashic-tool-icons-'));

// 1. 构建真实聊天宿主 harness 与插件 UI 产物
const build = spawnSync(process.execPath, [
  resolve(repo, 'node_modules/vite/bin/vite.js'), 'build',
  '--config', 'frontend/plugins/akashic_clients/chat/vite.harness.config.ts',
  '--outDir', root + '/dist',
], { cwd: repo, encoding: 'utf8' });
writeFileSync(root + '/build.log', build.stdout + build.stderr);
assert.equal(build.status, 0, '前端构建失败，见 ' + root + '/build.log');

const userInfoUiPath = resolve(repo, 'plugins/userinformation/plugin_ui.js');
assert.ok(existsSync(userInfoUiPath), '缺少 plugins/userinformation/plugin_ui.js，请先运行 npm run build:web-plugins');
const userInfoJs = readFileSync(userInfoUiPath, 'utf8');
const userInfoHash = createHash('sha256').update(userInfoJs).digest('hex');

const CODEX_SVG = {
  kind: 'svg',
  viewBox: '0 0 24 24',
  nodes: [
    { tag: 'path', attrs: { d: 'M12 2.5 20 7v10l-8 4.5L4 17V7Z' } },
    { tag: 'path', attrs: { d: 'm8.5 9.5 3 2.5-3 2.5' } },
    { tag: 'path', attrs: { d: 'M13 14.5h3' } },
  ],
};

const CLAUDE_SVG = {
  kind: 'svg',
  viewBox: '0 0 24 24',
  nodes: [
    { tag: 'path', attrs: { d: 'M12 3v18' } },
    { tag: 'path', attrs: { d: 'M3 12h18' } },
    { tag: 'path', attrs: { d: 'm5.6 5.6 12.8 12.8' } },
    { tag: 'path', attrs: { d: 'M5.6 18.4 18.4 5.6' } },
  ],
};

const STEPS = [
  { name: 'shell', owner: 'shell', icon: { kind: 'lucide', name: 'terminal' }, args: { command: 'pytest tests/test_tool_icons.py -v', description: '运行工具图标后端契约测试' }, result: '19 passed' },
  { name: 'write_stdin', owner: 'shell', icon: { kind: 'lucide', name: 'corner-down-left' }, args: { execution_id: 1, chars: '\n' }, result: 'done' },
  { name: 'read_file', owner: 'files', icon: { kind: 'lucide', name: 'file-text' }, args: { path: 'plugins/tools/icon.py' }, result: 'def normalize_tool_icon...' },
  { name: 'edit_file', owner: 'files', icon: { kind: 'lucide', name: 'pencil' }, args: { path: 'plugins/tools/plugin.py' }, result: 'updated' },
  { name: 'web_search', owner: 'standard_web', icon: { kind: 'lucide', name: 'search' }, args: { query: 'lucide-react tree shaking' }, result: '8 results' },
  { name: 'design_render', owner: 'design', icon: { kind: 'lucide', name: 'pen-tool' }, args: { jsx: '<Frame ... />' }, result: 'rendered' },
  { name: 'codemode', owner: 'codemode', icon: { kind: 'lucide', name: 'code-2' }, args: { code: 'return await tools.read_file(...)' }, result: 'ok' },
  { name: 'delegate_codex', owner: 'subagent_codex', icon: CODEX_SVG, args: { task: '审查插件契约与兼容性' }, result: 'accepted' },
  { name: 'delegate_claude_code', owner: 'subagent_claude_code', icon: CLAUDE_SVG, args: { task: '只读核对 UI 边界' }, result: 'accepted' },
  { name: 'user_information_list', owner: 'userinformation', icon: { kind: 'lucide', name: 'user-round-search' }, args: {}, result: '{"providers": ["fitbit@github", "steam@github"]}' },
  { name: 'user_information_query', owner: 'userinformation', icon: { kind: 'lucide', name: 'user-round-search' }, args: { provider: 'fitbit@github', query: 'sleep_report' }, result: '{"hours": 7.4}' },
  { name: 'user_information_query', owner: 'userinformation', icon: { kind: 'lucide', name: 'user-round-search' }, args: { provider: 'steam@github', query: 'recent_games' }, result: '{"games": []}' },
  { name: 'legacy_unregistered_tool', icon: { kind: 'future-kind' }, args: { description: '旧归档或未知图标类型，宽容兜底显示 Wrench' }, result: 'ok' },
];

const base = Date.UTC(2026, 10, 20, 7, 0, 0);
const at = (seconds) => new Date(base + seconds * 1000).toISOString();
const row = (id, seq, author, actor, body, seconds) => ({
  id, seq, session_id: 'tool-icons-demo', timestamp: at(seconds), author, actor, source: 'akashic', attachments: [], metadata: {}, body,
});

const ITEMS = [
  row('u-1', 0, 'user', 'user', { kind: 'input', parts: [{ kind: 'text', value: '展示一下各个插件工具图标改造后的实际效果' }] }, 0),
  row('a-1', 1, 'assistant', 'model', {
    kind: 'output',
    finish: 'continue',
    parts: [
      { kind: 'thinking.0', display: 'thinking', text: '依次调用各插件工具并验证图标投影。' },
      ...STEPS.map((s, idx) => ({
        kind: 'tool_call',
        binding_id: `b-${idx}`,
        name: s.name,
        ...(s.owner ? { owner: s.owner } : {}),
        ...(s.icon ? { icon: s.icon } : {}),
        arguments: s.args,
      })),
    ],
  }, 1),
  ...STEPS.map((s, idx) => row(`r-${idx}`, 2 + idx, 'tool', 'tool', {
    kind: 'tool_result',
    call_ref: { message_id: 'a-1', part_index: idx + 1 },
    outcome: 'success',
    parts: [{ kind: 'text', value: s.result }],
  }, 2 + idx)),
  row('a-2', 2 + STEPS.length, 'assistant', 'model', {
    kind: 'output',
    finish: 'complete',
    parts: [{ kind: 'text', value: '各插件工具已分别渲染各自的单色图标，其中 `delegate_codex` / `delegate_claude_code` 使用插件自带 SVG，`user_information_query` 根据 `provider` 动态切换图标，未声明图标的工具保持兜底扳手。' }],
  }, 20),
];

const LIVE_ITEMS = [
  { id: 'live-u', seq: 0, session_id: 'tool-icons-live', timestamp: at(0), author: 'user', actor: 'user', source: 'akashic', attachments: [], metadata: {}, body: { kind: 'input', parts: [{ kind: 'text', value: '运行态着色测试' }] } },
  { id: 'live-a', seq: 1, session_id: 'tool-icons-live', timestamp: at(1), author: 'assistant', actor: 'model', source: 'akashic', attachments: [], metadata: {}, body: {
    kind: 'output', finish: 'continue', parts: [
      { kind: 'tool_call', binding_id: 'b-live-0', name: 'shell', owner: 'shell', icon: { kind: 'lucide', name: 'terminal' }, arguments: { command: 'npm test' } },
      { kind: 'tool_call', binding_id: 'b-live-1', name: 'edit_file', owner: 'files', icon: { kind: 'lucide', name: 'pencil' }, arguments: { path: 'app.ts' } },
    ],
  } },
  { id: 'live-r0', seq: 2, session_id: 'tool-icons-live', timestamp: at(2), author: 'tool', actor: 'tool', source: 'akashic', attachments: [], metadata: {}, body: {
    kind: 'tool_result', call_ref: { message_id: 'live-a', part_index: 0 }, outcome: 'success', parts: [{ kind: 'text', value: 'ok' }],
  } },
];

const catalogPayload = {
  catalog_revision: 'a'.repeat(64),
  items: [
    {
      id: 'userinformation',
      revision: '1',
      module_sha256: userInfoHash,
      stylesheet_sha256: null,
      slots: ['tool.icon'],
    },
  ],
};

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
      items: [
        { key: 'tool-icons-demo', first_message_content: '展示一下各个插件工具图标改造后的实际效果', updated_at: ITEMS.at(-1).timestamp, message_count: ITEMS.length },
        { key: 'tool-icons-live', first_message_content: '运行态着色测试', updated_at: LIVE_ITEMS.at(-1).timestamp, message_count: LIVE_ITEMS.length },
      ],
      next_cursor: null,
    });
  }
  if (path === '/api/chat/sessions/tool-icons-demo/messages') {
    return json(res, { version: 2, items: ITEMS, through_seq: ITEMS.at(-1).seq, has_more: false, before_seq: null });
  }
  if (path === '/api/chat/sessions/tool-icons-live/messages') {
    return json(res, { version: 2, items: LIVE_ITEMS, through_seq: LIVE_ITEMS.at(-1).seq, has_more: false, before_seq: null });
  }
  if (path === '/api/chat/plugin-ui/catalog') return json(res, catalogPayload);
  if (path === '/api/chat/plugin-ui/asset') {
    res.writeHead(200, { 'content-type': 'text/javascript' });
    return res.end(userInfoJs);
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
  socket.on('message', (raw) => {
    const frame = JSON.parse(String(raw));
    if (frame.type === 'session.follow') {
      socket.send(JSON.stringify({
        type: 'session.following', version: 2, session_id: frame.session_id,
        request_id: frame.request_id,
        through_seq: frame.session_id === 'tool-icons-live' ? LIVE_ITEMS.at(-1).seq : ITEMS.at(-1).seq,
      }));
      socket.send(JSON.stringify({
        type: 'reply.status', version: 2, session_id: frame.session_id,
        snapshot_id: frame.session_id === 'tool-icons-live' ? 'live-snap' : 'settled-snap',
        available: true,
        items: frame.session_id === 'tool-icons-live'
          ? [{ session_id: frame.session_id, source: 'akashic', handle: 'h1', active: true, preview: null }]
          : [],
      }));
    }
  });
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
  const ctx = await browser.newContext({ viewport: { width: 1080, height: 860 }, deviceScaleFactor: 2 });
  const page = await ctx.newPage();
  page.setDefaultTimeout(15_000);
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));

  await page.goto(origin + '/?session=tool-icons-demo');
  await page.locator('.turn-process-summary').waitFor();
  // 展开整轮过程，查看所有 13 个工具行
  await page.locator('.turn-process-summary').click();
  await page.waitForFunction(() => {
    const svgs = [...document.querySelectorAll('.tool-step .tool-step-icon')];
    return svgs.length === 13 && new Set(svgs.map((s) => s.innerHTML)).size === 13;
  });
  await page.waitForTimeout(200);

  // 收集每个工具行图标的 SVG 结构、尺寸、颜色
  const rows = await page.evaluate(() => {
    return [...document.querySelectorAll('.tool-step')].map((el) => {
      const title = el.querySelector('.tool-step-title span')?.textContent?.trim() ?? '';
      const svg = el.querySelector('.tool-step-icon');
      const rect = svg?.getBoundingClientRect();
      const children = svg ? [...svg.children].map((c) => c.tagName.toLowerCase()) : [];
      return {
        title,
        width: rect ? Math.round(rect.width) : 0,
        height: rect ? Math.round(rect.height) : 0,
        inner: svg?.innerHTML ?? '',
        children,
        stroke: svg?.getAttribute('stroke') ?? '',
      };
    });
  });

  assert.equal(rows.length, 13, '应渲染 13 个工具步骤，实际：' + rows.length);
  for (const r of rows) {
    assert.equal(r.width, 14, `${r.title} 图标宽度应为 14px，实际 ${r.width}`);
    assert.equal(r.height, 14, `${r.title} 图标高度应为 14px，实际 ${r.height}`);
    assert.ok(r.inner.length > 0, `${r.title} SVG 内容不得为空`);
  }

  // 验证彼此区分度：13 个工具分别对应 13 个不同的图标！
  const distinctSvgs = new Set(rows.map((r) => r.inner));
  assert.equal(distinctSvgs.size, 13, '13 个工具图标应各不相同，实际去重后仅 ' + distinctSvgs.size + ' 个');

  // 验证自定义 SVG（delegate_codex 与 delegate_claude_code）
  const codexRow = rows.find((r) => r.title === 'delegate_codex');
  assert.deepEqual(codexRow.children, ['path', 'path', 'path'], 'delegate_codex 应渲染 3 条自定义 SVG path');
  assert.equal(codexRow.stroke, 'currentColor');

  const claudeRow = rows.find((r) => r.title === 'delegate_claude_code');
  assert.deepEqual(claudeRow.children, ['path', 'path', 'path', 'path'], 'delegate_claude_code 应渲染 4 条自定义 SVG path');

  // 截图 1：纸感亮色模式
  await page.screenshot({ path: root + '/tool-icons-light.png', fullPage: true });

  // 截图 2：墨纸暗色模式
  await page.evaluate(() => { document.documentElement.dataset.theme = 'dark'; });
  await page.waitForTimeout(150);
  await page.screenshot({ path: root + '/tool-icons-dark.png', fullPage: true });

  // 验证运行态着色（.tool-step.active .tool-step-icon 颜色区别于已完成图标）
  await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
  await page.goto(origin + '/?session=tool-icons-live');
  await page.locator('.tool-step.active .tool-step-icon').waitFor();
  const colors = await page.evaluate(() => {
    const done = document.querySelector('.tool-step:not(.active) .tool-step-icon');
    const active = document.querySelector('.tool-step.active .tool-step-icon');
    return {
      done: done ? getComputedStyle(done).color : '',
      active: active ? getComputedStyle(active).color : '',
    };
  });
  assert.ok(colors.done && colors.active && colors.done !== colors.active,
    `运行中工具图标应使用 trace 状态色（done=${colors.done}, active=${colors.active}）`);

  assert.deepEqual(errors, [], '页面出现未捕获错误: ' + JSON.stringify(errors));
  writeFileSync(root + '/results.json', JSON.stringify({ passed: true, rows: rows.map((r) => ({ title: r.title, width: r.width, height: r.height })) }, null, 2));
  console.log('verify-tool-icons passed! Evidence at:', root);
} finally {
  await browser.close();
  server.close();
}

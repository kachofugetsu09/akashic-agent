/**
 * 过程区折叠的真实浏览器验收：vite harness 构建真实聊天页，HTTP/WS 走本地夹具，
 * playwright-core 驱动真 Chromium，断言两条渲染路径与展开动效，并留下截图。
 *
 * 场景 A「运行中」：末条输出 finish=continue、会话没收到 reply.status → 轮次未落定，
 *   过程逐行渲染（TimelineProcess groupTools=true），6 步里前 3 步收进「3 个更早步骤」，
 *   尾部 3 步逐行可见。
 * 场景 B「已落定」：末条输出 finish=complete 且收到 reply.status → 整轮过程收进摘要行，
 *   展开后逐行恢复、不再套一层折叠（外层摘要已经收过一次）。
 *
 * 不调用模型、不碰真实后端。宿主缺 Chromium 依赖时用 CHROMIUM_PATH + LD_LIBRARY_PATH 指向自备副本；
 * 宿主没有字体时用 FONTCONFIG_FILE 指向自备字体，否则截图全是方框。
 *
 * 用法：
 *   node frontend/chat/src/verify-process-fold.mjs
 * 环境变量：
 *   CHROMIUM_PATH     浏览器可执行文件（默认 /usr/bin/chromium）
 *   LD_LIBRARY_PATH   自备 Chromium 的依赖目录
 *   FONTCONFIG_FILE   自备 fontconfig（宿主无字体时需要）
 *   EVIDENCE_DIR      证据目录（默认 mkdtemp，结束时打印）
 */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { existsSync, mkdirSync, readFileSync, writeFileSync, appendFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { resolve, dirname, extname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { WebSocketServer } from 'ws';
import { chromium } from 'playwright-core';
import { desktopModels } from '../../../scripts/webui-performance/fixtures.mjs';

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
const root = process.env.EVIDENCE_DIR
  ? (mkdirSync(process.env.EVIDENCE_DIR, { recursive: true }), resolve(process.env.EVIDENCE_DIR))
  : mkdtempSync(resolve(tmpdir(), 'akashic-process-fold-'));

// ---- 1. 构建真实聊天页（harness 入口，非发行包）----
const build = spawnSync(process.execPath, [
  resolve(repo, 'node_modules/vite/bin/vite.js'), 'build',
  '--config', 'frontend/chat/vite.harness.config.ts',
  '--outDir', root + '/dist',
], { cwd: repo, encoding: 'utf8' });
writeFileSync(root + '/build.log', build.stdout + build.stderr);
assert.equal(build.status, 0, '前端构建失败，见 ' + root + '/build.log');

// ---- 2. 夹具会话 ----
// 协议约束：带 tool_call 的输出必须是 finish=continue（readTimelineMessage 校验），
// 工具结果各自成一条 tool_result 消息，按 (message_id, part_index) 指回调用。
const STEPS = [
  { tool: 'read', args: { path: '/srv/app/core/turn.py' }, result: 'def run():\n    pass' },
  { tool: 'grep', args: { query: 'foldable' }, result: 'core/turn.py:12: foldable' },
  { tool: 'write', args: { path: '/srv/app/core/turn.py' }, result: 'wrote 42 bytes' },
  { tool: 'shell', args: { command: 'pytest -q' }, result: '3 passed' },
  { tool: 'read', args: { path: '/srv/app/README.md' }, result: '# 项目' },
  { tool: 'grep', args: { query: 'durationMs' }, result: 'message-view.tsx:1' },
];
const SESSIONS = { live: 'fold-live', settled: 'fold-settled' };
const base = Date.UTC(2026, 9, 22, 2, 0, 0);
const at = (seconds) => new Date(base + seconds * 1000).toISOString();

function buildHistory(session, settled) {
  const row = (id, seq, author, body, seconds) => ({
    id, seq, session_id: session, timestamp: at(seconds), author, source: 'akashic',
    attachments: [], metadata: {}, body,
  });
  const items = [row('fold-input', 0, 'user', { kind: 'input', parts: [{ kind: 'text', value: '看一下折叠' }] }, 0)];
  // 折叠按「单条消息的过程」计算：6 个步骤（思考/工具交替）必须落在同一条 output 里，
  // 否则每行各成一个 process-trace，谁也凑不满 3 步。带 tool_call 的输出只能是 continue。
  items.push(row('fold-out', 1, 'assistant', {
    kind: 'output',
    parts: STEPS.flatMap((step, index) => [
      { kind: `thinking.${index}`, display: 'thinking', text: `第 ${index + 1} 步：先想想 ${step.tool}` },
      { kind: 'tool_call', binding_id: 'fixture-tool', name: step.tool, arguments: step.args },
    ]),
    finish: 'continue',
  }, 1));
  STEPS.forEach((step, index) => {
    items.push(row(`fold-result-${index}`, 2 + index, 'tool', {
      kind: 'tool_result',
      parts: [{ kind: 'text', value: step.result }],
      call_ref: { message_id: 'fold-out', part_index: index * 2 + 1 },
      outcome: 'success',
    }, 2 + index));
  });
  const last = 2 + STEPS.length;
  // 已落定场景用一条正文收尾（同轮多段输出会合并成一条时间线行）；
  // 运行中场景停在 continue，等实时活动接管。
  items.push(row('fold-answer', last, 'assistant', settled
    ? { kind: 'output', parts: [{ kind: 'text', value: '看完了，六个步骤都在。' }], finish: 'complete' }
    : { kind: 'output', parts: [{ kind: 'text', value: '还在跑。' }], finish: 'continue' }, 40));
  return items;
}
const histories = {
  [SESSIONS.live]: buildHistory(SESSIONS.live, false),
  [SESSIONS.settled]: buildHistory(SESSIONS.settled, true),
};

// ---- 3. 夹具服务器：HTTP + /ws（升级自己处理，ws 的 path 选项不认 query）----
const sockets = new Set();
const events = [];
const log = (direction, data) => {
  const event = { at: new Date().toISOString(), direction, data };
  events.push(event);
  appendFileSync(root + '/frames.jsonl', JSON.stringify(event) + '\n');
};
const json = (res, data, status = 200) => {
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(JSON.stringify(data));
};
// 每个场景开页前先设定：连上 WS 后要不要立刻推一帧 reply.status（推了才算「回复状态已同步」）。
let armReplyStatus = null;

const server = createServer(async (req, res) => {
  const url = new URL(req.url, 'http://127.0.0.1');
  const path = url.pathname;
  if (path === '/lab/events') return json(res, events);
  if (path === '/lab/arm' && req.method === 'POST') {
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    armReplyStatus = JSON.parse(Buffer.concat(chunks) || '{}').replyStatus ?? null;
    return json(res, { armReplyStatus });
  }
  if (path === '/lab/frame' && req.method === 'POST') {
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    const frame = JSON.parse(Buffer.concat(chunks));
    log('sent', frame);
    for (const socket of sockets) socket.send(JSON.stringify(frame));
    return json(res, { clients: sockets.size });
  }
  if (path === '/api/shell/state') return json(res, { status: 'ready', configured: true, chatReady: true });
  if (path === '/api/chat/models') return json(res, desktopModels(1));
  if (path === '/api/chat/navigation/pins') return json(res, { pins: [], sessions: [] });
  if (path === '/api/chat/sessions') {
    return json(res, {
      items: Object.entries(histories).map(([key, items]) => ({
        key, first_message_content: '看一下折叠', updated_at: items.at(-1).timestamp, message_count: items.length,
      })),
      next_cursor: null,
    });
  }
  const history = path.match(/^\/api\/chat\/sessions\/([^/]+)\/messages$/);
  if (history) {
    const items = histories[decodeURIComponent(history[1])];
    if (!items) return json(res, { detail: 'session not found' }, 404);
    return json(res, { version: 2, items, through_seq: items.at(-1).seq, has_more: false, before_seq: null });
  }
  if (path === '/api/chat/plugin-ui/catalog') return json(res, { catalog_revision: '0'.repeat(64), items: [] });
  if (path === '/api/runtime/host-bridge') return json(res, { state: 'healthy', failures: 0 });
  if (path.startsWith('/api/')) { log('unexpected_http', path); return json(res, { error: 'unexpected fixture request' }, 404); }
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
  log('connection', 'open');
  socket.on('message', (raw) => log('received', JSON.parse(String(raw))));
  socket.on('close', () => { sockets.delete(socket); log('connection', 'close'); });
  if (armReplyStatus !== null) {
    const frame = { type: 'reply.status', version: 2, ...armReplyStatus };
    log('sent', frame);
    socket.send(JSON.stringify(frame));
  }
});
server.on('upgrade', (request, socket, head) => {
  const { pathname } = new URL(request.url, 'http://127.0.0.1');
  if (pathname !== '/ws') return socket.destroy();
  wss.handleUpgrade(request, socket, head, (websocket) => wss.emit('connection', websocket, request));
});
await new Promise((done) => server.listen(0, '127.0.0.1', done));
const origin = 'http://127.0.0.1:' + server.address().port;

// ---- 4. 浏览器 ----
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH ?? '/usr/bin/chromium',
  headless: true,
  args: ['--no-sandbox', '--disable-dev-shm-usage'],
});
const results = [];
const record = (scene, label, data) => {
  results.push({ scene, label, ...data });
  console.log('  [' + scene + '] ' + label + '  ' + JSON.stringify(data));
};

/** 折叠区当前状态：摘要行文字、成员是否在 DOM、高度、计算样式与无障碍属性。 */
const readFold = (page) => page.evaluate(() => {
  const group = document.querySelector('.tool-group');
  if (!group) return null;
  const summary = group.querySelector('.tool-group-summary');
  const disclosure = group.querySelector('.tool-step-disclosure');
  const inner = group.querySelector('.tool-step-disclosure-inner');
  const style = disclosure ? getComputedStyle(disclosure) : null;
  const innerStyle = inner ? getComputedStyle(inner) : null;
  return {
    label: summary?.innerText.replace(/\s+/g, ' ').trim() ?? null,
    expanded: summary?.getAttribute('aria-expanded') ?? null,
    mounted: Boolean(group.querySelector('.tool-group-items')),
    memberCount: group.querySelectorAll('.tool-group-items .flow-item').length,
    height: disclosure ? Math.round(disclosure.getBoundingClientRect().height * 100) / 100 : null,
    rowsTemplate: style?.gridTemplateRows ?? null,
    transitionDuration: style?.transitionDuration ?? null,
    innerOpacity: innerStyle?.opacity ?? null,
    innerDuration: innerStyle?.transitionDuration ?? null,
    hidden: disclosure?.getAttribute('aria-hidden') ?? null,
    inert: disclosure?.hasAttribute('inert') ?? null,
  };
});

/** 按帧采样展开/收起：中间帧数量是「真的在过渡」和「直接跳」的分界。
    悬停用真实指针移动（page.mouse.move），因为 React 的 onPointerEnter 走的是浏览器派发的
    pointerover/pointerenter 链，手工 dispatchEvent 在 React 17+ 的根监听下收不到。 */
async function sample(page, action, frames = 20) {
  const box = action === 'hover' ? await page.locator('.tool-group-summary').first().boundingBox() : null;
  if (box) await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  return page.evaluate(async ({ action, frames }) => {
    const group = document.querySelector('.tool-group');
    const disclosure = group.querySelector('.tool-step-disclosure');
    const button = group.querySelector('.tool-group-summary');
    if (action === 'click') button.click();
    const heights = [];
    for (let index = 0; index < frames; index += 1) {
      await new Promise((done) => requestAnimationFrame(done));
      heights.push(Math.round(disclosure.getBoundingClientRect().height * 100) / 100);
    }
    return { heights, final: heights.at(-1) };
  }, { action, frames });
}

/** 中间帧计数：不论展开还是收起，都用「起点与终点之间」的高度帧数来判，
    起点取首帧、终点取末帧（收起时末帧是 0）。 */
const middles = (heights) => {
  const from = heights[0];
  const to = heights.at(-1);
  const low = Math.min(from, to);
  const high = Math.max(from, to);
  return heights.filter((height) => height > low + 0.5 && height < high - 0.5).length;
};

async function openScene(scene, session) {
  armReplyStatus = scene === 'settled'
    ? { session_id: session, snapshot_id: 'fixture-snapshot', available: true, items: [] }
    : null;
  const context = await browser.newContext({ viewport: { width: 1200, height: 900 } });
  const page = await context.newPage();
  page.setDefaultTimeout(15_000);
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto(origin + '/?session=' + session);
  return { context, page, errors };
}

try {
  // ===== 场景 A：轮次未落定，过程逐行 + 步骤折叠 =====
  {
    const scene = 'live';
    const { context, page, errors } = await openScene(scene, SESSIONS.live);
    try {
      await page.locator('.tool-group').first().waitFor();
      const collapsed = await readFold(page);
      record(scene, '初始折叠', collapsed);
      // 6 步思考 + 6 个工具 = 12 步，尾部留 3 步，前 9 步收进计数。
      assert.match(collapsed.label, /^9 个更早步骤/, '摘要行应是「9 个更早步骤」，实际：' + collapsed.label);
      assert.equal(collapsed.expanded, 'false', '初始应为收起');
      assert.equal(collapsed.hidden, 'true', '收起时应 aria-hidden');
      assert.equal(collapsed.inert, true, '收起时应 inert');
      assert.equal(collapsed.height, 0, '收起高度应为 0');
      assert.equal(collapsed.mounted, false, '未交互前不应挂载成员');

      // 尾部三步逐行可见：折叠组之外只剩尾部 3 步 + 一段旁白。
      const tail = await page.evaluate(() =>
        [...document.querySelectorAll('.process-trace > .flow-item')].map((node) => ({
          // 第二个类名才是过程种类：第一个恒为 flow-item。
          kind: [...node.classList][1] ?? null,
          text: node.innerText.trim().split('\n')[0],
        })));
      record(scene, '尾部逐行可见', { count: tail.length, tail });
      assert.deepEqual(tail.map((item) => item.kind),
        ['flow-tool', 'flow-tool', 'flow-tool', 'flow-narration'], '尾部应是 shell/read/grep 三步加一段旁白，实际：' + JSON.stringify(tail));

      // 悬停预挂载：成员进 DOM 但仍 0fr，点下去才有起始高度
      await sample(page, 'hover', 2);
      const hovered = await readFold(page);
      record(scene, '悬停预挂载', { mounted: hovered.mounted, height: hovered.height });
      assert.equal(hovered.mounted, true, '悬停后应把成员挂进 DOM');
      assert.equal(hovered.height, 0, '预挂载后仍应收起（0fr）');

      // 展开：200ms，有中间帧
      const expand = await sample(page, 'click');
      record(scene, '展开采样', expand);
      assert.ok(middles(expand.heights) >= 2, '展开应有多帧中间高度，实际：' + expand.heights.join(','));
      const open = await readFold(page);
      record(scene, '展开后', open);
      assert.equal(open.expanded, 'true', '展开后 aria-expanded=true');
      assert.equal(open.inert, false, '展开后不应 inert');
      assert.equal(open.hidden, 'false', '展开后不应 aria-hidden');
      assert.equal(open.memberCount, 9, '展开后应逐行恢复 9 个被折叠的步骤');
      assert.equal(open.transitionDuration, '0.2s', '展开高度过渡应为 200ms，实际：' + open.transitionDuration);
      assert.equal(open.innerDuration, '0.2s, 0.2s', '展开内层应为 200ms，实际：' + open.innerDuration);
      assert.equal(open.innerOpacity, '1', '展开后内层不透明');
      await page.screenshot({ path: root + '/live-open.png', animations: 'disabled' });

      // 收起：120ms，比展开快，同样有中间帧
      const collapse = await sample(page, 'click');
      record(scene, '收起采样', collapse);
      assert.ok(middles(collapse.heights) >= 1, '收起也应有中间高度，实际：' + collapse.heights.join(','));
      const closed = await readFold(page);
      record(scene, '收起后', closed);
      assert.equal(closed.expanded, 'false', '收起后 aria-expanded=false');
      assert.equal(closed.height, 0, '收起后高度回到 0');
      assert.equal(closed.inert, true, '收起后应 inert');
      assert.equal(closed.transitionDuration, '0.12s', '收起高度过渡应为 120ms，实际：' + closed.transitionDuration);
      assert.equal(closed.innerDuration, '0.12s, 0.12s', '收起内层应为 120ms，实际：' + closed.innerDuration);
      assert.equal(closed.memberCount, 9, '收起后成员留在 DOM（高度为 0），不再卸载');
      await page.screenshot({ path: root + '/live-closed.png', animations: 'disabled' });

      // 再展开一次：走的是成员已在 DOM 的稳定路径；等过渡跑完再量高度。
      await page.evaluate(() => document.querySelector('.tool-group-summary').click());
      await page.waitForFunction(() => document.querySelector('.tool-group-summary')?.getAttribute('aria-expanded') === 'true');
      await page.waitForFunction(() => {
        const node = document.querySelector('.tool-group .tool-step-disclosure');
        return node !== null && node.getBoundingClientRect().height > 0;
      });
      const reopened = await readFold(page);
      record(scene, '再次展开', { height: reopened.height, memberCount: reopened.memberCount });
      assert.equal(reopened.memberCount, 9, '再次展开仍是 9 个成员');
      assert.ok(reopened.height > 0, '再次展开高度应大于 0，实际：' + reopened.height);

      assert.deepEqual(errors, [], '页面不应有运行时错误');
    } finally { await context.close(); }
  }

  // ===== 场景 B：轮次已落定，整轮过程收进摘要行，展开后不再二次折叠 =====
  {
    const scene = 'settled';
    const { context, page, errors } = await openScene(scene, SESSIONS.settled);
    try {
      await page.locator('.turn-process').first().waitFor();
      const summary = await page.evaluate(() => {
        const node = document.querySelector('.turn-process-summary');
        return { label: node?.innerText.replace(/\s+/g, ' ').trim() ?? null, expanded: node?.getAttribute('aria-expanded') ?? null };
      });
      record(scene, '整轮摘要行', summary);
      assert.match(summary.label ?? '', /思考 6 次，调用 6 个工具/, '摘要行应写明思考与工具数量，实际：' + summary.label);
      assert.equal(summary.expanded, 'false', '摘要行初始收起');

      // 落定后的过程视图里不该再有第二层折叠
      const before = await page.evaluate(() => ({
        toolGroups: document.querySelectorAll('.tool-group').length,
        processItems: document.querySelectorAll('.turn-process-items .flow-item').length,
      }));
      record(scene, '展开前', before);
      assert.equal(before.toolGroups, 0, '落定后不应出现折叠行（外层摘要已经收过一次）');

      await page.evaluate(() => document.querySelector('.turn-process-summary').click());
      await page.waitForFunction(() => document.querySelector('.turn-process-summary')?.getAttribute('aria-expanded') === 'true');
      // 落定场景的展开同样要等过渡跑完再量高度（aria 更新早于高度一帧）。
      await page.waitForFunction(() => {
        const node = document.querySelector('.turn-process .tool-step-disclosure');
        return node !== null && node.getBoundingClientRect().height > 0;
      });
      const after = await page.evaluate(() => ({
        toolGroups: document.querySelectorAll('.tool-group').length,
        processItems: document.querySelectorAll('.turn-process-items .flow-item').length,
        height: Math.round(document.querySelector('.turn-process .tool-step-disclosure').getBoundingClientRect().height * 100) / 100,
      }));
      record(scene, '展开后', after);
      assert.equal(after.toolGroups, 0, '展开后仍不套第二层折叠');
      assert.equal(after.processItems, 12, '展开后应逐行恢复 12 个过程项（6 步思考 + 6 个工具）');
      assert.ok(after.height > 0, '展开后高度应大于 0');
      await page.screenshot({ path: root + '/settled-open.png', animations: 'disabled' });

      assert.deepEqual(errors, [], '页面不应有运行时错误');
    } finally { await context.close(); }
  }

  writeFileSync(root + '/results.json', JSON.stringify(results, null, 2));
  console.log('PASS 过程区折叠');
} finally {
  await browser.close();
  for (const socket of sockets) socket.terminate();
  await new Promise((done) => server.close(done));
  wss.close();
  console.log('证据目录: ' + root);
}

/**
 * 折叠/展开平滑动效的真实浏览器验收（PR 1）：
 *   1. 未悬停直接首次点击展开（容器常驻，切 class 即过渡，不直接跳高）
 *   2. 吸顶状态点击「收起」（不再走 is-instant 瞬关，中间帧 > 0 且标题行留在原位）
 *   3. 单个工具详情（ToolStep）展开与收起均有中间帧（参数/结果在收起期间不提前卸载）
 *   4. 思考行（ThinkingRow）展开与收起均有中间帧
 *   5. 长提问（UserTextClamp）与长输出预览（PreviewText）双向平滑过渡
 *   6. 压缩边界 / 上下文提示（Disclosure）双向平滑过渡
 *   7. 展开途中立即打断收起（transition 可逆折返）
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
  : mkdtempSync(resolve(tmpdir(), 'akashic-smooth-disclosures-'));

const build = spawnSync(process.execPath, [
  resolve(repo, 'node_modules/vite/bin/vite.js'), 'build',
  '--config', 'frontend/plugins/akashic_clients/chat/vite.harness.config.ts',
  '--outDir', root + '/dist',
], { cwd: repo, encoding: 'utf8' });
writeFileSync(root + '/build.log', build.stdout + build.stderr);
assert.equal(build.status, 0, '前端构建失败，见 ' + root + '/build.log');

const base = Date.UTC(2026, 10, 17, 1, 0, 0);
const at = (seconds) => new Date(base + seconds * 1000).toISOString();

const longLines = Array.from({ length: 30 }, (_, index) => `line-${String(index + 1).padStart(2, '0')}: sample tool output for smooth collapse`).join('\n');
const longUserText = Array.from({ length: 24 }, (_, index) => `用户第 ${index + 1} 行长输入文本用于验证展开全部与收起的平滑高度过渡。`).join('\n');

const SESSION = 'smooth-fold';
const STEPS = Array.from({ length: 10 }, (_, index) => ({
  tool: index % 2 === 0 ? 'shell' : 'read',
  args: { command: `echo step-${index + 1}`, description: `步骤 ${index + 1} 描述` },
  result: index === 0 ? longLines : `result-${index + 1}\n${'detail line\n'.repeat(8)}`,
}));

const messages = [
  {
    id: 'm-user-long', seq: 0, session_id: SESSION, timestamp: at(0),
    author: 'user', actor: 'user', source: 'akashic', attachments: [], metadata: {},
    body: { kind: 'input', parts: [{ kind: 'text', value: longUserText }] },
  },
  {
    id: 'm-compaction', seq: 1, session_id: SESSION, timestamp: at(1),
    author: 'assistant', actor: 'model', source: 'akashic', attachments: [], metadata: {},
    body: {
      kind: 'output', finish: 'complete',
      parts: [{
        kind: 'compaction.boundary',
        value: { reference: 'compaction-2', trigger: 'idle', mode: 'background', tokens_before: 42000, tokens_after: 12000, generation: 2, summary_preview: '压缩摘要第一行\n压缩摘要第二行' },
      }],
    },
  },
  {
    id: 'm-out', seq: 2, session_id: SESSION, timestamp: at(2),
    author: 'assistant', actor: 'model', source: 'akashic', attachments: [], metadata: {},
    body: {
      kind: 'output', finish: 'continue',
      parts: [
        { kind: 'context.notice', value: { text: '上下文提示：已自动加载项目约定。' } },
        ...STEPS.flatMap((step, index) => [
          { kind: `thinking.${index}`, display: 'thinking', text: `第 ${index + 1} 段思考：\n` + '详细推理内容。\n'.repeat(12) },
          { kind: 'tool_call', binding_id: 'fixture-tool', name: step.tool, arguments: step.args },
        ]),
      ],
    },
  },
  ...STEPS.map((step, index) => ({
    id: `m-res-${index}`, seq: 3 + index, session_id: SESSION, timestamp: at(3 + index),
    author: 'tool', actor: 'tool', source: 'akashic', attachments: [], metadata: {},
    body: {
      kind: 'tool_result', outcome: 'success',
      call_ref: { message_id: 'm-out', part_index: 1 + index * 2 + 1 },
      parts: [{ kind: 'text', value: step.result }],
    },
  })),
  {
    id: 'm-final', seq: 3 + STEPS.length, session_id: SESSION, timestamp: at(45),
    author: 'assistant', actor: 'model', source: 'akashic', attachments: [], metadata: {},
    body: {
      kind: 'output', finish: 'complete',
      parts: [{ kind: 'text', value: Array.from({ length: 18 }, (_, i) => `总结段落 ${i + 1}：这里留出足够的页面高度，确保吸顶折叠收起时滚动空间充足。`).join('\n\n') }],
    },
  },
];

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
      items: [{ key: SESSION, first_message_content: '平滑折叠测试', updated_at: messages.at(-1).timestamp, message_count: messages.length }],
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
      snapshot_id: 'smooth-fold-snap', available: true, items: [],
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

const middles = (values) => {
  const low = Math.min(values[0], values.at(-1));
  const high = Math.max(values[0], values.at(-1));
  return values.filter((value) => value > low + 0.5 && value < high - 0.5).length;
};

async function sampleBoxHeight(page, triggerSelector, targetSelector, frames = 22) {
  return page.evaluate(async ({ triggerSelector, targetSelector, frames }) => {
    const trigger = document.querySelector(triggerSelector);
    const target = document.querySelector(targetSelector);
    trigger.click();
    const heights = [];
    for (let i = 0; i < frames; i += 1) {
      await new Promise((resolve) => requestAnimationFrame(resolve));
      heights.push(Math.round(target.getBoundingClientRect().height * 100) / 100);
    }
    return heights;
  }, { triggerSelector, targetSelector, frames });
}

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH ?? '/usr/bin/chromium',
  headless: true,
  args: ['--no-sandbox', '--disable-dev-shm-usage'],
});

try {
  const context = await browser.newContext({ viewport: { width: 1200, height: 800 } });
  const page = await context.newPage();
  page.setDefaultTimeout(15_000);
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (msg) => { if (msg.type() === 'error') console.error('[browser]', msg.text()); });

  await page.goto(`${origin}/?session=${SESSION}`);
  await page.locator('.turn-process-summary').waitFor();
  await page.waitForTimeout(200);

  // 1. 直接点开 TurnProcessSummary（不先 hover）：首次展开也应有中间帧
  const firstExpand = await sampleBoxHeight(page, '.turn-process-summary', '.turn-process .tool-step-disclosure');
  console.log('[1] direct first expand TurnProcessSummary:', JSON.stringify(firstExpand));
  assert.ok(middles(firstExpand) >= 1, `未悬停首次展开应有中间过渡帧，实际：${firstExpand}`);
  assert.ok(firstExpand.at(-1) > 200, '展开后应达到完整高度');

  // 2. 点击途中打断（先收起，4 帧后再点展开）：应平稳折返而不归零跳
  const interrupt = await page.evaluate(async () => {
    const trigger = document.querySelector('.turn-process-summary');
    const target = document.querySelector('.turn-process .tool-step-disclosure');
    const heights = [];
    trigger.click();
    for (let i = 0; i < 4; i += 1) {
      await new Promise((resolve) => requestAnimationFrame(resolve));
      heights.push(Math.round(target.getBoundingClientRect().height * 100) / 100);
    }
    trigger.click();
    for (let i = 0; i < 18; i += 1) {
      await new Promise((resolve) => requestAnimationFrame(resolve));
      heights.push(Math.round(target.getBoundingClientRect().height * 100) / 100);
    }
    return heights;
  });
  console.log('[2] interrupt collapse -> expand:', JSON.stringify(interrupt));
  assert.ok(interrupt[3] < interrupt[0], '打断前高度应正在下降');
  assert.ok(interrupt[3] > 20, '打断时不应已瞬间归零');
  assert.ok(interrupt.at(-1) > interrupt[3], '打断后应从当前位置重新长回');

  // 3. 展开内部的第一个 ToolStep，并测试它自身的收起也有中间帧（且 PreviewText 双向平滑）
  const stepExpand = await sampleBoxHeight(page, '.turn-process-items .tool-step-summary', '.turn-process-items .tool-step .tool-step-disclosure');
  console.log('[3a] ToolStep expand:', JSON.stringify(stepExpand));
  assert.ok(middles(stepExpand) >= 1, `ToolStep 展开应有中间帧，实际：${stepExpand}`);

  const previewExpand = await sampleBoxHeight(page, '.tool-result-more', '.tool-result-preview pre');
  console.log('[3b] PreviewText expand:', JSON.stringify(previewExpand));
  assert.ok(middles(previewExpand) >= 1, `PreviewText 展开应有中间帧，实际：${previewExpand}`);

  const previewCollapse = await sampleBoxHeight(page, '.tool-result-more', '.tool-result-preview pre');
  console.log('[3c] PreviewText collapse:', JSON.stringify(previewCollapse));
  assert.ok(middles(previewCollapse) >= 1, `PreviewText 收起应有中间帧，实际：${previewCollapse}`);

  const stepCollapse = await sampleBoxHeight(page, '.turn-process-items .tool-step-summary', '.turn-process-items .tool-step .tool-step-disclosure');
  console.log('[3d] ToolStep collapse:', JSON.stringify(stepCollapse));
  assert.ok(middles(stepCollapse) >= 1, `ToolStep 收起应有中间帧，实际：${stepCollapse}`);

  // 4. ThinkingRow 展开与收起均有中间帧
  const thinkingExpand = await sampleBoxHeight(page, '.thinking-row .process-trigger', '.thinking-row .tool-step-disclosure');
  console.log('[4a] ThinkingRow expand:', JSON.stringify(thinkingExpand));
  assert.ok(middles(thinkingExpand) >= 1, `ThinkingRow 展开应有中间帧，实际：${thinkingExpand}`);

  const thinkingCollapse = await sampleBoxHeight(page, '.thinking-row .process-trigger', '.thinking-row .tool-step-disclosure');
  console.log('[4b] ThinkingRow collapse:', JSON.stringify(thinkingCollapse));
  assert.ok(middles(thinkingCollapse) >= 1, `ThinkingRow 收起应有中间帧，实际：${thinkingCollapse}`);

  // 5. 滚动让 TurnProcessSummary 吸顶（stuck = true），再点「收起」：不能瞬关，且标题行保持在原位置附近
  const stickyCollapse = await page.evaluate(async () => {
    const turn = document.querySelector('.turn-process');
    const sticky = turn.querySelector('.tool-step-sticky');
    const disclosure = turn.querySelector('.tool-step-disclosure');
    const trigger = turn.querySelector('.turn-process-summary');
    const scroller = document.querySelector('.conversation-scroll');
    // 滚入 turn-process 内部约 120px，触发 position: sticky
    const turnRect = turn.getBoundingClientRect();
    const scrollerRect = scroller.getBoundingClientRect();
    scroller.scrollTop += turnRect.top - scrollerRect.top + 120;
    await new Promise((resolve) => requestAnimationFrame(resolve));
    const beforeTop = sticky.getBoundingClientRect().top;
    const beforeDelta = beforeTop - turn.getBoundingClientRect().top;
    const heights = [];
    const stickyTops = [];
    const paintedTops = [];
    trigger.click();
    const observer = new ResizeObserver(() => {
      paintedTops.push(Math.round(sticky.getBoundingClientRect().top * 100) / 100);
    });
    await new Promise((resolve) => requestAnimationFrame(resolve));
    observer.observe(disclosure);
    for (let i = 0; i < 22; i += 1) {
      await new Promise((resolve) => requestAnimationFrame(resolve));
      heights.push(Math.round(disclosure.getBoundingClientRect().height * 100) / 100);
      stickyTops.push(Math.round(sticky.getBoundingClientRect().top * 100) / 100);
    }
    observer.disconnect();
    return { beforeTop, beforeDelta, expandedAfter: trigger.getAttribute("aria-expanded"), cls: disclosure.className, heights, stickyTops, paintedTops, finalHeight: heights.at(-1) };
  });
  console.log('[5] sticky TurnProcessSummary collapse:', JSON.stringify(stickyCollapse));
  assert.ok(stickyCollapse.beforeDelta > 1, '应先进入吸顶状态');
  assert.ok(middles(stickyCollapse.heights) >= 1, `吸顶收起应平滑过渡而不是瞬关，实际：${stickyCollapse.heights}`);
  assert.equal(stickyCollapse.finalHeight, 0, '收起完成后高度应归零');
  const maxDrift = Math.max(...stickyCollapse.paintedTops.map((top) => Math.abs(top - stickyCollapse.beforeTop)));
  assert.ok(maxDrift <= 2, `吸顶收起期间标题行绘制前不应跳离原位，最大偏差 ${maxDrift}px`);

  // 6. UserTextClamp 展开与收起均有中间帧
  const userExpand = await sampleBoxHeight(page, '.user-text-toggle', '.user-text-clamp');
  console.log('[6a] UserTextClamp expand:', JSON.stringify(userExpand));
  assert.ok(middles(userExpand) >= 1, `长提问展开应有中间帧，实际：${userExpand}`);

  const userCollapse = await page.evaluate(async () => {
    document.querySelector('.user-text-toggle').click();
    const heights = [];
    const toggles = [];
    for (let i = 0; i < 22; i += 1) {
      await new Promise((resolve) => requestAnimationFrame(resolve));
      heights.push(Math.round(document.querySelector('.user-text-clamp').getBoundingClientRect().height * 100) / 100);
      toggles.push(document.querySelector('.user-text-toggle') !== null);
    }
    return { heights, toggles, state: document.querySelector('.user-text-clamp').dataset.state, inline: document.querySelector('.user-text-clamp').style.maxHeight };
  });
  console.log('[6b] UserTextClamp collapse:', JSON.stringify(userCollapse));
  assert.ok(middles(userCollapse.heights) >= 1, `长提问收起应有中间帧，实际：${userCollapse.heights}`);
  assert.ok(userCollapse.toggles.every(Boolean), `收起途中按钮不应闪退，实际：${userCollapse.toggles}`);
  assert.equal(userCollapse.state, 'clamped', '收起完成后应回到截断态并保留渐隐');
  assert.equal(userCollapse.inline, '', '收起完成后内联 max-height 应清掉');

  // 7. CompactionBoundaryRow 展开与收起均有中间帧（ContextNoticeRow 共用同一 Disclosure）
  const compactExpand = await sampleBoxHeight(page, '.compaction-boundary__trigger', '.compaction-boundary .tool-step-disclosure');
  console.log('[7a] CompactionBoundaryRow expand:', JSON.stringify(compactExpand));
  assert.ok(middles(compactExpand) >= 1, `压缩边界展开应有中间帧，实际：${compactExpand}`);
  const compactCollapse = await sampleBoxHeight(page, '.compaction-boundary__trigger', '.compaction-boundary .tool-step-disclosure');
  console.log('[7b] CompactionBoundaryRow collapse:', JSON.stringify(compactCollapse));
  assert.ok(middles(compactCollapse) >= 1, `压缩边界收起应有中间帧，实际：${compactCollapse}`);

  assert.deepEqual(errors, [], `页面不应有运行时错误：${JSON.stringify(errors)}`);
  console.log('PASS verify-smooth-disclosures');
} finally {
  await browser.close();
  for (const socket of sockets) socket.terminate();
  await new Promise((done) => server.close(done));
  wss.close();
  console.log('证据目录: ' + root);
}

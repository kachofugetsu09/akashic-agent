/**
 * 录折叠动效的 GIF：真 Chromium 里点开/收起，用 CDP screencast 抓帧，ffmpeg 拼 GIF。
 *
 * 为什么不用 Playwright 的 recordVideo：它内部调随包发的精简 ffmpeg 并写死 `-c:v vp8`，
 * 那个 build 只编 vp9，报 `Unknown encoder 'vp8'`；换成系统 ffmpeg 又会卡在
 * `-speed 8` 这类 vp8 专用参数上不退出。screencast 是浏览器直接吐 JPEG，绕开整条编码链，
 * 而且不受截图往返的 IO 影响，帧率稳定得多。
 *
 * 用法：
 *   ~/.cache/akashic-dev/mjs-verify/run-verify.sh frontend/chat/src/record-process-fold-gif.mjs
 * 产物（EVIDENCE_DIR）：
 *   fold-live.gif     运行中：折叠 → 悬停 → 展开（200ms）→ 收起（120ms）
 *   fold-settled.gif  已落定：整轮摘要展开
 *   frames/<name>/    原始 JPEG 帧，想重拼不用重录
 */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { existsSync, mkdirSync, readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
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
  : mkdtempSync(resolve(tmpdir(), 'akashic-fold-gif-'));
const ffmpeg = process.env.FFMPEG_PATH ?? '/usr/bin/ffmpeg';

// ---- 构建 ----
const build = spawnSync(process.execPath, [
  resolve(repo, 'node_modules/vite/bin/vite.js'), 'build',
  '--config', 'frontend/chat/vite.harness.config.ts',
  '--outDir', root + '/dist',
], { cwd: repo, encoding: 'utf8' });
writeFileSync(root + '/build.log', build.stdout + build.stderr);
assert.equal(build.status, 0, '前端构建失败，见 ' + root + '/build.log');

// ---- 夹具（与 verify-process-fold.mjs 同构）----
const STEPS = [
  { tool: 'read', args: { path: '/srv/app/core/turn.py' }, result: 'def run():\n    pass' },
  { tool: 'grep', args: { query: 'foldable' }, result: 'core/turn.py:12: foldable' },
  { tool: 'write', args: { path: '/srv/app/core/turn.py' }, result: 'wrote 42 bytes' },
  { tool: 'shell', args: { command: 'pytest -q' }, result: '3 passed' },
  { tool: 'read', args: { path: '/srv/app/README.md' }, result: '# 项目' },
  { tool: 'grep', args: { query: 'durationMs' }, result: 'message-view.tsx:1' },
];
const SESSIONS = { live: 'gif-live', settled: 'gif-settled' };
const base = Date.UTC(2026, 9, 22, 2, 0, 0);
const at = (s) => new Date(base + s * 1000).toISOString();

function buildHistory(session, settled) {
  const row = (id, seq, author, body, seconds) => ({
    id, seq, session_id: session, timestamp: at(seconds), author, source: 'akashic',
    attachments: [], metadata: {}, body,
  });
  const items = [row('fold-input', 0, 'user', { kind: 'input', parts: [{ kind: 'text', value: '看一下折叠' }] }, 0)];
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
  items.push(row('fold-answer', 2 + STEPS.length, 'assistant', settled
    ? { kind: 'output', parts: [{ kind: 'text', value: '看完了，六个步骤都在。' }], finish: 'complete' }
    : { kind: 'output', parts: [{ kind: 'text', value: '还在跑。' }], finish: 'continue' }, 40));
  return items;
}
const histories = {
  [SESSIONS.live]: buildHistory(SESSIONS.live, false),
  [SESSIONS.settled]: buildHistory(SESSIONS.settled, true),
};

const sockets = new Set();
const json = (res, data, status = 200) => {
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(JSON.stringify(data));
};
let armReplyStatus = null;

const server = createServer(async (req, res) => {
  const url = new URL(req.url, 'http://127.0.0.1');
  const path = url.pathname;
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
  if (path.startsWith('/api/')) return json(res, { error: 'unexpected fixture request' }, 404);
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
  socket.on('close', () => sockets.delete(socket));
  if (armReplyStatus !== null) socket.send(JSON.stringify({ type: 'reply.status', version: 2, ...armReplyStatus }));
});
server.on('upgrade', (request, socket, head) => {
  if (new URL(request.url, 'http://127.0.0.1').pathname !== '/ws') return socket.destroy();
  wss.handleUpgrade(request, socket, head, (ws) => wss.emit('connection', ws, request));
});
await new Promise((done) => server.listen(0, '127.0.0.1', done));
const origin = 'http://127.0.0.1:' + server.address().port;

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH ?? '/usr/bin/chromium',
  headless: true,
  args: ['--no-sandbox', '--disable-dev-shm-usage'],
});

/**
 * CDP screencast：浏览器在画面变化时吐 JPEG 帧，并记录每帧的相对时间戳。
 * 比反复 page.screenshot 快得多（后者要等编码 + 往返 + 写盘），帧率才够表现 120ms 的收起。
 */
async function startScreencast(page, dir) {
  mkdirSync(dir, { recursive: true });
  const cdp = await page.context().newCDPSession(page);
  const frames = [];
  cdp.on('Page.screencastFrame', async ({ data, sessionId, metadata }) => {
    // metadata.timestamp 是秒级 epoch；保留它，后面按真实间隔生成 GIF，
    // 否则静止段被压掉、按固定帧率摊平会让播放变快。
    const index = frames.length;
    frames.push(metadata.timestamp);
    writeFileSync(`${dir}/${String(index).padStart(5, '0')}.jpg`, Buffer.from(data, 'base64'));
    await cdp.send('Page.screencastFrameAck', { sessionId }).catch(() => {});
  });
  await cdp.send('Page.startScreencast', { format: 'jpeg', quality: 80, maxWidth: 760, maxHeight: 520, everyNthFrame: 1 });
  return {
    async stop() {
      await cdp.send('Page.stopScreencast').catch(() => {});
      await cdp.detach().catch(() => {});
      return frames;
    },
  };
}

/** 把光标平滑移到元素上：悬停预挂载是动效的一部分，得让观众看到指针过去。 */
async function glideTo(page, selector, ms = 450) {
  const box = await page.locator(selector).first().boundingBox();
  assert.ok(box, '找不到 ' + selector);
  const target = { x: box.x + Math.min(box.width / 2, 200), y: box.y + box.height / 2 };
  const steps = 20;
  for (let i = 1; i <= steps; i += 1) {
    const t = i / steps;
    const ease = t * t * (3 - 2 * t);
    await page.mouse.move(20 + (target.x - 20) * ease, 20 + (target.y - 20) * ease);
    await page.waitForTimeout(ms / steps);
  }
}

async function capture(name, session, scene, script) {
  armReplyStatus = scene === 'settled'
    ? { session_id: session, snapshot_id: 'gif-snapshot', available: true, items: [] }
    : null;
  const context = await browser.newContext({
    // 只录消息区，不录侧栏：GIF 小、注意力集中在折叠行。
    viewport: { width: 760, height: 520 },
  });
  const page = await context.newPage();
  page.setDefaultTimeout(20_000);
  await page.goto(origin + '/?session=' + session);
  await page.locator('.tool-group, .turn-process').first().waitFor();
  await page.waitForTimeout(600);

  const framesDir = root + '/frames/' + name;
  const rec = await startScreencast(page, framesDir);
  await script(page);
  const stamps = await rec.stop();
  await context.close();
  assert.ok(stamps.length >= 8, '帧太少（' + stamps.length + '），screencast 没跑起来');

  // screencast 只在画面变化时吐帧，静止段没有帧。按每帧真实持续时长写 concat 清单，
  // 让 ffmpeg 按时间轴还原，而不是当成等间隔序列（那样会把播放加速）。
  const list = root + '/' + name + '.concat.txt';
  const lines = [];
  for (let i = 0; i < stamps.length; i += 1) {
    const span = i + 1 < stamps.length ? stamps[i + 1] - stamps[i] : 1 / 15;
    // 夹一下：单帧最长 0.5s（防止暂停段把 GIF 撑长），最短 1/30s。
    const duration = Math.min(Math.max(span, 1 / 30), 0.5);
    lines.push(`file '${framesDir}/${String(i).padStart(5, '0')}.jpg'`, `duration ${duration.toFixed(4)}`);
  }
  // concat demuxer 要求末帧重复一次，否则最后一帧的 duration 不生效。
  lines.push(`file '${framesDir}/${String(stamps.length - 1).padStart(5, '0')}.jpg'`);
  writeFileSync(list, lines.join('\n') + '\n');

  const gif = root + '/' + name + '.gif';
  const palette = root + '/palette-' + name + '.png';
  const pass1 = spawnSync(ffmpeg, [
    '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', list,
    '-vf', 'fps=15,scale=640:-1:flags=lanczos,palettegen=max_colors=128:stats_mode=diff',
    palette,
  ], { encoding: 'utf8' });
  assert.equal(pass1.status, 0, 'palettegen 失败：' + pass1.stderr);
  const pass2 = spawnSync(ffmpeg, [
    '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', list, '-i', palette,
    '-lavfi', 'fps=15,scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle',
    '-loop', '0', gif,
  ], { encoding: 'utf8' });
  assert.equal(pass2.status, 0, 'paletteuse 失败：' + pass2.stderr);
  const size = readFileSync(gif).length;
  const span = stamps.at(-1) - stamps[0];
  console.log(`${name}: ${stamps.length} 帧 / ${span.toFixed(2)}s → ${gif}  ${(size / 1024).toFixed(0)} KB`);
  return gif;
}

try {
  // A：折叠 → 悬停（预挂载）→ 展开（200ms）→ 停住 → 收起（120ms）
  await capture('fold-live', SESSIONS.live, 'live', async (page) => {
    await page.waitForTimeout(500);
    await glideTo(page, '.tool-group-summary');
    await page.waitForTimeout(600);
    await page.locator('.tool-group-summary').first().click();
    await page.waitForTimeout(1600);        // 展开 + 让人读清逐行恢复
    await page.locator('.tool-group-summary').first().click();
    await page.waitForTimeout(1000);        // 收起 + 尾巴
  });

  // B：落定后整轮摘要展开（不再套第二层折叠）
  await capture('fold-settled', SESSIONS.settled, 'settled', async (page) => {
    await page.waitForTimeout(500);
    await glideTo(page, '.turn-process-summary');
    await page.waitForTimeout(600);
    await page.locator('.turn-process-summary').first().click();
    await page.waitForTimeout(1800);
  });

  console.log('证据目录: ' + root);
} finally {
  await browser.close();
  for (const socket of sockets) socket.terminate();
  await new Promise((done) => server.close(done));
  wss.close();
}

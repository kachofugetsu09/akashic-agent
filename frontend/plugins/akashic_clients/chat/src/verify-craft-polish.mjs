/**
 * 界面细节打磨与 shell_ui 折叠的真实浏览器验收（PR 3）：
 *   1. 正文图片 1px 透明内描边（outline: 1px solid ...; outline-offset: -1px）与媒体附件卡片 ::after 内描边
 *   2. 浮层阴影替代实线边框（border-width: 0px + box-shadow 含 1px 透明描边环）
 *   3. 标题 text-wrap: balance、正文 text-wrap: pretty、动态数字 tabular-nums
 *   4. shell_ui 插件库存折叠（ShellDisclosure）双向 0fr <-> 1fr 平滑过渡（含「技术信息」嵌套）
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
import { build as esbuild } from 'esbuild';
import { desktopModels } from '../../../../../scripts/webui-performance/fixtures.mjs';

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '../../../../..');
const root = process.env.EVIDENCE_DIR
  ? (mkdirSync(process.env.EVIDENCE_DIR, { recursive: true }), resolve(process.env.EVIDENCE_DIR))
  : mkdtempSync(resolve(tmpdir(), 'akashic-craft-polish-'));

const build = spawnSync(process.execPath, [
  resolve(repo, 'node_modules/vite/bin/vite.js'), 'build',
  '--config', 'frontend/plugins/akashic_clients/chat/vite.harness.config.ts',
  '--outDir', root + '/dist',
], { cwd: repo, encoding: 'utf8' });
writeFileSync(root + '/build.log', build.stdout + build.stderr);
assert.equal(build.status, 0, '前端构建失败，见 ' + root + '/build.log');

// 同时打包 shell_ui PluginInventory 测试页面，真实验证 ShellDisclosure 与样式
const shellEntry = resolve(root, 'shell-entry.tsx');
writeFileSync(shellEntry, `
import React from "react";
import { createRoot } from "react-dom/client";
import ${JSON.stringify(resolve(repo, 'packages/akashic-theme/src/material-tokens.css'))};
import ${JSON.stringify(resolve(repo, 'packages/akashic-theme/src/brand-tokens.css'))};
import ${JSON.stringify(resolve(repo, 'frontend/plugins/shell_ui/src/style.css'))};
import { PluginInventory } from ${JSON.stringify(resolve(repo, 'frontend/plugins/shell_ui/src/plugin-inventory.tsx'))};
const plugins = [
  { id: "sample-plugin", name: "示例插件", description: "用于验证列表行平滑展开与收起", source: "external", enabled: true, locked: null, phase: "active", version: "1.2.3", missing_services: [], dependents: [], error: null },
  { id: "base-plugin", name: "基础运行组件", description: "内建基础设施", source: "builtin", enabled: true, locked: "management-required", phase: "active", version: "0.9.0", missing_services: [], dependents: ["sample-plugin"], error: null },
];
const inventory = { current: { plugins }, error: null, subscribe: () => () => {}, refresh: async () => ({ plugins }) };
const http = { request: async () => new Response(JSON.stringify({ plugins }), { status: 200 }) };
createRoot(document.getElementById("root")).render(
  <div className="shell-settings-root" style={{ padding: "24px", background: "var(--ak-paper-canvas)", "--ak-rule-subtle": "rgb(0 0 0 / 0.08)" } as React.CSSProperties}>
    <PluginInventory http={http} inventory={inventory} active={false} />
  </div>
);
`);
await esbuild({
  entryPoints: [shellEntry],
  bundle: true,
  format: 'esm',
  outfile: root + '/dist/shell-test.js',
  loader: { '.woff2': 'empty' },
  nodePaths: [resolve(repo, 'node_modules')],
  define: { 'process.env.NODE_ENV': '"production"' },
});
writeFileSync(root + '/dist/shell-test.html', `<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="/shell-test.css"></head><body><div id="root"></div><script type="module" src="/shell-test.js"></script></body></html>`);

const SVG_URL = '/api/chat/artifacts/img-att';
const base = Date.UTC(2026, 10, 17, 1, 0, 0);
const at = (seconds) => new Date(base + seconds * 1000).toISOString();
const SESSION = 'craft-polish';

const messages = [
  {
    id: 'm-user', seq: 0, session_id: SESSION, timestamp: at(0),
    author: 'user', actor: 'user', source: 'akashic',
    attachments: [{ kind: 'image', artifact_id: 'img-att', filename: 'white-card.png', media_type: 'image/png', size_bytes: 120, sha256: 'b'.repeat(64) }],
    metadata: {},
    body: { kind: 'input', parts: [{ kind: 'text', value: '这是一段很长的提问文本用于确认段落换行策略。'.repeat(20) }] },
  },
  {
    id: 'm-out', seq: 1, session_id: SESSION, timestamp: at(1),
    author: 'assistant', actor: 'model', source: 'akashic', attachments: [], metadata: {},
    body: {
      kind: 'output', finish: 'continue',
      parts: [
        { kind: 'tool_call', binding_id: 'fixture-tool', name: 'read', arguments: { path: 'a.txt' } },
      ],
    },
  },
  {
    id: 'm-res', seq: 2, session_id: SESSION, timestamp: at(2),
    author: 'tool', actor: 'tool', source: 'akashic', attachments: [], metadata: {},
    body: {
      kind: 'tool_result', outcome: 'success',
      call_ref: { message_id: 'm-out', part_index: 0 },
      parts: [{ kind: 'text', value: 'ok' }],
    },
  },
  {
    id: 'm-reply', seq: 3, session_id: SESSION, timestamp: at(3),
    author: 'assistant', actor: 'model', source: 'akashic', attachments: [], metadata: {},
    body: { kind: 'output', finish: 'complete', parts: [{ kind: 'text', value: `正文说明段落。\n\n![示例图](${SVG_URL})` }] },
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
      items: [{ key: SESSION, first_message_content: '工艺细节测试', updated_at: messages.at(-1).timestamp, message_count: messages.length }],
      next_cursor: null,
    });
  }
  if (path === `/api/chat/sessions/${SESSION}/messages`) {
    return json(res, { version: 2, items: messages, through_seq: messages.at(-1).seq, has_more: false, before_seq: null });
  }
  if (path === '/api/chat/artifacts/img-att') {
    res.writeHead(200, { 'content-type': 'image/svg+xml' });
    res.end('<svg xmlns="http://www.w3.org/2000/svg" width="160" height="100"><rect width="160" height="100" fill="#ffffff"/></svg>');
    return;
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
      snapshot_id: 'craft-snap', available: true, items: [],
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

const middles = (values, eps = 0.5) => {
  const low = Math.min(...values);
  const high = Math.max(...values);
  return values.filter((value) => value > low + eps && value < high - eps).length;
};

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH ?? '/usr/bin/chromium',
  headless: true,
  args: ['--no-sandbox', '--disable-dev-shm-usage'],
});

try {
  // 1. 聊天页面细节样式检查
  {
    const context = await browser.newContext({ viewport: { width: 1200, height: 800 } });
    const page = await context.newPage();
    page.setDefaultTimeout(15_000);
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));

    await page.goto(`${origin}/?session=${SESSION}`);
    await page.locator('.agent-content img').waitFor();
    await page.keyboard.press('Control+f');
    await page.locator('.thread-find').waitFor();
    await page.locator('.turn-process-summary').click();
    await page.locator('.turn-process-items .tool-step-summary').click();

    const styles = await page.evaluate(() => {
      const img = document.querySelector('.agent-content img');
      const attSurface = document.querySelector('.ak-attachment.is-media .ak-attachment__surface');
      const findCard = document.querySelector('.thread-find');
      const p = document.querySelector('.agent-content p');
      const userClamp = document.querySelector('.user-text-clamp');
      const h4 = document.querySelector('.tool-detail-section h4');
      const stepState = document.querySelector('.tool-step-state');
      const imgStyle = getComputedStyle(img);
      const attAfter = attSurface ? getComputedStyle(attSurface, '::after') : null;
      const findStyle = getComputedStyle(findCard);
      return {
        imgOutline: { style: imgStyle.outlineStyle, width: imgStyle.outlineWidth, offset: imgStyle.outlineOffset, color: imgStyle.outlineColor },
        attAfterShadow: attAfter?.boxShadow ?? null,
        findBorder: findStyle.borderTopWidth,
        findShadow: findStyle.boxShadow,
        pWrap: getComputedStyle(p).textWrap,
        userWrap: getComputedStyle(userClamp).textWrap,
        h4Wrap: h4 ? getComputedStyle(h4).textWrap : null,
        stepTabular: stepState ? getComputedStyle(stepState).fontVariantNumeric : null,
      };
    });
    console.log('[1] Chat craft styles:', JSON.stringify(styles));
    assert.equal(styles.imgOutline.style, 'solid');
    assert.equal(styles.imgOutline.width, '1px');
    assert.equal(styles.imgOutline.offset, '-1px');
    assert.notEqual(styles.attAfterShadow, 'none');
    assert.equal(styles.findBorder, '0px');
    assert.notEqual(styles.findShadow, 'none');
    assert.equal(styles.pWrap, 'pretty');
    assert.equal(styles.userWrap, 'pretty');
    assert.equal(styles.h4Wrap, 'balance');
    assert.equal(styles.stepTabular, 'tabular-nums');

    // 检查空白态标题 text-wrap: balance
    await page.locator('.chat-sidebar__new').click();
    await page.locator('.home-hero__title').waitFor();
    const homeTitleWrap = await page.locator('.home-hero__title').evaluate((el) => getComputedStyle(el).textWrap);
    assert.equal(homeTitleWrap, 'balance');

    assert.deepEqual(errors, []);
    await context.close();
  }

  // 2. shell_ui 插件库存页（ShellDisclosure 双向过渡与样式）
  {
    const context = await browser.newContext({ viewport: { width: 960, height: 700 } });
    const page = await context.newPage();
    page.setDefaultTimeout(15_000);
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));

    await page.goto(`${origin}/shell-test.html`);
    await page.locator('.shell-inventory-head').first().waitFor();

    const rowExpand = await page.evaluate(async () => {
      const btn = document.querySelector('.shell-inventory-head');
      const box = document.querySelector('.shell-inventory-row .shell-disclosure');
      btn.click();
      const heights = [];
      for (let i = 0; i < 20; i += 1) {
        await new Promise((resolve) => requestAnimationFrame(resolve));
        heights.push(Math.round(box.getBoundingClientRect().height * 100) / 100);
      }
      return heights;
    });
    console.log('[2a] shell_ui row expand:', JSON.stringify(rowExpand));
    assert.ok(middles(rowExpand) >= 1, `shell_ui 行展开应平滑过渡，实际：${rowExpand}`);

    const techExpand = await page.evaluate(async () => {
      const btn = document.querySelector('.shell-inventory-tech-trigger');
      const box = document.querySelector('.shell-inventory-tech .shell-disclosure');
      btn.click();
      const heights = [];
      for (let i = 0; i < 20; i += 1) {
        await new Promise((resolve) => requestAnimationFrame(resolve));
        heights.push(Math.round(box.getBoundingClientRect().height * 100) / 100);
      }
      return heights;
    });
    console.log('[2b] shell_ui tech info expand:', JSON.stringify(techExpand));
    assert.ok(middles(techExpand) >= 1, `技术信息展开应平滑过渡，实际：${techExpand}`);

    const rowCollapse = await page.evaluate(async () => {
      const btn = document.querySelector('.shell-inventory-head');
      const box = document.querySelector('.shell-inventory-row .shell-disclosure');
      btn.click();
      const heights = [];
      for (let i = 0; i < 20; i += 1) {
        await new Promise((resolve) => requestAnimationFrame(resolve));
        heights.push(Math.round(box.getBoundingClientRect().height * 100) / 100);
      }
      return heights;
    });
    console.log('[2c] shell_ui row collapse:', JSON.stringify(rowCollapse));
    assert.ok(middles(rowCollapse) >= 1, `shell_ui 行收起应平滑过渡，实际：${rowCollapse}`);
    assert.equal(rowCollapse.at(-1), 0);

    const shellStyles = await page.evaluate(() => {
      const card = document.querySelector('.shell-inventory-rows');
      const name = document.querySelector('.shell-inventory-head strong');
      const status = document.querySelector('.shell-inventory-status');
      return {
        cardBorder: getComputedStyle(card).borderTopWidth,
        cardShadow: getComputedStyle(card).boxShadow,
        nameWrap: getComputedStyle(name).textWrap,
        statusTabular: status ? getComputedStyle(status).fontVariantNumeric : null,
      };
    });
    console.log('[2d] shell_ui craft styles:', JSON.stringify(shellStyles));
    assert.equal(shellStyles.cardBorder, '0px');
    assert.notEqual(shellStyles.cardShadow, 'none');
    assert.equal(shellStyles.nameWrap, 'balance');

    assert.deepEqual(errors, []);
    await context.close();
  }

  console.log('PASS verify-craft-polish');
} finally {
  await browser.close();
  for (const socket of sockets) socket.terminate();
  await new Promise((done) => server.close(done));
  wss.close();
  console.log('证据目录: ' + root);
}

/** PR #5 真实浏览器验收：SubagentWorkbench 双态输入框、外部只读态、composer.status 失败横幅与血缘交互。 */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { existsSync, mkdirSync, readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { resolve, dirname, extname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { chromium } from 'playwright-core';

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const root = process.env.EVIDENCE_DIR
  ? (mkdirSync(process.env.EVIDENCE_DIR, { recursive: true }), resolve(process.env.EVIDENCE_DIR))
  : mkdtempSync(resolve(tmpdir(), 'akashic-subagent-wb-'));

// 构建一个真实挂载 SubagentWorkbench + runner-ui composer.status 的独立 harness 页面
writeFileSync(root + '/index.html', '<html data-theme="paper-light"><head><link rel="stylesheet" href="/assets/fixture.css"></head><body style="margin:0"><div class="akashic-chat" style="display:flex;flex-direction:column;gap:10px;height:100vh;padding:16px"><div id="banner"></div><div id="wb" style="flex:1 1 auto;min-height:0"></div></div><script type="module" src="/assets/fixture.js"></script></body></html>');
writeFileSync(root + '/fixture.tsx', `
import { createRoot } from 'react-dom/client';
import { initializeTheme } from ${JSON.stringify(resolve(repo, 'packages/akashic-theme/src/theme-runtime.ts'))};
initializeTheme();
import { SubagentWorkbench } from ${JSON.stringify(resolve(repo, 'frontend/plugins/subagent/src/workbench.tsx'))};
import runnerUi from ${JSON.stringify(resolve(repo, 'frontend/plugins/subagent/src/runner-ui.tsx'))};
import ${JSON.stringify(resolve(repo, 'frontend/plugins/akashic_clients/chat/src/styles.css'))};

const openedIntents: unknown[] = [];
(window as any).__openedIntents = openedIntents;
let intentCb: ((intent: any) => any) | null = null;

const fakeHttp = {
  request: (url: string, init?: RequestInit) => fetch(url, init),
};

const fakeView = {
  active: true,
  onActiveChange: () => () => {},
  session: 'sess-wb-1',
  onSessionChange: () => () => {},
  requestAttention: () => {},
  reveal: () => {},
  onIntent: (cb: (intent: any) => any) => { intentCb = cb; return () => {}; },
  quote: () => 'inserted' as const,
  instance: null,
  onClose: () => () => {},
  instances: [],
  openNew: () => {},
  cycle: () => false,
  setTitle: () => {},
  setIndicator: () => {},
};

createRoot(document.getElementById('wb')!).render(
  <SubagentWorkbench ctx={{ http: fakeHttp } as any} view={fakeView as any} />
);

const bannerHost = document.getElementById('banner')!;
let pushBannerPayload: ((v: any) => void) | null = null;
runnerUi.slots['chat.notice']!.mount(bannerHost, {
  slot: 'chat.notice',
  sessionId: 'sess-wb-1',
  onInvalidate: () => () => {},
  reportError: () => {},
  onShown: () => () => {},
  capabilities: {},
  subscribe: (cb) => {
    pushBannerPayload = cb;
    cb({
      subagents: [
        { job_id: 'sub-run-1', label: '重构后端池', session_id: 'subagent:sub-run-1', parent_session_id: 'sess-wb-1', depth: 1, read_only: false, fork: true, status: 'running', active: true, error: '' },
        { job_id: 'sub-fail-2', label: '修复构建', session_id: 'subagent:sub-fail-2', parent_session_id: 'sess-wb-1', depth: 2, read_only: true, status: 'failed', active: false, error: 'Exit code 2', fail_seq: 1 },
      ],
      runs: [
        { id: 'ext-1', label: 'Codex 审查', runner: 'codex', status: 'completed', result: 'LGTM', progress: true, activities: [{ kind: 'status', text: 'Codex 完成只读检查' }] },
      ],
    });
    (window as any).__pushBanner = (v: any) => pushBannerPayload?.(v);
    return () => {};
  },
  query: async () => ({}),
  workbench: {
    accepts: (name: string) => name === 'subagent.open.v1',
    open: async (name: string, payload: unknown) => {
      openedIntents.push({ name, payload });
      intentCb?.({ id: 'i1', name, session: 'sess-wb-1', payload });
      return 'handled';
    },
  },
});
`);

mkdirSync(root + '/dist', { recursive: true });
writeFileSync(root + '/dist/index.html', readFileSync(root + '/index.html'));
writeFileSync(root + '/chat-ui-shim.tsx', "import { createRoot } from 'react-dom/client'; export function render(host: Element, node: any) { const r = createRoot(host); r.render(node); return () => r.unmount(); } export function ToolText({ text }: { text: string }) { return <span>{text}</span>; }");
await build({
  entryPoints: [root + '/fixture.tsx'],
  outfile: root + '/dist/fixture.js',
  bundle: true,
  format: 'esm',
  jsx: 'automatic',
  absWorkingDir: repo,
  nodePaths: [resolve(repo, 'node_modules')],
  loader: { '.css': 'css', '.woff2': 'empty', '.woff': 'empty', '.ttf': 'empty' },
  alias: { '@akashic/chat-ui-v1': root + '/chat-ui-shim.tsx' },
});

const sentMessages = [];
const messageRequests = [];
const server = createServer(async (req, res) => {
  const url = new URL(req.url, 'http://127.0.0.1');
  if (url.pathname === '/api/dashboard/subagent/tree') {
    res.writeHead(200, { 'content-type': 'application/json' });
    return res.end(JSON.stringify({
      session: 'sess-wb-1',
      subagents: [
        { job_id: 'sub-run-1', label: '重构后端池', session_id: 'subagent:sub-run-1', parent_session_id: 'sess-wb-1', depth: 1, read_only: false, fork: true, model_id: 'deepseek-chat', cwd: '/srv/app', status: 'running', active: true },
        { job_id: 'sub-fail-2', label: '修复构建', session_id: 'subagent:sub-fail-2', parent_session_id: 'subagent:sub-run-1', depth: 2, read_only: true, status: 'failed', active: false, error: 'Exit code 2' },
      ],
      runs: [
        { id: 'ext-1', label: 'Codex 审查', runner: 'codex', status: 'completed', result: 'LGTM', progress: true, activities: [{ kind: 'status', text: 'Codex 完成只读检查' }] },
      ],
    }));
  }
  if (url.pathname === '/api/dashboard/subagent/messages') {
    const jid = url.searchParams.get('job_id');
    messageRequests.push({ jid, after: url.searchParams.get('after_seq'), before: url.searchParams.get('before_seq') });
    const beforeSeq = url.searchParams.get('before_seq');
    if (jid === 'sub-run-1' && beforeSeq === '10') {
      res.writeHead(200, { 'content-type': 'application/json' });
      return res.end(JSON.stringify({
        job_id: jid, status: 'running', active: true, has_more: false,
        messages: [
          { seq: 0, message_id: 'old-0', author: 'parent', kind: 'input', timestamp: '2026-11-17T01:59:00Z', parts: [{ kind: 'text', value: '更早的历史上下文' }] },
        ],
      }));
    }
    res.writeHead(200, { 'content-type': 'application/json' });
    return res.end(JSON.stringify({
      job_id: jid,
      status: jid === 'sub-fail-2' ? 'failed' : 'running',
      active: jid === 'sub-run-1',
      has_more: jid === 'sub-run-1',
      messages: jid === 'sub-run-1' ? [
        ...Array.from({ length: 14 }, (_, k) => ({
          seq: 10 + k,
          message_id: `m-${k}`,
          author: k % 2 ? 'subagent' : 'parent',
          kind: k % 2 ? 'output' : 'input',
          timestamp: '2026-11-17T02:00:05Z',
          parts: k === 12
            ? [
                { kind: 'tool_call', binding_id: 'shell', arguments: { command: 'ls skills/', description: '核对可用技能目录' } },
                { kind: 'tool_call', binding_id: 'read', arguments: { path: 'plugins/subagent/runtime.py' } },
                { kind: 'tool_call', binding_id: 'search', arguments: { query: 'EventMail' } },
                { kind: 'tool_call', binding_id: 'shell', arguments: { command: 'pytest tests/test_subagent_workbench_and_ui.py', description: '运行子任务回归测试' } },
                { kind: 'tool_call', binding_id: 'read', arguments: { path: 'frontend/plugins/subagent/src/workbench.tsx' } },
              ]
            : [{ kind: 'text', value: k === 13 ? '正在检查 runtime.py...' : `历史步骤 #${k} 内容占位行` }],
        })),
        ...sentMessages.filter(m => m.job_id === jid).map((m, i) => ({
          seq: 30 + i, message_id: `sent-${i}`, author: 'parent', kind: 'input', timestamp: '2026-11-17T02:00:06Z', parts: [{ kind: 'text', value: m.message }],
        })),
      ] : [
        { seq: 0, message_id: 'm0', author: 'parent', kind: 'input', timestamp: '2026-11-17T02:00:00Z', parts: [{ kind: 'text', value: '请完成 mail-watch 全链路开发与安装验收：核对 hookhub 过滤规则、补齐增量读取接口、创建独立插件仓库并以 submodule 接入 akashic-fleet，最后在隔离实例里验证激活、定时拉取、去重游标与真实 EventMail 投递链路。'.repeat(2) }] },
        { seq: 1, message_id: 'm1', author: 'subagent', kind: 'output', timestamp: '2026-11-17T02:00:05Z', parts: [{ kind: 'text', value: '构建报错 Exit code 2' }] },
      ],
    }));
  }
  if (url.pathname === '/api/dashboard/subagent/send' && req.method === 'POST') {
    const chunks = [];
    for await (const c of req) chunks.push(c);
    const body = JSON.parse(Buffer.concat(chunks).toString());
    sentMessages.push(body);
    res.writeHead(200, { 'content-type': 'application/json' });
    return res.end(JSON.stringify({ job_id: body.job_id, mode: 'steered', status: 'queued' }));
  }
  const rel = url.pathname === '/' ? 'index.html' : url.pathname.replace(/^\/assets\//, '');
  const file = resolve(root + '/dist', rel);
  if (!existsSync(file)) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { 'content-type': ({ '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' })[extname(file)] ?? 'application/octet-stream' });
  res.end(readFileSync(file));
});

await new Promise((done) => server.listen(0, '127.0.0.1', done));
const origin = 'http://127.0.0.1:' + server.address().port;

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH ?? '/usr/bin/chromium',
  headless: true,
  args: ['--no-sandbox', '--disable-dev-shm-usage'],
});

try {
  const ctx = await browser.newContext({ viewport: { width: 1100, height: 820 } });
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(origin + '/');

  // 1. 验证 composer.status 失败提醒横幅出现，且点击 "在工作台查看" 切换到对应子任务
  await page.locator('.subagent-composer-alert').waitFor();
  assert.match(await page.locator('.subagent-composer-alert').innerText(), /修复构建/);
  await page.locator('.subagent-composer-alert__open').click();
  await page.waitForFunction(() => document.body.innerText.includes('构建报错 Exit code 2'));
  // 1a. 长委派指令默认折叠，点击才展开
  const briefTrigger = page.locator('.subagent-wb__brief-trigger');
  await briefTrigger.waitFor();
  assert.equal(await briefTrigger.getAttribute('aria-expanded'), 'false');
  await briefTrigger.click();
  assert.equal(await briefTrigger.getAttribute('aria-expanded'), 'true');
  await briefTrigger.click();
  // 1b. 已结束的子任务只读一次消息，不再每 1.2s 轮询
  const settledBefore = messageRequests.filter((r) => r.jid === 'sub-fail-2').length;
  await page.waitForTimeout(3000);
  const settledPolls = messageRequests.filter((r) => r.jid === 'sub-fail-2').length - settledBefore;
  console.log('[1b] settled child message polls in 3s:', settledPolls);
  assert.equal(settledPolls, 0, `已结束的子任务不应继续轮询消息，3 秒内又请求了 ${settledPolls} 次`);

  // 2. 验证单焦点切换器 + 工具步骤折叠 + 可续聊/Steer
  await page.locator('.subagent-wb__switcher').click();
  await page.locator('.subagent-wb__pill').first().click();
  await page.waitForFunction(() => document.body.innerText.includes('正在检查 runtime.py'));
  // 2b. 工具调用早期步骤默认折叠成计数行，尾部保留 3 条
  const foldRow = page.locator('.subagent-wb__fold-row');
  await foldRow.waitFor();
  assert.equal(await page.locator('.subagent-wb__tool-item').count(), 3);
  await page.locator('.subagent-wb__composer textarea').fill('优先核对锁顺序');
  await page.locator('.subagent-wb__composer button[type="submit"]').click();
  await page.waitForFunction(() => document.body.innerText.includes('优先核对锁顺序'));
  assert.equal(sentMessages.length, 1);
  // 2c. 运行中的子任务首读之后按 after_seq 增量拉取，不重复取整页
  const runPolls = messageRequests.filter((r) => r.jid === 'sub-run-1' && r.before === null);
  console.log('[2c] running child polls:', JSON.stringify(runPolls.map((r) => r.after)));
  // 每次选中先读最近一页（after 为空），之后只增量拉取
  const sinceSelect = runPolls.slice(runPolls.map((r) => r.after).lastIndexOf(null) + 1);
  assert.ok(sinceSelect.length > 0 && sinceSelect.every((r) => Number(r.after) >= 23), `选中后的轮询应带 after_seq，实际：${runPolls.map((r) => r.after)}`);
  // 2d. 验证加载更早消息 + 向上滚动时轮询不会强拉回底部
  await page.locator('.subagent-wb__load-older').click();
  await page.waitForFunction(() => document.body.innerText.includes('更早的历史上下文'));
  await page.evaluate(() => {
    const s = document.querySelector('.subagent-wb__stream');
    if (s) s.scrollTop = 0;
  });
  await page.waitForTimeout(1600);
  const stayedTop = await page.evaluate(() => document.querySelector('.subagent-wb__stream')?.scrollTop ?? -1);
  assert.equal(stayedTop, 0, 'scrollTop must stay at 0 when user scrolled up during tree poll');
  await page.screenshot({ path: root + '/workbench-live.png', animations: 'disabled' });

  // 3. 切换到外部只读代理：验证显示只读横幅且无发送输入框
  await page.locator('.subagent-wb__switcher').click();
  await page.locator('.subagent-wb__pill').nth(2).click();
  await page.locator('.subagent-wb__readonly-composer').waitFor();
  assert.equal(await page.locator('.subagent-wb__composer').count(), 0);

  // 4. 关闭提醒横幅后，若同一任务经冷恢复再次失败（fail_seq 变化），应重新弹出提醒
  await page.locator('.subagent-composer-alert__dismiss').click();
  assert.equal(await page.locator('.subagent-composer-alert').count(), 0);
  await page.evaluate(() => {
    window.__pushBanner({
      subagents: [{ job_id: 'sub-fail-2', label: '修复构建', session_id: 'subagent:sub-fail-2', parent_session_id: 'sess-wb-1', depth: 2, read_only: true, status: 'failed', active: false, error: 'Second failure', fail_seq: 5 }],
      runs: [],
    });
  });
  await page.locator('.subagent-composer-alert').waitFor();
  await page.screenshot({ path: root + '/workbench-readonly.png', animations: 'disabled' });
  // 5. 移动端 390x844 溢出检查
  await page.setViewportSize({ width: 390, height: 844 });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
  assert.equal(overflow, false, 'mobile horizontal overflow');
  assert.deepEqual(errors, []);
  console.log('PASS verify-workbench (lineage + failure banner + live steer + external readonly + mobile)');
  await ctx.close();
} finally {
  await browser.close();
  await new Promise((done) => server.close(done));
}

/** 用真实页面检查窄屏阅读和返回路径；输出报告与逐页截图，不提交业务写操作。 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { chmod, mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { chromium } from "playwright-core";

const { values } = parseArgs({
  options: {
    url: { type: "string" },
    output: { type: "string", default: "/tmp/akashic-narrow-ui" },
    browser: {
      type: "string",
      default: process.env.CHROME_BIN ?? "/opt/google/chrome/chrome",
    },
    widths: { type: "string", default: "320,360,390,412,448,760,1024,1365" },
    height: { type: "string", default: "914" },
    page: { type: "string", default: "all" },
    large: { type: "boolean", default: false },
    rtl: { type: "boolean", default: false },
    candidate: { type: "boolean", default: false },
    plugins: {
      type: "string",
      default: "shell-ui,conversation-ui,onboarding,akasha,models,computer",
    },
    bundles: { type: "string", default: "" },
  },
});
assert(values.url, "请显式提供 --url，选择本次验收的服务");
assert(["all", "models"].includes(values.page), "page 只接受 all 或 models");
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const output = resolve(values.output);
const bundles = new Map(
  values.bundles
    .split(",")
    .filter(Boolean)
    .map((item) => item.split("=")),
);
await mkdir(output, { recursive: true, mode: 0o700 });
await chmod(output, 0o700);
const report = {
  url: values.url,
  candidate: values.candidate,
  screens: [],
  failures: [],
  blockedRequests: [],
  pages: [],
};
const browser = await chromium.launch({
  executablePath: values.browser,
  headless: true,
});
let screenshot = 0;

/** 只放行已沿服务端 owner 核对的读取；同一 query 入口也有保存和删除。 */
function readQuery(query) {
  const plugin = query?.plugin_id?.split("@")[0];
  const payload = query?.payload;
  if (!payload || typeof payload !== "object" || Array.isArray(payload))
    return false;
  if (plugin === "projects" && query.method === "project.list")
    return Object.keys(payload).length === 0;
  // plugins/akasha/plugin.py 的策略读取与召回记录投影。
  if (plugin === "akasha")
    return (
      query.method === "scope.policy.get" || query.method === "recall.turn"
    );
  // 外部插件源码的只读 Message/Compaction 与 Observe 投影。
  return (
    (plugin === "status_commands" && query.method === "memory.status") ||
    (plugin === "observe" && query.method === "kvcache.message_usage")
  );
}

/** 检查当前阅读区、可见控件和长正文，并保存私有截图。 */
async function measure(page, label) {
  const view = page.locator(".shell-view.is-active");
  await view.waitFor();
  const file = `${String(++screenshot).padStart(3, "0")}-${label}.png`;
  await page.screenshot({
    path: resolve(output, file),
    animations: "disabled",
    mask: [page.locator('input[type="password"]')],
  });
  const size = await view.evaluate((e) => ({
    width: e.clientWidth,
    scrollWidth: e.scrollWidth,
    height: e.clientHeight,
  }));
  assert(
    size.width >= page.viewportSize().width - 1,
    `${label}: 页面未占满宽度`,
  );
  assert(size.scrollWidth <= size.width + 1, `${label}: 页面横向溢出`);
  const reading = [];
  for (const frame of page.frames()) {
    if (
      frame.parentFrame() &&
      !(await (await frame.frameElement()).isVisible())
    )
      continue;
    const geometry = await frame.evaluate(() => {
      const regions = [
        ...document.querySelectorAll(
          '.shell-view.is-active,[role="dialog"],dialog[open]',
        ),
      ];
      if (!regions.length) regions.push(document.body);
      const visible = (e) => {
        const r = e.getBoundingClientRect();
        return (
          !!(r.width && r.height) && getComputedStyle(e).visibility !== "hidden"
        );
      };
      const controls = regions.flatMap((root) => [
        ...root.querySelectorAll("button,input,select,textarea"),
      ]);
      const clipped = controls
        .filter(visible)
        .filter((e) => {
          if (
            e.closest(
              ".product-band__nav,.onboarding-progress,.meme-dashboard__sidebar,.model-capsule__rails",
            )
          )
            return false;
          const r = e.getBoundingClientRect();
          return (
            r.bottom > 0 &&
            r.top < innerHeight &&
            (r.left < -1 || r.right > innerWidth + 1)
          );
        })
        .map((e) => ({ tag: e.tagName, class: String(e.className) }));
      const reading = regions
        .flatMap((root) => [
          ...root.querySelectorAll(
            ".detail-content,.feedback-step p,.observe-error-message,.akasha-evidence-main p,.plain-message-response",
          ),
        ])
        .filter(visible)
        .filter((e) => e.textContent.length > 100 && !e.closest("pre"))
        .map((e) => ({
          class: String(e.className),
          fontSize: parseFloat(getComputedStyle(e).fontSize),
          lineHeight: parseFloat(getComputedStyle(e).lineHeight),
        }));
      return { width: innerWidth, clipped, reading };
    });
    assert.deepEqual(geometry.clipped, [], `${label}: 可见控件被横向裁切`);
    for (const text of geometry.reading) {
      assert(text.fontSize >= 16, `${label}: 长正文小于 16px`);
      assert(text.lineHeight >= text.fontSize * 1.4, `${label}: 正文行距过小`);
    }
    reading.push(...geometry.reading);
  }
  report.screens.push({
    label,
    viewport: page.viewportSize(),
    ...size,
    reading,
    screenshot: file,
  });
}

/** 滚动到真实控件并检查中心点，避免把被遮挡误判成可操作。 */
async function hit(locator) {
  await locator.evaluate((e) =>
    e.scrollIntoView({
      behavior: "instant",
      block: "center",
      inline: "nearest",
    }),
  );
  return locator.evaluate((e) => {
    const r = e.getBoundingClientRect(),
      x = r.x + r.width / 2,
      y = r.y + r.height / 2;
    const target =
      e.getRootNode() instanceof ShadowRoot ? e.getRootNode().host : e;
    const top = document.elementFromPoint(x, y);
    return (
      r.width > 0 &&
      r.height > 0 &&
      x >= 0 &&
      x < innerWidth &&
      y >= 0 &&
      y < innerHeight &&
      (top === target || target.contains(top))
    );
  });
}

/** 通过 Shell 的原生入口切换页面。 */
async function main(page, label, selector) {
  // 全新 workspace 首跑时整屏引导会遮住导航；像用户一样关掉它。
  await dismissOnboarding(page);
  // 只有一个页面时 Shell 不渲染顶栏，默认页面已经是当前页。
  const band = page.locator(".product-band__nav");
  if (await band.count())
    await band.getByRole("button", { name: label, exact: true }).click();
  await page
    .locator(`.shell-view.is-active ${selector}`)
    .waitFor({ state: "visible" });
}

/** 整屏引导打开时像用户一样离开：欢迎页选“直接聊天”，其余步骤选“稍后再说”。 */
async function dismissOnboarding(page) {
  const surface = page.locator("dialog.onboarding-surface[open]");
  if (!(await surface.count())) return;
  await surface
    .getByRole("button", { name: /^(稍后再说|我已经配好了，直接聊天)$/ })
    .first()
    .click();
  await surface.waitFor({ state: "detached" });
}

/** 遍历设置面板：每个分节、每张插件卡片的详情，只作本地编辑并放弃。 */
async function settings(page, prefix) {
  await page.evaluate(() => { window.location.hash = "plugins"; });
  const panel = page.locator("dialog.shell-settings-dialog[open]");
  await panel.waitFor();
  const names = await panel.locator("nav button").allTextContents();
  for (const [index, name] of names.entries()) {
    // 窄屏是两级页面：先回到分节列表再选。
    const back = panel.locator(".shell-settings-back");
    if (await back.isVisible()) await back.click();
    await panel.locator("nav button", { hasText: name }).click();
    await page.waitForTimeout(400);
    await measure(page, `${prefix}-settings-${index}`);
    if (await panel.locator(".shell-plugin-cards").count()) await pluginCards(page, prefix, panel);
    const rerun = panel.getByRole("button", { name: "重新走一遍引导", exact: true });
    if (await rerun.count()) {
      await rerun.click();
      await onboarding(page, prefix);
      await panel.locator(".onboarding-overview").waitFor();
    }
  }
  await panel.getByRole("button", { name: "关闭设置", exact: true }).click();
  await panel.waitFor({ state: "detached" }).catch(() => {});
}

/** 逐张打开插件卡片：详情页、开启后的字段与高级设置；离开时未保存修改应被拦下。 */
async function pluginCards(page, prefix, panel) {
  const count = await panel.locator(".shell-plugin-card").count();
  for (let i = 0; i < count; i++) {
    await panel.locator(".shell-plugin-card").nth(i).click();
    await panel.locator(".shell-settings-mount .config-form form").first().waitFor();
    await measure(page, `${prefix}-plugin-${i}`);
    const toggle = panel.locator('.config-toggle input[role="switch"]').first();
    let edited = false;
    if ((await toggle.count()) && (await toggle.isEnabled()) && !(await toggle.isChecked())) {
      await toggle.check();
      edited = true;
      await measure(page, `${prefix}-plugin-${i}-enabled-fields`);
    }
    const advance = panel.locator(".shell-settings-mount details summary");
    for (let j = 0; j < (await advance.count()); j++) {
      await advance.nth(j).click();
      await measure(page, `${prefix}-plugin-${i}-advanced-${j}`);
    }
    const crumb = panel.locator(".shell-settings-crumb");
    await ((await crumb.isVisible()) ? crumb : panel.locator(".shell-settings-back")).click();
    if (edited) {
      const confirm = page.locator("dialog.config-dialog[open]");
      await confirm.waitFor();
      await measure(page, `${prefix}-plugin-${i}-dirty-confirm`);
      const leave = confirm.getByRole("button", { name: "放弃并离开", exact: true });
      assert(await hit(leave));
      await leave.click();
    }
    await panel.locator(".shell-plugin-cards").waitFor();
  }
}

/** 走整屏引导的欢迎页与首个配置步骤，检查动作可达后用“稍后再说”离开。 */
async function onboarding(page, prefix) {
  const surface = page.locator("dialog.onboarding-surface[open]");
  await surface.getByRole("button", { name: "开始", exact: true }).waitFor();
  await measure(page, `${prefix}-onboarding-welcome`);
  await surface.getByRole("button", { name: "开始", exact: true }).click();
  await surface
    .locator(".onboarding-column .settings-connection-card,.onboarding-column .config-form form")
    .first()
    .waitFor();
  await measure(page, `${prefix}-onboarding-step`);
  for (const button of await surface
    .locator(".onboarding-footer button:not(:disabled):not([hidden]),.onboarding-top button")
    .all())
    assert(await hit(button), "引导动作不可达");
  await dismissOnboarding(page);
}

/** 检查插件自己拥有的阅读路径，不触发删除或发送。 */
async function customPanel(page, prefix, id) {
  const view = page.locator(".shell-view.is-active");
  if (id === "fitbit-health") {
    await view.locator("[data-fitbit-content]:not([hidden])").waitFor();
    await view.locator(".fitbit-dashboard__predictions summary").click();
    await measure(page, `${prefix}-fitbit-predictions`);
  }
  if (id === "meme") {
    const categories = view.locator("button.meme-category");
    await categories.first().waitFor();
    // 分类只读取图片；不触发相邻的删除按钮。
    await categories.last().click();
    await view
      .locator('.meme-state[role="status"]')
      .waitFor({ state: "hidden" });
    await measure(page, `${prefix}-meme-last-category`);
  }
  if (id === "observe") {
    const portal = view.getByRole("button", {
      name: "查看错误分析",
      exact: true,
    });
    await portal.waitFor();
    await portal.click();
    const dialog = page.getByRole("dialog", { name: /错误 ·/ });
    await dialog.waitFor();
    await measure(page, `${prefix}-observe-errors`);
    const row = dialog.locator(".observe-error-list > div > button").first();
    if (await row.count()) {
      await row.click();
      await dialog.locator(".observe-error-detail").waitFor();
      for (const tab of ["趋势", "Traceback", "现场"]) {
        const button = dialog
          .getByRole("button", { name: new RegExp("^" + tab) })
          .first();
        await button.click();
        await measure(page, `${prefix}-observe-${tab}`);
      }
      const back = dialog.getByRole("button", {
        name: "‹ 错误列表",
        exact: true,
      });
      if (await back.isVisible()) {
        assert(await hit(back));
        await back.click();
      }
    }
    const close = dialog.getByRole("button", {
      name: "关闭错误分析",
      exact: true,
    });
    assert(await hit(close));
    await close.click();
  }
}

/** 检查真实聊天、只读历史、弹窗和工具，不发送输入。 */
async function chat(page, prefix, narrow) {
  await main(page, "对话", "iframe");
  const frame = page.frameLocator(".shell-view.is-active iframe");
  await frame.locator("textarea:not([disabled])").waitFor();
  if (values.large || values.rtl)
    await frame.locator("html").evaluate(
      (root, { large, rtl }) => {
        if (large) root.style.fontSize = "200%";
        if (rtl) root.dir = "rtl";
      },
      { large: values.large, rtl: values.rtl },
    );
  assert.deepEqual(
    await frame.locator('[role="alert"]').allTextContents(),
    [],
    "真实聊天页出现错误提示",
  );
  await measure(page, `${prefix}-chat`);
  const model = frame.locator(".model-capsule__trigger");
  await model.click();
  const picker = frame.getByRole("dialog", { name: "选择模型", exact: true });
  await picker.waitFor();
  await measure(page, `${prefix}-chat-model-picker`);
  const tabs = picker.getByRole("tab");
  for (let i = 0; i < (await tabs.count()); i++) {
    await tabs.nth(i).click();
    assert(await hit(tabs.nth(i)));
  }
  const choices = picker.locator(".model-capsule__option");
  assert(await hit(choices.first()), "模型首项不可达");
  assert(await hit(choices.last()), "模型末项不可达");
  await measure(page, `${prefix}-chat-model-last`);
  const effort = picker.locator(".model-capsule__effort-entry");
  if (await effort.count()) {
    await effort.click();
    await measure(page, `${prefix}-chat-effort-picker`);
  }
  await frame
    .getByRole("button", { name: "关闭模型选择", exact: true })
    .click();
  if (narrow)
    await frame.getByRole("button", { name: "打开导航", exact: true }).click();
  await measure(page, `${prefix}-chat-navigation`);
  const navigation = narrow
    ? frame.locator(".compact-navigation-dialog")
    : frame.locator(".chat-sidebar").first();
  const create = navigation.getByRole("button", {
    name: "新建项目",
    exact: true,
  });
  if (await create.count()) {
    await create.click();
    const dialog = frame.getByRole("dialog", { name: "创建项目", exact: true });
    await dialog.waitFor();
    await measure(page, `${prefix}-project-dialog`);
    const memory = dialog.getByRole("button", { name: /记忆设置/ });
    if (await memory.count()) {
      await memory.click();
      await measure(page, `${prefix}-project-memory-menu`);
      await frame.locator("body").press("Escape");
    }
    await frame.locator("body").press("Escape");
    if (narrow)
      await frame
        .getByRole("button", { name: "打开导航", exact: true })
        .click();
  }
  const recent = navigation.locator(".conversation-session").first();
  if (await recent.count()) {
    await recent.click();
    await frame.locator(".message-row").first().waitFor();
    await measure(page, `${prefix}-chat-history`);
  } else if (narrow) await frame.locator("body").press("Escape");
  // 工具只展开与关闭；不点击远程桌面，不发送按键或剪贴板。
  const toggle = page.locator(
    ".shell-view.is-active .conversation-tools-toggle",
  );
  if (await toggle.count()) {
    await toggle.click();
    const tools = page.locator(".shell-view.is-active .conversation-tools");
    const tabs = tools.getByRole("tab");
    for (let i = 0; i < (await tabs.count()); i++) {
      await tabs.nth(i).click();
      await measure(page, `${prefix}-tool-${i}`);
      const clipboard = tools.getByRole("button", {
        name: "打开剪贴板",
        exact: true,
      });
      if (await clipboard.count()) {
        await clipboard.click();
        await measure(page, `${prefix}-computer-clipboard`);
        const controls = tools.locator(
          ".computer-clipboard button,.computer-clipboard textarea",
        );
        for (let j = 0; j < (await controls.count()); j++)
          assert(await hit(controls.nth(j)), "剪贴板控件不可达");
        await tools
          .getByRole("button", { name: "关闭剪贴板", exact: true })
          .click();
      }
    }
    const close = tools.getByRole("button", {
      name: "关闭工具区",
      exact: true,
    });
    assert(await hit(close));
    await close.click();
  }
}

/** 遍历实际连接与模板，检查表单操作和关闭。 */
async function models(page, prefix) {
  // 模型是设置面板里的分节，经深链接直接进入。
  await page.evaluate(() => { window.location.hash = "models"; });
  const view = page.locator("dialog.shell-settings-dialog[open]");
  await view.locator(".settings-page").waitFor();
  await view.locator(".settings-connection-card").first().waitFor();
  await measure(page, `${prefix}-models`);
  const cards = view.locator("button.settings-connection-card");
  const count = await cards.count();
  for (let i = 0; i < count; i++) {
    await cards.nth(i).click();
    const dialog = page.locator("dialog.settings-scrim[open]");
    await dialog.waitFor();
    await measure(page, `${prefix}-connection-${i}`);
    const summaries = dialog.locator("summary");
    for (let j = 0; j < (await summaries.count()); j++) {
      await summaries.nth(j).click();
      await measure(page, `${prefix}-connection-${i}-section-${j}`);
    }
    const editors = dialog.locator(".settings-model-expand");
    for (let j = 0; j < (await editors.count()); j++) {
      await editors.nth(j).click();
      await dialog.locator(".settings-model-detail").nth(j).scrollIntoViewIfNeeded();
      await measure(page, `${prefix}-connection-${i}-model-${j}`);
    }
    const manual = dialog.locator("[data-manual-toggle]");
    if (await manual.count()) {
      await manual.click();
      await measure(page, `${prefix}-connection-${i}-manual`);
    }
    const controls = dialog.locator("input,select,textarea,button");
    for (let j = 0; j < (await controls.count()); j++)
      if (await controls.nth(j).isVisible())
        assert(await hit(controls.nth(j)), `连接 ${i}: 表单控件不可达`);
    await page.keyboard.press("Escape");
    await dialog.waitFor({ state: "hidden" });
  }
  await view.getByRole("button", { name: "添加向量模型", exact: true }).click();
  const dialog = page.getByRole("dialog", {
    name: "添加向量模型",
    exact: true,
  });
  await measure(page, `${prefix}-embedding`);
  const cancel = dialog.getByRole("button", { name: "取消", exact: true });
  assert(await hit(cancel));
  await cancel.click();
  await view.getByRole("button", { name: "关闭设置", exact: true }).click();
}

try {
  for (const width of values.widths.split(",").map(Number)) {
    assert(
      Number.isInteger(width) && width >= 320,
      "widths 只接受不小于 320 的 CSS 像素宽度",
    );
    const context = await browser.newContext({
      viewport: { width, height: Number(values.height) },
      hasTouch: width <= 760,
      isMobile: width <= 760,
    });
    // 候选 HTML 由浏览器拦截提供；仅给指定服务 origin 授予本地网络访问。
    if (values.candidate)
      await context.grantPermissions(["local-network-access"], {
        origin: new URL(values.url).origin,
      });
    // 方法与 owner 必须同时命中；不能把所有 POST query 都当作读取。
    await context.route("**/*", async (route) => {
      const req = route.request(),
        url = new URL(req.url());
      let q;
      try {
        q = req.postDataJSON();
      } catch {}
      const read =
        req.method() === "POST" &&
        url.pathname === "/api/chat/plugin-ui/query" &&
        readQuery(q);
      if (!["GET", "HEAD"].includes(req.method()) && !read) {
        report.blockedRequests.push({
          method: req.method(),
          path: url.pathname,
          query: q?.method,
        });
        return route.abort();
      }
      const chatDocument =
        url.pathname === "/chat" || url.pathname === "/chat/";
      const chatAsset =
        url.pathname.startsWith("/assets/") &&
        new URL(req.frame().url()).pathname.replace(/\/$/, "") === "/chat";
      if (values.candidate && (chatDocument || chatAsset)) {
        const file = chatDocument
          ? "index.html"
          : url.pathname.slice("/assets/".length);
        assert(!file.includes("/"), "候选资源必须是构建目录的单个文件");
        const body = await readFile(resolve(root, "static/chat", file));
        const type = file.endsWith(".html")
          ? "text/html"
          : file.endsWith(".css")
            ? "text/css"
            : file.endsWith(".js")
              ? "text/javascript"
              : "application/octet-stream";
        return route.fulfill({ body, contentType: type });
      }
      if (values.candidate && url.pathname === "/api/chat/web-ui/bootstrap") {
        const response = await route.fetch(),
          data = await response.json();
        for (const m of data.modules) {
          const id = m.pluginId.split("@")[0];
          if (!values.plugins.split(",").includes(id) && !bundles.has(id))
            continue;
          const folder =
            bundles.get(id) ??
            resolve(root, "plugins", id.replaceAll("-", "_"));
          m.module = await readFile(resolve(folder, "web_module.js"), "utf8");
          m.stylesheet = await readFile(
            resolve(folder, "web_module.css"),
            "utf8",
          );
          m.moduleBytes = Buffer.byteLength(m.module);
          m.stylesheetBytes = Buffer.byteLength(m.stylesheet);
          // 替换候选制品时同步其现有发布元数据，不增加运行时校验。
          m.moduleSha256 = createHash("sha256").update(m.module).digest("hex");
          m.stylesheetSha256 = createHash("sha256")
            .update(m.stylesheet)
            .digest("hex");
        }
        return route.fulfill({ response, json: data });
      }
      return route.continue();
    });
    const page = await context.newPage();
    page.on("pageerror", (error) =>
      report.failures.push({
        width,
        error: String(error),
        stack: error.stack,
        lastScreen: report.screens.at(-1)?.label,
      }),
    );
    try {
      await page.goto(values.url, { waitUntil: "domcontentloaded" });
      // 全新 workspace 首跑时整屏引导会异步弹出并遮住导航；
      // 给它一个出现窗口，像用户一样关掉；看过后同一 context 内不再复现。
      if (
        await page
          .locator("dialog.onboarding-surface[open]")
          .waitFor({ timeout: 5000 })
          .then(() => true)
          .catch(() => false)
      )
        await dismissOnboarding(page);
      if (values.large || values.rtl)
        await page.evaluate(
          ({ large, rtl }) => {
            if (large) document.documentElement.style.fontSize = "200%";
            if (rtl) document.documentElement.dir = "rtl";
          },
          { large: values.large, rtl: values.rtl },
        );
      if (values.page === "all") {
        await settings(page, String(width));
      }
      await models(page, String(width));
      if (values.page === "all") await chat(page, String(width), width <= 820);
    } catch (error) {
      report.failures.push({ width, error: String(error) });
      await page.screenshot({ path: resolve(output, `${width}-failure.png`) });
    }
    await context.close();
  }
} finally {
  await writeFile(
    resolve(output, "report.json"),
    JSON.stringify(report, null, 2),
  );
  await browser.close();
}
console.log(
  JSON.stringify({
    screens: report.screens.length,
    failures: report.failures,
    report: resolve(output, "report.json"),
  }),
);
assert.deepEqual(report.blockedRequests, [], "验收触发了未授权的 HTTP 写请求");
assert.deepEqual(
  report.failures,
  [],
  "窄屏验收失败；按 report.json 和截图修复，不缩减页面清单",
);

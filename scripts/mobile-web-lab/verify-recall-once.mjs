import assert from "node:assert/strict";
import { build } from "esbuild";
import { chromium } from "playwright-core";
import { startMobileWebLabServer } from "./server.mjs";

const fixture = await build({ entryPoints: ["frontend/chat/src/mobile-lab-fixtures.ts"],
  bundle: true, write: false, format: "esm" });
const { createLabSnapshot } = await import(`data:text/javascript;base64,${Buffer.from(fixture.outputFiles[0].text).toString("base64")}`);
const plugin = await build({ entryPoints: ["frontend/plugins/akasha/src/mobile.js"],
  bundle: true, write: false, format: "esm", loader: { ".css": "empty" } });
const lab = await startMobileWebLabServer({ root: process.env.AKASHIC_MOBILE_WEB_LAB_ROOT });
const browser = await chromium.launch({ executablePath: process.env.AKASHIC_PERF_CHROMIUM || "/usr/bin/chromium", headless: true });
try {
  for (const width of [320, 412]) {
    // Lab 只允许 self；这里模拟 Android 的 appassets 资源，并加载真实 Akasha renderer。
    const page = await browser.newPage({ viewport: { width, height: 860 }, bypassCSP: true });
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.route("https://appassets.androidplatform.net/plugin-ui/**", route => route.fulfill({
      body: plugin.outputFiles[0].text, contentType: "text/javascript", headers: { "access-control-allow-origin": "*" },
    }));
    await page.goto(`${lab.origin}/mobile-lab-frame.html?generation_id=browser-lab&nonce=browser-lab`);
    await page.waitForFunction(() => Boolean(window.AkashicMobile));
    await page.evaluate(() => {
      window.AkashicNative.queryPluginUi = requestId => queueMicrotask(() =>
        window.AkashicMobile.receivePluginUiResult({ requestId, resultJson: JSON.stringify({ pending: false,
          items: [{ hits: [{ lane: "dense", messages: [{ preview: "召回内容" }] }] }],
        }) }));
      window.AkashicMobile.receivePluginCatalog({ catalogRevision: "a".repeat(64), updating: false,
        plugins: [{ id: "akasha", revision: "b".repeat(64), slots: ["turn.before_reasoning"],
          navigation: { label: "Akasha Inspector", description: "召回记录" },
          moduleUrl: "https://appassets.androidplatform.net/plugin-ui/akasha/module.js" }] });
    });
    const base = createLabSnapshot("stream");
    const [input, call, result] = base.messages.slice(2, 5);
    const second = { ...structuredClone(call), id: "second-call", seq: 5 };
    second.body.parts[0].value.thinking = "第二次调用前的思考";
    const secondResult = { ...structuredClone(result), id: "second-result", seq: 6 };
    secondResult.body.call_ref.message_id = second.id;
    const final = { ...structuredClone(base.messages.at(-1)), id: "final-answer", seq: 7 };
    final.body.parts.unshift({ kind: "model.facts", value: { call_record_id: "final-call-record", thinking: "最终思考" } });
    const phases = [
      { messages: [input], draft: call.id, thinking: "" },
      { messages: [input], draft: call.id, thinking: "第一次思考" },
      { messages: [input, call, result], draft: second.id, thinking: "第二次调用前的思考" },
      { messages: [input, call, result, second, secondResult], draft: final.id, thinking: "最终思考" },
      { messages: [input, call, result, second, secondResult, final] },
    ];
    for (const [index, phase] of phases.entries()) {
      const snapshot = structuredClone(base);
      snapshot.messages = phase.messages;
      snapshot.throughSeq = phase.messages.at(-1).seq;
      snapshot.projectionGeneration += index;
      snapshot.history = { hasOlder: false, isLatest: true, loading: false };
      snapshot.sessions[0].isRunning = Boolean(phase.draft);
      snapshot.composer.isStreaming = Boolean(phase.draft);
      snapshot.composer.canStop = Boolean(phase.draft);
      snapshot.composer.canSend = !phase.draft;
      snapshot.replyStatus.items = phase.draft ? [{ ...snapshot.replyStatus.items[0],
        preview: { message_id: phase.draft, text: "", thinking: phase.thinking } }] : [];
      await page.evaluate(snapshot => window.AkashicMobile.receiveSnapshot(snapshot), snapshot);
      if (!phase.draft) {
        await page.waitForFunction(() => document.body.innerText.includes("已思考")).catch(async error => { console.log(await page.locator("body").innerText()); throw error; });
        await page.getByRole("button", { name: /已思考/ }).click();
      }
      await page.locator(".akasha-mobile-recall-group").first().waitFor();
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      assert.equal(await page.locator(".akasha-mobile-recall-group").count(), 1, `${width}px phase ${index}`);
      if (index >= 2) assert.equal(await page.locator(".tool-step-title").filter({ hasText: "read" }).count() > 0, true);
    }
    assert.deepEqual(errors, []);
    await page.close();
  }
  console.log("Recall stays single through waiting, thinking, two tool calls and final history at 320/412px");
} finally {
  await browser.close();
  await lab.close();
}

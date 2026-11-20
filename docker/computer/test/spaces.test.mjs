import test from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { BrowserBackend, SPACE_SENTINEL_URL } from "../driver/cdp.mjs";
import { ComputerDriver } from "../driver/runtime.mjs";
import { BackgroundSpaces, SpaceInputError } from "../driver/spaces.mjs";

class FakeCdpConnection extends EventEmitter {
  constructor(handler) {
    super();
    this.handler = handler;
    this.calls = [];
  }
  async send(method, params = {}, sessionId) {
    this.calls.push({ method, params, sessionId });
    return await this.handler(method, params, sessionId, this);
  }
  close() {}
}

test("BackgroundSpaces validates inputs strictly", async () => {
  const browser = new BrowserBackend();
  browser.browser = new FakeCdpConnection(async () => ({}));
  const spaces = new BackgroundSpaces({ browser, maxConcurrency: 2 });

  await assert.rejects(spaces.run(null), SpaceInputError);
  await assert.rejects(spaces.run({}), SpaceInputError);
  await assert.rejects(spaces.run({ url: "https://example.com", script: "return 1;" }), SpaceInputError);
  await assert.rejects(spaces.run({ url: "file:///etc/passwd" }), SpaceInputError);
  await assert.rejects(spaces.run({ url: "javascript:alert(1)" }), SpaceInputError);
  await assert.rejects(spaces.run({ url: "https://example.com", waitUntil: "bogus" }), SpaceInputError);
  await assert.rejects(spaces.run({ url: "https://example.com", timeoutMs: 10 }), SpaceInputError);
  await assert.rejects(spaces.run({ script: "   " }), SpaceInputError);
});

test("BackgroundSpaces runs declarative fetch in hidden background tab and hides it from listTabs", async () => {
  const browser = new BrowserBackend();
  let nextTarget = 1;
  const liveTargets = new Map([
    ["user-front", { targetId: "user-front", type: "page", url: "https://www.bilibili.com/", title: "Bilibili" }],
  ]);
  let selectorPolls = 0;

  browser.browser = new FakeCdpConnection(async (method, params, sessionId, conn) => {
    if (method === "Target.getTargets") {
      return { targetInfos: [...liveTargets.values()] };
    }
    if (method === "Browser.getWindowForTarget") {
      return { windowId: 1 };
    }
    if (method === "Target.createTarget") {
      assert.equal(params.url, SPACE_SENTINEL_URL);
      assert.equal(params.background, true);
      assert.equal(params.newWindow, undefined);
      const targetId = `bg-${nextTarget++}`;
      liveTargets.set(targetId, { targetId, type: "page", url: params.url, title: "" });
      return { targetId };
    }
    if (method === "Target.attachToTarget") {
      return { sessionId: `sess-${params.targetId}` };
    }
    if (method === "Page.enable" || method === "Runtime.enable" || method === "Page.addScriptToEvaluateOnNewDocument") {
      return {};
    }
    if (method === "Page.navigate") {
      const tid = sessionId.replace("sess-", "");
      liveTargets.set(tid, { targetId: tid, type: "page", url: params.url, title: "X" });
      queueMicrotask(() => conn.emit("event", { sessionId, method: "Page.domContentEventFired", params: {} }));
      return { frameId: "f1", loaderId: "l1" };
    }
    if (method === "Runtime.evaluate") {
      if (params.expression.includes("document.querySelector")) {
        selectorPolls++;
        return { result: { value: selectorPolls >= 2 } };
      }
      if (params.expression === "JSON.stringify({ ok: true })") {
        return { result: { value: '{"ok":true}' } };
      }
      return { result: { value: null } };
    }
    if (method === "Target.closeTarget") {
      liveTargets.delete(params.targetId);
      return { success: true };
    }
    throw new Error(`Unexpected CDP method: ${method}`);
  });

  const spaces = new BackgroundSpaces({ browser, maxConcurrency: 2 });
  const beforeTabs = await browser.listTabs();
  assert.deepEqual(beforeTabs.map((t) => t.url), ["https://www.bilibili.com/"]);

  const runPromise = spaces.run({
    space: "x-feed",
    url: "https://x.com/kcfg258",
    waitFor: "article",
    eval: "JSON.stringify({ ok: true })",
  });

  // While running, listTabs must never expose the background space target.
  await new Promise((r) => setTimeout(r, 20));
  const midTabs = await browser.listTabs();
  assert.deepEqual(midTabs.map((t) => t.url), ["https://www.bilibili.com/"]);

  const out = await runPromise;
  assert.deepEqual(out, { space: "x-feed", result: '{"ok":true}' });
  assert.equal(liveTargets.has("bg-1"), false);
  assert.equal(spaces.activeCount, 0);
  assert.ok(
    browser.browser.calls.some((c) => c.method === "Page.addScriptToEvaluateOnNewDocument" && c.params.source.includes("visibilityState")),
  );
});

test("BackgroundSpaces supports script mode, cleans up on script error, and bounds concurrency", async () => {
  const browser = new BrowserBackend();
  let nextTarget = 1;
  const openTargets = new Set();
  let currentRunning = 0;
  let peakRunning = 0;

  browser.browser = new FakeCdpConnection(async (method, params, sessionId, conn) => {
    if (method === "Target.createTarget") {
      const targetId = `bg-${nextTarget++}`;
      openTargets.add(targetId);
      return { targetId };
    }
    if (method === "Target.attachToTarget") {
      return { sessionId: `sess-${params.targetId}` };
    }
    if (method === "Page.enable" || method === "Runtime.enable" || method === "Page.addScriptToEvaluateOnNewDocument") {
      return {};
    }
    if (method === "Page.navigate") {
      currentRunning++;
      peakRunning = Math.max(peakRunning, currentRunning);
      setTimeout(() => {
        currentRunning--;
        conn.emit("event", { sessionId, method: "Page.domContentEventFired", params: {} });
      }, 40);
      return { frameId: "f1", loaderId: "l1" };
    }
    if (method === "Runtime.evaluate") {
      if (params.expression === "document.title") return { result: { value: "Title" } };
      if (params.expression === "location.href") return { result: { value: "https://example.com/" } };
      return { result: { value: "ok" } };
    }
    if (method === "Target.closeTarget") {
      openTargets.delete(params.targetId);
      return { success: true };
    }
    return {};
  });

  const spaces = new BackgroundSpaces({ browser, maxConcurrency: 2 });

  // 4 concurrent runs with maxConcurrency=2
  const results = await Promise.all([
    spaces.run({ space: "s1", script: "await page.goto('https://example.com/1'); return 1;" }),
    spaces.run({ space: "s2", script: "await page.goto('https://example.com/2'); return 2;" }),
    spaces.run({ space: "s3", script: "await page.goto('https://example.com/3'); return 3;" }),
    spaces.run({ space: "s4", script: "await page.goto('https://example.com/4'); return 4;" }),
  ]);
  assert.deepEqual(results.map((r) => r.result), [1, 2, 3, 4]);
  assert.equal(peakRunning, 2);
  assert.equal(openTargets.size, 0);

  // Script error still closes target and releases slot
  await assert.rejects(
    spaces.run({ script: "await page.goto('https://example.com/err'); throw new Error('boom');" }),
    /boom/,
  );
  assert.equal(openTargets.size, 0);
  assert.equal(spaces.activeCount, 0);

  // Aborting while queued removes entry cleanly
  const hold1 = spaces.run({ script: "await sleep(120); return 'h1';" });
  const hold2 = spaces.run({ script: "await sleep(120); return 'h2';" });
  const ac = new AbortController();
  const abortedQueued = spaces.run({ script: "return 'never';" }, ac.signal);
  assert.equal(spaces.queuedCount, 1);
  ac.abort(new Error("caller gone"));
  await assert.rejects(abortedQueued, /caller gone/);
  assert.equal(spaces.queuedCount, 0);
  assert.deepEqual(await Promise.all([hold1, hold2]), [
    { space: "default", result: "h1" },
    { space: "default", result: "h2" },
  ]);
  assert.equal(spaces.activeCount, 0);
});

test("ComputerDriver suppresses UI events for hidden spaces and restores front on background external popups", async () => {
  let allowExternalFocus = false;
  const driver = new ComputerDriver({ shouldAllowExternalFocus: () => allowExternalFocus });
  const liveTargets = new Map([
    ["front-page", { targetId: "front-page", type: "page", url: "https://www.bilibili.com/", title: "Bilibili" }],
  ]);
  const activated = [];

  const fakeConn = new FakeCdpConnection(async (method, params) => {
    if (method === "Target.getTargets") return { targetInfos: [...liveTargets.values()] };
    if (method === "Browser.getWindowForTarget") return { windowId: 1 };
    if (method === "Target.setDiscoverTargets") return {};
    if (method === "Target.activateTarget") {
      activated.push(params.targetId);
      return {};
    }
    return {};
  });

  driver.browser.start = async () => {
    driver.browser.browser = fakeConn;
  };
  await driver.start();
  assert.equal(driver.browser.screen.front, "front-page");

  let targetEmits = 0;
  driver.on("targets", () => {
    targetEmits++;
  });

  // 1. Hidden background space create/destroy must emit zero targets events and never touch front.
  driver.browser.hiddenTargets.add("space-tab-1");
  fakeConn.emit("event", {
    method: "Target.targetCreated",
    params: { targetInfo: { targetId: "space-tab-1", type: "page", url: SPACE_SENTINEL_URL } },
  });
  fakeConn.emit("event", {
    method: "Target.targetDestroyed",
    params: { targetId: "space-tab-1" },
  });
  assert.equal(targetEmits, 0);
  assert.equal(driver.browser.hiddenTargets.has("space-tab-1"), false);
  assert.equal(activated.length, 0);

  // 2. External non-opener page (e.g. opencli --window background) restores user's front page.
  fakeConn.emit("event", {
    method: "Target.targetCreated",
    params: { targetInfo: { targetId: "opencli-popup", type: "page", url: "https://h5.waimai.meituan.com/" } },
  });
  await new Promise((r) => setTimeout(r, 200));
  assert.ok(activated.length >= 1);
  assert.ok(activated.every((id) => id === "front-page"));

  // 3. When foreground external focus is allowed, front is not forced back.
  activated.length = 0;
  allowExternalFocus = true;
  fakeConn.emit("event", {
    method: "Target.targetCreated",
    params: { targetInfo: { targetId: "opencli-fg", type: "page", url: "https://example.com/" } },
  });
  await new Promise((r) => setTimeout(r, 200));
  assert.equal(activated.length, 0);

  // 4. Child popup opened by a hidden background space inherits hidden state and never steals front.
  allowExternalFocus = false;
  driver.browser.hiddenTargets.add("bg-root");
  fakeConn.emit("event", {
    method: "Target.targetCreated",
    params: { targetInfo: { targetId: "bg-popup", type: "page", openerId: "bg-root", url: "https://example.com/pop" } },
  });
  await new Promise((r) => setTimeout(r, 200));
  assert.equal(driver.browser.screen.front, "front-page");
  assert.equal(driver.browser.hiddenTargets.has("bg-popup"), true);
  assert.equal(driver.browser.hiddenTargetRoots.get("bg-popup"), "bg-root");
  assert.ok(activated.every((id) => id === "front-page"));
});

test("BackgroundSpaces enforces timeout on hanging/sync-loop scripts, cleans popup subtree, and surfaces close failures", async () => {
  const browser = new BrowserBackend();
  let nextTarget = 1;
  const liveTargets = new Map();
  let failCloseFor = null;

  browser.browser = new FakeCdpConnection(async (method, params) => {
    if (method === "Target.getTargets") {
      return { targetInfos: [...liveTargets.values()] };
    }
    if (method === "Target.createTarget") {
      const targetId = `bg-${nextTarget++}`;
      liveTargets.set(targetId, { targetId, type: "page", url: params.url });
      return { targetId };
    }
    if (method === "Target.attachToTarget") {
      return { sessionId: `sess-${params.targetId}` };
    }
    if (method === "Target.closeTarget") {
      if (params.targetId === failCloseFor) {
        return { success: false };
      }
      liveTargets.delete(params.targetId);
      return { success: true };
    }
    return {};
  });

  const spaces = new BackgroundSpaces({ browser, maxConcurrency: 1 });

  // 1. Hanging Promise + synchronous infinite loop must be forcefully killed at timeoutMs and free slot.
  await assert.rejects(
    spaces.run({ script: "await new Promise(() => {});", timeoutMs: 1000 }),
    /aborted|Timeout/i,
  );
  assert.equal(spaces.activeCount, 0);
  assert.equal(liveTargets.size, 0);

  await assert.rejects(
    spaces.run({ script: "while (true) {}", timeoutMs: 1000 }),
    /aborted|Timeout/i,
  );
  assert.equal(spaces.activeCount, 0);
  assert.equal(liveTargets.size, 0);

  // 2. Child popup created during script run is closed on exit.
  const childRes = await spaces.run({
    script: "await page.sleep(20); return 'done';",
    timeoutMs: 2000,
  });
  assert.equal(childRes.result, "done");

  // Inject popup creation during next run
  const origSend = browser.browser.send.bind(browser.browser);
  browser.browser.send = async (method, params, sessionId, timeoutMs) => {
    const out = await origSend(method, params, sessionId, timeoutMs);
    if (method === "Runtime.enable") {
      const rootId = [...liveTargets.keys()].at(-1);
      liveTargets.set("bg-child-1", { targetId: "bg-child-1", type: "page", openerId: rootId, url: "https://example.com/pop" });
      browser.browser.emit("event", {
        method: "Target.targetCreated",
        params: { targetInfo: { targetId: "bg-child-1", type: "page", openerId: rootId, url: "https://example.com/pop" } },
      });
    }
    return out;
  };
  await spaces.run({ script: "return 'with-popup';", timeoutMs: 2000 });
  assert.equal(liveTargets.size, 0);
  browser.browser.send = origSend;

  // 3. Target.closeTarget failure throws instead of returning ok, and keeps target in hiddenTargets.
  failCloseFor = "bg-5";
  await assert.rejects(
    spaces.run({ script: "return 'ok';", timeoutMs: 2000 }),
    /success=false|cleanup/i,
  );
  assert.equal(spaces.activeCount, 0);
  assert.equal(browser.hiddenTargets.has("bg-5"), true);
  assert.deepEqual(await browser.listTabs(), []);

  // 4. Late Target.createTarget resolution after caller abort still gets cleaned up.
  failCloseFor = null;
  browser.hiddenTargets.clear();
  browser.hiddenTargetRoots.clear();
  liveTargets.clear();
  const baseSend = browser.browser.send.bind(browser.browser);
  browser.browser.send = async (method, params, sessionId, timeoutMs) => {
    if (method === "Target.createTarget") {
      await new Promise((r) => setTimeout(r, 80));
    }
    return await baseSend(method, params, sessionId, timeoutMs);
  };
  const ac = new AbortController();
  const latePromise = spaces.run({ script: "return 'never';", timeoutMs: 2000 }, ac.signal);
  setTimeout(() => ac.abort(new Error("aborted before createTarget reply")), 20);
  await assert.rejects(latePromise, /aborted before createTarget reply/);
  assert.equal(liveTargets.size, 0);
  assert.equal(browser.hiddenTargets.size, 0);
  assert.equal(spaces.activeCount, 0);

  // 5. Popup opened during slow closeTarget is drained in cleanup loop before release.
  let delayedFirstClose = false;
  browser.browser.send = async (method, params, sessionId, timeoutMs) => {
    if (method === "Target.closeTarget" && !delayedFirstClose) {
      delayedFirstClose = true;
      const rootId = params.targetId;
      await new Promise((r) => setTimeout(r, 40));
      liveTargets.set("bg-late-popup", { targetId: "bg-late-popup", type: "page", openerId: rootId, url: "https://example.com/late" });
      browser.browser.emit("event", {
        method: "Target.targetCreated",
        params: { targetInfo: { targetId: "bg-late-popup", type: "page", openerId: rootId, url: "https://example.com/late" } },
      });
    }
    return await baseSend(method, params, sessionId, timeoutMs);
  };
  await spaces.run({ script: "return 'ok';", timeoutMs: 2000 });
  assert.equal(liveTargets.size, 0);
  assert.equal(browser.hiddenTargets.size, 0);
  assert.equal(browser.hiddenTargetRoots.size, 0);

  // 6. Unbounded popup spawning past drain pass limit throws cleanup error instead of returning ok.
  let chainSeq = 0;
  browser.browser.send = async (method, params, sessionId, timeoutMs) => {
    if (method === "Target.closeTarget") {
      const closedId = params.targetId;
      const out = await baseSend(method, params, sessionId, timeoutMs);
      chainSeq++;
      const childId = `bg-chain-${chainSeq}`;
      liveTargets.set(childId, { targetId: childId, type: "page", openerId: closedId, url: "https://example.com/chain" });
      browser.browser.emit("event", {
        method: "Target.targetCreated",
        params: { targetInfo: { targetId: childId, type: "page", openerId: closedId, url: "https://example.com/chain" } },
      });
      return out;
    }
    return await baseSend(method, params, sessionId, timeoutMs);
  };
  await assert.rejects(
    spaces.run({ script: "return 'ok';", timeoutMs: 2000 }),
    /did not drain remaining targets/i,
  );
  assert.equal(spaces.activeCount, 0);
  assert.deepEqual(await browser.listTabs(), []);
  browser.browser.send = baseSend;
});

import test from "node:test";
import assert from "node:assert/strict";
import { ComputerDriver } from "../driver/runtime.mjs";

// 在一次性 Computer 容器中运行；页面只覆盖 Cua Basic 未涉及的生命周期边界。
test("source driver: browser APIs, tab ownership, cancellation and stale work", async () => {
  const driver = new ComputerDriver();
  await driver.start();
  let serial = 0;
  const run = (code, session = "owner", turn = "one", timeoutMs = 15000) =>
    driver.run({
      context: {
        session_id: session,
        turn_id: turn,
        call_id: String(++serial),
      },
      code,
      timeoutMs,
    });
  const text = (result) =>
    result.content
      .filter((x) => x.type === "text")
      .map((x) => x.text)
      .join("\n");
  try {
    await driver.cancel("before-admission");
    await assert.rejects(
      driver.run({
        context: {
          session_id: "never",
          turn_id: "one",
          call_id: "before-admission",
        },
        code: "throw new Error('must not run')",
      }),
      /cancelled before admission/,
    );
    assert.equal(driver.sessions.has("never"), false);
    const page =
      '<title>Driver boundary</title><input aria-label="Name"><select aria-label="Choice"><option value="a">Alpha</option><option value="b">Beta</option></select><button onclick="document.title=\'Clicked\'">Save</button>';
    const result = await run(
      `var tab = await browser.tabs.new(); await tab.goto(${JSON.stringify("data:text/html," + encodeURIComponent(page))}); nodeRepl.write(await tab.ax.get());`,
    );
    assert.match(text(result), /text field.*Name/);
    const fill = await run(
      'await tab.playwright.getByRole("textbox",{name:"Name"}).fill("花月"); await tab.playwright.getByRole("combobox").selectOption("b"); nodeRepl.write(await tab.playwright.domSnapshot());',
    );
    assert.match(text(fill), /花月/);
    assert.match(text(fill), /Beta/);
    const ax = await run(
      String.raw`var state = await tab.ax.get(); var field = Number(state.match(/(\d+) text field[^\n]*Name/)[1]); await tab.ax.setValue(field,"AX value"); nodeRepl.write(await tab.ax.get());`,
    );
    assert.match(text(ax), /AX value/);
    assert.match(
      text(
        await run(
          'await tab.clipboard.writeText("clip花月"); nodeRepl.write(await tab.clipboard.readText());',
        ),
      ),
      /clip花月/,
    );
    const both = await run('await tab.ax.write("both");');
    assert.match(text(both), /AX value/);
    assert.ok(
      both.content.some((x) => x.type === "image" && x.data.length > 100),
    );
    assert.match(
      text(await run("nodeRepl.write(await tab.dom_cua.get_visible_dom());")),
      /Save/,
    );
    const id = text(await run("nodeRepl.write(tab.id);")).trim();
    await assert.rejects(
      run(
        `await (await browser.tabs.get(${JSON.stringify(id)})).close();`,
        "other",
      ),
      /current Session/,
    );
    await run(
      "await tab.markDeliverable(); var scratch = await browser.tabs.new();",
    );
    const scratchId = text(await run("nodeRepl.write(scratch.id);")).trim();
    await driver.run({
      context: {
        session_id: "owner",
        turn_id: "one",
        call_id: String(++serial),
      },
      endTurn: true,
    });
    const tabs = await driver.browser.listTabs();
    assert.ok(tabs.some((t) => String(t.id) === id));
    assert.ok(!tabs.some((t) => String(t.id) === scratchId));
    await run(
      `var human = (await browser.user.openTabs()).find(t=>String(t.id)===${JSON.stringify(id)}); var claimed = await browser.user.claimTab(human); await claimed.close();`,
      "other",
      "two",
    );
    const started = performance.now();
    await assert.rejects(
      run("undefinedComputerVariable;", "errors"),
      /ReferenceError/,
    );
    assert.ok(
      performance.now() - started < 5000,
      "JS errors must not become timeout",
    );
    await run('nodeRepl.write("ready");', "cancel");
    await assert.rejects(run("while(true) {}", "cancel", "one", 500), /timed out/);
    assert.equal(driver.active, null);
    assert.equal(driver.closed, false);
    // 超时重置的是 JS 绑定，浏览器和截图照常可用。
    assert.ok(
      (
        await run(
          "var tab = await browser.tabs.new(); await nodeRepl.emitImage(await tab.screenshot({}));",
          "cancel",
        )
      ).content.some((x) => x.type === "image"),
    );
    // 不同 Session 并行；同一 Session 同时只有一个调用。
    const slow = run("await new Promise(r => setTimeout(r, 800)); nodeRepl.write('slow');", "parallel-a");
    assert.match(text(await run("nodeRepl.write('fast');", "parallel-b")), /fast/);
    await assert.rejects(run("1;", "parallel-a"), /already has a running/);
    assert.match(text(await slow), /slow/);
    // 匿名浏览器是同一 Chromium 里的隔离 Context：不带个人 profile 的 cookie，也不出现在主目录里。
    const cdp = (method, params) => driver.browser.browser.send(method, params);
    await cdp("Storage.setCookies", { cookies: [{ name: "who", value: "profile", domain: "example.test", path: "/" }] });
    const mainTabs = (await driver.browser.listTabs()).length;
    await run("var anon = await agent.browsers.create(); var page = await anon.tabs.new();", "anon");
    const [contextId] = driver.anonymous.contexts;
    assert.ok(contextId);
    assert.ok((await cdp("Storage.getCookies", {})).cookies.some((item) => item.name === "who"));
    assert.equal((await cdp("Storage.getCookies", { browserContextId: contextId })).cookies.length, 0);
    assert.equal((await driver.browser.listTabs()).length, mainTabs);
    assert.ok((await driver.targets()).some((target) => target.anonymous));
    await driver.run({ context: { session_id: "anon", turn_id: "one", call_id: String(++serial) }, endTurn: true });
    assert.equal(driver.anonymous.contexts.size, 0);
  } catch (error) {
    console.error(error);
    throw error;
  } finally {
    await driver.close();
  }
});

test("cancel releases direct and emulated CDP touch inputs", async () => {
  const driver = new ComputerDriver();
  await driver.start();
  const context = {
    session_id: "touch",
    turn_id: "one",
    call_id: "touch-direct",
  };
  const tab = await driver.browser.call("createTab", {}, context);
  const target = { tabId: tab.id };
  const cdp = (method, commandParams) =>
    driver.browser.execute({ target, method, commandParams }, context);
  try {
    // A reader must not become the owner of another Session's later input.
    const setup = await driver.browser.execute({ target, method: "Runtime.evaluate", commandParams: {
      expression: "document.body.style.height='2000px'; window.touchEvents=[]; for(const name of ['touchstart','touchend','touchcancel']) addEventListener(name,()=>touchEvents.push(name));",
    } }, { ...context, session_id: "observer" });
    assert.equal(setup.exceptionDetails, undefined, JSON.stringify(setup));
    await cdp("Emulation.setTouchEmulationEnabled", {
      enabled: true,
      maxTouchPoints: 1,
    });
    for (const method of [
      "Input.dispatchTouchEvent",
      "Input.emulateTouchFromMouseEvent",
    ]) {
      await cdp("Emulation.setEmitTouchEventsForMouse", {
        enabled: method === "Input.emulateTouchFromMouseEvent",
      });
      await cdp("Runtime.evaluate", {
        expression:
          "window.touchStarted=new Promise(resolve=>addEventListener('touchstart',resolve,{once:true})); window.touchReleased=new Promise(resolve=>{addEventListener('touchend',resolve,{once:true}); addEventListener('touchcancel',resolve,{once:true})});",
      });
      let started;
      const ready = new Promise((resolve) => {
        started = resolve;
      });
      const input =
        method === "Input.dispatchTouchEvent"
          ? { type: "touchStart", touchPoints: [{ x: 100, y: 100 }] }
          : { type: "mousePressed", x: 100, y: 100, button: "left" };
      const running = driver.run({
        context: { ...context, call_id: method },
        task: async (signal) => {
          await cdp(method, input);
          await cdp("Runtime.evaluate", {
            expression: "touchStarted.then(()=>true)",
            awaitPromise: true,
          });
          started();
          await new Promise((_, reject) =>
            signal.addEventListener("abort", () => reject(signal.reason), {
              once: true,
            }),
          );
        },
      });
      await Promise.race([ready, running]);
      const rejected = assert.rejects(running, /cancelled/);
      await driver.cancel(method);
      await rejected;
      const events = await cdp("Runtime.evaluate", {
        expression: "touchReleased.then(()=>touchEvents.splice(0))",
        awaitPromise: true,
        returnByValue: true,
      });
      assert.ok(
        events.result.value.includes("touchstart"),
        `${method}: ${JSON.stringify(events)}`,
      );
      assert.ok(
        events.result.value.some((x) =>
          ["touchcancel", "touchend"].includes(x),
        ),
        `${method}: ${JSON.stringify(events)}`,
      );
      assert.equal(driver.closed, false);
    }
    await driver.browser.endTurn(context);
  } finally {
    await driver.close();
  }
});

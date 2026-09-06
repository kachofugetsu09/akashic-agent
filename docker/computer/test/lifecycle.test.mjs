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
    const pid = driver.desktop.process.pid;
    await assert.rejects(
      run(
        "var handle = sky.drag_handle(); await handle.start({x:300,y:300}); while(true) {}",
        "cancel",
        "one",
        500,
      ),
      /timed out/,
    );
    assert.equal(driver.active, null);
    assert.equal(driver.closed, false);
    assert.notEqual(driver.desktop.process.pid, pid);
    assert.ok(
      (
        await run(
          "await nodeRepl.emitImage((await sky.get_screenshot())[0].bytes);",
          "cancel",
        )
      ).content.some((x) => x.type === "image"),
    );
    const calls = [];
    const nativeCall = driver.desktop.call.bind(driver.desktop);
    driver.desktop.call = (method, args) => {
      calls.push(method);
      return nativeCall(method, args);
    };
    await run(
      "setTimeout(()=>sky.move({x:400,y:400}).catch(()=>{}),100); void 0;",
      "late",
    );
    await run("await new Promise(resolve=>setTimeout(resolve,250));", "late");
    assert.ok(
      !calls.includes("move"),
      "a timer from a settled call must not gain the next call identity",
    );
  } catch (error) {
    console.error(error);
    throw error;
  } finally {
    await driver.close();
  }
});

test("unresponsive native release closes admission within its deadline", async () => {
  const driver = new ComputerDriver();
  await driver.start();
  const child = driver.desktop.process;
  child.kill("SIGSTOP");
  const started = performance.now();
  await assert.rejects(
    driver.run({
      context: { session_id: "stalled-native", turn_id: "one", call_id: "one" },
      code: "42;",
      timeoutMs: 15000,
    }),
    /release is uncertain/,
  );
  assert.ok(performance.now() - started < 11000);
  assert.equal(driver.closed, true);
  assert.equal(driver.active, null);
  await assert.rejects(
    driver.run({
      context: { session_id: "next", turn_id: "one", call_id: "two" },
      code: "42;",
    }),
    /stopped/,
  );
  await assert.rejects(driver.close(), /without a release receipt/);
  assert.equal(child.signalCode, "SIGKILL");
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
    const setup = await cdp("Runtime.evaluate", {
      expression:
        "document.body.style.height='2000px'; window.touchEvents=[]; for(const name of ['touchstart','touchend','touchcancel']) addEventListener(name,()=>touchEvents.push(name));",
    });
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

import { randomUUID } from "node:crypto";
import { Worker } from "node:worker_threads";
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { EventEmitter } from "node:events";
import { BrowserBackend } from "./cdp.mjs";
import { AnonymousBrowsers } from "./anonymous.mjs";

/** 一个容器只有一个带个人 profile 的浏览器；各 Session 隔离 JS 绑定并各自串行，Session 之间并行。 */
export class ComputerDriver extends EventEmitter {
  constructor({ anonymousIdleMs = 600000 } = {}) {
    super();
    this.anonymous = new AnonymousBrowsers(anonymousIdleMs);
    this.browser = new BrowserBackend(undefined, undefined, null, this.anonymous.contexts);
    this.anonymous.on("targets", () => this.emit("targets"));
    this.anonymous.on("closed", ({ browserId, reason, context }) => {
      this.emit("targets");
      this.sessions.get(context.session_id)?.worker.postMessage({ kind: "browserClosed", browserId, reason });
    });
    this.anonymous.on("cleanupError", error => {
      this.closed = true;
      this.rejectAll(error);
      console.error("Anonymous browser cleanup failed:", error.message);
    });
  }
  sessions = new Map();
  // session_id → 正在执行的调用；同一 Session 一次一个。
  actives = new Map();
  closed = false;
  cancelledCalls = new Map();
  control = null;
  controlVersion = 0;
  agentJobs = new Set();

  /** 人工接管先阻止新操作，再等待已发操作和输入释放。 */
  async takeControl(id, target = "", signal) {
    if (this.closed) throw new Error("Computer driver is stopped");
    if (this.control) throw new Error("Another viewer controls this Computer");
    const control = { id, target, ready: false, jobs: new Set() };
    control.done = new Promise((resolve) => { control.resume = resolve; });
    control.instance = this.anonymous.instances.get(target.split(":")[0]);
    if (control.instance) this.anonymous.hold(control.instance);
    this.control = control;
    this.controlVersion++;
    this.emit("control");
    this.pauseClock();
    try {
      await waitForSignal(Promise.allSettled([...this.agentJobs]), signal);
      if (this.control !== control || control.release) throw new Error("Control request was cancelled");
      control.preparing = (async () => {
        await this.browser.releaseInputs();
        for (const active of this.actives.values()) await this.anonymous.releaseInputs(active.context);
      })();
      await waitForSignal(control.preparing, signal);
      if (this.control !== control || control.release) throw new Error("Control request was cancelled");
      control.ready = true;
    } catch (error) {
      await this.releaseControl(id);
      throw error;
    }
  }

  /** 先禁止新输入，排空人工已发操作，再恢复 Agent；并发释放共用同一结果。 */
  async releaseControl(id) {
    const control = this.control;
    if (control?.id !== id) return;
    if (control.release) return control.release;
    const hadInput = control.ready || control.preparing;
    control.ready = false;
    control.release = (async () => {
      if (hadInput) {
        await control.preparing;
        await Promise.allSettled([...control.jobs]);
        await this.browser.releaseInputs();
        for (const instance of this.anonymous.instances.values())
          if (instance.backend && !instance.closing) await instance.backend.releaseInputs();
      }
      if (this.control !== control) throw new Error("Computer control owner changed during release");
      this.control = null;
      this.emit("control");
      if (control.instance) this.anonymous.unhold(control.instance);
      control.releaseWorkload?.();
      control.resume();
      this.resumeClock();
    })();
    try { await control.release; }
    catch (error) { this.closed = true; this.rejectAll(error); throw error; }
  }

  async waitForControl(signal) {
    while (this.control) {
      signal?.throwIfAborted();
      const control = this.control;
      await waitForSignal(control.done, signal);
    }
    signal?.throwIfAborted();
  }

  /** 通用 CDP 可能执行脚本，所有 Agent RPC 都经过同一个权限边界。 */
  async agentOperation(task, signal) {
    do { await this.waitForControl(signal); } while (this.control);
    const work = task();
    this.agentJobs.add(work);
    // OpenCLI 刷新等不经 tracked 的操作也要让面板知道 Agent 在用。
    this.emit("activity");
    try { return await work; } finally { this.agentJobs.delete(work); this.emit("activity"); }
  }

  // 测试与诊断读取的单一视图；并发时返回任意一个正在执行的调用。
  get active() {
    return this.actives.values().next().value ?? null;
  }

  findActive(callId) {
    return [...this.actives.values()].find(active => active.context.call_id === callId) ?? null;
  }

  rejectAll(error) {
    for (const active of this.actives.values()) active.reject(error);
  }

  // 人工接管期间暂停全部调用的计时，交还后从剩余时间继续。
  pauseClock() {
    for (const active of this.actives.values()) {
      if (!active.timer) continue;
      clearTimeout(active.timer);
      active.remaining = Math.max(1, active.remaining - (Date.now() - active.startedAt));
      active.timer = null;
    }
  }

  resumeClock(only) {
    for (const active of only ? [only] : this.actives.values()) {
      if (this.control || active.timer || active.cancelled) continue;
      active.startedAt = Date.now();
      active.timer = setTimeout(() => active.reject(new Error("Computer call timed out")), active.remaining);
    }
  }
  /** 目录直接来自存活页面：主浏览器的标签在前，匿名 Context 的标签在后；front 是屏幕最前的那个。 */
  async targets() {
    const groups = [{ id: "main", backend: this.browser, instance: null }];
    for (const instance of this.anonymous.instances.values())
      if (instance.backend && !instance.closing) groups.push({ id: instance.id, backend: instance.backend, instance });
    const targets = [];
    for (const { id, backend, instance } of groups)
      for (const tab of await backend.listTabs()) {
        const { targetId, created, status } = backend.tab(tab.id);
        targets.push({ id: `${id}:${tab.id}`, title: tab.title, url: tab.url,
          anonymous: Boolean(instance), agent: Boolean(created), kept: Boolean(status),
          front: targetId === this.browser.screen.front,
          sessionId: instance?.context.session_id ?? created?.session_id ?? null,
          turn: instance || created ? JSON.stringify([
            (instance?.context ?? created).session_id, (instance?.context ?? created).turn_id]) : null });
      }
    return targets;
  }

  async viewTarget(id) {
    if (typeof id !== "string" || id.length > 128) throw new TypeError("Invalid view target");
    const [group, tabText] = id.split(":");
    const instance = group === "main" ? null : this.anonymous.instances.get(group);
    const backend = group === "main" ? this.browser : instance?.backend;
    if (!backend || instance?.closing) throw new Error("Browser target has closed");
    const tabId = Number(tabText);
    const tabs = await backend.listTabs();
    if (!tabs.some(tab => tab.id === tabId)) throw new Error("Browser page has closed");
    return { backend, tabId, targetId: backend.tab(tabId).targetId };
  }

  /** 把一个标签提到屏幕最前；所有观看者看到的是同一块屏幕。 */
  async show(id) {
    const { targetId } = await this.viewTarget(id);
    await this.browser.screen.show(this.browser.browser, targetId);
    // 接管期间切到另一个匿名浏览器时，由接管持有它，人工操作不被闲置回收打断。
    const control = this.control, instance = this.anonymous.instances.get(id.split(":")[0]);
    if (control && control.instance !== instance) {
      if (control.instance) this.anonymous.unhold(control.instance);
      control.instance = instance;
      if (instance) this.anonymous.hold(instance);
    }
    this.emit("targets");
  }

  /** 人工导航：在标签里打开地址、后退、前进或刷新；临时附着，用完就断开。 */
  async navigate(id, action, url = "") {
    const { targetId } = await this.viewTarget(id);
    const connection = this.browser.browser;
    const { sessionId } = await connection.send("Target.attachToTarget", { targetId, flatten: true });
    try {
      if (action === "go") await connection.send("Page.navigate", { url }, sessionId);
      else if (action === "reload") await connection.send("Page.reload", {}, sessionId);
      else {
        const { currentIndex, entries } = await connection.send("Page.getNavigationHistory", {}, sessionId);
        const entry = entries[currentIndex + (action === "back" ? -1 : 1)];
        if (entry) await connection.send("Page.navigateToHistoryEntry", { entryId: entry.id }, sessionId);
      }
    } finally {
      await connection.send("Target.detachFromTarget", { sessionId }).catch(() => {});
    }
  }

  /** 人工新开标签：个人浏览器里开一个窗口并提到最前，返回它的观看 id。 */
  async openPage(url = "about:blank") {
    const connection = this.browser.browser;
    const { targetId } = await connection.send("Target.createTarget", { url, newWindow: true });
    await this.browser.screen.place(connection, targetId);
    await this.browser.screen.show(connection, targetId);
    const tab = (await this.browser.listTabs()).find(item => this.browser.tab(item.id).targetId === targetId);
    this.emit("targets");
    return tab ? `main:${tab.id}` : "";
  }

  async closePage(id) {
    const { targetId } = await this.viewTarget(id);
    await this.browser.browser.send("Target.closeTarget", { targetId });
  }

  // 最前的标签被关掉时，把主浏览器的第一个标签提上来；一个都不剩就补开空白页，屏幕不会停在不可知的窗口。
  async recoverFront(targetId) {
    const screen = this.browser.screen;
    if (screen.front !== targetId) return;
    screen.front = null;
    try {
      const connection = this.browser.browser;
      const [first] = await this.browser.listTabs();
      const next = first ? this.browser.tab(first.id).targetId
        : (await connection.send("Target.createTarget", { url: "about:blank", newWindow: true })).targetId;
      if (!first) await screen.place(connection, next);
      await screen.show(connection, next);
    } catch (error) {
      console.error("Browser front recovery failed:", error.message);
    }
    this.emit("targets");
  }

  async start() {
    this.browser.on("cursor", (state) => this.emit("cursor", state));
    await this.browser.start();
    this.anonymous.main = this.browser;
    // 1. 页面增删、换标题都要刷新目录；最前的页面关闭时补一个。
    this.browser.browser.on("event", ({ method, params }) => {
      if (!method?.startsWith("Target.target")) return;
      if (method === "Target.targetDestroyed") void this.recoverFront(params.targetId);
      if (method === "Target.targetCreated" && params.targetInfo.type === "page" && params.targetInfo.openerId)
        this.browser.screen.front = params.targetInfo.targetId;
      this.emit("targets");
    });
    await this.browser.browser.send("Target.setDiscoverTargets", { discover: true });
    // 2. 启动时屏幕上只有个人 profile 的首个窗口。
    const [first] = await this.browser.listTabs();
    if (first) this.browser.screen.front = this.browser.tab(first.id).targetId;
    this.browser.on("event", (event) => {
      for (const session of this.sessions.values())
        session.worker.postMessage({ kind: "event", event });
    });
    this.browser.on("disconnected", () => {
      this.closed = true;
      this.rejectAll(
        new Error("Chromium disconnected; restart the Computer workload"),
      );
    });
  }
  async session(id) {
    if (this.sessions.has(id)) return this.sessions.get(id);
    if (this.sessions.size >= 16)
      throw new Error(
        "Computer has 16 live JS sessions; reset an unused session first",
      );
    const directory = await mkdtemp(join(tmpdir(), "akashic-computer-driver-"));
    const pipePath = join(directory, "cdp");
    await writeFile(pipePath, "", { mode: 0o600 });
    const worker = new Worker(new URL("./worker.mjs", import.meta.url), {
      workerData: { directory, pipePath },
    });
    const session = { worker, directory, id };
    this.sessions.set(id, session);
    worker.on("message", (message) => this.message(session, message));
    worker.on("error", (error) => {
      const active = this.actives.get(id);
      if (active?.session === session) active.reject(error);
    });
    worker.on("exit", (code) => {
      const active = this.actives.get(id);
      if (active?.session === session)
        active.reject(new Error(`JS worker exited: ${code}`));
    });
    return session;
  }
  message(session, message) {
    const active = this.actives.get(session.id);
    if (message.kind === "ready" || message.kind === "metadata") return;
    if (message.kind === "result") {
      if (active?.session === session) {
        if (message.error) {
          active.scriptError = message.scriptError === true;
          active.reject(
            Object.assign(new Error(message.error), {
              content: message.content,
            }),
          );
        } else active.resolve(message.content);
      }
      return;
    }
    if (message.kind !== "browser")
      throw new Error(`Invalid worker message: ${message.kind}`);
    if (
      !active ||
      active.cancelled ||
      active.session !== session ||
      active.context.call_id !== message.callId
    ) {
      session.worker.postMessage({
        kind: "reply",
        id: message.id,
        error: "This Computer call has ended",
      });
      return;
    }
    const work = this.agentOperation(() =>
      this.browserCall(session, message, active.context),
      active.controller.signal);
    active.pending.add(work);
    work
      .then(
        (result) =>
          session.worker.postMessage({ kind: "reply", id: message.id, result }),
        (error) =>
          session.worker.postMessage({
            kind: "reply",
            id: message.id,
            error: error.message,
          }),
      )
      .finally(() => active.pending.delete(work));
  }

  async browserCall(session, message, context) {
    if (message.method === "createBrowser") {
      let id;
      id = await this.anonymous.create(context, (event) =>
        session.worker.postMessage({ kind: "event", event, browserId: id }),
      );
      return id;
    }
    const { browserId, params } = message.params;
    if (browserId === "primary")
      return this.browser.call(message.method, params, context);
    return this.anonymous.call(browserId, message.method, params, context);
  }
  /** 调用结束先 drain 和释放输入；完整结算的脚本错误保留 Session。 */
  async run(
    { context, code = "", endTurn = false, timeoutMs = 60_000, task },
    signal,
  ) {
    if (endTurn) {
      while (this.actives.get(context?.session_id))
        await deadline(this.actives.get(context.session_id).done, 145000, "Wait for active call");
      signal?.throwIfAborted();
      if (!this.sessions.has(context?.session_id)) task = async () => [];
    }
    if (this.closed) throw new Error("Computer driver is stopped");
    if (this.actives.has(context?.session_id))
      throw new Error("This Session already has a running Computer call; retry after it settles");
    if (
      !context ||
      !["session_id", "turn_id", "call_id"].every(
        (key) =>
          typeof context[key] === "string" &&
          context[key].length > 0 &&
          context[key].length <= 256,
      )
    ) {
      throw new TypeError(
        "Computer call requires session_id, turn_id and call_id",
      );
    }
    if (typeof code !== "string" || Buffer.byteLength(code) > 128 * 1024)
      throw new TypeError("Computer code exceeds 128 KiB");
    if (!Number.isInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 110_000)
      throw new TypeError("timeoutMs must be 1..110000");
    if (this.cancelledCalls.has(context.call_id)) {
      this.cancelledCalls.delete(context.call_id);
      throw new Error(
        "Computer call cancelled before admission; no actions were sent",
      );
    }
    signal?.throwIfAborted();
    const active = {
      context,
      pending: new Set(),
      cancelled: false,
      controller: new AbortController(),
      remaining: timeoutMs,
      timer: null,
    };
    this.actives.set(context.session_id, active);
    active.done = new Promise((resolve) => {
      active.finish = resolve;
    });
    const result = new Promise((resolve, reject) =>
      Object.assign(active, { resolve, reject }),
    );
    // 接管等待期间取消也会拒绝 result，立即安装观察者，避免未处理拒绝。
    void result.catch(() => {});
    const abort = () => {
      const error = signal.reason ?? new Error("Computer caller disconnected");
      active.controller.abort(error);
      active.reject(error);
    };
    let content = [],
      failure, reset = false;
    try {
      if (!task) active.session = await this.session(context.session_id);
      signal?.addEventListener("abort", abort, { once: true });
      if (signal?.aborted) abort();
      this.resumeClock(active);
      await this.waitForControl(active.controller.signal);
      if (task) {
        const work = this.agentOperation(() => task(active.controller.signal, context), active.controller.signal);
        active.pending.add(work);
        work
          .then(active.resolve, active.reject)
          .finally(() => active.pending.delete(work));
      } else {
        active.session.worker.postMessage({
          kind: "run",
          context,
          code,
          endTurn,
        });
      }
      content = await result;
      await this.waitForControl(active.controller.signal);
    } catch (error) {
      failure = error;
      reset = !active.scriptError || endTurn || active.controller.signal.aborted;
      active.cancelled = true;
      active.controller.abort(error);
      if (reset && active.session) await active.session.worker.terminate();
    } finally {
      clearTimeout(active.timer);
      signal?.removeEventListener("abort", abort);
      // 浏览器已送出的 CDP 命令有自己的有界超时；结算后松开仍按住的键鼠。
      try {
        await Promise.allSettled([...active.pending]);
        await deadline(this.browser.releaseInputs(context.session_id), 11000, "Browser release");
        await deadline(this.anonymous.releaseInputs(context), 11000, "Anonymous input release");
        if (endTurn)
          await deadline(this.browser.endTurn(context), 11000, "Turn cleanup");
        if (endTurn || reset)
          await this.anonymous.cleanup(
            (instance) =>
              instance.context.session_id === context.session_id &&
              (reset || instance.context.turn_id === context.turn_id),
          );
      } catch (error) {
        this.closed = true;
        reset = true;
        this.browser.close();
        failure = new AggregateError(
          failure ? [failure, error] : [error],
          "Computer input release is uncertain; restart the workload",
        );
      }
      if (reset && active.session) {
        await active.session.worker.terminate();
        this.sessions.delete(context.session_id);
        await rm(active.session.directory, { recursive: true, force: true });
      }
      this.actives.delete(context.session_id);
      active.finish();
    }
    if (failure) {
      this.emit("cursor", { point: null, error: "操作中断，位置标记已清除" });
      failure.message +=
        reset ? "; earlier effects may remain; JS bindings for this session were reset"
          : "; earlier effects may remain; JS bindings and browser pages were kept; inspect the current page before retrying";
      throw failure;
    }
    return { content, call_id: context.call_id };
  }
  async perform(task, signal) {
    const callId = randomUUID();
    const result = await this.run(
      {
        context: {
          session_id: "legacy-page",
          turn_id: callId,
          call_id: callId,
        },
        timeoutMs: 30000,
        task,
      },
      signal,
    );
    return result.content;
  }
  async cancel(callId) {
    if (this.closed)
      throw new Error("Input release is uncertain; driver is stopped");
    if (typeof callId !== "string" || !callId.length || callId.length > 256)
      throw new TypeError("cancel requires a valid call_id");
    const active = this.findActive(callId);
    if (!active) {
      const now = Date.now();
      for (const [id, until] of this.cancelledCalls)
        if (until < now) this.cancelledCalls.delete(id);
      if (this.cancelledCalls.size >= 4096)
        throw new Error(
          "Too many pending cancellations; release was not confirmed",
        );
      this.cancelledCalls.set(callId, now + 300000);
      return;
    }
    active.reject(new Error("Computer call cancelled"));
    await active.done;
    if (this.closed)
      throw new Error("Input release is uncertain; restart the workload");
  }
  async reset(sessionId) {
    if (this.actives.has(sessionId))
      throw new Error("Cannot reset while this Session has a running Computer call");
    await this.anonymous.cleanup(
      (instance) => instance.context.session_id === sessionId,
    );
    const session = this.sessions.get(sessionId);
    if (!session) return;
    await session.worker.terminate();
    this.sessions.delete(sessionId);
    await rm(session.directory, { recursive: true, force: true });
  }
  /** 0115：close 失败后由重启调用。只结束本 driver 创建的 worker 和匿名进程，不碰主 profile。 */
  async abandon() {
    this.closed = true;
    this.rejectAll(new Error("Computer is restarting"));
    const control = this.control;
    this.control = null;
    this.emit("control");
    // 占用只归还一次：仍在等待的旧 releaseControl 稍后结束时不能再归还。
    const releaseWorkload = control?.releaseWorkload;
    if (control) control.releaseWorkload = null;
    releaseWorkload?.();
    control?.resume?.();
    for (const session of this.sessions.values()) await session.worker.terminate();
    await this.anonymous.abandon();
    this.browser.close();
    for (const session of this.sessions.values())
      await rm(session.directory, { recursive: true, force: true });
    this.sessions.clear();
  }

  async close() {
    this.closed = true;
    const actives = [...this.actives.values()];
    this.rejectAll(new Error("Computer driver is stopping"));
    if (this.control) await this.releaseControl(this.control.id);
    await Promise.all(actives.map(active => active.done));
    for (const session of this.sessions.values())
      await session.worker.terminate();
    await this.anonymous.cleanup(() => true);
    await deadline(this.browser.releaseInputs(), 11000, "Browser release");
    await this.browser.browser.send("Browser.close");
    this.browser.close();
    for (const session of this.sessions.values())
      await rm(session.directory, { recursive: true, force: true });
    this.sessions.clear();
  }
}

/** 清理也有截止时间，超时后关闭 admission，不能把不确定状态报成释放成功。 */
async function deadline(work, ms, label) {
  let timer;
  try {
    return await Promise.race([
      work,
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(`${label} timed out`)), ms);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

/** 等待期间上游取消立即退出；已送出的操作仍由原 owner 结算。 */
async function waitForSignal(work, signal) {
  signal?.throwIfAborted();
  let abort;
  try {
    return await Promise.race([work, new Promise((_, reject) => {
      abort = () => reject(signal.reason);
      signal?.addEventListener("abort", abort, { once: true });
    })]);
  } finally { if (abort) signal?.removeEventListener("abort", abort); }
}

import { EventEmitter } from "node:events";
import { randomUUID } from "node:crypto";
import { BrowserBackend } from "./cdp.mjs";

/** 匿名浏览器是主 Chromium 里的隔离 BrowserContext：不带个人 profile 的 cookie 与存储，也不另起进程。 */
export class AnonymousBrowsers extends EventEmitter {
  constructor(idleMs = 600000) {
    super();
    this.idleMs = idleMs;
  }
  instances = new Map();
  // 主浏览器的目录据此排除匿名标签。
  contexts = new Set();
  // 主后端：借它的浏览器级连接创建、销毁 Context，并共用同一块屏幕。
  main = null;

  async create(context, onEvent) {
    if (this.instances.size >= 8)
      throw new Error("Computer has 8 anonymous browsers; close one first");
    // 1. 先保留名额，并发创建也不能超过上限。
    const id = randomUUID();
    const instance = { id, context, backend: null, contextId: null, closing: false, busy: 0, idleTimer: null };
    this.instances.set(id, instance);
    try {
      // 2. 在主 Chromium 里建空的隔离 Context；扩展默认不进入这类 Context。
      const { browserContextId } = await this.main.browser.send("Target.createBrowserContext", { disposeOnDetach: false });
      instance.contextId = browserContextId;
      this.contexts.add(browserContextId);
      const backend = new BrowserBackend(undefined, `Anonymous ${id}`, browserContextId, undefined, this.main.screen);
      instance.backend = backend;
      await backend.start();
      this.emit("targets");
      backend.on("event", onEvent);
      this.armIdle(instance);
      return id;
    } catch (error) {
      try {
        await this.close(id, context);
      } catch (cleanupError) {
        throw new AggregateError([error, cleanupError],
          `Anonymous browser failed: ${error.message}; cleanup failed: ${cleanupError.message}`);
      }
      throw error;
    }
  }

  get(id, context, allowClosing = false) {
    const instance = this.instances.get(id);
    if (!instance || (instance.closing && !allowClosing))
      throw new Error("Anonymous browser is closed or unknown");
    if (
      instance.context.session_id !== context.session_id ||
      instance.context.turn_id !== context.turn_id
    )
      throw new Error("Anonymous browser belongs to another Turn");
    return instance;
  }

  async call(id, method, params, context) {
    if (method === "closeBrowser") return this.close(id, context);
    const instance = this.get(id, context);
    this.hold(instance);
    try { return await instance.backend.call(method, params, context); }
    finally { this.unhold(instance); }
  }

  async releaseInputs(context) {
    for (const instance of this.instances.values())
      if (instance.context.session_id === context.session_id && instance.backend && !instance.closing)
        await instance.backend.releaseInputs();
  }

  /** 输入或人工接管持有实例；观看画面和目录不刷新闲置期限。 */
  hold(instance) {
    clearTimeout(instance.idleTimer);
    instance.busy++;
  }

  unhold(instance) {
    instance.busy--;
    this.armIdle(instance);
  }

  armIdle(instance) {
    clearTimeout(instance.idleTimer);
    if (instance.closing || instance.busy) return;
    instance.idleTimer = setTimeout(() => {
      this.close(instance.id, instance.context, "idle").catch(error => this.emit("cleanupError", error));
    }, this.idleMs);
    instance.idleTimer.unref();
  }

  /** 关闭一个 Context 连同它的标签；不影响主 profile 和其他 Context。 */
  async close(id, context, reason = "closed") {
    const instance = this.get(id, context, true);
    if (instance.closing) return instance.done;
    instance.closing = true;
    clearTimeout(instance.idleTimer);
    this.emit("closed", { browserId: id, reason, context: instance.context });
    instance.done = (async () => {
      try {
        if (instance.backend?.browser) await instance.backend.releaseInputs();
      } finally {
        instance.backend?.close();
        try {
          if (instance.contextId && this.main?.browser)
            await this.main.browser.send("Target.disposeBrowserContext", { browserContextId: instance.contextId });
        } finally {
          this.contexts.delete(instance.contextId);
          this.instances.delete(id);
          this.emit("targets");
        }
      }
    })();
    return instance.done;
  }

  /** 正常关闭失败后的强制收尾：通知所属 Session，丢掉连接；Context 随浏览器重启消失。 */
  async abandon() {
    for (const instance of this.instances.values()) {
      clearTimeout(instance.idleTimer);
      instance.closing = true;
      instance.backend?.close();
      this.emit("closed", { browserId: instance.id, reason: "restart", context: instance.context });
    }
    this.instances.clear();
    this.contexts.clear();
  }

  async cleanup(match) {
    for (const instance of [...this.instances.values()].filter(match))
      await this.close(instance.id, instance.context);
  }
}

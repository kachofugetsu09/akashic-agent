import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { BrowserBackend } from "./cdp.mjs";

/** 匿名浏览器共享独立 headless 进程，各 Context 隔离身份与页面存储。 */
export class AnonymousBrowsers {
  instances = new Map();
  engine = null;
  starting = null;
  stopping = null;

  async create(context, onEvent) {
    if (this.instances.size >= 8)
      throw new Error("Computer has 8 anonymous browsers; close one first");
    // 1. 先保留名额，并发创建也不能超过上限或重复启动 headless 进程。
    const id = randomUUID();
    const instance = {
      id,
      context,
      backend: null,
      contextId: null,
      closing: false,
    };
    this.instances.set(id, instance);
    try {
      if (this.stopping) await this.stopping;
      if (!this.engine)
        this.starting ??= this.startEngine().finally(() => {
          this.starting = null;
        });
      if (this.starting) await this.starting;
      const engine = this.engine;
      if (this.exited()) throw new Error("Anonymous Chromium exited; close its browsers before recreating");
      // 2. 创建空的隐身 Context，不读取或复制个人 profile 和扩展。
      const { browserContextId } = await engine.backend.browser.send(
        "Target.createBrowserContext",
      );
      instance.contextId = browserContextId;
      const backend = new BrowserBackend(
        engine.url,
        `Anonymous ${id}`,
        browserContextId,
      );
      instance.backend = backend;
      await backend.start();
      backend.on("event", onEvent);
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

  /** 临时数据目录只属于这份 headless 进程；主 profile 始终由主 owner 管理。 */
  async startEngine() {
    const directory = await mkdtemp(join(tmpdir(), "akashic-anonymous-"));
    const child = spawn(
      "chromium",
      [
        "--headless=new",
        `--user-data-dir=${directory}/profile`,
        "--remote-debugging-address=127.0.0.1",
        "--remote-debugging-port=0",
        "--disable-setuid-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--disable-quic",
        "--no-first-run",
        "--no-default-browser-check",
        "about:blank",
      ],
      {
        detached: true,
        stdio: ["ignore", "ignore", "pipe"],
        env: {
          ...process.env,
          HOME: directory,
          XDG_CONFIG_HOME: `${directory}/config`,
          XDG_CACHE_HOME: `${directory}/cache`,
        },
      },
    );
    const engine = { directory, child, backend: null, stderr: "" };
    engine.exited = new Promise((resolve) => child.once("exit", resolve));
    child.stderr.on("data", (chunk) => {
      engine.stderr = (engine.stderr + chunk).slice(-4096);
    });
    this.engine = engine;
    try {
      await new Promise((resolve, reject) => {
        child.once("spawn", resolve);
        child.once("error", reject);
      });
      const deadline = Date.now() + 10000;
      let port;
      while (Date.now() < deadline) {
        if (this.exited())
          throw new Error(`Anonymous Chromium exited: ${engine.stderr}`);
        try {
          port = Number(
            (
              await readFile(`${directory}/profile/DevToolsActivePort`, "utf8")
            ).split("\n")[0],
          );
          break;
        } catch (error) {
          if (error.code !== "ENOENT") throw error;
        }
        await sleep(50);
      }
      if (!Number.isInteger(port) || port < 1 || port > 65535)
        throw new Error("Anonymous Chromium did not publish a CDP port");
      engine.url = `http://127.0.0.1:${port}`;
      const backend = new BrowserBackend(engine.url);
      await backend.start();
      engine.backend = backend;
    } catch (error) {
      await this.stopEngine();
      throw error;
    }
  }

  exited() {
    return (
      this.engine.child.exitCode !== null ||
      this.engine.child.signalCode !== null
    );
  }

  get(id, context) {
    const instance = this.instances.get(id);
    if (!instance || instance.closing)
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
    if (this.exited())
      throw new Error("Anonymous Chromium exited; close its browsers before recreating");
    return instance.backend.call(method, params, context);
  }

  async releaseInputs(context) {
    if (!this.engine || this.exited()) return;
    for (const instance of this.instances.values())
      if (instance.context.session_id === context.session_id)
        await instance.backend.releaseInputs();
  }

  /** 关闭 Context 不影响其他 Context；最后一个关闭时释放 headless 进程。 */
  async close(id, context) {
    const instance = this.get(id, context);
    instance.closing = true;
    try {
      if (instance.backend && !this.exited())
        await instance.backend.releaseInputs();
    } finally {
      try {
        instance.backend?.close();
      } finally {
        try {
          if (instance.contextId && !this.exited())
            await this.engine.backend.browser.send(
              "Target.disposeBrowserContext",
              { browserContextId: instance.contextId },
            );
        } finally {
          this.instances.delete(id);
          if (!this.instances.size && this.engine) {
            this.stopping = this.stopEngine().finally(() => { this.stopping = null; });
            await this.stopping;
          }
        }
      }
    }
  }

  async stopEngine() {
    const engine = this.engine;
    engine.backend?.close();
    const child = engine.child;
    if (child.pid && !this.exited()) {
      process.kill(-child.pid, "SIGTERM");
      const exited = await Promise.race([
        engine.exited.then(() => true),
        sleep(5000).then(() => false),
      ]);
      if (!exited) {
        process.kill(-child.pid, "SIGKILL");
        await engine.exited;
      }
    }
    await rm(engine.directory, { recursive: true, force: true });
    this.engine = null;
  }

  async cleanup(match) {
    // 顺序关闭，最后一个 Context 独占 headless 的停机与目录删除。
    for (const instance of [...this.instances.values()].filter(match))
      await this.close(instance.id, instance.context);
  }
}

import { spawn } from "node:child_process";
import { setTimeout as sleep } from "node:timers/promises";
import { performance } from "node:perf_hooks";

/** 主 profile 只有这个 owner 能启动；请求与 Turn 阻止空闲回收。 */
export class ComputerLifecycle {
  state = "sleeping";
  error = "";
  users = 0;
  turns = new Set();
  lastUsed = performance.now();
  transition = null;
  child = null;

  constructor({ start, stop, idleMs }) {
    this.startDriver = start;
    this.stopDriver = stop;
    this.idleMs = idleMs;
    this.timer = setInterval(
      () => {
        if (
          this.state === "ready" &&
          !this.users &&
          !this.turns.size &&
          performance.now() - this.lastUsed >= this.idleMs
        ) {
          void this.stop().catch((error) =>
            console.error("Computer idle stop:", error),
          );
        }
      },
      Math.min(idleMs, 1000),
    );
  }

  status() {
    return {
      state: this.state,
      error: this.error,
      users: this.users,
      turns: this.turns.size,
      idleMs: this.idleMs,
    };
  }

  touch() {
    if (this.state === "ready") this.lastUsed = performance.now();
  }

  async use(task) {
    const release = await this.acquire();
    try { return await task(); }
    finally { release(); }
  }

  async acquire() {
    this.users++;
    try {
      await this.wake();
    } catch (error) {
      this.users--;
      throw error;
    }
    return () => {
      this.users--;
      this.touch();
    };
  }

  async wake() {
    if (this.transition) await this.transition;
    if (this.state === "ready") return;
    if (this.state === "failed") throw new Error(this.error);
    this.state = "starting";
    this.transition = this.start().finally(() => {
      this.transition = null;
    });
    await this.transition;
  }

  /** 先确认真正就绪；启动失败也必须结束旧进程才能再次使用 profile。 */
  async start() {
    const child = spawn(
      "/opt/computer/start.sh",
      ["--desktop-session", "--runtime"],
      {
        detached: true,
        stdio: "inherit",
      },
    );
    this.child = child;
    this.exited = new Promise((resolve) =>
      child.once("exit", (code, signal) => {
        if (this.state === "ready") {
          this.state = "failed";
          this.error = `Computer runtime exited (${code ?? signal})`;
        }
        resolve();
      }),
    );
    const spawned = new Promise((resolve, reject) => {
      child.once("spawn", resolve);
      child.once("error", reject);
    });
    try {
      await spawned;
      await this.startDriver(
        () => child.exitCode !== null || child.signalCode !== null,
      );
      this.state = "ready";
      this.touch();
    } catch (error) {
      this.state = "stopping";
      try {
        await this.stopRuntime();
      } finally {
        this.state = "failed";
        this.error = `Computer startup failed: ${error.message}`;
      }
      throw error;
    }
  }

  /** Driver 先释放输入，再等 Chromium 正常退出以落盘登录状态。 */
  async stop() {
    if (this.transition) await this.transition;
    if (!this.child) return;
    this.state = "stopping";
    this.transition = (async () => {
      try {
        await this.stopRuntime();
        this.state = "sleeping";
        this.error = "";
      } catch (error) {
        this.state = "failed";
        this.error = `Computer stop failed: ${error.message}`;
        throw error;
      }
    })().finally(() => {
      this.transition = null;
    });
    await this.transition;
  }

  async stopRuntime() {
    try {
      await this.stopDriver();
    } finally {
      if (this.child?.pid) await this.stopProcess();
    }
  }

  async stopProcess() {
    const child = this.child;
    if (child.exitCode === null && child.signalCode === null) {
      // Driver 已发送 Browser.close；等待浏览器落盘，再由 shell 关闭桌面。
      const exited = await Promise.race([
        this.exited.then(() => true),
        sleep(15000).then(() => false),
      ]);
      if (!exited) {
        process.kill(-child.pid, "SIGKILL");
        await this.exited;
        this.child = null;
        throw new Error(
          "Computer runtime required SIGKILL; profile flush was not confirmed",
        );
      }
    }
    this.child = null;
  }

  async close() {
    clearInterval(this.timer);
    await this.stop();
  }
}

export function duration(name, defaultMs) {
  const value = Number(process.env[name] ?? defaultMs);
  if (!Number.isSafeInteger(value) || value <= 0)
    throw new Error(`${name} must be positive milliseconds`);
  return value;
}

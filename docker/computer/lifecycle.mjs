import { spawn } from "node:child_process";
import { setTimeout as sleep } from "node:timers/promises";
import { performance } from "node:perf_hooks";

/** 主 profile 只有这个 owner 能启动；请求与 Turn 阻止空闲回收。 */
export class ComputerLifecycle {
  #state = "sleeping";
  error = "";
  users = 0;
  turns = new Set();
  lastUsed = performance.now();
  transition = null;
  child = null;

  constructor({ start, stop, abandon, idleMs, onChange = () => {} }) {
    this.changed = onChange;
    this.startDriver = start;
    this.stopDriver = stop;
    this.abandonDriver = abandon;
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

  // 状态变化通知订阅者；error 总与 state 一起读取，由订阅方合并同一轮的多次变化。
  get state() {
    return this.#state;
  }

  set state(value) {
    this.#state = value;
    this.changed();
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

  fail(error) {
    this.state = "failed";
    this.error = error.message;
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
      ["--runtime"],
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
      this.error = "";
      this.touch();
    } catch (error) {
      this.state = "stopping";
      try {
        await this.stopRuntime();
      } catch (cleanupError) {
        this.fail(new AggregateError([error, cleanupError],
          `Computer startup failed: ${error.message}; cleanup failed: ${cleanupError.message}`));
        throw new Error(this.error);
      }
      this.state = "sleeping";
      this.error = `Computer startup failed: ${error.message}`;
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
      else this.child = null;
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
    if (child.signalCode)
      throw new Error(`Computer runtime exited by ${child.signalCode}; profile flush was not confirmed`);
  }

  /** 0115：人工从 failed 恢复。旧运行时整组结束、确认无进程后，才允许再次打开主 profile。 */
  async restart() {
    if (this.transition) await this.transition;
    if (this.state !== "failed")
      throw new Error(`Computer is ${this.state}; only a failed Computer can be restarted`);
    this.state = "stopping";
    this.transition = (async () => {
      // 1. 能正常收尾就正常收尾，让还活着的 Chromium 有机会落盘；原生输入卡住时
      //    收尾可能永远等不到回执，所以给出期限。
      try {
        await withDeadline(this.stopDriver(), RESTART_STOP_MS);
      } catch (error) {
        // 2. 失败或超时就强制结束 driver 自己的子进程；旧输入随下面的 X server 一起消失。
        console.error("Computer restart: driver stop failed:", error.message);
        await this.abandonDriver();
      }
      // 3. 整组结束运行时（Xvnc、桌面、Chromium、推流），并确认组内没有存活进程。
      await this.killRuntime();
      this.state = "sleeping";
      this.error = "";
    })().catch((error) => {
      this.state = "failed";
      this.error = `Computer restart failed: ${error.message}`;
      throw error;
    }).finally(() => {
      this.transition = null;
    });
    await this.transition;
    // 4. 和普通唤醒走同一条路径；启动失败按原规则回到休眠或 failed。
    await this.wake();
  }

  async killRuntime() {
    const child = this.child;
    if (!child?.pid) return;
    try {
      process.kill(-child.pid, "SIGKILL");
    } catch (error) {
      if (error.code !== "ESRCH") throw error;
    }
    await this.exited;
    const deadline = performance.now() + 5000;
    while (groupAlive(child.pid)) {
      if (performance.now() > deadline)
        throw new Error("Computer runtime processes did not exit after SIGKILL");
      await sleep(50);
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

// driver.close 内部各步已有期限（原生输入 4s、浏览器释放 11s），这里覆盖剩下无期限的等待。
const RESTART_STOP_MS = 20000;

async function withDeadline(work, ms) {
  let timer;
  try {
    return await Promise.race([
      work,
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(`driver stop timed out after ${ms}ms`)), ms);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

function groupAlive(pid) {
  try {
    process.kill(-pid, 0);
    return true;
  } catch (error) {
    if (error.code === "ESRCH") return false;
    throw error;
  }
}

import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { EventEmitter, once } from "node:events";

/** Native 连接只由 supervisor 持有；JS 超时不能绕开输入释放。 */
export class DesktopBackend extends EventEmitter {
  pending = new Map();
  nextId = 1;
  start() {
    this.process = spawn(
      process.env.COMPUTER_DESKTOP_BIN ?? "/opt/computer/bin/akashic-desktop",
      [],
      { stdio: ["pipe", "pipe", "pipe"] },
    );
    const child = this.process;
    this.stderr = "";
    this.process.stderr.on("data", (chunk) => {
      this.stderr = (this.stderr + chunk).slice(-8192);
    });
    createInterface({ input: this.process.stdout }).on("line", (line) => {
      const message = JSON.parse(line);
      const pending = this.pending.get(message.id);
      if (!pending) return;
      this.pending.delete(message.id);
      if (message.error) pending.reject(new Error(message.error));
      else pending.resolve(message.result);
    });
    child.on("error", (error) => {
      if (this.process === child) this.fail(error);
    });
    child.on("exit", (code, signal) => {
      if (this.process === child)
        this.fail(
          new Error(
            `Desktop process exited (${code ?? signal}): ${this.stderr}`,
          ),
        );
    });
    child.stdin.on("error", (error) => {
      if (this.process === child) this.fail(error);
    });
    return this.call("release");
  }
  fail(error) {
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
  }
  /** 原生输入保持原回执，位置读取只生成独立的只读反馈。 */
  async call(method, input = {}) {
    if (
      !this.process ||
      this.process.exitCode != null ||
      this.process.signalCode != null
    )
      return Promise.reject(new Error("Desktop backend is not running"));
    const id = this.nextId++;
    const result = await new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.process.stdin.write(JSON.stringify({ id, method, input }) + "\n");
    });
    // 已完成的输入保持原回执；位置反馈失败单独报告，不重试输入。
    if (this.listenerCount("cursor") &&
        ["click", "move", "drag", "drag_handle", "scroll"].includes(method)) {
      try {
        const point = await this.call("get_pointer");
        this.emit("cursor", { point: { ...point, kind: method === "click" ? "click" : "move" } });
      } catch (error) {
        console.error("Computer cursor read failed:", error.message);
        this.emit("cursor", { point: null, error: "操作位置读取失败" });
      }
    }
    return result;
  }
  async cancel() {
    if (!this.process) return;
    if (this.process.exitCode != null || this.process.signalCode != null) {
      if (this.process.exitCode !== 0)
        throw new Error(
          `Desktop exited without a release receipt: ${this.stderr}`,
        );
      return;
    }
    const exited = once(this.process, "exit");
    this.process.kill("SIGTERM");
    this.process.stdin.end();
    let timer;
    try {
      await Promise.race([
        exited,
        new Promise((_, reject) => {
          timer = setTimeout(() => {
            this.process.kill("SIGKILL");
            reject(
              new Error("Desktop release timed out; input state is uncertain"),
            );
          }, 4000);
        }),
      ]);
      if (this.process.signalCode || this.process.exitCode !== 0)
        throw new Error(`Desktop release was not confirmed: ${this.stderr}`);
    } finally {
      clearTimeout(timer);
    }
  }
}

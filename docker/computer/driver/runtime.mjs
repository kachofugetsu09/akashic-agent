import { randomUUID } from "node:crypto";
import { Worker } from "node:worker_threads";
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { BrowserBackend } from "./cdp.mjs";
import { DesktopBackend } from "./desktop.mjs";

/** 一个容器拥有一份输入状态；各 Session 仅隔离 JS 绑定，不复制浏览器 profile。 */
export class ComputerDriver {
  browser = new BrowserBackend();
  desktop = new DesktopBackend();
  sessions = new Map();
  active = null;
  closed = false;
  cancelledCalls = new Map();
  async start() {
    await this.browser.start();
    await deadline(this.desktop.start(), 4000, "Native startup");
    this.browser.on("event", (event) => {
      for (const session of this.sessions.values())
        session.worker.postMessage({ kind: "event", event });
    });
    this.browser.on("disconnected", () => {
      this.closed = true;
      this.active?.reject(
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
      if (this.active?.session === session) this.active.reject(error);
    });
    worker.on("exit", (code) => {
      if (this.active?.session === session)
        this.active.reject(new Error(`JS worker exited: ${code}`));
    });
    return session;
  }
  message(session, message) {
    const active = this.active;
    if (message.kind === "ready" || message.kind === "metadata") return;
    if (message.kind === "result") {
      if (active?.session === session) {
        if (message.error)
          active.reject(
            Object.assign(new Error(message.error), {
              content: message.content,
            }),
          );
        else active.resolve(message.content);
      }
      return;
    }
    if (!["browser", "desktop"].includes(message.kind))
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
    const backend = message.kind === "browser" ? this.browser : this.desktop;
    const work = backend.call(message.method, message.params, active.context);
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
  /** 调用结束时先 drain 再 release；异常会使本 Session 的 JS 对象失效。 */
  async run(
    { context, code = "", endTurn = false, timeoutMs = 60_000, task },
    signal,
  ) {
    if (endTurn) {
      while (this.active)
        await deadline(this.active.done, 145000, "Wait for active call");
      signal?.throwIfAborted();
      if (!this.sessions.has(context?.session_id)) task = async () => [];
    }
    if (this.closed) throw new Error("Computer driver is stopped");
    if (this.active)
      throw new Error("Computer is busy; retry after the current call settles");
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
    const active = {
      context,
      pending: new Set(),
      cancelled: false,
      controller: new AbortController(),
    };
    this.active = active;
    active.done = new Promise((resolve) => {
      active.finish = resolve;
    });
    const result = new Promise((resolve, reject) =>
      Object.assign(active, { resolve, reject }),
    );
    let timer;
    const abort = () =>
      active.reject(signal.reason ?? new Error("Computer caller disconnected"));
    let content = [],
      failure;
    try {
      if (!task) active.session = await this.session(context.session_id);
      signal?.addEventListener("abort", abort, { once: true });
      if (signal?.aborted) abort();
      timer = setTimeout(
        () => active.reject(new Error("Computer call timed out")),
        timeoutMs,
      );
      if (task) {
        const work = task(active.controller.signal);
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
    } catch (error) {
      failure = error;
      active.cancelled = true;
      active.controller.abort(error);
      if (active.session) await active.session.worker.terminate();
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
      // Native 先响应取消；浏览器已送出的 CDP 命令有自己的有界超时。
      try {
        if (failure) await this.desktop.cancel();
        await Promise.allSettled([...active.pending]);
        if (!failure)
          await deadline(this.desktop.call("release"), 4000, "Native release");
        await deadline(this.browser.releaseInputs(), 11000, "Browser release");
        if (endTurn)
          await deadline(this.browser.endTurn(context), 11000, "Turn cleanup");
      } catch (error) {
        this.closed = true;
        await this.desktop.cancel().catch(() => {});
        this.browser.close();
        failure = new AggregateError(
          failure ? [failure, error] : [error],
          "Computer input release is uncertain; restart the workload",
        );
      }
      if (failure && active.session) {
        await active.session.worker.terminate();
        this.sessions.delete(context.session_id);
        await rm(active.session.directory, { recursive: true, force: true });
      }
      if (failure && !this.closed) {
        try {
          await deadline(this.desktop.start(), 4000, "Native restart");
        } catch (error) {
          this.closed = true;
          failure = new AggregateError(
            [failure, error],
            "Native restart failed",
          );
        }
      }
      this.active = null;
      active.finish();
    }
    if (failure) {
      failure.message +=
        "; earlier effects may remain; JS bindings for this session were reset";
      throw failure;
    }
    return { content, call_id: context.call_id };
  }
  async perform(task, signal, kind = "browser") {
    const callId = randomUUID();
    const result = await this.run(
      {
        context: {
          session_id: `legacy-${kind}`,
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
  async input(method, input, signal) {
    return this.perform(
      () => this.desktop.call(method, input),
      signal,
      "desktop",
    );
  }

  async cancel(callId) {
    if (this.closed)
      throw new Error("Input release is uncertain; driver is stopped");
    if (typeof callId !== "string" || !callId.length || callId.length > 256)
      throw new TypeError("cancel requires a valid call_id");
    if (this.active?.context.call_id !== callId) {
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
    const active = this.active;
    active.reject(new Error("Computer call cancelled"));
    await active.done;
    if (this.closed)
      throw new Error("Input release is uncertain; restart the workload");
  }
  async reset(sessionId) {
    if (this.active)
      throw new Error("Cannot reset while a Computer call is running");
    const session = this.sessions.get(sessionId);
    if (!session) return;
    await session.worker.terminate();
    this.sessions.delete(sessionId);
    await rm(session.directory, { recursive: true, force: true });
  }
  async close() {
    this.closed = true;
    if (this.active) {
      const active = this.active;
      active.reject(new Error("Computer driver is stopping"));
      await active.done;
    }
    for (const session of this.sessions.values())
      await session.worker.terminate();
    await this.desktop.cancel();
    await deadline(this.browser.releaseInputs(), 11000, "Browser release");
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

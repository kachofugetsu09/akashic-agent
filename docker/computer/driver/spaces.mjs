import { setTimeout as sleep } from "node:timers/promises";
import { Worker } from "node:worker_threads";
import { SPACE_SENTINEL_URL } from "./cdp.mjs";

export class SpaceInputError extends Error {}

function assertAllowedUrl(rawUrl) {
  if (typeof rawUrl !== "string" || rawUrl.length === 0 || rawUrl.length > 8192) {
    throw new SpaceInputError("url must be a non-empty string of at most 8192 characters");
  }
  let parsed;
  try {
    parsed = new URL(rawUrl);
  } catch {
    throw new SpaceInputError("url must be a valid http(s) URL");
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new SpaceInputError("Only http and https URLs are allowed in background spaces");
  }
  return parsed.href;
}

function validateInteger(value, name, min, max, fallback) {
  if (value === undefined) return fallback;
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new SpaceInputError(`${name} must be an integer between ${min} and ${max}`);
  }
  return value;
}

function validatePayload(payload) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new SpaceInputError("request body must be an object");
  }
  const space =
    payload.space === undefined
      ? "default"
      : typeof payload.space === "string" && payload.space.trim().length > 0 && payload.space.length <= 128
        ? payload.space.trim()
        : (() => {
            throw new SpaceInputError("space must be a non-empty string of at most 128 characters");
          })();

  const timeoutMs = validateInteger(payload.timeoutMs, "timeoutMs", 1000, 110000, 45000);
  const spoofVisibility =
    payload.spoofVisibility === undefined
      ? true
      : typeof payload.spoofVisibility === "boolean"
        ? payload.spoofVisibility
        : (() => {
            throw new SpaceInputError("spoofVisibility must be a boolean");
          })();

  const hasScript = payload.script !== undefined;
  const hasUrl = payload.url !== undefined;
  if (hasScript && hasUrl) {
    throw new SpaceInputError("Provide either script or url, not both");
  }
  if (!hasScript && !hasUrl) {
    throw new SpaceInputError("Either script or url is required");
  }

  if (hasScript) {
    if (typeof payload.script !== "string" || payload.script.trim().length === 0 || payload.script.length > 131072) {
      throw new SpaceInputError("script must be a non-empty string of at most 131072 characters");
    }
    return { mode: "script", space, timeoutMs, spoofVisibility, script: payload.script };
  }

  const url = assertAllowedUrl(payload.url);
  const waitUntil = payload.waitUntil === undefined ? "domcontentloaded" : payload.waitUntil;
  if (!["domcontentloaded", "load", "none"].includes(waitUntil)) {
    throw new SpaceInputError("waitUntil must be domcontentloaded, load, or none");
  }
  let waitFor = null;
  if (payload.waitFor !== undefined) {
    if (typeof payload.waitFor !== "string" || payload.waitFor.trim().length === 0 || payload.waitFor.length > 1024) {
      throw new SpaceInputError("waitFor must be a non-empty CSS selector string of at most 1024 characters");
    }
    waitFor = payload.waitFor;
  }
  const waitTimeoutMs = validateInteger(payload.waitTimeoutMs, "waitTimeoutMs", 100, 60000, 15000);
  const waitMs = validateInteger(payload.waitMs, "waitMs", 0, 30000, 0);
  let evalExpr = null;
  if (payload.eval !== undefined) {
    if (typeof payload.eval !== "string" || payload.eval.trim().length === 0 || payload.eval.length > 65536) {
      throw new SpaceInputError("eval must be a non-empty string of at most 65536 characters");
    }
    evalExpr = payload.eval;
  }

  return {
    mode: "declarative",
    space,
    timeoutMs,
    spoofVisibility,
    url,
    waitUntil,
    waitFor,
    waitTimeoutMs,
    waitMs,
    evalExpr,
  };
}

function combineSignals(callerSignal, timeoutSignal) {
  if (!callerSignal) return timeoutSignal;
  if (typeof AbortSignal.any === "function") return AbortSignal.any([callerSignal, timeoutSignal]);
  const controller = new AbortController();
  const abort = (src) => controller.abort(src.reason);
  if (callerSignal.aborted) abort(callerSignal);
  else if (timeoutSignal.aborted) abort(timeoutSignal);
  else {
    callerSignal.addEventListener("abort", () => abort(callerSignal), { once: true });
    timeoutSignal.addEventListener("abort", () => abort(timeoutSignal), { once: true });
  }
  return controller.signal;
}

function raceWithSignal(work, signal) {
  if (!signal) return work;
  if (signal.aborted) return Promise.reject(signal.reason ?? new Error("Space execution aborted"));
  let onAbort;
  const aborted = new Promise((_, reject) => {
    onAbort = () => reject(signal.reason ?? new Error("Space execution aborted"));
    signal.addEventListener("abort", onAbort, { once: true });
  });
  return Promise.race([work, aborted]).finally(() => {
    if (onAbort) signal.removeEventListener("abort", onAbort);
  });
}

function buildEvalExpression(fnOrExpr, isFn, arg) {
  if (isFn || typeof fnOrExpr === "function") {
    const src = typeof fnOrExpr === "function" ? fnOrExpr.toString() : String(fnOrExpr);
    return `(${src})(${JSON.stringify(arg)})`;
  }
  return String(fnOrExpr);
}

const VISIBILITY_SPOOF_SCRIPT = `(() => {
  try {
    Object.defineProperty(document, "visibilityState", { configurable: true, get: () => "visible" });
    Object.defineProperty(document, "hidden", { configurable: true, get: () => false });
  } catch {}
})();`;

function isAlreadyGoneError(err) {
  const msg = String(err?.message ?? err ?? "");
  return /No target with given id|Target closed|Session closed|Not found/i.test(msg);
}

/**
 * ego-lite 风格的后台 Task Space 运行时：
 * - 在同一 Chromium 进程（共享主 profile cookie/登录态）内用 background: true 开隐身标签；
 * - 不建新 X11 窗口、不碰 Screen.front、不进入 agentOperation 锁，全程不抢前台焦点与键鼠；
 * - 通过有界并发池跑单次脚本（Worker 强隔离，可强杀超时死循环）或声明式抓取，结束后校验整棵目标树关闭。
 */
export class BackgroundSpaces {
  constructor({ browser, maxConcurrency = 4 } = {}) {
    if (!browser) throw new TypeError("BackgroundSpaces requires a browser backend");
    const concurrency = Number.isInteger(maxConcurrency) && maxConcurrency >= 1 && maxConcurrency <= 16 ? maxConcurrency : 4;
    this.browser = browser;
    this.maxConcurrency = concurrency;
    this.activeCount = 0;
    this.queue = [];
  }

  get queuedCount() {
    return this.queue.length;
  }

  async #acquire(signal) {
    signal?.throwIfAborted();
    if (this.activeCount < this.maxConcurrency) {
      this.activeCount++;
      return;
    }
    await new Promise((resolve, reject) => {
      const entry = {
        resolve: () => {
          signal?.removeEventListener("abort", onAbort);
          resolve();
        },
        reject: (err) => {
          signal?.removeEventListener("abort", onAbort);
          reject(err);
        },
      };
      const onAbort = () => {
        const idx = this.queue.indexOf(entry);
        if (idx >= 0) this.queue.splice(idx, 1);
        reject(signal.reason ?? new Error("Space execution aborted while queued"));
      };
      signal?.addEventListener("abort", onAbort, { once: true });
      this.queue.push(entry);
    });
  }

  #release() {
    while (this.queue.length > 0) {
      const next = this.queue.shift();
      next.resolve();
      return;
    }
    this.activeCount = Math.max(0, this.activeCount - 1);
  }

  async #closeTargetVerified(connection, tid) {
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const res = await connection.send("Target.closeTarget", { targetId: tid }, undefined, 5000);
        if (res && res.success === false) {
          throw new Error(`Target.closeTarget returned success=false for ${tid}`);
        }
        return true;
      } catch (err) {
        if (isAlreadyGoneError(err)) return true;
        if (attempt === 0) {
          await sleep(50);
          continue;
        }
        try {
          const targets = await connection.send("Target.getTargets", {}, undefined, 2000);
          const stillAlive = (targets?.targetInfos ?? []).some((item) => item.targetId === tid);
          if (!stillAlive) return true;
        } catch {
          // If Target.getTargets itself failed, assume target cleanup is unconfirmed.
        }
        throw err instanceof Error ? err : new Error(String(err));
      }
    }
    return false;
  }

  async #runScriptWorker(script, space, page, runSignal) {
    runSignal.throwIfAborted();
    const worker = new Worker(new URL("./space-worker.mjs", import.meta.url), {
      workerData: { script, space },
    });
    let finished = false;
    let terminating = null;
    const stopWorker = () => {
      if (terminating) return terminating;
      terminating = worker.terminate().catch(() => 0);
      return terminating;
    };

    try {
      const value = await new Promise((resolve, reject) => {
        const cleanupListeners = () => {
          runSignal.removeEventListener("abort", onAbort);
        };
        const onAbort = () => {
          if (finished) return;
          finished = true;
          cleanupListeners();
          void stopWorker();
          reject(runSignal.reason ?? new Error("Space script aborted"));
        };
        runSignal.addEventListener("abort", onAbort, { once: true });

        worker.on("message", async (msg) => {
          if (finished || runSignal.aborted) return;
          if (msg?.kind === "result") {
            finished = true;
            cleanupListeners();
            resolve(msg.value);
            return;
          }
          if (msg?.kind === "error") {
            finished = true;
            cleanupListeners();
            reject(new Error(msg.error || "Space script error"));
            return;
          }
          if (msg?.kind === "rpc") {
            const { id, method, args = [] } = msg;
            try {
              let out;
              if (method === "evaluate") {
                const [exprOrFn, isFn, arg] = args;
                out = await page.evaluate(buildEvalExpression(exprOrFn, isFn, arg));
              } else if (method === "waitForFunction") {
                const [exprOrFn, isFn, timeoutMs, arg] = args;
                out = await page.waitForFunction(buildEvalExpression(exprOrFn, isFn, arg), timeoutMs);
              } else if (typeof page[method] === "function") {
                out = await page[method](...args);
              } else {
                throw new Error(`Unsupported page RPC method: ${method}`);
              }
              if (!finished && !runSignal.aborted) {
                worker.postMessage({ kind: "reply", id, value: out });
              }
            } catch (err) {
              if (!finished && !runSignal.aborted) {
                worker.postMessage({
                  kind: "reply",
                  id,
                  error: err instanceof Error ? err.message : String(err),
                });
              }
            }
          }
        });

        worker.on("error", (err) => {
          if (finished) return;
          finished = true;
          cleanupListeners();
          reject(err);
        });

        worker.on("exit", (code) => {
          if (finished) return;
          finished = true;
          cleanupListeners();
          reject(new Error(`Space script worker exited unexpectedly with code ${code}`));
        });
      });
      runSignal.throwIfAborted();
      return value;
    } finally {
      finished = true;
      await stopWorker();
    }
  }

  async run(payload, callerSignal) {
    const spec = validatePayload(payload);
    const runSignal = combineSignals(callerSignal, AbortSignal.timeout(spec.timeoutMs));
    await this.#acquire(runSignal);

    const connection = this.browser.browser;
    if (!connection) {
      this.#release();
      throw new Error("Browser CDP connection is not ready");
    }

    let rootTargetId = null;
    const ownedSet = new Set();
    const ownedOrder = [];
    const registerOwned = (tid, isChild = false) => {
      if (!tid || ownedSet.has(tid)) return;
      ownedSet.add(tid);
      ownedOrder.push(tid);
      this.browser.hiddenTargets.add(tid);
      if (isChild && rootTargetId) {
        this.browser.hiddenTargetRoots.set(tid, rootTargetId);
      }
    };

    const sessionListeners = new Set();
    const onEvent = (event) => {
      if (event?.method === "Target.targetCreated" && event.params?.targetInfo) {
        const info = event.params.targetInfo;
        if (info.openerId && ownedSet.has(info.openerId)) {
          registerOwned(info.targetId, true);
        }
      }
      for (const listener of [...sessionListeners]) listener(event);
    };
    connection.on("event", onEvent);

    let primaryError = null;
    let createPromise = null;
    let cleanupStarted = false;
    const closedTargets = new Set();
    try {
      runSignal.throwIfAborted();
      createPromise = connection.send("Target.createTarget", {
        url: SPACE_SENTINEL_URL,
        background: true,
      });
      createPromise.then(
        ({ targetId }) => {
          if (!targetId) return;
          if (!rootTargetId) rootTargetId = targetId;
          registerOwned(targetId, false);
          if (cleanupStarted && !closedTargets.has(targetId)) {
            void this.#closeTargetVerified(connection, targetId)
              .then(() => {
                closedTargets.add(targetId);
                this.browser.hiddenTargets.delete(targetId);
                this.browser.hiddenTargetRoots.delete(targetId);
              })
              .catch(() => {});
          }
        },
        () => {},
      );
      ({ targetId: rootTargetId } = await raceWithSignal(createPromise, runSignal));
      registerOwned(rootTargetId, false);
      runSignal.throwIfAborted();

      const { sessionId } = await raceWithSignal(
        connection.send("Target.attachToTarget", {
          targetId: rootTargetId,
          flatten: true,
        }),
        runSignal,
      );

      const cdp = (method, params = {}, callTimeoutMs = 15000) =>
        raceWithSignal(connection.send(method, params, sessionId, callTimeoutMs), runSignal);

      await cdp("Page.enable");
      await cdp("Runtime.enable");
      if (spec.spoofVisibility) {
        await cdp("Page.addScriptToEvaluateOnNewDocument", {
          source: VISIBILITY_SPOOF_SCRIPT,
        });
      }

      const waitEvent = (methodName, timeoutMs) =>
        new Promise((resolve, reject) => {
          if (runSignal.aborted) {
            reject(runSignal.reason);
            return;
          }
          let timer;
          const cleanup = () => {
            clearTimeout(timer);
            runSignal.removeEventListener("abort", onAbort);
            sessionListeners.delete(listener);
          };
          const onAbort = () => {
            cleanup();
            reject(runSignal.reason ?? new Error("Space execution aborted"));
          };
          const listener = (event) => {
            if (event.sessionId === sessionId && event.method === methodName) {
              cleanup();
              resolve(event.params ?? {});
            }
          };
          timer = setTimeout(() => {
            cleanup();
            reject(new Error(`Timed out waiting for ${methodName}`));
          }, timeoutMs);
          runSignal.addEventListener("abort", onAbort, { once: true });
          sessionListeners.add(listener);
        });

      const runEval = async (expression, { awaitPromise = true, timeoutMs = 15000 } = {}) => {
        runSignal.throwIfAborted();
        const res = await cdp(
          "Runtime.evaluate",
          { expression, awaitPromise, returnByValue: true },
          timeoutMs,
        );
        if (res.exceptionDetails) {
          const text =
            res.exceptionDetails.exception?.description ||
            res.exceptionDetails.text ||
            "Evaluation failed";
          throw new Error(text);
        }
        return res.result?.value;
      };

      const boundedSleep = async (ms) => {
        const delay = Number(ms);
        if (!Number.isFinite(delay) || delay <= 0) return;
        await sleep(Math.min(delay, 30000), undefined, { signal: runSignal });
      };

      const page = {
        async goto(rawUrl, opts = {}) {
          const targetUrl = assertAllowedUrl(rawUrl);
          const waitUntil = opts.waitUntil ?? "domcontentloaded";
          const gotoTimeout = validateInteger(opts.timeoutMs, "goto.timeoutMs", 500, 60000, 30000);
          if (!["domcontentloaded", "load", "none"].includes(waitUntil)) {
            throw new SpaceInputError("waitUntil must be domcontentloaded, load, or none");
          }
          const eventPromise =
            waitUntil === "none"
              ? null
              : waitEvent(
                  waitUntil === "load" ? "Page.loadEventFired" : "Page.domContentEventFired",
                  gotoTimeout,
                );
          if (eventPromise) eventPromise.catch(() => {});
          const nav = await cdp("Page.navigate", { url: targetUrl }, gotoTimeout);
          if (nav.errorText) {
            throw new Error(`Navigation failed: ${nav.errorText}`);
          }
          if (eventPromise && nav.loaderId) {
            await eventPromise;
          }
          return { url: await page.url(), title: await page.title() };
        },
        async waitForSelector(selector, timeoutMs = 15000) {
          if (typeof selector !== "string" || !selector.trim()) {
            throw new SpaceInputError("selector must be a non-empty string");
          }
          const budget = validateInteger(timeoutMs, "waitForSelector.timeoutMs", 50, 60000, 15000);
          const deadlineAt = Date.now() + budget;
          const expr = `Boolean(document.querySelector(${JSON.stringify(selector)}))`;
          while (Date.now() < deadlineAt) {
            runSignal.throwIfAborted();
            try {
              if (await runEval(expr, { awaitPromise: false, timeoutMs: 2000 })) {
                return true;
              }
            } catch (err) {
              if (runSignal.aborted) throw runSignal.reason ?? err;
              if (!/Execution context was destroyed|Cannot find context/i.test(String(err.message))) {
                throw err;
              }
            }
            await sleep(100, undefined, { signal: runSignal });
          }
          throw new Error(`Timed out waiting for selector: ${selector}`);
        },
        async waitForFunction(fnOrExpr, timeoutMs = 15000, arg) {
          const budget = validateInteger(timeoutMs, "waitForFunction.timeoutMs", 50, 60000, 15000);
          const body = buildEvalExpression(fnOrExpr, typeof fnOrExpr === "function", arg);
          const deadlineAt = Date.now() + budget;
          while (Date.now() < deadlineAt) {
            runSignal.throwIfAborted();
            try {
              const ok = await runEval(body, { awaitPromise: true, timeoutMs: 3000 });
              if (ok) return ok;
            } catch (err) {
              if (runSignal.aborted) throw runSignal.reason ?? err;
              if (!/Execution context was destroyed|Cannot find context/i.test(String(err.message))) {
                throw err;
              }
            }
            await sleep(100, undefined, { signal: runSignal });
          }
          throw new Error("Timed out waiting for condition");
        },
        async evaluate(fnOrExpr, arg) {
          const expression = buildEvalExpression(fnOrExpr, typeof fnOrExpr === "function", arg);
          return await runEval(expression, { awaitPromise: true, timeoutMs: 15000 });
        },
        async click(selector) {
          if (typeof selector !== "string" || !selector.trim()) {
            throw new SpaceInputError("selector must be a non-empty string");
          }
          const ok = await runEval(
            `(() => { const el = document.querySelector(${JSON.stringify(selector)}); if (!el) return false; el.click(); return true; })()`,
            { awaitPromise: false },
          );
          if (!ok) throw new Error(`Element not found for click: ${selector}`);
          return true;
        },
        async fill(selector, value) {
          if (typeof selector !== "string" || !selector.trim()) {
            throw new SpaceInputError("selector must be a non-empty string");
          }
          const ok = await runEval(
            `(() => {
              const el = document.querySelector(${JSON.stringify(selector)});
              if (!el) return false;
              const val = ${JSON.stringify(String(value ?? ""))};
              const proto = el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
              const setter = Object.getOwnPropertyDescriptor(proto, "value")?.set;
              if (setter) setter.call(el, val);
              else el.value = val;
              el.dispatchEvent(new Event("input", { bubbles: true }));
              el.dispatchEvent(new Event("change", { bubbles: true }));
              return true;
            })()`,
            { awaitPromise: false },
          );
          if (!ok) throw new Error(`Element not found for fill: ${selector}`);
          return true;
        },
        async title() {
          return (await runEval("document.title", { awaitPromise: false })) ?? "";
        },
        async url() {
          return (await runEval("location.href", { awaitPromise: false })) ?? "";
        },
        async content() {
          return (await runEval("document.documentElement ? document.documentElement.outerHTML : ''", { awaitPromise: false })) ?? "";
        },
        sleep: boundedSleep,
      };

      let result;
      if (spec.mode === "script") {
        result = await this.#runScriptWorker(spec.script, spec.space, page, runSignal);
      } else {
        result = await raceWithSignal(
          (async () => {
            await page.goto(spec.url, { waitUntil: spec.waitUntil });
            if (spec.waitFor) await page.waitForSelector(spec.waitFor, spec.waitTimeoutMs);
            if (spec.waitMs > 0) await boundedSleep(spec.waitMs);
            return spec.evalExpr
              ? await page.evaluate(spec.evalExpr)
              : { url: await page.url(), title: await page.title() };
          })(),
          runSignal,
        );
      }

      runSignal.throwIfAborted();
      return { space: spec.space, result };
    } catch (error) {
      primaryError = error;
      throw error;
    } finally {
      cleanupStarted = true;
      const cleanupErrors = [];
      if (createPromise && !rootTargetId) {
        await Promise.race([
          createPromise.then(
            ({ targetId }) => {
              if (targetId) {
                rootTargetId = targetId;
                registerOwned(targetId, false);
              }
            },
            () => {},
          ),
          sleep(5000),
        ]);
      }

      if (rootTargetId) {
        const failedTargets = new Set();
        const discoverSubtree = async () => {
          for (const [childId, rootId] of this.browser.hiddenTargetRoots.entries()) {
            if (rootId === rootTargetId) registerOwned(childId, true);
          }
          try {
            const all = await connection.send("Target.getTargets", {}, undefined, 2000);
            let added = true;
            while (added) {
              added = false;
              for (const info of all?.targetInfos ?? []) {
                if (info.openerId && ownedSet.has(info.openerId) && !ownedSet.has(info.targetId)) {
                  registerOwned(info.targetId, true);
                  added = true;
                }
              }
            }
          } catch {
            // Best effort discovery; close whatever we already tracked.
          }
        };

        let drained = false;
        for (let pass = 0; pass < 10; pass++) {
          await discoverSubtree();
          const pending = [...ownedOrder]
            .reverse()
            .filter((tid) => !closedTargets.has(tid) && !failedTargets.has(tid));
          if (pending.length === 0) {
            drained = true;
            break;
          }
          for (const tid of pending) {
            try {
              await this.#closeTargetVerified(connection, tid);
              closedTargets.add(tid);
              this.browser.hiddenTargets.delete(tid);
              this.browser.hiddenTargetRoots.delete(tid);
            } catch (err) {
              // 关闭未确认时保留在 hiddenTargets，避免泄露到 listTabs()。
              failedTargets.add(tid);
              cleanupErrors.push(err instanceof Error ? err : new Error(String(err)));
            }
          }
        }
        if (!drained) {
          await discoverSubtree();
          const remaining = [...ownedOrder].filter((tid) => !closedTargets.has(tid) && !failedTargets.has(tid));
          if (remaining.length > 0) {
            cleanupErrors.push(new Error(`Background space cleanup did not drain remaining targets: ${remaining.join(", ")}`));
          }
        }
      }
      connection.off("event", onEvent);
      this.#release();
      if (cleanupErrors.length > 0) {
        if (primaryError) {
          throw new AggregateError([primaryError, ...cleanupErrors], "Background space run and target cleanup failed");
        }
        throw cleanupErrors.length === 1
          ? cleanupErrors[0]
          : new AggregateError(cleanupErrors, "Background space target cleanup failed");
      }
    }
  }
}

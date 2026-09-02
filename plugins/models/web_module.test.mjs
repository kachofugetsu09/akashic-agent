import assert from "node:assert/strict";
import test from "node:test";

import {
  capabilitySummary,
  createDialogAuthOwner,
  createLatestCatalogRead,
  modelsForRole,
  readJsonResponse,
} from "./web_module.js";

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return {promise, resolve};
}

test("a stale catalog response cannot replace the newest catalog", async () => {
  const first = deferred();
  const second = deferred();
  const requests = [first, second];
  const applied = [];
  const signals = [];
  let index = 0;
  const owner = createLatestCatalogRead(
    (signal) => {
      signals.push(signal);
      return requests[index++].promise;
    },
    (catalog) => applied.push(catalog.revision),
  );

  const oldRead = owner.run();
  const newRead = owner.run();
  assert.equal(signals[0].aborted, true);
  second.resolve({revision: 2});
  await newRead;
  first.resolve({revision: 1});
  await oldRead;

  assert.deepEqual(applied, [2]);
});

test("a closed catalog owner cannot start another read", async () => {
  let reads = 0;
  let activeSignal;
  const activeRead = deferred();
  const owner = createLatestCatalogRead((signal) => {
    reads += 1;
    activeSignal = signal;
    return activeRead.promise;
  }, () => {});

  const pending = owner.run();
  owner.close();
  await owner.run();
  assert.equal(activeSignal.aborted, true);
  activeRead.resolve({revision: 1});
  await pending;

  assert.equal(reads, 1);
});

test("an auth attempt returned after dialog close is cancelled", async () => {
  const cancelled = [];
  const owner = createDialogAuthOwner(async (attemptId) => cancelled.push(attemptId));

  await owner.close();
  await assert.rejects(owner.add("attempt-late"), /登录面板已关闭/u);
  assert.throws(() => owner.checkFinish("attempt-late"), /登录面板已关闭/u);
  assert.deepEqual(cancelled, ["attempt-late"]);
});

test("a failed auth cancellation remains available for retry", async () => {
  let calls = 0;
  const owner = createDialogAuthOwner(async () => {
    calls += 1;
    if (calls === 1) throw new Error("temporary failure");
  });
  await owner.add("attempt-retry");

  await assert.rejects(owner.close(), /temporary failure/u);
  await owner.cancel("attempt-retry");

  assert.equal(calls, 2);
});

test("capability summary separates confirmed vision from unknown", () => {
  const models = [
    {
      capabilities: {inputModalities: ["text", "image"]},
      capabilitySources: {inputModalities: "litellm-remote@sha256:test"},
    },
    {
      capabilities: {inputModalities: ["text"]},
      capabilitySources: {inputModalities: "unknown"},
    },
  ];

  assert.equal(capabilitySummary(models), "2 个模型 · 1 个可看图 · 1 个待识别");
  assert.deepEqual(modelsForRole(models, "vision"), [models[0]]);
  assert.equal(modelsForRole(models, "default"), models);
});

test("HTTP response errors stay actionable and reject invalid success bodies", async () => {
  await assert.rejects(
    readJsonResponse(new Response("Internal Server Error", {status: 500})),
    /请求失败：500/u,
  );
  await assert.rejects(
    readJsonResponse(new Response(JSON.stringify({code: "forbidden_contract"}), {
      status: 403,
      headers: {"Content-Type": "application/json"},
    })),
    /服务已更新，请刷新页面后重试/u,
  );
  await assert.rejects(
    readJsonResponse(new Response("[]", {
      status: 200,
      headers: {"Content-Type": "application/json"},
    })),
    /服务返回了无效 JSON/u,
  );
});

import assert from "node:assert/strict";
import test from "node:test";

import {
  MOBILE_PLUGIN_QUERY_MAX_PENDING,
  MobilePluginQueryQueue,
} from "./mobile-plugin-query-queue.ts";

function request(ownerId, interactive = true) {
  return { ownerId, interactive, started: false };
}

function drain(queue, sent) {
  while (true) {
    const next = queue.startNext();
    if (!next) return;
    sent.push(next[0]);
  }
}

test("synchronous query bursts cap total pending before the native bridge", () => {
  const queue = new MobilePluginQueryQueue((item) => item.interactive);
  const sent = [];
  for (let index = 0; index < MOBILE_PLUGIN_QUERY_MAX_PENDING; index += 1) {
    queue.enqueue(`request-${index}`, request("owner-a"));
    drain(queue, sent);
  }

  assert.equal(queue.pendingCount, 128);
  assert.equal(queue.activeCount, 4);
  assert.equal(queue.queuedCount, 124);
  assert.deepEqual(sent, ["request-0", "request-1", "request-2", "request-3"]);
  assert.throws(
    () => queue.enqueue("request-overflow", request("owner-a")),
    /插件未完成请求已达上限（最多 128 个）/,
  );
  assert.equal(queue.pendingCount, 128);
  assert.equal(queue.queuedCount, 124);
});

test("a reply releases one pending slot and starts the next queued request", () => {
  const queue = new MobilePluginQueryQueue((item) => item.interactive);
  const sent = [];
  for (let index = 0; index < MOBILE_PLUGIN_QUERY_MAX_PENDING; index += 1) {
    queue.enqueue(`request-${index}`, request("owner-a"));
    drain(queue, sent);
  }

  assert.equal(queue.complete("request-0")?.ownerId, "owner-a");
  drain(queue, sent);

  assert.equal(queue.pendingCount, 127);
  assert.equal(queue.activeCount, 4);
  assert.equal(queue.queuedCount, 123);
  assert.equal(sent.at(-1), "request-4");
  queue.enqueue("request-replacement", request("owner-b"));
  drain(queue, sent);
  assert.equal(queue.pendingCount, 128);
  assert.equal(queue.queuedCount, 124);
});

test("owner cancellation releases both active and queued capacity", () => {
  const queue = new MobilePluginQueryQueue((item) => item.interactive, 6, 2, 1);
  const sent = [];
  ["a-0", "a-1", "a-2"].forEach((requestId) => {
    queue.enqueue(requestId, request("owner-a"));
    drain(queue, sent);
  });
  ["b-0", "b-1", "b-2"].forEach((requestId) => {
    queue.enqueue(requestId, request("owner-b"));
    drain(queue, sent);
  });
  assert.equal(queue.pendingCount, 6);
  assert.equal(queue.activeCount, 2);
  assert.equal(queue.queuedCount, 4);

  const removed = queue.removeOwner("owner-a");
  assert.deepEqual(removed.map(([requestId]) => requestId), ["a-0", "a-1", "a-2"]);
  assert.deepEqual(removed.map(([, item]) => item.started), [true, true, false]);
  drain(queue, sent);

  assert.equal(queue.pendingCount, 3);
  assert.equal(queue.activeCount, 2);
  assert.equal(queue.queuedCount, 1);
  ["c-0", "c-1", "c-2"].forEach((requestId) => {
    queue.enqueue(requestId, request("owner-c"));
    drain(queue, sent);
  });
  assert.equal(queue.pendingCount, 6);
  assert.throws(
    () => queue.enqueue("still-full", request("owner-c")),
    /未完成请求已达上限（最多 6 个）/,
  );
});

test("invalid scheduler budgets and duplicate identities fail loudly", () => {
  const queue = new MobilePluginQueryQueue((item) => item.interactive, 2, 1, 1);
  queue.enqueue("same", request("owner"));

  assert.throws(() => queue.enqueue("same", request("owner")), /身份重复或为空/);
  assert.throws(
    () => new MobilePluginQueryQueue((item) => item.interactive, 0, 1, 1),
    /总量预算无效/,
  );
  assert.throws(
    () => new MobilePluginQueryQueue((item) => item.interactive, 1, 2, 1),
    /并发预算无效/,
  );
});

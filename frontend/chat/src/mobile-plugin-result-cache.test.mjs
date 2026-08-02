import assert from "node:assert/strict";
import test from "node:test";

import { MobilePluginResultCache } from "./mobile-plugin-result-cache.ts";

test("entry quota evicts the least recently used immutable result", () => {
  const cache = new MobilePluginResultCache(2, 100);
  cache.set("first", '{"value":1}');
  cache.set("second", '{"value":2}');

  assert.equal(cache.get("first"), '{"value":1}');
  cache.set("third", '{"value":3}');

  assert.equal(cache.get("second"), undefined);
  assert.equal(cache.get("first"), '{"value":1}');
  assert.equal(cache.get("third"), '{"value":3}');
  assert.equal(cache.size, 2);
  assert.equal(cache.byteSize, 22);
});

test("byte quota evicts oldest results until the aggregate fits", () => {
  const cache = new MobilePluginResultCache(10, 10);
  cache.set("first", "1111");
  cache.set("second", "2222");
  cache.set("third", "3333");

  assert.equal(cache.get("first"), undefined);
  assert.equal(cache.get("second"), "2222");
  assert.equal(cache.get("third"), "3333");
  assert.equal(cache.byteSize, 8);
});

test("oversized result fails loudly without evicting unrelated entries", () => {
  const cache = new MobilePluginResultCache(10, 10);
  cache.set("kept", "safe");

  assert.throws(
    () => cache.set("oversized", "12345678901"),
    /超过总字节预算/,
  );
  assert.equal(cache.get("oversized"), undefined);
  assert.equal(cache.get("kept"), "safe");
  assert.equal(cache.byteSize, 4);
});

test("clear drops all catalog-scoped results and resets byte accounting", () => {
  const cache = new MobilePluginResultCache(10, 10);
  cache.set("first", "data");

  cache.clear();

  assert.equal(cache.get("first"), undefined);
  assert.equal(cache.size, 0);
  assert.equal(cache.byteSize, 0);
});

test("invalid cache invariants fail loudly", () => {
  const cache = new MobilePluginResultCache();

  assert.throws(() => cache.set("", "invalid"), /身份为空/);
  assert.throws(() => new MobilePluginResultCache(0, 1), /条数预算无效/);
  assert.throws(() => new MobilePluginResultCache(1, 0), /字节预算无效/);
});

test("byte quota measures UTF-8 content rather than JavaScript character count", () => {
  const cache = new MobilePluginResultCache(10, 4);

  cache.set("chinese", "中");
  cache.set("ascii", "a");

  assert.equal(cache.byteSize, 4);
});

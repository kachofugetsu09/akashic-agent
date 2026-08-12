import assert from "node:assert/strict";
import test from "node:test";

import { createUuid } from "./browser-uuid.ts";

test("uses the native UUID implementation when the browser exposes it", () => {
  const value = createUuid({
    randomUUID: () => "11111111-2222-4333-8444-555555555555",
    getRandomValues: () => { throw new Error("fallback must not run"); },
  });
  assert.equal(value, "11111111-2222-4333-8444-555555555555");
});

test("creates an RFC 4122 UUID with getRandomValues on plain HTTP", () => {
  const value = createUuid({
    getRandomValues: (bytes) => {
      bytes.fill(0xab);
      return bytes;
    },
  });
  assert.equal(value, "abababab-abab-4bab-abab-abababababab");
  assert.match(value, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/u);
});

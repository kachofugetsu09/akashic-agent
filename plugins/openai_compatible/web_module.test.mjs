import assert from "node:assert/strict";
import test from "node:test";

import {createDiscoveryOwner} from "./web_module.js";

test("a newer discovery aborts and invalidates the older result", () => {
  const owner = createDiscoveryOwner();
  const first = owner.start("endpoint-a");
  const second = owner.start("endpoint-b");

  assert.equal(first.signal.aborted, true);
  assert.equal(first.isCurrent("endpoint-a"), false);
  assert.equal(second.isCurrent("endpoint-a"), false);
  assert.equal(second.isCurrent("endpoint-b"), true);
});

test("field changes and dialog close invalidate pending discovery", () => {
  const owner = createDiscoveryOwner();
  const changed = owner.start("before-change");
  owner.invalidate();
  assert.equal(changed.signal.aborted, true);
  assert.equal(changed.isCurrent("before-change"), false);

  const closing = owner.start("before-close");
  owner.close();
  assert.equal(closing.signal.aborted, true);
  assert.equal(closing.isCurrent("before-close"), false);
  assert.throws(() => owner.start("closed"), /已关闭/u);
});

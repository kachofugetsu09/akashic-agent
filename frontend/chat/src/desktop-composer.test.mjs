import assert from "node:assert/strict";
import test from "node:test";

import { nextComposerExpanded } from "./composer-layout.ts";

test("composer expansion stays owned by the draft until it is cleared", () => {
  assert.equal(nextComposerExpanded(false, "short", () => false), false);
  assert.equal(nextComposerExpanded(false, "wrapped draft", () => true), true);

  let measured = false;
  assert.equal(nextComposerExpanded(true, "wrapped draft plus one", () => {
    measured = true;
    return false;
  }), true);
  assert.equal(measured, false);
  assert.equal(nextComposerExpanded(true, "", () => true), false);
});

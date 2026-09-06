import test from "node:test";
import assert from "node:assert/strict";
import { createAccessibilityCore } from "../driver/accessibility.mjs";

test("AX text preserves tree parents and identities across a changed snapshot", () => {
  const core = createAccessibilityCore();
  const node = (id, name, parentIndex) => ({
    nodeID: String(id),
    backendDOMNodeID: id,
    role: "button",
    name,
    parentIndex,
    properties: {},
  });
  const before = {
    tab: { id: 1 },
    warnings: [],
    nodes: [
      node(1, "root", -1),
      node(2, "left", 0),
      node(3, "right", 0),
      node(4, "left child", 1),
    ],
  };
  const first = core.buildRevision(null, before);
  assert.equal(
    first.text,
    "0 button root\n\t1 button left\n\t\t3 button left child\n\t2 button right\n",
  );
  const second = core.buildRevision(first, {
    ...before,
    nodes: [...before.nodes, node(5, "new child", 2)],
  });
  assert.equal(first.identityForElement(3), second.identityForElement(3));
  assert.notEqual(second.identityForElement(4), second.identityForElement(3));
});

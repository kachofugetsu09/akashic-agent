import assert from "node:assert/strict";
import test from "node:test";

import { browserPoint } from "./web_module.js";

test("browserPoint ignores letterbox and maps the shown browser image", () => {
  const bounds = { left: 10, top: 20, width: 640, height: 800 };

  assert.equal(browserPoint(bounds, 330, 100), null);
  assert.deepEqual(browserPoint(bounds, 10, 220), { x: 0, y: 0 });
  assert.deepEqual(browserPoint(bounds, 649, 619), { x: 1278, y: 798 });
});

test("browserPoint maps a full-size browser image", () => {
  const bounds = { left: 0, top: 0, width: 1280, height: 800 };

  assert.deepEqual(browserPoint(bounds, 640, 400), { x: 640, y: 400 });
});

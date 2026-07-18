import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";

const [, , modulePath, navigationValue, slotsJson] = process.argv;
if (!modulePath || !navigationValue || !slotsJson) {
  throw new Error("usage: mobile_plugin_ui_contract.mjs MODULE NAVIGATION SLOTS_JSON");
}

const expectedNavigation = navigationValue === "true";
assert.ok(expectedNavigation || navigationValue === "false", "navigation 必须是布尔值");
const expectedSlots = JSON.parse(slotsJson);
assert.ok(Array.isArray(expectedSlots), "slots 必须是数组");

const loaded = await import(pathToFileURL(modulePath).href);
const definition = loaded.default;
assert.ok(definition && typeof definition === "object", "插件必须默认导出定义对象");

const slots = definition.slots ?? {};
assert.ok(slots && typeof slots === "object" && !Array.isArray(slots), "插件 slots 无效");
assert.deepEqual(Object.keys(slots).sort(), [...expectedSlots].sort(), "插件 slots 与发布锁不一致");
for (const [name, renderer] of Object.entries(slots)) {
  assert.ok(renderer && typeof renderer === "object", `插件 renderer 无效: ${name}`);
  assert.equal(typeof renderer.mount, "function", `插件 renderer 缺少 mount: ${name}`);
}

const hasDashboard = definition.dashboard !== undefined;
assert.equal(hasDashboard, expectedNavigation, "插件 dashboard 与 catalog navigation 不一致");
if (hasDashboard) {
  assert.ok(definition.dashboard && typeof definition.dashboard === "object", "插件 dashboard 无效");
  assert.equal(typeof definition.dashboard.mount, "function", "插件 dashboard 缺少 mount");
}

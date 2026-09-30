/** Actual Models host/provider modules with synthetic HTTP in JSDOM.
 * Run: node scripts/model_workspace_scenario.mjs
 * Uses no real browser, accounts, credentials, backend, or workspace data.
 */
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";
import { activate } from "../frontend/plugins/models/src/module.js";
import { activate as activateCodex } from "../plugins/codex/web_module.js";
import { activate as activateOpenCode } from "../plugins/opencode_go/web_module.js";

const settle = async () => {
  for (let i = 0; i < 5; i += 1) await new Promise(resolve => setTimeout(resolve, 0));
};
const checks = [];

function providerEntry(activateProvider) {
  let entry;
  activateProvider({ui: {inject: (_id, mount) => mount({register(value) {
    entry = value;
    return () => {};
  }})}});
  return entry;
}

async function mount(provider, initialCatalog = null) {
  const dom = new JSDOM('<main id="host"></main>', {url: "http://model-workspace.local"});
  for (const name of ["window", "document", "Option", "Event", "HTMLElement", "FormData"]) {
    globalThis[name] = dom.window[name];
  }
  globalThis.IntersectionObserver = class { observe() {} disconnect() {} };
  dom.window.HTMLElement.prototype.getClientRects = () => [{width: 100, height: 100}];
  // JSDOM has no top layer; only verify the host's explicit lifecycle operations.
  dom.window.HTMLDialogElement.prototype.showModal = function () { this.open = true; };
  dom.window.HTMLDialogElement.prototype.close = function () {
    if (!this.open) return;
    this.open = false;
    queueMicrotask(() => this.dispatchEvent(new Event("close")));
  };
  const timers = [];
  dom.window.setTimeout = callback => { timers.push(callback); return timers.length; };
  dom.window.clearTimeout = () => {};
  dom.window.confirm = () => true;
  const catalog = initialCatalog ?? {revision: 1, connections: [], models: [], roleBindings: {}, defaultEmbeddingModelId: null};
  const commands = [];
  let entry, connectionId;
  activate({
    ui: {inject: (_id, mount) => mount({register(value) { entry = value; return () => {}; }})},
    http: {async request(_url, init) {
      let result = catalog;
      if (init?.method === "POST") {
        const payload = JSON.parse(init.body);
        commands.push(payload);
        if (!["start_auth", "cancel_auth"].includes(payload.type)) {
          assert.equal(payload.expected_revision, catalog.revision, "writes use the last committed catalog revision");
        }
        if (payload.type === "start_auth") {
          connectionId = payload.connection_id;
          result = {revision: catalog.revision, status: "pending", attemptId: "fixture-attempt", challenge: {interval: 5}};
        } else if (payload.type === "finish_auth") {
          catalog.connections.push({id: connectionId, name: "Fixture", driverId: provider.id, availability: "available"});
          result = {revision: ++catalog.revision, status: "committed"};
        } else if (payload.type === "sync_models") {
          assert.equal(catalog.connections.length, 1, "sync follows committed authentication");
          if (!initialCatalog) catalog.models.push({id: "fixture-model", connectionId, kind: "chat", model: "fixture-chat", availability: "available", capabilities: {inputModalities: ["text"]}, capabilitySources: {inputModalities: "fixture"}});
          result = {revision: ++catalog.revision, status: "committed"};
        } else if (payload.type === "set_default") {
          catalog.roleBindings[payload.role] = payload.model_id;
          result = {revision: ++catalog.revision, status: "committed"};
        } else if (payload.type === "cancel_auth") {
          result = {revision: catalog.revision, status: "cancelled"};
        } else throw new Error(`Unexpected command ${payload.type}`);
      }
      return new Response(JSON.stringify(result), {headers: {"content-type": "application/json"}});
    }},
  });
  const dispose = entry.render(document.querySelector("#host"), {child: () => ({
    entries: [provider], render: (_id, host, props) => provider.render(host, null, props),
  })});
  await settle();
  return {catalog, commands, timers, async close() { dispose(); await settle(); dom.window.close(); }};
}

for (const activateProvider of [activateOpenCode, activateCodex]) {
  const provider = providerEntry(activateProvider);
  const fixture = await mount(provider);
  try {
    document.querySelector("[data-providers] button").click();
    await settle();
    if (provider.id === "codex") {
      document.querySelector("[data-start]").click();
      await settle();
      assert.equal(fixture.timers.length, 1);
      fixture.timers.shift()();
    } else {
      document.querySelector(".settings-dialog-form").dispatchEvent(new Event("submit", {cancelable: true}));
    }
    await settle();
    assert.deepEqual(fixture.commands.map(command => command.type), ["start_auth", "finish_auth", "sync_models", "set_default"]);
    assert.equal(fixture.catalog.roleBindings.default, "fixture-model");
    assert.equal(document.querySelector("dialog[open]"), null);
    checks.push(`${provider.id}: first authentication commits, synchronizes models and sets the missing default`);
  } finally {
    await fixture.close();
  }
}

const roleFixture = await mount(providerEntry(activateCodex), {
  revision: 1,
  connections: [{id: "saved", name: "Saved", driverId: "codex", availability: "available"}],
  models: [{id: "saved-model", connectionId: "saved", kind: "chat", model: "saved-chat", availability: "available", capabilities: {inputModalities: ["text"]}, capabilitySources: {inputModalities: "fixture"}}],
  roleBindings: {default: "saved-model"}, defaultEmbeddingModelId: null,
});
try {
  document.querySelector(".settings-role-pick").click();
  await settle();
  assert.ok(document.querySelector(".settings-sheet-scrim[open]"));
  window.addEventListener("akashic:before-navigate", event => event.preventDefault(), {once: true});
  window.dispatchEvent(new Event("akashic:before-navigate", {cancelable: true}));
  await settle();
  assert.ok(document.querySelector(".settings-sheet-scrim[open]"), "a later leave guard may cancel navigation");
  const navigation = new Event("akashic:before-navigate", {cancelable: true});
  window.dispatchEvent(navigation);
  await settle();
  assert.equal(navigation.defaultPrevented, false);
  assert.equal(document.querySelector("dialog[open]"), null, "retained inactive pages must not own a modal");
  assert.deepEqual(roleFixture.commands, [], "abandoning the chooser never saves a selection");
  checks.push("Role chooser preserves cancelled navigation and releases accepted navigation without writing a binding");
} finally {
  await roleFixture.close();
}

for (const hasAvailable of [true, false]) {
  const provider = providerEntry(activateCodex);
  const makeModel = (id, availability) => ({id, connectionId: "saved", kind: "chat", model: id, availability, capabilities: {inputModalities: ["text"]}, capabilitySources: {inputModalities: "fixture"}});
  const fixture = await mount(provider, {
    revision: 1, connections: [{id: "saved", name: "Saved", driverId: "codex", availability: "available"}],
    models: [makeModel("disabled-first", "disabled"), ...(hasAvailable ? [makeModel("available-second", "available")] : [])],
    roleBindings: {}, defaultEmbeddingModelId: null,
  });
  try {
    document.querySelector("[data-connections] button").click();
    await settle();
    document.querySelector("[data-sync]").click();
    await settle();
    const bindings = fixture.commands.filter(command => command.type === "set_default");
    assert.deepEqual(bindings.map(command => command.model_id), hasAvailable ? ["available-second"] : []);
    checks.push(hasAvailable ? "Sync bootstraps a missing default from an available model, skipping opted-out rows" : "Sync leaves the default unset when every saved model is disabled");
  } finally {
    await fixture.close();
  }
}

console.log(JSON.stringify({passed: checks.length, boundary: "Actual UI modules, synthetic HTTP and JSDOM; no browser/backend validation", checks}, null, 2));

/** Actual Models host/provider modules with synthetic HTTP in JSDOM.
 * Run: node scripts/model_workspace_scenario.mjs
 * Uses no real browser, accounts, credentials, backend, or workspace data.
 */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import postcss from "postcss";
import { JSDOM } from "jsdom";
import { activate } from "../frontend/plugins/models/src/module.js";
import { activate as activateOpenAI } from "../frontend/plugins/openai_compatible/src/module.js";
import { activate as activateCodex } from "../frontend/plugins/codex/src/module.js";
import { activate as activateOpenCode } from "../frontend/plugins/opencode_go/src/module.js";

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

async function mount(provider, initialCatalog = null, {beforeWrite = () => {}, beforeRead = () => {}} = {}) {
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
      if (init?.method !== "POST") await beforeRead();
      let result = catalog;
      if (_url.endsWith("/discover_saved") || _url.endsWith("/discover")) {
        result = {models: [{kind: provider.id === "openai-compatible" ? null : "chat", model: "fixture-chat", capabilities: {inputModalities: ["text"]}, capabilitySources: {inputModalities: "fixture"}}]};
      } else if (init?.method === "POST") {
        const payload = JSON.parse(init.body);
        commands.push(payload);
        await beforeWrite(payload);
        if (!["start_auth", "cancel_auth"].includes(payload.type)) {
          assert.equal((payload.connection?.expected_revision ?? payload.expected_revision), catalog.revision, "writes use the last committed catalog revision");
        }
        if (payload.type === "create_connection_with_model") {
          const connection = payload.connection;
          const model = payload.model;
          connectionId = connection.connection_id;
          catalog.connections.push({id: connectionId, name: connection.name, driverId: provider.id, availability: "available"});
          catalog.models.push({id: model.model_id, connectionId, kind: model.kind, model: model.model, availability: "available", capabilities: {inputModalities: ["text"]}, capabilitySources: {inputModalities: "fixture"}});
          result = {revision: ++catalog.revision, status: "committed"};
        } else if (payload.type === "start_auth") {
          connectionId = payload.connection_id;
          result = {revision: catalog.revision, status: "pending", attemptId: "fixture-attempt", challenge: {interval: 5}};
        } else if (payload.type === "finish_auth") {
          catalog.connections.push({id: connectionId, name: "Fixture", driverId: provider.id, availability: "available"});
          result = {revision: ++catalog.revision, status: "committed"};
        } else if (payload.type === "sync_models") {
          assert.equal(catalog.connections.length, 1, "sync follows committed authentication");

          result = {revision: ++catalog.revision, status: "committed"};
        } else if (payload.type === "set_default") {
          catalog.roleBindings[payload.role] = payload.model_id;
          result = {revision: ++catalog.revision, status: "committed"};
        } else if (payload.type === "add_model") {
          catalog.models.push({id: payload.model_id, connectionId: payload.connection_id, kind: payload.kind, model: payload.model, availability: "available", capabilities: {inputModalities: ["text"]}, capabilitySources: {inputModalities: "fixture"}});
          result = {revision: ++catalog.revision, status: "committed"};
        } else if (payload.type === "remove_model") {
          catalog.models = catalog.models.filter(model => model.id !== payload.model_id);
          for (const role of Object.keys(catalog.roleBindings)) if (catalog.roleBindings[role] === payload.model_id) delete catalog.roleBindings[role];
          result = {revision: ++catalog.revision, status: "committed"};
        } else if (payload.type === "update_model") {
          const model = catalog.models.find(model => model.id === payload.model_id);
          Object.assign(model.capabilities, {contextWindow: payload.context_window,
            maxOutputTokens: payload.max_output_tokens, inputModalities: payload.image_input ? ["text", "image"] : ["text"]});
          result = {revision: ++catalog.revision, status: "committed"};
        } else if (payload.type === "verify_model") {
          result = {revision: catalog.revision, status: "verified"};
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
    assert.deepEqual(fixture.commands.map(command => command.type), ["start_auth", "finish_auth"]);
    assert.equal(fixture.catalog.models.length, 0, "authentication and discovery never adopt candidates");
    const choice = document.querySelector(".settings-sheet-row input");
    assert.equal(choice.checked, false);
    choice.click();
    document.querySelector(".settings-sheet-foot .settings-primary-button").click();
    await settle();
    assert.deepEqual(fixture.commands.map(command => command.type), ["start_auth", "finish_auth", "add_model", "set_default"]);
    assert.equal(fixture.commands[2].discovery_owned, true);
    assert.equal(fixture.catalog.roleBindings.default, fixture.catalog.models[0].id);
    assert.equal(document.querySelector("dialog[open]"), null);
    checks.push(`${provider.id}: authentication opens unchecked candidates; explicit choice saves and sets the missing default`);
  } finally {
    await fixture.close();
  }
}

for (const manual of [false, true]) {
  const fixture = await mount(providerEntry(activateOpenAI));
  try {
    document.querySelector("[data-providers] button").click();
    await settle();
    const form = document.querySelector(".settings-dialog-form");
    form.elements.name.value = "Fixture";
    form.elements.endpoint.value = "https://fixture.invalid/v1";
    form.elements.apiKey.value = "fixture-key";
    if (manual) {
      document.querySelector("[data-manual]").click();
      form.elements.manualModel.value = "fixture-chat";
      form.elements.manualConfirm.checked = true;
    } else {
      document.querySelector("[data-discover]").click();
      await settle();
      document.querySelector(".settings-sheet-row input").click();
      document.querySelector(".settings-sheet-foot .settings-primary-button").click();
      await settle();
    }
    form.dispatchEvent(new Event("submit", {cancelable: true}));
    await settle();
    const saved = fixture.commands.find(command => command.type === "create_connection_with_model");
    assert.ok(saved);
    assert.equal(saved.model.discovery_owned, !manual, "known directory choices retain discovery ownership; manual choices do not");
    if (!manual) {
      document.querySelector("[data-connections] button").click();
      await settle();
      assert.ok(document.querySelector("[data-sync]"), "selected unknown-purpose directory entries support refresh");
      document.querySelector(".settings-model-toggle").click();
      await settle();
      document.querySelector("[data-probe]").click();
      await settle();
      document.querySelector(".settings-sheet-row input").click();
      document.querySelector(".settings-sheet-foot .settings-primary-button").click();
      await settle();
      assert.equal(fixture.commands.find(command => command.type === "add_model").discovery_owned, true);
    }
    checks.push(`OpenAI-compatible first selection: ${manual ? "manual" : "discovered"} capability ownership`);
  } finally { await fixture.close(); }
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
  document.querySelector("[data-connections] button").click();
  await settle();
  const toggleTarget = document.querySelector(".settings-model-toggle");
  const toggle = toggleTarget.querySelector("input");
  assert.equal(toggle.labels[0], toggleTarget, "the full target labels only its checkbox");
  assert.equal(toggle.getAttribute("aria-label"), "选择 saved-chat");
  const verify = document.querySelector(".settings-model-row button");
  assert.equal(toggleTarget.contains(verify), false);
  verify.click();
  await settle();
  assert.equal(toggle.checked, true, "verification must not toggle availability");
  assert.deepEqual(roleFixture.commands.map(command => command.type), ["verify_model"]);
  toggleTarget.click();
  await settle();
  assert.deepEqual(roleFixture.commands.map(command => command.type), ["verify_model", "remove_model"]);
  assert.equal(roleFixture.catalog.models.length, 0);
  checks.push("Model toggle has a dedicated associated label; verification stays separate and label activation removes the bound model once");
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
    checks.push(hasAvailable ? "Sync bootstraps a missing default from an available model, skipping unavailable rows" : "Sync leaves the default unset when every saved model is disabled");
  } finally {
    await fixture.close();
  }
}

// Selection changes operate only on the discovered scope and reject stale dialogs.
for (const action of ["cancel", "clear", "stale"]) {
  const fixture = await mount(providerEntry(activateCodex), {
    revision: 1, connections: [{id: "saved", name: "Saved", driverId: "codex", availability: "available"}],
    models: ["fixture-chat", "manual-chat"].map(model => ({id: model, connectionId: "saved", kind: "chat", model, availability: "available", capabilities: {inputModalities: ["text"]}, capabilitySources: {inputModalities: "fixture"}})),
    roleBindings: {default: "fixture-chat"}, defaultEmbeddingModelId: null,
  });
  try {
    document.querySelector("[data-connections] button").click();
    await settle();
    document.querySelector("[data-probe]").click();
    await settle();
    const checkbox = document.querySelector(".settings-sheet-row input");
    assert.equal(checkbox.checked, true);
    assert.equal(checkbox.disabled, false, "default references do not lock selection");
    checkbox.click();
    if (action === "stale") fixture.catalog.revision += 1;
    document.querySelector(`.settings-sheet-foot .settings-${action === "cancel" ? "secondary" : "primary"}-button`).click();
    await settle();
    assert.deepEqual(fixture.commands.map(command => command.type), action === "cancel" ? [] : ["remove_model"]);
    assert.equal(fixture.catalog.models.length, action === "clear" ? 1 : 2, "stale CAS and cancel preserve saved models");
    assert.ok(fixture.catalog.models.some(model => model.model === "manual-chat"), "unlisted manual models stay selected");
    if (action === "clear") {
      document.querySelector("[data-probe]").click();
      await settle();
      assert.equal(document.querySelector(".settings-sheet-row input").checked, false, "removed candidates stay unchecked on rediscovery");
      document.querySelector(".settings-sheet-foot .settings-secondary-button").click();
    }
    checks.push(`Model selection ${action}: explicit scope, default references and revision checks`);
  } finally { await fixture.close(); }
}

// Ordinary external providers receive this public API through the same mount
// contract as built-ins. Locked candidates must not depend on checked defaults.
let externalUi;
const externalFixture = await mount({
  id: "external-fixture", label: "External fixture", detail: "Public API fixture",
  render(_host, _view, props) { externalUi = props.ui; return () => {}; },
});
try {
  document.querySelector("[data-providers] button").click();
  await settle();
  const candidates = Object.freeze([
    Object.freeze({kind: "chat", model: "locked"}),
    Object.freeze({kind: "chat", model: "optional"}),
  ]);
  const options = Object.freeze({locked: Object.freeze(["locked"])});
  const before = JSON.stringify({candidates, options});
  const selected = externalUi.pickModels(candidates, options);
  const boxes = [...document.querySelectorAll(".settings-sheet-row input")];
  assert.equal(boxes[0].checked, true);
  assert.equal(boxes[0].disabled, true);
  assert.equal(boxes[1].checked, false);
  const tools = [...document.querySelectorAll(".settings-sheet-tools button")];
  tools.find(button => button.textContent === "全选").click();
  assert.deepEqual(boxes.map(box => box.checked), [true, true]);
  tools.find(button => button.textContent === "全不选").click();
  assert.deepEqual(boxes.map(box => box.checked), [true, false]);
  document.querySelector(".settings-sheet-foot .settings-primary-button").click();
  const result = await selected;
  assert.deepEqual(result, [candidates[0]]);
  assert.equal(result[0], candidates[0], "selection returns the original candidate");
  assert.equal(JSON.stringify({candidates, options}), before, "caller inputs stay immutable");
  assert.deepEqual(externalFixture.commands, [], "picking never saves without a provider action");

  const cancelled = externalUi.pickModels(candidates, options);
  document.querySelector(".settings-sheet-foot .settings-secondary-button").click();
  assert.equal(await cancelled, null);
  assert.equal(document.querySelector(".settings-sheet-scrim"), null);
  assert.equal(JSON.stringify({candidates, options}), before);
  assert.deepEqual(externalFixture.commands, [], "cancelling a locked selection never saves");
  checks.push("External provider pickModels keeps locked candidates checked through bulk actions; confirm/cancel preserve inputs and never auto-save");
} finally {
  await externalFixture.close();
}

// 参数草稿通过真实宿主生命周期验证；写入延迟由 promise 控制。
const editCatalog = {revision: 1,
  connections: [{id: "saved", name: "Saved", driverId: "codex", availability: "available"}],
  models: ["first", "second"].map(id => ({id, connectionId: "saved", kind: "chat", model: id,
    availability: "available", capabilities: {contextWindow: 200000, maxOutputTokens: 32000, inputModalities: ["text"]}, capabilitySources: {inputModalities: "fixture"}})),
  roleBindings: {}, defaultEmbeddingModelId: null};
let releaseWrite, failWrite = false, failRead = false;
const waitingWrite = new Promise(resolve => { releaseWrite = resolve; });
const editFixture = await mount(providerEntry(activateCodex), editCatalog, {
  beforeWrite: async payload => {
    if (payload.type !== "update_model") return;
    await waitingWrite;
    if (failWrite) throw new Error("fixture save failed");
  },
  beforeRead: () => { if (failRead) throw new Error("fixture read failed"); },
});
try {
  assert.ok(document.querySelector("[data-connections] button"), document.body.textContent);
  document.querySelector("[data-connections] button").click();
  await settle();
  const editors = [...document.querySelectorAll(".settings-model-entry")];
  for (const editor of editors) editor.querySelector(".settings-model-expand").click();
  const context = editor => editor.querySelector('[name="contextWindow"]');
  const save = editor => editor.querySelector("[data-detail-save]");
  const fill = (input, value) => { input.value = value; input.dispatchEvent(new Event("input", {bubbles: true})); };
  fill(context(editors[0]), "1M");
  fill(context(editors[1]), "777K");
  const confirmations = [];
  window.confirm = message => { confirmations.push(message); return false; };
  const dialog = document.querySelector("dialog[open]");
  dialog.dispatchEvent(new Event("cancel", {cancelable: true}));
  assert.equal(confirmations.length, 1);
  assert.ok(dialog.open);
  const navigate = new Event("akashic:before-navigate", {cancelable: true});
  window.dispatchEvent(navigate);
  assert.ok(navigate.defaultPrevented);
  const unload = new Event("beforeunload", {cancelable: true});
  window.dispatchEvent(unload);
  assert.ok(unload.defaultPrevented);
  editors[1].querySelector('.settings-model-toggle input').click();
  assert.ok(editors[1].querySelector('.settings-model-toggle input').checked, "cancelled removal keeps the dirty row");
  save(editors[0]).click();
  assert.ok(context(editors[0]).disabled, "submitted values cannot change during the write");
  fill(context(editors[1]), "888K");
  context(editors[1]).focus();
  releaseWrite();
  await settle();
  assert.equal(context(editors[0]).value, "1M");
  assert.ok(save(editors[0]).disabled, "saved row is clean");
  assert.equal(context(editors[1]).value, "888K");
  assert.equal(document.activeElement, context(editors[1]));
  assert.ok(editors.every(editor => editor.isConnected && !editor.querySelector('.settings-model-detail').hidden));
  dialog.dispatchEvent(new Event("cancel", {cancelable: true}));
  assert.ok(dialog.open, "saving one row must not clear another row's leave protection");
  checks.push("Model drafts survive partial saves and keep focus; close/navigation/unload and row removal protect unsaved inputs");

  failWrite = true;
  save(editors[1]).click();
  await settle();
  assert.equal(context(editors[1]).value, "888K");
  assert.match(editors[1].querySelector('[data-detail-status]').textContent, /fixture save failed/);
  assert.equal(editFixture.catalog.models[1].capabilities.contextWindow, 200000);
  assert.equal(save(editors[1]).disabled, false);
  failWrite = false;
  failRead = true;
  save(editors[1]).click();
  await settle();
  assert.equal(context(editors[1]).value, "888K");
  assert.match(editors[1].querySelector('[data-detail-status]').textContent, /参数已提交.*读取最新状态失败/);
  assert.equal(editFixture.catalog.models[1].capabilities.contextWindow, 888000);
  failRead = false;
  window.confirm = () => true;
  dialog.dispatchEvent(new Event("cancel", {cancelable: true}));
  await settle();
  document.querySelector('[data-connections] button').click();
  await settle();
  const reopened = document.querySelectorAll('.settings-model-entry')[1];
  assert.equal(context(reopened).value, "888K");
  checks.push("Save failure retains draft; committed write with failed read reports uncertainty locally and reopening reads the saved value");
} finally { await editFixture.close(); }

for (const activateProvider of [activateOpenAI, activateOpenCode]) {
  const provider = providerEntry(activateProvider);
  const catalog = structuredClone(editCatalog);
  catalog.connections[0].driverId = provider.id;
  const fixture = await mount(provider, catalog);
  try {
    document.querySelector('[data-connections] button').click();
    await settle();
    const edit = async value => {
      const input = document.querySelector('[name="contextWindow"]');
      input.value = value;
      input.dispatchEvent(new Event('input', {bubbles: true}));
      input.dispatchEvent(new Event('change', {bubbles: true}));
      input.dispatchEvent(new window.KeyboardEvent('keydown', {key: 'Enter', bubbles: true, cancelable: true}));
      await settle();
    };
    let confirmations = 0;
    window.confirm = () => { confirmations++; return false; };
    await edit('2M');
    assert.equal(fixture.catalog.models[0].capabilities.contextWindow, 2000000);
    document.querySelector('dialog[open]').dispatchEvent(new Event('cancel', {cancelable: true}));
    await settle();
    assert.equal(confirmations, 0, 'model input must not mark the provider form dirty');
    assert.equal(document.querySelector('dialog[open]'), null);
    document.querySelector('[data-connections] button').click();
    await settle();
    document.querySelector('form [name="name"]').dispatchEvent(new Event('input', {bubbles: true}));
    await edit('3M');
    document.querySelector('dialog[open]').dispatchEvent(new Event('cancel', {cancelable: true}));
    assert.equal(confirmations, 1, 'saving a model must not clear an actual provider draft');
    assert.ok(document.querySelector('dialog[open]'));
    checks.push(`${provider.id}: model/provider drafts remain independent; Enter saves parameters`);
  } finally { await fixture.close(); }
}

const stylesheet = postcss.parse(await readFile(new URL("../frontend/plugins/models/src/style.css", import.meta.url), "utf8"));
const selectorStyle = selector => {
  const declarations = {};
  stylesheet.walkRules(rule => {
    if (rule.selector === selector) rule.walkDecls(declaration => { declarations[declaration.prop] = declaration.value; });
  });
  return declarations;
};
assert.equal(selectorStyle(".settings-page")["--models-hit-target"], "44px");
for (const [selector, properties] of [
  [".settings-model-toggle", ["min-width", "min-height"]],
  [".settings-sheet-row", ["min-height"]],
]) {
  for (const property of properties) assert.equal(selectorStyle(selector)[property], "var(--models-hit-target)");
}
for (const selector of [".settings-role-pick", ".settings-sheet", ".settings-model-manual input", ".settings-sheet-main strong", ".settings-model-main strong"]) {
  for (const [property, value] of Object.entries(selectorStyle(selector))) {
    assert.ok(!value.includes("--md-sys-"), `${selector} ${property} uses the paper contract`);
    if (["font-size", "box-shadow"].includes(property)) assert.ok(value.startsWith("var("));
  }
}
checks.push("Static CSS contract: model toggle and picker rows declare 44px minimum targets and new surfaces use semantic tokens (geometry not browser-measured)");

console.log(JSON.stringify({passed: checks.length, boundary: "Actual UI modules, synthetic HTTP and JSDOM; static CSS only, no browser/backend validation", checks}, null, 2));

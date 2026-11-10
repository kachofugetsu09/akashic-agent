/** Manual scenario: actual models module + delayed synthetic HTTP in JSDOM.
 * Run: node scripts/model_settings_refresh_scenario.mjs
 * No real backend, browser layout, account writes, or new unit-test suite.
 */
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";
import { activate } from "../frontend/plugins/models/src/module.js";

const checks = [];
const settle = () => new Promise(resolve => setTimeout(resolve, 0));
const catalog = (model = "a", revision = 1) => ({
  revision, connections: [{id: "fixture", name: "Fixture", driverId: "fixture", availability: "available"}],
  models: ["a", "b"].map(id => ({id, model: id, connectionId: "fixture", kind: "chat", availability: "available",
    capabilities: {inputModalities: ["text"]}, capabilitySources: {inputModalities: "declared"}})),
  roleBindings: {default: model, agent: "a", fast: "a"}, defaultEmbeddingModelId: null,
});

async function mount() {
  const dom = new JSDOM('<main id="host"></main>', {url: "http://model-settings.local"});
  for (const name of ["window", "document", "Option", "Event", "HTMLElement"]) globalThis[name] = dom.window[name];
  globalThis.IntersectionObserver = class { observe() {} disconnect() {} };
  dom.window.HTMLElement.prototype.getClientRects = () => [{width: 100, height: 100}];
  // JSDOM has no native top layer. Exercise explicit dialog close events only.
  dom.window.HTMLDialogElement.prototype.showModal = function () { this.open = true; };
  dom.window.HTMLDialogElement.prototype.close = function () {
    if (!this.open) return;
    this.open = false;
    queueMicrotask(() => this.dispatchEvent(new Event("close")));
  };
  const requests = [];
  const dirty = [];
  let entry;
  activate({
    ui: {inject: (_id, mount) => mount({register: value => { entry = value; return () => {}; }})},
    http: {request: (url, init) => new Promise((resolve, reject) => requests.push({
      url, init, reject,
      // Ignore abort deliberately: stale-result guards must also hold after physical completion.
      resolve: (data, status = 200) => resolve(new Response(JSON.stringify(data), {status, headers: {"content-type": "application/json"}})),
    }))},
  });
  const dispose = entry.render(document.querySelector("#host"), {child: () => ({entries: []})}, {dirty: value => dirty.push(value)});
  requests[0].resolve(catalog()); await settle();
  const pickers = () => [...document.querySelectorAll("[data-bindings] .settings-role-pick")];
  const picker = () => pickers()[0];
  const displayed = () => picker().querySelector(".settings-role-pick-value").textContent.split(" · ")[0];
  const change = async value => {
    picker().click();
    const choice = [...document.querySelectorAll(".settings-sheet-row")]
      .find(row => row.querySelector("strong").textContent === value);
    assert.ok(choice, `the chooser exposes ${value}`);
    choice.click();
    await settle();
    assert.equal(document.querySelector("dialog[open]"), null);
    assert.equal(JSON.parse(posts().at(-1).init.body).model_id, value);
  };
  const focus = () => window.dispatchEvent(new Event("focus"));
  const search = () => document.querySelector("[data-search] input").dispatchEvent(new Event("input"));
  const posts = () => requests.filter(request => request.init?.method === "POST");
  const error = () => document.querySelector("[data-error]");
  return {dom, requests, dirty, dispose, picker, pickers, displayed, change, focus, search, posts, error};
}

let fixture;
try {
  fixture = await mount();
  fixture.focus(); const oldRead = fixture.requests.at(-1);
  await fixture.change("b"); const save = fixture.requests.at(-1); const original = fixture.picker();
  assert.equal(oldRead.init.signal.aborted, true);
  fixture.focus(); fixture.search();
  assert.equal(fixture.requests.at(-1), save);
  oldRead.resolve(catalog("a", 0)); await settle();
  // The sheet closes on selection; the button keeps the last confirmed label
  // until readback, while the submitted B remains locked and cannot be repeated.
  assert.equal(fixture.picker(), original); assert.equal(fixture.displayed(), "a");
  assert.equal(JSON.parse(save.init.body).model_id, "b");
  assert.ok(fixture.pickers().every(select => select.disabled));
  assert.match(document.querySelector("[data-bindings]").textContent, /正在保存/);
  original.click(); assert.equal(fixture.posts().length, 1);
  const navigation = new Event("akashic:before-navigate", {cancelable: true});
  window.dispatchEvent(navigation); assert.equal(navigation.defaultPrevented, false);
  save.resolve({status: "applied", revision: 2}); await settle();
  const verification = fixture.requests.at(-1);
  fixture.search(); assert.equal(fixture.picker(), original); assert.equal(original.disabled, true);
  verification.resolve(catalog("b", 2)); await settle();
  assert.equal(fixture.displayed(), "b"); assert.equal(fixture.picker().disabled, false);
  assert.match(document.querySelector("[data-toast-region]").textContent, /保存请求已完成/);
  assert.deepEqual(fixture.dirty, []);
  checks.push("Pending save survives focus, search and late pre-command reads; duplicates blocked without trapping navigation; actual catalog unlocks controls");
  fixture.dispose(); fixture.dom.window.close();

  for (const rejected of [false, true]) {
    fixture = await mount(); await fixture.change("b");
    if (rejected) fixture.requests.at(-1).resolve({detail: "fixture conflict"}, 409);
    else fixture.requests.at(-1).reject(new Error("fixture lost response"));
    await settle();
    assert.equal(fixture.displayed(), "a"); assert.equal(fixture.picker().disabled, true);
    assert.equal(JSON.parse(fixture.posts()[0].init.body).model_id, "b");
    fixture.requests.at(-1).resolve(catalog(rejected ? "a" : "b", 2)); await settle();
    assert.equal(fixture.displayed(), rejected ? "a" : "b");
    assert.equal(fixture.picker().disabled, false);
    assert.match(fixture.error().textContent, /保存请求未成功确认/);
    assert.equal(document.querySelector("[data-toast-region]").textContent, "");
    assert.equal(fixture.posts().length, 1);
    checks.push(rejected ? "409 reconciles actual old binding without automatic resubmit" : "Lost POST response may have committed: readback displays actual B without false failure rollback or success claim");
    fixture.dispose(); fixture.dom.window.close();
  }

  fixture = await mount(); await fixture.change("b");
  fixture.requests.at(-1).resolve({status: "applied", revision: 2}); await settle();
  fixture.requests.at(-1).resolve({detail: "fixture unavailable"}, 503); await settle();
  fixture.search(); assert.equal(fixture.displayed(), "a"); assert.equal(fixture.picker().disabled, true);
    assert.equal(JSON.parse(fixture.posts()[0].init.body).model_id, "b");
  assert.match(fixture.error().textContent, /尚未核对/);
  fixture.error().querySelector("button").click(); const staleVerification = fixture.requests.at(-1);
  fixture.focus(); const latest = fixture.requests.at(-1);
  assert.equal(staleVerification.init.signal.aborted, true);
  latest.resolve(catalog("b", 2)); await settle();
  staleVerification.resolve(catalog("a", 1)); await settle();
  assert.equal(fixture.displayed(), "b"); assert.equal(fixture.picker().disabled, false);
  assert.equal(fixture.error().hidden, true); assert.equal(fixture.posts().length, 1);
  checks.push("Failed verification retains the submitted choice, current-label lock and retry; superseded read cannot restore old binding");
  await fixture.change("a"); assert.equal(fixture.posts().length, 2);
  assert.equal(document.querySelector("[data-toast-region]").textContent, "");
  assert.equal(JSON.parse(fixture.requests.at(-1).init.body).expected_revision, 2);
  checks.push("Next explicit save uses authoritative revision after reconciliation");
  const lateSave = fixture.requests.at(-1); const count = fixture.requests.length;
  fixture.dispose(); const dirtyCount = fixture.dirty.length;
  lateSave.resolve({status: "applied", revision: 3}); await settle();
  assert.equal(fixture.requests.length, count); assert.equal(fixture.dirty.length, dirtyCount);
  fixture.dom.window.close(); fixture = null;
  checks.push("Unmount during POST ignores late completion and performs no new read or parent-state update");
  fixture = await mount();
  assert.equal(fixture.displayed(), "a"); assert.equal(fixture.picker().disabled, false);
  await fixture.change("b"); fixture.requests.at(-1).resolve({status: "applied", revision: 2}); await settle();
  const abandonedRead = fixture.requests.at(-1);
  fixture.dispose(); assert.equal(abandonedRead.init.signal.aborted, true);
  abandonedRead.resolve(catalog("b", 2)); await settle();
  assert.equal(document.querySelector("#host").textContent, "");
  assert.deepEqual(fixture.dirty, []); fixture.dom.window.close(); fixture = null;
  checks.push("Reentry loads actual catalog without old lock; unmount during verification ignores late catalog");
  console.log(JSON.stringify({passed: checks.length, boundary: "Actual module in JSDOM; delayed synthetic HTTP, no real browser/backend", checks}, null, 2));
} finally {
  if (fixture) { fixture.dispose(); fixture.dom.window.close(); }
}

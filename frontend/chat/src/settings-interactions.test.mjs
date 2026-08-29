import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { applyConnection, cancelConnectionAuth, createConnectionDraft, groupConnections, loadSettingsState } from "./settings-data.ts";

const app = await readFile(new URL("./settings-app.tsx", import.meta.url), "utf8");
const dialog = await readFile(new URL("./settings-connection-dialog.tsx", import.meta.url), "utf8");
const connection = await readFile(new URL("./use-settings-connection.ts", import.meta.url), "utf8");
const data = await readFile(new URL("./settings-data.ts", import.meta.url), "utf8");

const runtime = (id, sourceId, sourceName) => ({
  id, sourceId, sourceName, provider: "fixture", model: id, baseUrl: "https://example.com",
  catalogProvider: "fixture", contextWindow: 1, maxOutputTokens: 1, inputModalities: ["text"],
  reasoningEffort: "medium", supportedReasoningEfforts: ["medium"],
  credential: { id: `credential-${id}`, configured: true, source: "workspace" },
});

test("settings connection grouping is pure and preserves source ownership", () => {
  const groups = groupConnections([
    runtime("model-a", "source-a", "账号 A"),
    runtime("model-b", "source-a", "账号 A"),
    runtime("model-c", "source-c", "账号 C"),
  ], "model-b");
  assert.deepEqual(groups.map((group) => [group.sourceId, group.runtimes.map((item) => item.id)]), [
    ["source-a", ["model-a", "model-b"]],
  ]);
});

test("editing a connection never projects a stored credential secret", () => {
  const existing = groupConnections([runtime("model-a", "source-a", "账号 A")], "")[0];
  const draft = createConnectionDraft({ kind: "api", provider: "fixture", name: "Fixture", detail: "", baseUrl: "" }, existing);
  assert.equal(draft.apiKey, "");
  assert.equal(draft.credentialId, "credential-model-a");
});

test("editing an API connection keeps its private endpoint when unchanged", async () => {
  const commands = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (_url, init = {}) => {
    if (!init.body) {
      return new Response(JSON.stringify({
        revision: 7,
        connections: [{
          id: "source-a", name: "Account A", driverId: "openai-compatible",
          authIdentity: "account-a", availability: "available",
        }],
        models: [{
          id: "chat-a", connectionId: "source-a", kind: "chat", model: "wire-a",
          defaultReasoningEffort: null, availability: "available",
          capabilities: {
            contextWindow: 64000, maxOutputTokens: null, inputModalities: ["text"],
            supportedReasoningEfforts: [], embeddingDimensions: null,
          },
        }],
        roleBindings: { default: "chat-a" },
        defaultEmbeddingModelId: null,
      }), { status: 200 });
    }
    const command = JSON.parse(init.body);
    commands.push(command);
    return new Response(JSON.stringify({
      revision: 8, status: "committed", attemptId: null, challenge: null,
    }), { status: 200 });
  };
  try {
    const state = await loadSettingsState();
    const existing = groupConnections(state.runtimes, "")[0];
    const draft = createConnectionDraft(
      { kind: "api", provider: "", name: "Custom API", detail: "", baseUrl: "" },
      existing,
    );
    assert.equal(draft.baseUrl, "");
    await applyConnection(draft, state, new AbortController().signal);
  } finally {
    globalThis.fetch = originalFetch;
  }
  assert.equal(commands.length, 1);
  assert.equal(commands[0].type, "update_connection");
  assert.equal("endpoint" in commands[0], false);
});

test("settings page delegates modal form and transport lifecycle", () => {
  assert.match(app, /<SettingsConnectionDialog/);
  assert.doesNotMatch(app, /createPortal|settings-dialog-body|startCodexLogin|discoverConnectionModels/);
  assert.match(dialog, /<Dialog open/);
  assert.match(dialog, /onCloseAutoFocus/);
  assert.match(connection, /if \(saveRef\.current\) return/);
  assert.match(connection, /if \(loginRef\.current \|\| codexLogin\?\.status === "waiting"\) return/);
  assert.match(connection, /controller\.abort\(\)/);
  assert.match(connection, /cancelConnectionAuth\(loginAttemptRef\.current\)/);
});

test("Radix owns dialog title and description identities", () => {
  assert.match(dialog, /<DialogTitle>\{title\}<\/DialogTitle>/);
  assert.doesNotMatch(dialog, /<DialogTitle id=/);
});

test("all model reads and writes cross the plugin control surface", () => {
  assert.match(data, /\/api\/settings\/model\/catalog/);
  assert.match(data, /\/api\/settings\/model\/command/);
  assert.doesNotMatch(data, /\/api\/settings\/(?:state|models|apply|roles|embedding-models|codex-login)/);
  assert.match(data, /item\.kind === "embedding" && item\.availability === "available"/);
});

test("editing OpenCode reauthenticates with the submitted fields before sync", async () => {
  const commands = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (_url, init) => {
    const command = JSON.parse(init.body);
    commands.push(command);
    const receipt = command.type === "start_auth"
      ? { revision: 7, status: "pending", attemptId: "attempt-a", challenge: {} }
      : { revision: command.type === "finish_auth" ? 8 : 9, status: "committed", attemptId: null, challenge: null };
    return new Response(JSON.stringify(receipt), { status: 200 });
  };
  try {
    await applyConnection({
      sourceId: "opencode-a", sourceName: "新名称", kind: "opencode-go", provider: "opencode-go",
      baseUrl: "https://new.example/v1", apiKey: "new-secret", credentialId: "account-a",
      model: "", reasoningEffort: "",
    }, {
      modelRevision: 7,
      catalog: {
        revision: 7,
        connections: [{ id: "opencode-a", name: "旧名称", driverId: "opencode-go", authIdentity: "account-a", availability: "available" }],
        models: [], roleBindings: { default: "chat-a" }, defaultEmbeddingModelId: null,
      },
    }, new AbortController().signal);
  } finally {
    globalThis.fetch = originalFetch;
  }
  assert.deepEqual(commands.map((item) => item.type), ["start_auth", "finish_auth", "sync_models"]);
  assert.deepEqual(commands[0].input, {
    api_key: "new-secret", endpoint: "https://new.example/v1", name: "新名称", auth_identity: "account-a",
  });
});

test("closing a waiting login sends one keepalive cancellation", async () => {
  let request;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    request = { url, init };
    return new Response(JSON.stringify({ revision: 7, status: "cancelled" }), { status: 200 });
  };
  try {
    await cancelConnectionAuth("attempt-a");
  } finally {
    globalThis.fetch = originalFetch;
  }
  assert.equal(request.url, "/api/settings/model/command");
  assert.equal(request.init.keepalive, true);
  assert.deepEqual(JSON.parse(request.init.body), { type: "cancel_auth", attempt_id: "attempt-a" });
});

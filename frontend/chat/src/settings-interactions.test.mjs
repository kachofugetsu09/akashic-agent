import assert from "node:assert/strict";
import test from "node:test";

import {
  applyConnection,
  availableChatModels,
  cancelConnectionAuth,
  createConnectionDraft,
  groupConnections,
  loadModelCatalog,
} from "./settings-data.ts";

const model = (id, sourceId, sourceName) => ({
  id,
  sourceId,
  sourceName,
  provider: "fixture",
  model: id,
  baseUrl: "https://example.com",
  reasoningEffort: "medium",
  credentialId: `credential-${id}`,
});

test("settings connection grouping is pure and preserves source ownership", () => {
  const groups = groupConnections([
    model("model-a", "source-a", "账号 A"),
    model("model-b", "source-a", "账号 A"),
    model("model-c", "source-c", "账号 C"),
  ], "model-b");
  assert.deepEqual(groups.map((group) => [group.sourceId, group.models.map((item) => item.id)]), [
    ["source-a", ["model-a", "model-b"]],
  ]);
});

test("editing a connection never projects a stored credential secret", () => {
  const existing = groupConnections([model("model-a", "source-a", "账号 A")], "")[0];
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
    const catalog = await loadModelCatalog();
    const existing = groupConnections(availableChatModels(catalog), "")[0];
    const draft = createConnectionDraft(
      { kind: "api", provider: "", name: "Custom API", detail: "", baseUrl: "" },
      existing,
    );
    assert.equal(draft.baseUrl, "");
    await applyConnection(draft, catalog, new AbortController().signal);
  } finally {
    globalThis.fetch = originalFetch;
  }
  assert.equal(commands.length, 1);
  assert.equal(commands[0].type, "update_connection");
  assert.equal("endpoint" in commands[0], false);
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
      revision: 7,
      connections: [{ id: "opencode-a", name: "旧名称", driverId: "opencode-go", authIdentity: "account-a", availability: "available" }],
      models: [], roleBindings: { default: "chat-a" }, defaultEmbeddingModelId: null,
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

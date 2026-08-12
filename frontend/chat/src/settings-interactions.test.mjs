import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { createConnectionDraft, groupConnections } from "./settings-data.ts";

const app = await readFile(new URL("./settings-app.tsx", import.meta.url), "utf8");
const dialog = await readFile(new URL("./settings-connection-dialog.tsx", import.meta.url), "utf8");
const connection = await readFile(new URL("./use-settings-connection.ts", import.meta.url), "utf8");

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

test("settings page delegates modal form and transport lifecycle", () => {
  assert.match(app, /<SettingsConnectionDialog/);
  assert.doesNotMatch(app, /createPortal|settings-dialog-body|startCodexLogin|discoverConnectionModels/);
  assert.match(dialog, /<Dialog open/);
  assert.match(dialog, /onCloseAutoFocus/);
  assert.match(connection, /if \(discoverRef\.current\) return/);
  assert.match(connection, /if \(loginRef\.current\) return/);
  assert.match(connection, /controller\.abort\(\)/);
});

test("Radix owns dialog title and description identities", () => {
  assert.match(dialog, /<DialogTitle>\{title\}<\/DialogTitle>/);
  assert.doesNotMatch(dialog, /<DialogTitle id=/);
});

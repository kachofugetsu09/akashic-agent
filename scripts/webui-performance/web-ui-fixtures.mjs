import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const MODULES = [
  {
    pluginId: "shell-ui",
    directory: "shell_ui",
    requires: ["web.root.v1", "shell.rail-actions.v1"],
    provides: ["shell.pages.v1", "shell.rail-actions.v1", "shell.settings.v1", "shell.settings-plugins.v1"],
    contractDigests: {
      "shell.rail-actions.v1": "911133603ab7d50f616775e0352f90506345d61ee12d1865a63d18a656bb045c",
      "shell.settings.v1": "1f4dd2eaee9118c799590a36745975150c46979a3f62b431f73aa29e949ce189",
      "shell.settings-plugins.v1": "1823b19c778297893495ca9193d688e24c99d079effbaf5ecf0d1148ae0a1aa2",
    },
  },
  {
    pluginId: "conversation-ui",
    directory: "conversation_ui",
    requires: ["shell.pages.v1"],
    provides: ["conversation.tools.v2"],
    contractDigests: {
      "conversation.tools.v2": "dcbf5db629d4fa6fed4019ad91663f9af94c5b9a02f9e67c94b0745e40f19049",
    },
  },
  {
    pluginId: "models",
    directory: "models",
    requires: ["shell.settings.v1"],
    provides: ["models.connection-types.v1"],
    contractDigests: {
      "models.connection-types.v1": "8c304d85090a65a4a66cd777a5c2a88e6908de147964dc91e7f84336979ea561",
      "shell.settings.v1": "1f4dd2eaee9118c799590a36745975150c46979a3f62b431f73aa29e949ce189",
    },
  },
  {
    pluginId: "onboarding",
    directory: "onboarding",
    requires: ["shell.settings.v1"],
    provides: [],
    contractDigests: {
      "shell.settings.v1": "1f4dd2eaee9118c799590a36745975150c46979a3f62b431f73aa29e949ce189",
    },
  },
];

export function createWebUiBootstrap(repoRoot) {
  const modules = MODULES.map((descriptor) => {
    const root = resolve(repoRoot, "plugins", descriptor.directory);
    const module = readFileSync(resolve(root, "web_module.js"), "utf8");
    const stylesheet = readFileSync(resolve(root, "web_module.css"), "utf8");
    const contract = JSON.stringify({
      contractDigests: descriptor.contractDigests,
      provides: descriptor.provides,
      requires: descriptor.requires,
    });
    return {
      ...descriptor,
      generationId: "fixture-generation",
      module,
      moduleSha256: sha256(module),
      moduleBytes: Buffer.byteLength(module),
      stylesheet,
      stylesheetSha256: sha256(stylesheet),
      stylesheetBytes: Buffer.byteLength(stylesheet),
      contractSha256: sha256(contract),
    };
  });
  const catalogId = sha256(JSON.stringify(modules));
  return {
    bootstrap: { schemaVersion: 1, snapshotId: "fixture-snapshot", catalogId, modules },
    state: { snapshotId: "fixture-snapshot", catalogId, updating: false },
  };
}

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

import type { ComposerSuggestProvider, ComposerSuggestion, PluginUiDefinition } from "@akashic/chat-ui-v1";

function readCommand(row: unknown): ComposerSuggestion {
  if (!row || typeof row !== "object") throw new Error("命令目录格式错误");
  const value = row as Record<string, unknown>;
  if (typeof value.name !== "string" || !value.name || typeof value.description !== "string"
    || (value.input_hint !== null && typeof value.input_hint !== "string")) {
    throw new Error("命令目录格式错误");
  }
  return { token: value.name, label: value.name,
    description: value.description + (value.input_hint ? ` · ${value.input_hint}` : "") };
}

const commands: ComposerSuggestProvider = {
  keepOpenOnMatch: true,
  label: "命令", sigils: ["/"], prefix: "/", empty: "没有可用命令",
  async load(context) {
    const payload = await context.query("commands.list", {}, { cache: "none" });
    if (!Array.isArray(payload.items)) throw new Error("命令目录格式错误");
    return payload.items.map(readCommand);
  },
};

const definition: PluginUiDefinition = { slots: { "composer.suggest": commands } };
export default definition;

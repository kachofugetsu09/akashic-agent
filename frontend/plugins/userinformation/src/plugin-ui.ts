import type { PluginUiDefinition, ToolIconFacts } from "@akashic/chat-ui-v1";

const PROVIDER_ICONS: Record<string, string> = {
  fitbit: "heart-pulse",
  steam: "gamepad-2",
  watch: "map-pin",
  location: "map-pin",
};

function resolveToolIcon(block: ToolIconFacts): string | null {
  if (block.name !== "user_information_query") return null;
  if (!block.arguments || typeof block.arguments !== "object" || Array.isArray(block.arguments)) return null;
  const args = block.arguments as { provider?: unknown; query?: unknown };
  if (typeof args.provider !== "string") return null;
  const key = args.provider.split("@")[0];
  if (key === "fitbit" && typeof args.query === "string" && args.query.includes("sleep")) {
    return "moon";
  }
  return PROVIDER_ICONS[key] ?? null;
}

const definition: PluginUiDefinition = {
  slots: {
    "tool.icon": { resolve: resolveToolIcon },
  },
};

export default definition;

import type { ComposerSuggestProvider, ComposerSuggestion, PluginUiDefinition } from "@akashic/chat-ui-v1";

const SOURCE_LABEL: Record<string, string> = { builtin: "内置", workspace: "工作区", user: "我的" };

// 技能目录的一行；字段来自本插件的 skills.list。
function readSkill(row: unknown): ComposerSuggestion {
  if (!row || typeof row !== "object") throw new Error("技能目录格式错误");
  const value = row as Record<string, unknown>;
  if (typeof value.name !== "string" || !value.name || typeof value.available !== "boolean"
    || !["display_name", "description", "source", "missing"].every((key) => typeof value[key] === "string")) {
    throw new Error("技能目录格式错误");
  }
  const source = value.source as string;
  return { token: value.name, label: value.display_name as string || value.name, description: value.description as string,
    unavailable: value.available ? undefined : value.missing as string || "依赖缺失", badge: SOURCE_LABEL[source] ?? source };
}

// 输入 / 或 @ 选技能；发出的 /技能名 由本插件后端识别并把正文并入当轮。
const skills: ComposerSuggestProvider = {
  label: "技能", sigils: ["/", "@"], prefix: "/", empty: "没有已安装的技能",
  async load(context) {
    const payload = await context.query("skills.list", {}, { cache: "none" });
    if (!Array.isArray(payload.items)) throw new Error("技能目录格式错误");
    return payload.items.map(readSkill);
  },
};

const definition: PluginUiDefinition = {
  slots: {
    "composer.suggest": skills,
  },
};
export default definition;

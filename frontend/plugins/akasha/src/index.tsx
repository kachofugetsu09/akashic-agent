import { registerForm } from "../../shared/src/configuration";
import type { WebHostContextV1, WebUiDisposer } from "@akashic/web-ui-v1";

// Akasha 的 Web 入口只有设置分节；每轮召回展示由 message_ui.js 经插件界面插槽提供。
export function activate(ctx: WebHostContextV1): WebUiDisposer {
  return registerForm(ctx, {id: "akasha", title: "长期情景记忆", description: "在对话互动中沉淀情景记忆与关联；关闭后已有记忆仍保留，但暂停学习。"});
}

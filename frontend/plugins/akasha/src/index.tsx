import { registerForm, Field } from "../../shared/src/configuration";
import type { WebHostContextV1, WebUiDisposer } from "@akashic/web-ui-v1";

// Akasha 的 Web 入口只有设置分节；每轮召回展示由 message_ui.js 经插件界面插槽提供。
// 学习动力学参数改动后已发布的记忆图不能连续发布，表单只放召回侧参数。
export function activate(ctx: WebHostContextV1): WebUiDisposer {
  return registerForm(ctx, {id: "akasha", title: "长期情景记忆", description: "在对话互动中沉淀情景记忆与关联；关闭后已有记忆仍保留，但暂停学习。", family: "memory", familyLabel: "记忆",
    fields: ({values, change, status}) => <details><summary>召回参数</summary>
      <Field label="每轮最多召回条数" name="context_recall_limit" type="number" inputMode="numeric" min={1} max={40} step={1} value={values.context_recall_limit} fallback={status.defaults?.context_recall_limit} change={change} hint="1–40；调小能减少无关记忆" />
      <Field label="每轮注入记忆的字数上限" name="inject_max_chars" type="number" inputMode="numeric" min={1} step={1} value={values.inject_max_chars} fallback={status.defaults?.inject_max_chars} change={change} hint="召回内容超出时截断；调大会占用更多上下文" />
    </details>,
  });
}

import type { WebHostContextV1 } from "@akashic/web-ui-v1";
import { registerForm, Field, Check } from "../../shared/src/configuration";

// MEMORY.md / SELF.md 的篇幅、按话题浮现与定期整理；留空即用默认值。
export function activate(ctx: WebHostContextV1) {
  return registerForm(ctx, {id: "markdown_memory", title: "长期记忆文档", description: "把对话里值得长期记住的事提炼进 MEMORY.md、SELF.md 和主题文档。", family: "memory", familyLabel: "记忆",
    switchable: false,
    fields: ({values, change, status}) => <>
      <Field label="用户记忆篇幅（字）" name="memory_budget" type="number" inputMode="numeric" min={1} step={1} value={values.memory_budget} fallback={status.defaults?.memory_budget} change={change} hint="MEMORY.md 三个用户章节合计；超出时会合并精简旧条目" />
      <Field label="每轮最多浮现的主题文档" name="surface_limit" type="number" inputMode="numeric" min={0} step={1} value={values.surface_limit} fallback={status.defaults?.surface_limit} change={change} hint="和当前话题相关的主题文档会自动带进上下文；填 0 关闭" />
      <details><summary>浮现与整理</summary>
        <Field label="浮现的最低相关度" name="surface_min_score" type="number" inputMode="decimal" min={0} step="any" value={values.surface_min_score} fallback={status.defaults?.surface_min_score} change={change} hint="调高只带最相关的文档" />
        <Field label="浮现内容字数上限" name="surface_chars" type="number" inputMode="numeric" min={1} step={1} value={values.surface_chars} fallback={status.defaults?.surface_chars} change={change} />
        <Check label="定期整理" name="consolidate" value={values.consolidate} fallback={status.defaults?.consolidate} change={change} hint="把重复或相互矛盾的旧条目合并为一条，不引入新事实" />
        <Field label="整理间隔（小时）" name="consolidate_hours" type="number" inputMode="numeric" min={1} step={1} value={values.consolidate_hours} fallback={status.defaults?.consolidate_hours} change={change} />
        <Field label="累计多少条新草稿才整理" name="consolidate_min_drafts" type="number" inputMode="numeric" min={1} step={1} value={values.consolidate_min_drafts} fallback={status.defaults?.consolidate_min_drafts} change={change} />
      </details></>,
  });
}

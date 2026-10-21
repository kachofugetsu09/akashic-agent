import type { WebHostContextV1 } from "@akashic/web-ui-v1";
import { registerForm, Field } from "../../shared/src/configuration";

// 对话回复的运行参数；留空即用默认值。
export function activate(ctx: WebHostContextV1) {
  return registerForm(ctx, {id: "reply", title: "对话回复", description: "普通对话里一轮回复最多走多少步、每次输出多长。", family: "agent", familyLabel: "对话与智能体",
    switchable: false,
    fields: ({values, change, status}) => <>
      <Field label="每轮最多步数" name="max_steps" type="number" min={0} step={1} inputMode="numeric" value={values.max_steps} fallback={status.defaults?.max_steps ?? "不限"} change={change} hint="一步是一次模型输出；填 0 不限制。复杂任务被提前打断时调大" />
      <Field label="单次输出 token 上限" name="max_output_tokens" type="number" min={1} step={1} inputMode="numeric" value={values.max_output_tokens} fallback={status.defaults?.max_output_tokens ?? "不限"} change={change} hint="留空不额外限制，由模型自身上限决定" />
    </>,
  });
}

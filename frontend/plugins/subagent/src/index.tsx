import type { WebHostContextV1 } from "@akashic/web-ui-v1";
import { registerForm, Field } from "../../shared/src/configuration";

// 子任务的运行参数；留空即用默认值。
export function activate(ctx: WebHostContextV1) {
  return registerForm(ctx, {id: "subagent", title: "子任务", description: "模型派出的独立子任务：同时能跑几个、每个最多走多少步。", family: "agent", familyLabel: "对话与智能体",
    switchable: false,
    fields: ({values, change, status}) => <>
      <Field label="同时运行的子任务数" name="max_active" type="number" min={1} max={20} step={1} inputMode="numeric" value={values.max_active} fallback={status.defaults?.max_active ?? "不限"} change={change} hint="1–20；超出时新的子任务会被拒绝" />
      <Field label="每个子任务最多步数" name="max_steps" type="number" min={1} step={1} inputMode="numeric" value={values.max_steps} fallback={status.defaults?.max_steps ?? "不限"} change={change} hint="一步是一次模型输出" />
      <Field label="单次输出 token 上限" name="max_output_tokens" type="number" min={1} step={1} inputMode="numeric" value={values.max_output_tokens} fallback={status.defaults?.max_output_tokens ?? "不限"} change={change} hint="留空不额外限制，由模型自身上限决定" />
      <Field label="子任务结果字数上限" name="result_chars" type="number" min={1000} max={1000000} step={1} inputMode="numeric" value={values.result_chars} fallback={status.defaults?.result_chars} change={change} hint="前台子任务交回的结果超出时截断" />
    </>,
  });
}

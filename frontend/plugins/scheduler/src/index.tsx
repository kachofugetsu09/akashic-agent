import type { WebHostContextV1 } from "@akashic/web-ui-v1";
import { registerForm, Field } from "../../shared/src/configuration";

// 定时任务的运行参数；留空即用默认值。
export function activate(ctx: WebHostContextV1) {
  return registerForm(ctx, {id: "scheduler", title: "定时任务", description: "到点自动执行的任务：最多保留几个、每次执行最多走多少步。", family: "agent", familyLabel: "对话与智能体",
    switchable: false,
    fields: ({values, change, status}) => <>
      <Field label="活动定时任务上限" name="max_active_jobs" type="number" min={1} max={200} step={1} inputMode="numeric" value={values.max_active_jobs} fallback={status.defaults?.max_active_jobs ?? "不限"} change={change} hint="1–200；达到上限后新建会提示先移除旧任务" />
      <Field label="每次执行最多步数" name="max_steps" type="number" min={1} step={1} inputMode="numeric" value={values.max_steps} fallback={status.defaults?.max_steps ?? "不限"} change={change} hint="一步是一次模型输出" />
      <Field label="单次输出 token 上限" name="max_output_tokens" type="number" min={1} step={1} inputMode="numeric" value={values.max_output_tokens} fallback={status.defaults?.max_output_tokens ?? "不限"} change={change} hint="留空不额外限制，由模型自身上限决定" />
    </>,
  });
}

import type { WebHostContextV1 } from "@akashic/web-ui-v1";
import { registerForm, Field, Select } from "../../shared/src/configuration";

// 后台命令、子任务结束后怎样回到原会话；留空即用默认值。
export function activate(ctx: WebHostContextV1) {
  return registerForm(ctx, {id: "jobs", title: "后台任务通知", description: "后台命令、子任务结束时，结果怎样交回原来的对话。", family: "agent", familyLabel: "对话与智能体",
    switchable: false,
    fields: ({values, change, status}) => <>
      <Select label="空闲时任务完成" name="completion_delivery" value={values.completion_delivery} change={change} fallbackLabel="默认（唤醒原会话处理结果）"
        options={[{value: "wakeup", label: "唤醒原会话处理结果"}, {value: "quiet", label: "不唤醒，下次对话时再告知"}]} hint="对话进行中完成的任务总会并入当前回合" />
      <Field label="连续唤醒上限" name="max_consecutive_wakes" type="number" inputMode="numeric" min={1} step={1} value={values.max_consecutive_wakes} fallback={status.defaults?.max_consecutive_wakes ?? "不限"} change={change} hint="任务接连完成时，最多连续自动唤醒几次；你发言后重新计数" />
    </>,
  });
}

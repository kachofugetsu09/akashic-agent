import type { WebHostContextV1 } from "@akashic/web-ui-v1";
import { registerForm, Field, ModelField } from "../../shared/src/configuration";

// 上下文压缩的运行参数；留空即用默认值。
export function activate(ctx: WebHostContextV1) {
  return registerForm(ctx, {id: "compaction", title: "上下文压缩", description: "对话变长后，较早的内容会压缩成摘要；这里决定保留多少最近原文。", family: "agent", familyLabel: "对话与智能体",
    switchable: false,
    fields: ({values, change, status}) => <>
      <ModelField label="摘要模型" name="model" value={values.model} change={change} status={status} hint="跟随对话时窗口与对话一致；换成窗口更小的模型，一次能摘要的原文会变少" />
      <Field label="保留最近原文（token）" name="keep_recent_tokens" type="number" min={1} step={1} inputMode="numeric" value={values.keep_recent_tokens} fallback={status.defaults?.keep_recent_tokens ?? "不限"} change={change} hint="调大保留更多细节，但每轮消耗更多上下文" />
      <Field label="空闲压缩等待时长（分钟）" name="idle_trigger_minutes" type="number" min={0} step={1} inputMode="numeric" value={values.idle_trigger_minutes} fallback={status.defaults?.idle_trigger_minutes ?? 60} change={change} hint="距离上一轮结束超过此时长（服务端 KV 缓存已过期）且达到下方体积时，在新请求前提前压缩；填 0 关闭" />
      <Field label="空闲压缩触发体积（token）" name="idle_trigger_tokens" type="number" min={1} step={1} inputMode="numeric" value={values.idle_trigger_tokens} fallback={status.defaults?.idle_trigger_tokens ?? 200000} change={change} hint="闲置唤醒时若当前上下文达到此体积，先压缩再发送以规避长上下文冷启动" />
    </>,
  });
}

import type { WebHostContextV1 } from "@akashic/web-ui-v1";
import { registerForm, Field, ModelField } from "../../shared/src/configuration";

// 上下文压缩的运行参数；留空即用默认值。
export function activate(ctx: WebHostContextV1) {
  return registerForm(ctx, {id: "compaction", title: "上下文压缩", description: "对话变长后，较早的内容会压缩成摘要；这里决定保留多少最近原文。", family: "agent", familyLabel: "对话与智能体",
    switchable: false,
    fields: ({values, change, status}) => <>
      <ModelField label="摘要模型" name="model" value={values.model} change={change} status={status} hint="跟随对话时窗口与对话一致；换成窗口更小的模型，一次能摘要的原文会变少" />
      <Field label="保留最近原文（token）" name="keep_recent_tokens" type="number" min={1} step={1} inputMode="numeric" value={values.keep_recent_tokens} fallback={status.defaults?.keep_recent_tokens ?? "不限"} change={change} hint="调大保留更多细节，但每轮消耗更多上下文" />
    </>,
  });
}

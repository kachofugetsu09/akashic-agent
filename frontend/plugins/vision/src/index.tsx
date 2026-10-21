import type { WebHostContextV1 } from "@akashic/web-ui-v1";
import { registerForm, ModelField } from "../../shared/src/configuration";

// 看图工具只有一项设置：由哪个能看图的模型读图。
export function activate(ctx: WebHostContextV1) {
  return registerForm(ctx, {id: "vision", title: "看图", description: "对话模型不能直接看图时，交给一个能看图的模型读图并回答问题。", family: "agent", familyLabel: "对话与智能体",
    switchable: false,
    fields: ({values, change, status}) => <>
      <ModelField label="看图模型" name="model" value={values.model} change={change} status={status} hint="只列出能接收图片的模型" />
    </>,
  });
}

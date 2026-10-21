import type { WebHostContextV1 } from "@akashic/web-ui-v1";
import { registerForm, Field } from "../../shared/src/configuration";

// shell 工具在模型不指定时的超时与输出上限；留空即用默认值。
export function activate(ctx: WebHostContextV1) {
  return registerForm(ctx, {id: "shell", title: "命令执行", description: "shell 工具运行命令时的默认超时与输出上限，模型可按需要另行指定。", family: "agent", familyLabel: "对话与智能体",
    switchable: false,
    fields: ({values, change, status}) => <>
      <Field label="默认超时（秒）" name="timeout" type="number" inputMode="numeric" min={1} max={14400} step={1} value={values.timeout} fallback={status.defaults?.timeout} change={change} hint="1–14400；到时结束整个进程组" />
      <Field label="默认输出上限（token）" name="max_output_tokens" type="number" inputMode="numeric" min={100} max={200000} step={1} value={values.max_output_tokens} fallback={status.defaults?.max_output_tokens} change={change} hint="超出时截断，调大会占用更多上下文" />
    </>,
  });
}

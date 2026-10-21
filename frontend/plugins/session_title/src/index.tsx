import type { WebHostContextV1 } from "@akashic/web-ui-v1";
import { registerForm, Field } from "../../shared/src/configuration";

// 会话标题的运行参数；留空即用默认值。
export function activate(ctx: WebHostContextV1) {
  return registerForm(ctx, {id: "session_title", title: "会话标题", description: "根据第一条消息自动生成会话标题。", family: "agent", familyLabel: "对话与智能体",
    switchable: false,
    fields: ({values, change, status}) => <>
      <Field label="标题最长字数" name="max_title_chars" type="number" min={1} max={200} step={1} inputMode="numeric" value={values.max_title_chars} fallback={status.defaults?.max_title_chars ?? "不限"} change={change} hint="1–200" />
    </>,
  });
}

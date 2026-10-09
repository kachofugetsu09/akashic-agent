import type { WebHostContextV1 } from "@akashic/web-ui-v1";
import { registerForm, Field } from "../../shared/src/configuration";

export function activate(ctx: WebHostContextV1) {
  return registerForm(ctx, {id: "telegram_channel", title: "Telegram 消息接入", description: "接收 Telegram 对话消息并在原对话中自动回复。", family: "telegram", familyLabel: "Telegram",
    fields: ({values, change, status}) => <><Field label="Bot Token" name="token" type="password" value={values.token} change={change} required={!status.has_token} hint={status.has_token ? "已保存凭据；留空保持不变" : "在 Telegram @BotFather 中创建机器人获取"} />
      <Field label="允许的用户" name="allow_from" value={Array.isArray(values.allow_from) ? values.allow_from.join(", ") : ""} change={(_, value) => change("allow_from", String(value).split(",").map(s => s.trim()).filter(Boolean))} hint="填用户名（不含 @），多个用逗号隔开；留空允许所有人" /></>,
  });
}

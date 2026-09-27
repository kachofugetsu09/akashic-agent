import type { WebHostContextV1 } from "@akashic/web-ui-v1";
import { registerForm, Field } from "../../shared/src/configuration";

export function activate(ctx: WebHostContextV1) {
  return registerForm(ctx, {id: "telegram_channel", title: "Telegram 接入", description: "接收 Telegram 消息并在原对话中回复。",
    fields: ({values, change, status}) => <><Field label="Bot token" name="token" type="password" value={values.token} change={change} required={!status.has_token} hint={status.has_token ? "已保存凭据；留空保留原 token" : "在 Telegram 的 @BotFather 中创建机器人，取得 token"} />
      <Field label="允许的用户名" name="allow_from" value={Array.isArray(values.allow_from) ? values.allow_from.join(", ") : ""} change={(_, value) => change("allow_from", String(value).split(",").map(s => s.trim()).filter(Boolean))} hint="多个用户名用英文逗号分隔；不含 @。留空允许所有用户。" /></>,
  });
}

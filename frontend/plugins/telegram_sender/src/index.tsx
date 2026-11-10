import type { WebHostContextV1 } from "@akashic/web-ui-v1";
import { registerForm, Field } from "../../shared/src/configuration";

export function activate(ctx: WebHostContextV1) {
  return registerForm(ctx, {id: "telegram_sender", title: "Telegram 消息推送", description: "配置独立推送能力，供主动提醒与定时通知使用。", family: "telegram", familyLabel: "Telegram",
    fields: ({values, change, status}) => <><Field label="Bot Token" name="token" type="password" value={values.token} change={change} required={!status.has_token} hint={status.has_token ? "已保存凭据；留空保持不变" : "使用 @BotFather 创建的机器人 Token"} />
      <details><summary>高级网络设置</summary><Field label="API 地址" name="api_base" value={values.api_base} change={change} required /><Field label="超时时间（秒）" name="timeout_seconds" type="number" value={values.timeout_seconds} change={change} /></details></>,
  });
}

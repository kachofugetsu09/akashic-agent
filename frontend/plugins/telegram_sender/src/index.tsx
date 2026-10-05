import type { WebHostContextV1 } from "@akashic/web-ui-v1";
import { registerForm, Field } from "../../shared/src/configuration";

export function activate(ctx: WebHostContextV1) {
  return registerForm(ctx, {id: "telegram_sender", title: "Telegram 发送", description: "配置独立发送能力，供主动联系等功能选择。", family: "telegram", familyLabel: "Telegram",
    fields: ({values, change, status}) => <><Field label="Bot token" name="token" type="password" value={values.token} change={change} required={!status.has_token} hint={status.has_token ? "留空保留已保存的 token" : "使用 @BotFather 创建的机器人 token"} />
      <details><summary>高级设置</summary><Field label="API 地址" name="api_base" value={values.api_base} change={change} required /><Field label="连接超时（秒）" name="timeout_seconds" type="number" value={values.timeout_seconds} change={change} /></details></>,
  });
}

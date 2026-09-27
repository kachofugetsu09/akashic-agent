import type { WebHostContextV1 } from "@akashic/web-ui-v1";
import { registerForm, Field } from "../../shared/src/configuration";

export function activate(ctx: WebHostContextV1) {
  return registerForm(ctx, {id: "qq_channel", title: "QQ 接入", description: "使用已有 NapCat / NcatBot 接收 QQ 消息。",
    fields: ({values, change}) => <><Field label="机器人 QQ 号" name="bot_uin" value={values.bot_uin} change={change} required />
      <Field label="允许的用户 QQ 号" name="allow_from" value={Array.isArray(values.allow_from) ? values.allow_from.join(", ") : ""} change={(_, value) => change("allow_from", String(value).split(",").map(s => s.trim()).filter(Boolean))} hint="多个 QQ 号用英文逗号分隔；留空允许所有用户。" />
      <details><summary>高级设置</summary><Field label="启动超时（秒）" name="websocket_open_timeout_seconds" type="number" value={values.websocket_open_timeout_seconds} change={change} /></details></>,
  });
}

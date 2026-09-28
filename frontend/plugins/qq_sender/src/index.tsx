import type { WebHostContextV1 } from "@akashic/web-ui-v1";
import { registerForm, Field } from "../../shared/src/configuration";

export function activate(ctx: WebHostContextV1) {
  return registerForm(ctx, {id: "qq_sender", title: "QQ 发送", description: "连接已有 OneBot WebSocket API，用于独立发送。",
    fields: ({values, change, status}) => <><Field label="OneBot WebSocket 地址" name="endpoint" value={values.endpoint} change={change} required hint="例如 ws://127.0.0.1:3001；使用 API 地址，不填 /event 地址。" />
      <Field label="访问 token（可选）" name="token" type="password" value={values.token} change={change} hint={status.has_token ? "留空保留已保存凭据" : "与 OneBot 服务的访问凭据一致"} /></>,
  });
}

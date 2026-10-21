import type { WebHostContextV1 } from "@akashic/web-ui-v1";
import { registerForm, Field } from "../../shared/src/configuration";

export function activate(ctx: WebHostContextV1) {
  return registerForm(ctx, {id: "web_search_searxng", title: "SearXNG 网页搜索", family: "web", familyLabel: "Web",
    description: "连接自建 SearXNG。保存地址后，在 Web 工具的搜索提供方中选择 SearXNG。",
    switchable: false,
    fields: ({values, change, status}) => <>
      <Field label="服务地址" name="endpoint" type="url" inputMode="url" value={values.endpoint} change={change}
        hint="填写实例根地址，可含部署子路径；不含 /search。实例须启用 JSON 格式。留空停用此搜索连接。" />
      <Field label="超时时间（秒）" name="timeout_seconds" type="number" min={1} max={120} step="any" inputMode="decimal"
        value={values.timeout_seconds} fallback={status.defaults?.timeout_seconds} change={change} />
    </>,
  });
}

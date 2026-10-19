import type { WebHostContextV1 } from "@akashic/web-ui-v1";
import { registerForm, Field } from "../../shared/src/configuration";

// Exa 搜索提供方的端点与凭据；搜索工具本身归 standard_web。
export function activate(ctx: WebHostContextV1) {
  return registerForm(ctx, {id: "web_search_exa", title: "Exa 网页搜索", description: "为 web_search 工具提供 Exa 搜索；可换成自建或代理的 Exa MCP 端点。", family: "web", familyLabel: "Web",
    switchable: false,
    fields: ({values, change, status}) => <>
      <Field label="端点" name="endpoint" type="url" inputMode="url" value={values.endpoint} change={change} required hint="Exa MCP 地址，默认 https://mcp.exa.ai/mcp" />
      <Field label="API key" name="api_key" type="password" value={values.api_key} change={change} hint={status.has_token ? "已保存凭据；留空保持不变" : "可选；不填时使用公开额度"} />
      <label className="config-field"><span>搜索类型</span><select name="search_type" value={String(values.search_type ?? "auto")} onChange={event => change("search_type", event.target.value)}>
        <option value="auto">均衡</option><option value="fast">快速</option><option value="deep">深度</option></select></label>
      <details><summary>高级网络设置</summary>
        <label className="config-field"><span>实时抓取</span><select name="livecrawl" value={String(values.livecrawl ?? "fallback")} onChange={event => change("livecrawl", event.target.value)}>
          <option value="fallback">缓存优先</option><option value="preferred">优先实时</option></select></label>
        <Field label="超时时间（秒）" name="timeout_seconds" type="number" value={values.timeout_seconds} change={change} />
      </details></>,
  });
}

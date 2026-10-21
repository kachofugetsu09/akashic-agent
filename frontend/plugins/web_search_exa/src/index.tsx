import type { WebHostContextV1 } from "@akashic/web-ui-v1";
import { registerForm, Field, Select, type Choice } from "../../shared/src/configuration";

const SEARCH_TYPES: Choice[] = [{value: "auto", label: "均衡"}, {value: "fast", label: "快速"}, {value: "deep", label: "深度"}];
const LIVECRAWL: Choice[] = [{value: "fallback", label: "缓存优先"}, {value: "preferred", label: "优先实时"}];
const named = (options: Choice[], value: unknown) => `默认（${options.find(option => option.value === value)?.label ?? String(value ?? "")}）`;

// Exa 搜索提供方的端点与凭据；搜索工具本身归 standard_web。
export function activate(ctx: WebHostContextV1) {
  return registerForm(ctx, {id: "web_search_exa", title: "Exa 网页搜索", description: "为 web_search 工具提供 Exa 搜索；可填 API key 提高额度，或换成自建、代理的 Exa MCP 端点。", family: "web", familyLabel: "Web",
    switchable: false,
    fields: ({values, change, status}) => <>
      <Field label="API key" name="api_key" type="password" value={values.api_key} change={change} hint={status.has_token ? "已保存凭据；留空保持不变" : "可选；不填时使用公开额度"} />
      <Select label="搜索类型" name="search_type" value={values.search_type} change={change} options={SEARCH_TYPES} fallbackLabel={named(SEARCH_TYPES, status.defaults?.search_type)} />
      <details><summary>高级网络设置</summary>
        <Field label="端点" name="endpoint" type="url" inputMode="url" value={values.endpoint} fallback={status.defaults?.endpoint} change={change} hint="Exa MCP 地址；留空使用官方端点" />
        <Select label="实时抓取" name="livecrawl" value={values.livecrawl} change={change} options={LIVECRAWL} fallbackLabel={named(LIVECRAWL, status.defaults?.livecrawl)} />
        <Field label="超时时间（秒）" name="timeout_seconds" type="number" min={1} max={120} step="any" inputMode="decimal" value={values.timeout_seconds} fallback={status.defaults?.timeout_seconds} change={change} />
      </details></>,
  });
}

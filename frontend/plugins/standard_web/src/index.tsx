import type { WebHostContextV1 } from "@akashic/web-ui-v1";
import { registerForm, Field, Select, type Choice } from "../../shared/src/configuration";

// 自动：只启用了一个提供方时直接用它；启用多个时需要在这里选。
function automatic(options: Choice[]): string {
  return options.length === 1 ? `自动（${options[0].label}）` : options.length ? "自动（启用多个时需选择）" : "自动（尚未启用提供方）";
}

export function activate(ctx: WebHostContextV1) {
  return registerForm(ctx, {id: "standard_web", title: "Web 工具", description: "web_search 与 web_fetch 用哪个提供方、模型不指定时的默认值。各提供方的端点与凭据在各自的设置里。", family: "web", familyLabel: "Web",
    switchable: false,
    fields: ({values, change, status}) => {
      const search = status.choices?.search_provider ?? [], fetch = status.choices?.fetch_provider ?? [];
      return <>
        <Select label="搜索提供方" name="search_provider" value={values.search_provider} change={change} options={search} fallbackLabel={automatic(search)} hint="想换搜索服务，先在插件列表启用对应的提供方插件" />
        <Select label="网页读取提供方" name="fetch_provider" value={values.fetch_provider} change={change} options={fetch} fallbackLabel={automatic(fetch)} />
        <Field label="默认搜索结果数" name="default_results" type="number" min={1} max={20} step={1} inputMode="numeric" value={values.default_results} fallback={status.defaults?.default_results} change={change} hint="1–20；模型可按需要另行指定" />
        <Field label="默认读取超时（秒）" name="default_fetch_timeout" type="number" min={1} max={120} step={1} inputMode="numeric" value={values.default_fetch_timeout} fallback={status.defaults?.default_fetch_timeout} change={change} hint="1–120；模型可按需要另行指定" />
      </>;
    },
  });
}

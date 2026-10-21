import type { WebHostContextV1 } from "@akashic/web-ui-v1";
import { registerForm, Field } from "../../shared/src/configuration";

// 直接 HTTP 读取的上限、代理与请求头；工具本身归 standard_web。
export function activate(ctx: WebHostContextV1) {
  return registerForm(ctx, {id: "web_fetch_http", title: "直接 HTTP 读取", description: "为 web_fetch 工具直接请求网页；可设置大小上限、代理和请求头。", family: "web", familyLabel: "Web",
    switchable: false,
    fields: ({values, change, status}) => <>
      <Field label="代理" name="proxy" type="url" inputMode="url" value={values.proxy} change={change} hint="如 http://127.0.0.1:7890 或 socks5://127.0.0.1:1080；留空沿用系统代理环境变量" />
      <Field label="返回给模型的字符上限" name="max_text_chars" type="number" min={1000} max={1000000} step={1} inputMode="numeric" value={values.max_text_chars} fallback={status.defaults?.max_text_chars} change={change} hint="超出部分截断；调大会占用更多上下文" />
      <Field label="响应大小上限（MB）" name="max_response_mb" type="number" min={0.1} max={100} step="any" inputMode="decimal" value={values.max_response_mb} fallback={status.defaults?.max_response_mb} change={change} />
      <details><summary>请求头</summary>
        <Field label="User-Agent" name="user_agent" value={values.user_agent} fallback={status.defaults?.user_agent} change={change} hint="有的网站会拦截非浏览器 UA，可换成常见浏览器的 UA" />
        <Field label="Accept-Language" name="accept_language" value={values.accept_language} fallback={status.defaults?.accept_language} change={change} />
      </details></>,
  });
}

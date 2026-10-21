import { useState } from "react";
import type { WebHostContextV1 } from "@akashic/web-ui-v1";
import { registerForm, request, Field, Check, ModelField, type FormProps } from "../../shared/src/configuration";

interface DeclaredField { name: string; label: string; kind: "text" | "lines" | "number" | "toggle" | "model"; hint: string; }
interface DeclaredForm { id: string; title: string; description: string; family: string; family_label: string; fields: DeclaredField[]; }

const ENDPOINT = "/api/dashboard/plugin_settings";

// 每行一项的列表字段：输入框保留原文便于编辑，提交的值去掉空行；全部清空即回到默认。
function Lines({field, values, change, status}: {field: DeclaredField} & FormProps) {
  const value = values[field.name] ?? status.defaults?.[field.name];
  const [text, setText] = useState(() => Array.isArray(value) ? value.join("\n") : "");
  return <label className="config-field"><span>{field.label}</span>
    <textarea name={field.name} rows={Math.min(Math.max(text.split("\n").length, 2), 8)} value={text}
      onChange={event => { setText(event.target.value);
        const items = event.target.value.split("\n").map(item => item.trim()).filter(Boolean);
        change(field.name, items.length ? items : null); }} />
    {field.hint && <small>{field.hint}</small>}</label>;
}

// 按字段类型选现成的表单控件；默认值取自插件配置模型，留空即回到默认。
function renderField(field: DeclaredField, props: FormProps) {
  const {values, change, status} = props;
  const fallback = status.defaults?.[field.name];
  const hint = field.hint || undefined;
  switch (field.kind) {
    case "model": return <ModelField key={field.name} label={field.label} name={field.name} value={values[field.name]} change={change} status={status} hint={hint} />;
    case "toggle": return <Check key={field.name} label={field.label} name={field.name} value={values[field.name]} change={change} hint={hint} fallback={fallback} />;
    case "number": return <Field key={field.name} label={field.label} name={field.name} type="number" inputMode="numeric" value={values[field.name]} change={change} hint={hint} fallback={fallback} />;
    case "lines": return <Lines key={field.name} field={field} {...props} />;
    default: return <Field key={field.name} label={field.label} name={field.name} value={values[field.name]} change={change} hint={hint} fallback={fallback} />;
  }
}

// 声明式表单在模块激活时一次读入；插件启停会让宿主换代并重新激活本模块。
export function activate(ctx: WebHostContextV1) {
  const disposers: (() => void)[] = [];
  let disposed = false;
  void request<{forms: DeclaredForm[]}>(ctx, `${ENDPOINT}/forms`).then(({forms}) => {
    if (disposed) return;
    for (const form of forms) {
      disposers.push(registerForm(ctx, {id: form.id, title: form.title, description: form.description, family: form.family,
        familyLabel: form.family_label, switchable: false, endpoint: `${ENDPOINT}/${form.id}/config`,
        fields: props => <>{form.fields.map(field => renderField(field, props))}</>}));
    }
  }).catch(error => console.warn("[plugin_settings] 读取声明式表单失败", error));
  return () => { disposed = true; disposers.splice(0).forEach(dispose => dispose()); };
}

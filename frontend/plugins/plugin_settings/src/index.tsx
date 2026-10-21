import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import type { WebHostContextV1 } from "@akashic/web-ui-v1";
import { Configuration, request, Field, Check, ModelField, type EmbedProps, type FormProps } from "../../shared/src/configuration";

interface DeclaredField { name: string; label: string; kind: "text" | "lines" | "number" | "toggle" | "model"; hint: string; }
interface DeclaredForm { id: string; title: string; description: string; fields: DeclaredField[]; }

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

// 宿主只接受激活时同步登记：登记一张固定卡片，打开时再读各插件声明的表单，每张表单各自保存。
function DeclaredForms({ctx, embed}: {ctx: WebHostContextV1; embed: EmbedProps}) {
  const [forms, setForms] = useState<DeclaredForm[] | null>(null);
  const [error, setError] = useState("");
  const dirty = useRef(new Set<string>());
  useEffect(() => {
    let alive = true;
    void request<{forms: DeclaredForm[]}>(ctx, `${ENDPOINT}/forms`).then(({forms}) => { if (alive) setForms(forms); },
      (reason: unknown) => { if (alive) setError(reason instanceof Error ? reason.message : "读取插件设置失败"); });
    return () => { alive = false; };
  }, [ctx]);
  if (!forms) return <p className="config-hint">{error || "正在读取…"}</p>;
  if (!forms.length) return <p className="config-hint">目前没有插件声明设置。</p>;
  if (embed.mode === "summary") return <span>{forms.map(form => form.title).join("、")}</span>;
  return <>{forms.map(form => <section key={form.id} className="declared-form">
    <h2>{form.title}</h2><p>{form.description}</p>
    <Configuration ctx={ctx} embed={{...embed, embedded: true, dirty: value => {
      if (value) dirty.current.add(form.id); else dirty.current.delete(form.id);
      embed.dirty?.(dirty.current.size > 0);
    }}} definition={{id: form.id, title: form.title, description: form.description, switchable: false,
      endpoint: `${ENDPOINT}/${form.id}/config`, fields: props => <>{form.fields.map(field => renderField(field, props))}</>}} />
  </section>)}</>;
}

export function activate(ctx: WebHostContextV1) {
  return ctx.ui.inject("shell.settings-plugins.v1", mount => mount.register({
    id: "plugin-settings", label: "插件设置", route: "plugin-settings",
    description: "没有自带界面的插件在这里调整设置，比如 GitHub Watch。",
    render(host, _view, props) {
      const root = createRoot(host);
      root.render(<DeclaredForms ctx={ctx} embed={(props ?? {}) as EmbedProps} />);
      return () => root.unmount();
    },
  }));
}

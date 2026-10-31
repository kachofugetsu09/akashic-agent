import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import { createRoot } from "react-dom/client";
import type { WebHostContextV1, WebResource } from "@akashic/web-ui-v1";
import { Configuration, request, Field, Check, ModelField, type EmbedProps, type FormProps, type Status } from "../../shared/src/configuration";

interface DeclaredField { name: string; label: string; kind: "text" | "lines" | "number" | "toggle" | "model" | "secret"; hint: string; }
interface DeclaredForm { id: string; title: string; description: string; fields: DeclaredField[]; }

const ENDPOINT = "/api/dashboard/plugin_settings";

// 每行一项的列表字段：输入框保留原文便于编辑，提交的值去掉空行；全部清空即回到默认。
const parseLines = (text: string) => { const items = text.split("\n").map(item => item.trim()).filter(Boolean); return items.length ? items : null; };
const joinLines = (value: unknown) => Array.isArray(value) ? value.join("\n") : "";

function Lines({field, values, change, status}: {field: DeclaredField} & FormProps) {
  const own = values[field.name] ?? null;
  const [text, setText] = useState(() => joinLines(own ?? status.defaults?.[field.name]));
  // 外部换了值（重新加载、放弃草稿、保存后核对）且与原文解析结果不同，才用新值覆盖原文；自己的输入总与解析结果一致，空行不会被吃掉。
  const external = JSON.stringify(own);
  useEffect(() => {
    if (JSON.stringify(parseLines(text)) === external) return;
    const next = joinLines(own ?? status.defaults?.[field.name]);
    if (next !== text) setText(next);
  }, [external]);
  return <label className="config-field"><span>{field.label}</span>
    <textarea name={field.name} rows={Math.min(Math.max(text.split("\n").length, 2), 8)} value={text}
      onChange={event => { setText(event.target.value); change(field.name, parseLines(event.target.value)); }} />
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
    case "secret": return <Field key={field.name} label={field.label} name={field.name} type="password" value={values[field.name]} change={change}
      hint={status.has_token ? "已保存凭据；留空保持不变" : hint} />;
    default: return <Field key={field.name} label={field.label} name={field.name} value={values[field.name]} change={change} hint={hint} fallback={fallback} />;
  }
}

// 宿主只接受激活时同步登记：登记一张固定卡片，打开时再读各插件声明的表单，每张表单各自保存。
function DeclaredForms({ctx, embed, declared, configs}: {ctx: WebHostContextV1; embed: EmbedProps; declared: WebResource<DeclaredForm[]>; configs: ReadonlyMap<string, WebResource<Status>>}) {
  const forms = useSyncExternalStore(declared.subscribe, () => declared.current ?? null);
  const failure = useSyncExternalStore(declared.subscribe, () => declared.error);
  const error = forms ? "" : failure ? failure.message || "读取插件设置失败" : "";
  const dirty = useRef(new Set<string>());
  useEffect(() => { void declared.refresh().catch(() => undefined); }, [declared]);
  if (!forms) return <p className="config-hint">{error || "正在读取…"}</p>;
  if (!forms.length) return <p className="config-hint">目前没有插件声明设置。</p>;
  if (embed.mode === "summary") return <span>{forms.map(form => form.title).join("、")}</span>;
  return <>{forms.map(form => <section key={form.id} className="declared-form">
    <h2>{form.title}</h2><p>{form.description}</p>
    <Configuration ctx={ctx} stored={configs.get(form.id)!} embed={{...embed, embedded: true, dirty: value => {
      if (value) dirty.current.add(form.id); else dirty.current.delete(form.id);
      embed.dirty?.(dirty.current.size > 0);
    }}} definition={{id: form.id, title: form.title, description: form.description, switchable: false,
      endpoint: `${ENDPOINT}/${encodeURIComponent(form.id)}/config`, fields: props => <>{form.fields.map(field => renderField(field, props))}</>}} />
  </section>)}</>;
}

export function activate(ctx: WebHostContextV1) {
  // 声明的表单清单随模块保留：插件配置列表重开时卡片摘要直接出现，不再闪“正在读取…”。
  const configs = new Map<string, WebResource<Status>>();
  const declared = ctx.data.resource<DeclaredForm[]>("forms", async signal => {
    const {forms} = await request<{forms: DeclaredForm[]}>(ctx, `${ENDPOINT}/forms`, {signal});
    // 配置与模块同寿命；重开表单复用已读取的值，草稿仍由每次挂载的 Configuration 拥有。
    for (const form of forms) {
      if (!configs.has(form.id)) configs.set(form.id, ctx.data.resource<Status>(`config.${form.id}`,
        signal => request<Status>(ctx, `${ENDPOINT}/${encodeURIComponent(form.id)}/config`, {signal})));
    }
    return forms;
  });
  return ctx.ui.inject("shell.settings-plugins.v1", mount => mount.register({
    id: "plugin-settings", label: "插件设置", route: "plugin-settings",
    description: "没有自带界面的插件在这里调整设置，比如 GitHub Watch。",
    render(host, _view, props) {
      const root = createRoot(host);
      root.render(<DeclaredForms ctx={ctx} embed={(props ?? {}) as EmbedProps} declared={declared} configs={configs} />);
      return () => root.unmount();
    },
  }));
}

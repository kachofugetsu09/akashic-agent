const DEEPSEEK_ICON = svgData(`<svg height="1em" style="flex:none;line-height:1" viewBox="0 0 24 24" width="1em" xmlns="http://www.w3.org/2000/svg"><title>DeepSeek</title><path d="M23.748 4.482c-.254-.124-.364.113-.512.234-.051.039-.094.09-.137.136-.372.397-.806.657-1.373.626-.829-.046-1.537.214-2.163.848-.133-.782-.575-1.248-1.247-1.548-.352-.156-.708-.311-.955-.65-.172-.241-.219-.51-.305-.774-.055-.16-.11-.323-.293-.35-.2-.031-.278.136-.356.276-.313.572-.434 1.202-.422 1.84.027 1.436.633 2.58 1.838 3.393.137.093.172.187.129.323-.082.28-.18.552-.266.833-.055.179-.137.217-.329.14a5.526 5.526 0 01-1.736-1.18c-.857-.828-1.631-1.742-2.597-2.458a11.365 11.365 0 00-.689-.471c-.985-.957.13-1.743.388-1.836.27-.098.093-.432-.779-.428-.872.004-1.67.295-2.687.684a3.055 3.055 0 01-.465.137 9.597 9.597 0 00-2.883-.102c-1.885.21-3.39 1.102-4.497 2.623C.082 8.606-.231 10.684.152 12.85c.403 2.284 1.569 4.175 3.36 5.653 1.858 1.533 3.997 2.284 6.438 2.14 1.482-.085 3.133-.284 4.994-1.86.47.234.962.327 1.78.397.63.059 1.236-.03 1.705-.128.735-.156.684-.837.419-.961-2.155-1.004-1.682-.595-2.113-.926 1.096-1.296 2.746-2.642 3.392-7.003.05-.347.007-.565 0-.845-.004-.17.035-.237.23-.256a4.173 4.173 0 001.545-.475c1.396-.763 1.96-2.015 2.093-3.517.02-.23-.004-.467-.247-.588zM11.581 18c-2.089-1.642-3.102-2.183-3.52-2.16-.392.024-.321.471-.235.763.09.288.207.486.371.739.114.167.192.416-.113.603-.673.416-1.842-.14-1.897-.167-1.361-.802-2.5-1.86-3.301-3.307-.774-1.393-1.224-2.887-1.298-4.482-.02-.386.093-.522.477-.592a4.696 4.696 0 011.529-.039c2.132.312 3.946 1.265 5.468 2.774.868.86 1.525 1.887 2.202 2.891.72 1.066 1.494 2.082 2.48 2.914.348.292.625.514.891.677-.802.09-2.14.11-3.054-.614zm1-6.44a.306.306 0 01.415-.287.302.302 0 01.2.288.306.306 0 01-.31.307.303.303 0 01-.304-.308zm3.11 1.596c-.2.081-.399.151-.59.16a1.245 1.245 0 01-.798-.254c-.274-.23-.47-.358-.552-.758a1.73 1.73 0 01.016-.588c.07-.327-.008-.537-.239-.727-.187-.156-.426-.199-.688-.199a.559.559 0 01-.254-.078c-.11-.054-.2-.19-.114-.358.028-.054.16-.186.192-.21.356-.202.767-.136 1.146.016.352.144.618.408 1.001.782.391.451.462.576.685.914.176.265.336.537.445.848.067.195-.019.354-.25.452z" fill="#4D6BFE"></path></svg>`);
const CLOSE_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M18 6 6 18"></path><path d="m6 6 12 12"></path></svg>`;
const EYE_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0"></path><circle cx="12" cy="12" r="3"></circle></svg>`;
const EYE_OFF_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m2 2 20 20"></path><path d="M6.71 6.71C4.7 8.1 3.24 10.06 2.06 11.65a1 1 0 0 0 0 .7c2.34 5.64 8.94 8.32 14.24 5.36"></path><path d="M10.73 5.08A10.66 10.66 0 0 1 21.94 11.65a1 1 0 0 1 0 .7 10.83 10.83 0 0 1-2.06 3.1"></path><path d="M14.12 14.12A3 3 0 0 1 9.88 9.88"></path></svg>`;
const SHIELD_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"></path><path d="m9 12 2 2 4-4"></path></svg>`;
const SPINNER_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="is-spinning" aria-hidden="true"><path d="M21 12a9 9 0 1 1-6.219-8.56"></path></svg>`;

export function activate(ctx) {
  return ctx.ui.inject("models.connection-types.v1", (mount) => mount.register({
    id: "openai-compatible",
    label: "OpenAI Compatible",
    detail: "API Key 与任意 OpenAI 格式服务",
    order: 30,
    editTemplateId: "custom-api",
    templates: [
      {
        id: "deepseek",
        label: "DeepSeek",
        detail: "官方 API",
        icon: DEEPSEEK_ICON,
        order: 30,
        defaults: {
          name: "DeepSeek",
          endpoint: "https://api.deepseek.com/v1",
          provider: "deepseek",
        },
      },
      {
        id: "custom-api",
        label: "自定义 API",
        detail: "连接任意兼容服务",
        order: 40,
        defaults: {name: "", endpoint: "", provider: "openai"},
      },
    ],
    render(host, _view, rawProps) {
      const props = requireProps(rawProps);
      const existing = props.state.connection;
      const defaults = props.state.template?.defaults ?? {};
      const provider = existing?.driverId ?? defaults.provider ?? "openai";
      const isDeepSeek = !existing && provider === "deepseek";
      const title = existing ? `编辑 ${existing.name}` : isDeepSeek ? "连接 DeepSeek" : "连接自定义 API";
      const description = isDeepSeek
        ? "填写 API Key 和模型名称；未知能力保持未知。"
        : "填写服务地址、凭据和模型名称。";
      host.innerHTML = `<header class="settings-dialog-header"><div class="settings-dialog-heading"><h2 class="settings-dialog-title">${escapeHtml(title)}</h2><p class="settings-dialog-description">${description}</p></div>
        <button type="button" class="settings-icon-button" aria-label="关闭" data-close>${CLOSE_ICON}</button></header>
        <form class="settings-dialog-form"><div class="settings-dialog-body"><div class="settings-form-grid">
          <label class="is-wide"><span>连接名称</span><input name="name" aria-label="连接名称" required autocomplete="organization" placeholder="${isDeepSeek ? "例如：DeepSeek 官方" : "例如：公司网关"}"></label>
          <label><span>Provider ID</span><input name="provider" aria-label="Provider ID" required placeholder="例如：openai"></label>
          <label><span>Base URL</span><input name="endpoint" aria-label="Base URL" required type="url" placeholder="https://api.example.com/v1"></label>
          <label class="settings-secret is-wide"><span>API Key</span><input name="apiKey" aria-label="API Key" type="password" ${existing ? "" : "required"} autocomplete="off" placeholder="sk-…"><button type="button" data-show-key aria-label="显示 API Key">${EYE_ICON}</button></label>
        </div>
        <section class="settings-model-discovery"><header><div><h3>模型</h3><p>填写服务使用的模型名称；连接和模型会由插件分别验证并保存。</p></div></header>
          <div class="settings-form-grid"><label class="is-wide"><span>模型名称</span><input name="model" aria-label="模型名称" required placeholder="${isDeepSeek ? "例如：deepseek-chat" : "例如：your-model-name"}"></label></div>
          <p>模型能力由 Provider 目录或后续显式设置补充，未知能力不会被猜测。</p>
        </section><p class="settings-inline-error" data-error role="alert" hidden></p></div>
        <footer class="settings-dialog-footer"><span class="settings-dialog-footer-note">${SHIELD_ICON}凭据保存后不会显示在页面中</span><button type="submit" class="settings-primary-button">保存连接</button></footer></form>`;
      const form = host.querySelector("form");
      form.elements.name.value = existing?.name ?? defaults.name ?? "";
      form.elements.endpoint.value = defaults.endpoint ?? "";
      form.elements.provider.value = provider;
      form.elements.model.value = props.state.models[0]?.model ?? "";
      const showKey = host.querySelector("[data-show-key]");
      showKey.addEventListener("click", () => {
        const visible = form.elements.apiKey.type === "text";
        form.elements.apiKey.type = visible ? "password" : "text";
        showKey.setAttribute("aria-label", visible ? "显示 API Key" : "隐藏 API Key");
        showKey.innerHTML = visible ? EYE_ICON : EYE_OFF_ICON;
      });
      host.querySelector("[data-close]").addEventListener("click", props.close);
      form.addEventListener("submit", (event) => {
        event.preventDefault();
        const data = new FormData(form);
        if (existing) {
          const endpoint = String(data.get("endpoint"));
          const apiKey = String(data.get("apiKey"));
          submit(form, props.actions.update({
            name: String(data.get("name")),
            endpoint: endpoint || null,
            credential: apiKey ? {driver: "api_key", access_token: apiKey} : null,
            driverConfig: null,
          }).then(() => props.changed("OpenAI Compatible 连接已更新")));
          return;
        }
        submit(form, props.actions.createManual({
            name: String(data.get("name")),
            endpoint: String(data.get("endpoint")),
            credential: {driver: "api_key", access_token: String(data.get("apiKey"))},
            driverConfig: {format_version: 1, catalog_provider_id: String(data.get("provider")), allow_unverified_manual: true},
          model: {
            kind: "chat",
            model: String(data.get("model")),
            capabilities: {
              context_window: null,
              max_output_tokens: null,
              input_modalities: ["text"],
              supports_tool_calls: null,
              supports_parallel_tool_calls: null,
              supported_reasoning_efforts: [],
              embedding_dimensions: null,
              embedding_normalization: null,
            },
            capability_sources: {},
            default_reasoning_effort: null,
            driver_config: {format_version: 1},
          },
        }).then(() => props.changed("OpenAI Compatible 连接已保存")));
      });
      queueMicrotask(() => form.elements.name.focus());
      return () => host.replaceChildren();
    },
  }));
}

function svgData(svg) {
  return `data:image/svg+xml,${encodeURIComponent(svg)}`;
}

function requireProps(value) {
  if (!value || typeof value !== "object" || typeof value.actions !== "object"
    || typeof value.actions.createManual !== "function" || typeof value.actions.update !== "function"
    || typeof value.close !== "function" || typeof value.changed !== "function" || !value.state) {
    throw new Error("models.connection-types.v1 props 无效");
  }
  if (value.state.connection !== null && value.state.connection !== undefined
    && (typeof value.state.connection !== "object" || typeof value.state.connection.id !== "string")) {
    throw new Error("models.connection-types.v1 connection 无效");
  }
  return value;
}

function submit(form, work) {
  const button = form.querySelector("[type=submit]");
  const error = form.querySelector("[data-error]");
  button.disabled = true;
  button.replaceChildren(htmlNode(SPINNER_ICON), "保存中");
  error.hidden = true;
  error.textContent = "";
  work.catch((reason) => {
    error.textContent = reason instanceof Error ? reason.message : String(reason);
    error.hidden = false;
  }).finally(() => {
    button.disabled = false;
    button.textContent = "保存连接";
  });
}

function htmlNode(markup) {
  const template = document.createElement("template");
  template.innerHTML = markup;
  return template.content.firstElementChild;
}

function escapeHtml(value) {
  const node = document.createElement("span");
  node.textContent = value;
  return node.innerHTML;
}

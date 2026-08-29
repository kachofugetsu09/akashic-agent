export function activate(ctx) {
  return ctx.ui.inject("models.connection-types.v1", (mount) => mount.register({
    id: "openai-compatible",
    label: "OpenAI Compatible",
    detail: "API Key 与任意 OpenAI 格式服务",
    order: 10,
    render(host, _view, rawProps) {
      const props = requireProps(rawProps);
      const existing = props.state.connection;
      const content = document.createElement("section");
      content.innerHTML = `<h2>${existing ? "编辑" : "添加"} OpenAI Compatible 连接</h2><form>
        <label>连接名称<input name="name" required autofocus autocomplete="organization" placeholder="例如：家庭网关"></label>
        <label>Base URL<input name="endpoint" type="url" ${existing ? "" : "required"} placeholder="${existing ? "留空保持不变" : "https://api.example.com/v1"}"></label>
        <label>API Key<input name="apiKey" type="password" ${existing ? "" : "required"} autocomplete="off" placeholder="${existing ? "留空保持不变" : "sk-…"}"></label>
        ${existing ? "" : `<label>Provider ID<input name="provider" required value="openai"></label>
        <label>模型名称<input name="model" required placeholder="例如：gpt-5"></label>
        <label>模型类型<select name="kind"><option value="chat">聊天模型</option><option value="embedding">向量模型</option></select></label>
        <label data-dimensions hidden>向量维度<input name="dimensions" type="number" min="1"></label>`}
        <p data-error role="alert"></p>
        <footer><button type="button" data-cancel>取消</button><button type="submit">保存连接</button></footer></form>`;
      host.replaceChildren(content);
      const form = content.querySelector("form");
      form.elements.name.value = existing?.name ?? "";
      const kind = form.elements.kind;
      if (kind) {
        const dimensions = form.querySelector("[data-dimensions]");
        const dimensionsInput = form.elements.dimensions;
        kind.addEventListener("change", () => {
          const enabled = kind.value === "embedding";
          dimensions.hidden = !enabled;
          dimensionsInput.required = enabled;
        });
      }
      form.querySelector("[data-cancel]").addEventListener("click", props.close);
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
        const modelKind = String(data.get("kind"));
        const dimensionsValue = Number(data.get("dimensions"));
        submit(form, props.actions.createManual({
            name: String(data.get("name")),
            endpoint: String(data.get("endpoint")),
            credential: {driver: "api_key", access_token: String(data.get("apiKey"))},
            driverConfig: {format_version: 1, catalog_provider_id: String(data.get("provider")), allow_unverified_manual: true},
          model: {
            kind: modelKind,
            model: String(data.get("model")),
            capabilities: {
              context_window: null,
              max_output_tokens: null,
              input_modalities: ["text"],
              supports_tool_calls: null,
              supports_parallel_tool_calls: null,
              supported_reasoning_efforts: [],
              embedding_dimensions: modelKind === "embedding" && dimensionsValue > 0 ? dimensionsValue : null,
              embedding_normalization: null,
            },
            capability_sources: {},
            default_reasoning_effort: null,
            driver_config: {format_version: 1},
          },
        }).then(() => props.changed("OpenAI Compatible 连接已保存")));
      });
      return () => host.replaceChildren();
    },
  }));
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
  error.textContent = "";
  work.catch((reason) => { error.textContent = reason instanceof Error ? reason.message : String(reason); })
    .finally(() => { button.disabled = false; });
}

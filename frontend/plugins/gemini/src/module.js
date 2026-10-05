export function activate(ctx) {
  return ctx.ui.inject("models.connection-types.v1", (types) => types.register({
    id: "gemini", label: "Gemini 原生 API", order: 35,
    detail: "连接 Google 或提供原生 Gemini 接口的网关", catalogSync: true,
    templates: [{id: "gemini-native", label: "Gemini 原生 API", detail: "Google 与原生接口网关", order: 35,
      defaults: {name: "Gemini", endpoint: "https://generativelanguage.googleapis.com/v1beta"}}],
    render(host, _view, dialog) {
      const props = dialog;
      if (!props?.state || typeof props.actions?.createManual !== "function") {
        throw new Error("models.connection-types.v1 props 无效");
      }
      const existing = props.state.connection;
      const defaults = props.state.template?.defaults ?? {};
      host.innerHTML = `<header class="settings-dialog-header"><div class="settings-dialog-heading">
        <h2 class="settings-dialog-title">${existing ? "编辑 Gemini 连接" : "连接 Gemini 原生 API"}</h2>
        <p class="settings-dialog-description">使用原生接口连接 Google 或你的网关。保存前会发送短消息验证模型。</p>
        </div><button type="button" class="settings-icon-button" data-close aria-label="关闭">×</button></header>
        <form class="settings-dialog-form"><div class="settings-dialog-body"><div class="settings-form-grid">
          <label class="is-wide"><span>连接名称</span><input name="name" required autocomplete="organization"></label>
          <label class="is-wide"><span>Base URL${existing ? "（留空保持不变）" : ""}</span><input name="endpoint" type="url" ${existing ? "" : "required"} placeholder="https://your-gateway.example/antigravity/v1beta"></label>
          <label class="is-wide"><span>API Key${existing ? "（留空保持不变）" : ""}</span><input name="apiKey" type="password" ${existing ? "" : "required"} autocomplete="off"></label>
          ${existing ? "" : '<label class="is-wide"><span>模型名称</span><input name="model" required placeholder="gemini-3.8-flash-high"></label>'}
        </div><p class="settings-inline-error" data-error role="alert" hidden></p></div>
        <footer class="settings-dialog-footer"><div class="settings-dialog-actions"><button type="submit" class="settings-primary-button">保存连接</button></div></footer></form>`;
      const form = host.querySelector("form");
      const error = host.querySelector("[data-error]");
      form.elements.name.value = existing?.name ?? defaults.name ?? "Gemini";
      form.elements.endpoint.value = existing ? "" : defaults.endpoint ?? "";
      form.addEventListener("input", () => props.dirty(true));
      host.querySelector("[data-close]").addEventListener("click", props.close);
      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        if (!form.reportValidity()) return;
        const data = new FormData(form);
        const apiKey = String(data.get("apiKey"));
        const endpoint = String(data.get("endpoint")).trim();
        const button = form.querySelector("[type=submit]");
        button.disabled = true;
        error.hidden = true;
        try {
          const name = String(data.get("name")).trim();
          if (existing) {
            await props.actions.update({name, endpoint: endpoint || null,
              credential: apiKey ? {driver: "api_key", access_token: apiKey} : null, driverConfig: null});
          } else {
            await props.actions.createManual({name, endpoint, credential: {driver: "api_key", access_token: apiKey},
              driverConfig: {format_version: 1, catalog_provider_id: "gemini"},
              model: {kind: "chat", model: String(data.get("model")).trim(),
                capabilities: {}, capability_sources: {}, driver_config: {format_version: 1}}});
          }
          props.changed("Gemini 原生连接已保存");
        } catch (reason) {
          error.textContent = reason instanceof Error ? reason.message : String(reason);
          error.hidden = false;
        } finally {
          button.disabled = false;
        }
      });
      return () => host.replaceChildren();
    },
  }));
}

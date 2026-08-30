export function activate(ctx) {
  return ctx.ui.inject("models.connection-types.v1", (mount) => mount.register({
    id: "opencode-go",
    label: "OpenCode Go",
    detail: "本机登录或 API Key",
    icon: "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M16 6H8v12h8V6zm4 16H4V2h16v20z'/%3E%3C/svg%3E",
    order: 30,
    render(host, _view, rawProps) {
      const props = requireProps(rawProps);
      const existing = props.state.connection;
      const content = document.createElement("section");
      content.innerHTML = `<h2>${existing ? "编辑" : "连接"} OpenCode Go</h2><p>${existing ? "留空 API Key 会保留现有凭据。" : "API Key 留空时读取本机 OpenCode 登录。"}</p><form><label>连接名称<input name="name" required autofocus autocomplete="organization"></label><label>Base URL<input name="endpoint" type="url" ${existing ? "placeholder=\"留空保持不变\"" : "value=\"https://opencode.ai/zen/go/v1\""}></label><label>API Key<input name="apiKey" type="password" autocomplete="off" ${existing ? "placeholder=\"留空保持不变\"" : ""}></label><p data-error role="alert"></p><footer><button type="button" data-cancel>取消</button><button type="submit">${existing ? "保存连接" : "保存并同步"}</button></footer></form>`;
      host.replaceChildren(content);
      const form = content.querySelector("form");
      form.elements.name.value = existing?.name ?? "OpenCode Go";
      content.querySelector("[data-cancel]").addEventListener("click", props.close);
      form.addEventListener("submit", (event) => {
        event.preventDefault();
        const data = new FormData(form);
        const input = {
          name: String(data.get("name")),
        };
        const endpoint = String(data.get("endpoint"));
        const apiKey = String(data.get("apiKey"));
        if (apiKey) {
          input.api_key = apiKey;
          if (endpoint) input.endpoint = endpoint;
        }
        if (existing) {
          submit(form, props.actions.update({
            name: input.name,
            endpoint: endpoint || null,
            credential: apiKey ? {driver: "api_key", access_token: apiKey} : null,
            driverConfig: null,
          }).then(() => props.actions.sync())
            .then(() => props.changed("OpenCode Go 连接已更新")));
          return;
        }
        submit(form, props.actions.startAuth(input)
          .then((started) => props.actions.finishAuth(started.attemptId))
          .then(() => props.actions.sync())
          .then(() => props.changed("OpenCode Go 已连接，模型已同步")));
      });
      return () => host.replaceChildren();
    },
  }));
}

function requireProps(value) {
  if (!value || typeof value !== "object" || typeof value.actions !== "object"
    || typeof value.actions.update !== "function" || typeof value.actions.startAuth !== "function"
    || typeof value.actions.finishAuth !== "function" || typeof value.actions.sync !== "function"
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

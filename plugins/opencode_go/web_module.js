const OPENCODE_ICON = `data:image/svg+xml,${encodeURIComponent(`<svg fill="currentColor" fill-rule="evenodd" height="1em" style="flex:none;line-height:1" viewBox="0 0 24 24" width="1em" xmlns="http://www.w3.org/2000/svg"><title>opencode</title><path d="M16 6H8v12h8V6zm4 16H4V2h16v20z"></path></svg>`)}`;
const CLOSE_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M18 6 6 18"></path><path d="m6 6 12 12"></path></svg>`;
const EYE_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0"></path><circle cx="12" cy="12" r="3"></circle></svg>`;
const EYE_OFF_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m2 2 20 20"></path><path d="M6.71 6.71C4.7 8.1 3.24 10.06 2.06 11.65a1 1 0 0 0 0 .7c2.34 5.64 8.94 8.32 14.24 5.36"></path><path d="M10.73 5.08A10.66 10.66 0 0 1 21.94 11.65a1 1 0 0 1 0 .7 10.83 10.83 0 0 1-2.06 3.1"></path><path d="M14.12 14.12A3 3 0 0 1 9.88 9.88"></path></svg>`;
const SHIELD_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"></path><path d="m9 12 2 2 4-4"></path></svg>`;
const SPINNER_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="is-spinning" aria-hidden="true"><path d="M21 12a9 9 0 1 1-6.219-8.56"></path></svg>`;

export function activate(ctx) {
  return ctx.ui.inject("models.connection-types.v1", (mount) => mount.register({
    id: "opencode-go",
    label: "OpenCode Go",
    detail: "本机登录或 API Key",
    icon: OPENCODE_ICON,
    connectionIcon: OPENCODE_ICON,
    order: 20,
    render(host, _view, rawProps) {
      const props = requireProps(rawProps);
      const existing = props.state.connection;
      host.innerHTML = `<header class="settings-dialog-header"><div class="settings-dialog-heading"><h2 class="settings-dialog-title">${existing ? `编辑 ${escapeHtml(existing.name)}` : "连接 OpenCode Go"}</h2><p class="settings-dialog-description">使用本机 OpenCode 登录或单独的 API Key，模型会自动同步。</p></div>
        <button type="button" class="settings-icon-button" aria-label="关闭" data-close>${CLOSE_ICON}</button></header>
        <form class="settings-dialog-form"><div class="settings-dialog-body"><div class="settings-form-grid">
          <label class="is-wide"><span>连接名称</span><input name="name" aria-label="连接名称" required autocomplete="organization"></label>
          <label class="is-wide"><span>Base URL</span><input name="endpoint" aria-label="Base URL" required type="url" placeholder="https://api.example.com/v1"></label>
          <label class="settings-secret is-wide"><span>API Key（可留空使用本机登录）</span><input name="apiKey" aria-label="API Key" type="password" autocomplete="off" placeholder="sk-…"><button type="button" data-show-key aria-label="显示 API Key">${EYE_ICON}</button></label>
        </div>
        <section class="settings-model-discovery settings-model-discovery--automatic"><header><div><h3>模型自动同步</h3><p>保存后读取账号当前可用的全部模型，无需手动选择。</p></div></header></section>
        <p class="settings-inline-error" data-error role="alert" hidden></p></div>
        <footer class="settings-dialog-footer"><span class="settings-dialog-footer-note">${SHIELD_ICON}凭据保存后不会显示在页面中</span><button type="submit" class="settings-primary-button">保存并同步模型</button></footer></form>`;
      const form = host.querySelector("form");
      form.elements.name.value = existing?.name ?? "OpenCode Go";
      form.elements.endpoint.value = existing ? "" : "https://opencode.ai/zen/go/v1";
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
      queueMicrotask(() => form.elements.name.focus());
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
  button.replaceChildren(htmlNode(SPINNER_ICON), "保存中");
  error.hidden = true;
  error.textContent = "";
  work.catch((reason) => {
    error.textContent = reason instanceof Error ? reason.message : String(reason);
    error.hidden = false;
  }).finally(() => {
    button.disabled = false;
    button.textContent = "保存并同步模型";
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

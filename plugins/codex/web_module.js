const CODEX_ICON = `data:image/svg+xml,${encodeURIComponent(`<svg height="1em" style="flex:none;line-height:1" viewBox="0 0 24 24" width="1em" xmlns="http://www.w3.org/2000/svg"><title>Codex</title><path d="M19.503 0H4.496A4.496 4.496 0 000 4.496v15.007A4.496 4.496 0 004.496 24h15.007A4.496 4.496 0 0024 19.503V4.496A4.496 4.496 0 0019.503 0z" fill="#fff"></path><path d="M9.064 3.344a4.578 4.578 0 012.285-.312c1 .115 1.891.54 2.673 1.275.01.01.024.017.037.021a.09.09 0 00.043 0 4.55 4.55 0 013.046.275l.047.022.116.057a4.581 4.581 0 012.188 2.399c.209.51.313 1.041.315 1.595a4.24 4.24 0 01-.134 1.223.123.123 0 00.03.115c.594.607.988 1.33 1.183 2.17.289 1.425-.007 2.71-.887 3.854l-.136.166a4.548 4.548 0 01-2.201 1.388.123.123 0 00-.081.076c-.191.551-.383 1.023-.74 1.494-.9 1.187-2.222 1.846-3.711 1.838-1.187-.006-2.239-.44-3.157-1.302a.107.107 0 00-.105-.024c-.388.125-.78.143-1.204.138a4.441 4.441 0 01-1.945-.466 4.544 4.544 0 01-1.61-1.335c-.152-.202-.303-.392-.414-.617a5.81 5.81 0 01-.37-.961 4.582 4.582 0 01-.014-2.298.124.124 0 00.006-.056.085.085 0 00-.027-.048 4.467 4.467 0 01-1.034-1.651 3.896 3.896 0 01-.251-1.192 5.189 5.189 0 01.141-1.6c.337-1.112.982-1.985 1.933-2.618.212-.141.413-.251.601-.33.215-.089.43-.164.646-.227a.098.098 0 00.065-.066 4.51 4.51 0 01.829-1.615 4.535 4.535 0 011.837-1.388zm3.482 10.565a.637.637 0 000 1.272h3.636a.637.637 0 100-1.272h-3.636zM8.462 9.23a.637.637 0 00-1.106.631l1.272 2.224-1.266 2.136a.636.636 0 101.095.649l1.454-2.455a.636.636 0 00.005-.64L8.462 9.23z" fill="url(#codex-gradient)"></path><defs><linearGradient gradientUnits="userSpaceOnUse" id="codex-gradient" x1="12" x2="12" y1="3" y2="21"><stop stop-color="#B1A7FF"></stop><stop offset=".5" stop-color="#7A9DFF"></stop><stop offset="1" stop-color="#3941FF"></stop></linearGradient></defs></svg>`)}`;
const CLOSE_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M18 6 6 18"></path><path d="m6 6 12 12"></path></svg>`;
const SHIELD_ICON = (size) => `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"></path><path d="m9 12 2 2 4-4"></path></svg>`;
const SPINNER_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="is-spinning" aria-hidden="true"><path d="M21 12a9 9 0 1 1-6.219-8.56"></path></svg>`;

export function activate(ctx) {
  return ctx.ui.inject("models.connection-types.v1", (mount) => mount.register({
    id: "codex",
    label: "Codex",
    detail: "ChatGPT 订阅登录",
    icon: CODEX_ICON,
    connectionIcon: CODEX_ICON,
    order: 10,
    render(host, _view, rawProps) {
      const props = requireProps(rawProps);
      const existing = props.state.connection;
      const title = existing ? `编辑 ${escapeHtml(existing.name)}` : "连接 Codex";
      host.innerHTML = `<header class="settings-dialog-header"><div class="settings-dialog-heading"><h2 class="settings-dialog-title">${title}</h2><p class="settings-dialog-description">授权 ChatGPT 订阅账号，保存后自动同步可用模型。</p></div>
        <button type="button" class="settings-icon-button" aria-label="关闭" data-close>${CLOSE_ICON}</button></header>
        <form class="settings-dialog-form"><div class="settings-dialog-body"><div class="settings-form-grid">
          <label class="is-wide"><span>连接名称</span><input name="name" aria-label="连接名称" disabled value="Codex"></label>
          <div class="settings-login-card is-wide">${SHIELD_ICON(20)}<span><strong>${existing ? "Codex 已登录" : "使用 ChatGPT 订阅登录"}</strong><small>授权凭据保存在当前 workspace，不会显示在页面中。</small></span><button type="button" data-start>${existing ? "重新登录" : "开始登录"}</button></div>
        </div><div class="settings-device-login" data-challenge role="status" hidden></div>
        <section class="settings-model-discovery settings-model-discovery--automatic"><header><div><h3>模型自动同步</h3><p>保存后读取账号当前可用的全部模型，无需手动选择。</p></div></header></section>
        <p class="settings-inline-error" data-error role="alert" hidden></p></div>
        <footer class="settings-dialog-footer"><span class="settings-dialog-footer-note">${SHIELD_ICON(15)}凭据保存后不会显示在页面中</span><button type="submit" class="settings-primary-button">保存并同步模型</button></footer></form>`;
      let timer = 0;
      let attemptId = "";
      let disposed = false;
      const form = host.querySelector("form");
      const error = host.querySelector("[data-error]");
      const start = host.querySelector("[data-start]");
      const submit = form.querySelector("[type=submit]");
      const finish = async () => {
        if (disposed) return;
        const receipt = await props.actions.finishAuth(attemptId);
        if (disposed) {
          if (receipt.status === "pending") await props.actions.cancelAuth(attemptId);
          return;
        }
        if (receipt.status === "pending") {
          timer = window.setTimeout(() => finish().catch(report), Number(receipt.challenge?.interval ?? 5) * 1000);
          return;
        }
        await props.actions.sync();
        attemptId = "";
        props.changed("Codex 登录完成，模型已同步");
      };
      const report = (reason) => {
        error.textContent = reason instanceof Error ? reason.message : String(reason);
        error.hidden = false;
        start.disabled = false;
      };
      start.addEventListener("click", () => {
        start.disabled = true;
        props.actions.startAuth({})
          .then((receipt) => {
            attemptId = receipt.attemptId;
            if (disposed) return props.actions.cancelAuth(attemptId);
            const challenge = host.querySelector("[data-challenge]");
            challenge.hidden = false;
            challenge.replaceChildren(document.createTextNode("验证码"));
            const code = document.createElement("strong");
            code.textContent = receipt.challenge?.user_code ?? "";
            const link = document.createElement("a");
            link.href = receipt.challenge?.verification_uri ?? "#";
            link.target = "_blank";
            link.rel = "noreferrer";
            link.textContent = "打开登录页面";
            challenge.append(code, link);
            timer = window.setTimeout(() => finish().catch(report), Number(receipt.challenge?.interval ?? 5) * 1000);
          })
          .catch(report);
      });
      host.querySelector("[data-close]").addEventListener("click", props.close);
      form.addEventListener("submit", (event) => {
        event.preventDefault();
        submit.disabled = true;
        submit.replaceChildren(htmlNode(SPINNER_ICON), "保存中");
        error.hidden = true;
        props.actions.sync()
          .then(() => props.changed("Codex 连接已保存，模型已同步"))
          .catch(report)
          .finally(() => {
            submit.disabled = false;
            submit.textContent = "保存并同步模型";
          });
      });
      queueMicrotask(() => start.focus());
      return () => {
        disposed = true;
        window.clearTimeout(timer);
        if (attemptId) void props.actions.cancelAuth(attemptId).catch((reason) => {
          console.error("取消 Codex 登录失败", reason);
        });
        host.replaceChildren();
      };
    },
  }));
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

function requireProps(value) {
  if (!value || typeof value !== "object" || typeof value.actions !== "object"
    || typeof value.actions.startAuth !== "function" || typeof value.actions.finishAuth !== "function"
    || typeof value.actions.cancelAuth !== "function" || typeof value.actions.sync !== "function"
    || typeof value.close !== "function" || typeof value.changed !== "function" || !value.state) {
    throw new Error("models.connection-types.v1 props 无效");
  }
  if (value.state.connection !== null && value.state.connection !== undefined
    && (typeof value.state.connection !== "object" || typeof value.state.connection.id !== "string")) {
    throw new Error("models.connection-types.v1 connection 无效");
  }
  return value;
}

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
    group: "subscription",
    catalogSync: true,
    render(host, _view, rawProps) {
      const props = requireProps(rawProps);
      const existing = props.state.connection;
      const title = existing ? `编辑 ${escapeHtml(existing.name)}` : "连接 Codex";
      host.innerHTML = `<header class="settings-dialog-header"><div class="settings-dialog-heading"><h2 class="settings-dialog-title">${title}</h2><p class="settings-dialog-description">授权 ChatGPT 订阅账号，登录完成后自动探测并选择要启用的模型。</p></div>
        <button type="button" class="settings-icon-button" aria-label="关闭" data-close>${CLOSE_ICON}</button></header>
        <form class="settings-dialog-form"><div class="settings-dialog-body"><div class="settings-form-grid">
          <label class="is-wide"><span>连接名称</span><input name="name" aria-label="连接名称" disabled value="Codex"></label>
          <div class="settings-login-card is-wide" data-login-card>${SHIELD_ICON(20)}<span><strong>${existing ? "Codex 已授权连接" : "使用 ChatGPT 订阅授权登录"}</strong><small>授权凭据保存在本地运行环境中，不会在页面中明文回显。</small></span><button type="button" data-start>${existing ? "重新授权" : "前往授权"}</button></div>
        </div><div class="settings-device-login" data-challenge role="status" aria-live="polite" hidden></div>
        <section class="settings-model-discovery settings-model-discovery--automatic"><header><div><h3>可用模型列表</h3><p>勾选需要开启的模型；后续可随时重新探测与增减。</p></div></header></section>
        <p class="settings-inline-error" data-error role="alert" hidden></p></div>
        <footer class="settings-dialog-footer"><span class="settings-dialog-footer-note">${SHIELD_ICON(15)}授权凭据保存后不会在页面中明文回显</span><button type="submit" class="settings-primary-button">探测并选择模型</button></footer></form>`;
      let timer = 0;
      let ticker = 0;
      let attemptId = "";
      let deadline = 0;
      let transientFailures = 0;
      let disposed = false;
      const form = host.querySelector("form");
      const error = host.querySelector("[data-error]");
      const start = host.querySelector("[data-start]");
      const card = host.querySelector("[data-login-card]");
      const challenge = host.querySelector("[data-challenge]");
      const submit = form.querySelector("[type=submit]");
      const stopWaiting = () => {
        window.clearTimeout(timer);
        window.clearInterval(ticker);
      };
      const showLogin = (state) => {
        challenge.hidden = false;
        card.hidden = true;
        challenge.dataset.state = state.kind;
        challenge.replaceChildren(loginView(state));
      };
      const resetLogin = () => {
        stopWaiting();
        challenge.hidden = true;
        card.hidden = false;
        start.disabled = false;
        start.focus();
      };
      // 失败停在状态卡里：写明原因并给出重新开始；过期与取消都会通知后端放弃这次尝试。
      const fail = (reason, expired = false) => {
        stopWaiting();
        const abandoned = attemptId;
        attemptId = "";
        if (abandoned) void props.actions.cancelAuth(abandoned).catch(() => {});
        showLogin({kind: "failed", expired, message: expired
          ? "验证码已过期，没有完成授权。重新开始会生成新的验证码。"
          : reason instanceof Error ? reason.message : String(reason)});
      };
      const poll = (interval) => {
        timer = window.setTimeout(() => finish().catch(retryOrFail), interval * 1000);
      };
      // 网络抖动或服务端 5xx 不算登录失败，最多连续重试三次；尝试已不存在说明已过期。
      const retryOrFail = (reason) => {
        if (disposed || !attemptId) return;
        const status = reason?.status ?? 0;
        if (String(reason?.message ?? "").includes("auth attempt 不存在")) return fail(reason, true);
        if ((status === 0 || status >= 500) && transientFailures < 3) {
          transientFailures += 1;
          poll(5);
          return;
        }
        fail(reason);
      };
      const finish = async () => {
        if (disposed) return;
        const receipt = await props.actions.finishAuth(attemptId);
        if (disposed) {
          if (receipt.status === "pending") await props.actions.cancelAuth(attemptId);
          return;
        }
        transientFailures = 0;
        if (receipt.status === "pending") {
          poll(Number(receipt.challenge?.interval ?? 5));
          return;
        }
        stopWaiting();
        attemptId = "";
        showLogin({kind: "done"});
        let selected = false;
        try {
          selected = await props.actions.selectModels();
        } catch (reason) {
          report(new Error(`Codex 已登录，但模型选择未完成：${reason instanceof Error ? reason.message : String(reason)} 可以点“探测并选择模型”重试。`));
          return;
        }
        props.changed(selected ? "Codex 登录完成，模型选择已保存" : "Codex 登录完成，模型选择未改动");
      };
      const tick = () => {
        const left = Math.max(0, deadline - Date.now());
        const remaining = challenge.querySelector("[data-remaining]");
        if (remaining) remaining.textContent = `验证码 ${formatClock(left)} 后过期`;
        if (!left) fail(null, true);
      };
      const report = (reason) => {
        error.textContent = reason instanceof Error ? reason.message : String(reason);
        error.hidden = false;
        start.disabled = false;
      };
      const begin = () => {
        start.disabled = true;
        error.hidden = true;
        showLogin({kind: "starting"});
        props.actions.startAuth({})
          .then((receipt) => {
            attemptId = receipt.attemptId;
            props.dirty(true);
            if (disposed) return props.actions.cancelAuth(attemptId);
            deadline = Date.now() + LOGIN_TTL_MS;
            transientFailures = 0;
            showLogin({kind: "waiting", code: receipt.challenge?.user_code ?? "", uri: receipt.challenge?.verification_uri ?? ""});
            challenge.querySelector("[data-open]")?.focus();
            ticker = window.setInterval(tick, 1000);
            tick();
            poll(Number(receipt.challenge?.interval ?? 5));
          })
          .catch((reason) => fail(reason));
      };
      start.addEventListener("click", begin);
      challenge.addEventListener("click", (event) => {
        const target = event.target instanceof Element ? event.target.closest("button") : null;
        if (!target) return;
        if (target.hasAttribute("data-restart")) begin();
        else if (target.hasAttribute("data-cancel")) {
          const abandoned = attemptId;
          attemptId = "";
          if (abandoned) void props.actions.cancelAuth(abandoned).catch(() => {});
          resetLogin();
        } else if (target.dataset.copy) void copyText(target);
      });
      host.querySelector("[data-close]").addEventListener("click", props.close);
      form.addEventListener("submit", (event) => {
        event.preventDefault();
        submit.disabled = true;
        submit.replaceChildren(htmlNode(SPINNER_ICON), "保存中");
        error.hidden = true;
        props.actions.selectModels()
          .then((selected) => props.changed(selected ? "模型选择已保存" : "模型选择未改动"))
          .catch(report)
          .finally(() => {
            submit.disabled = false;
            submit.textContent = "探测并选择模型";
          });
      });
      queueMicrotask(() => start.focus());
      return () => {
        disposed = true;
        stopWaiting();
        if (attemptId) void props.actions.cancelAuth(attemptId).catch((reason) => {
          console.error("取消 Codex 登录失败", reason);
        });
        host.replaceChildren();
      };
    },
  }));
}

// 后端放弃未完成登录的时限（models 插件 _AUTH_ATTEMPT_TTL_SECONDS），倒计时按它显示。
const LOGIN_TTL_MS = 15 * 60 * 1000;
const COPY_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect width="14" height="14" x="8" y="8" rx="2"></rect><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"></path></svg>`;
const ALERT_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"></circle><path d="M12 8v4"></path><path d="M12 16h.01"></path></svg>`;
const CHECK_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5"></path></svg>`;

// 登录状态卡：发起中、等待授权、已授权、失败各给一句现状和下一步，取消与重新开始都在卡内。
function loginView(state) {
  if (state.kind === "starting") {
    return htmlNode(`<div class="settings-device-head">${SPINNER_ICON}<span><strong>正在向 ChatGPT 申请验证码…</strong></span></div>`);
  }
  if (state.kind === "done") {
    return htmlNode(`<div class="settings-device-head is-done">${CHECK_ICON}<span><strong>ChatGPT 账号已授权</strong><small>正在读取可用模型，稍后请勾选要开启的模型。</small></span></div>`);
  }
  if (state.kind === "failed") {
    return htmlNode(`<div class="settings-device-head is-failed">${ALERT_ICON}<span><strong>${state.expired ? "验证码已过期" : "登录未完成"}</strong><small>${escapeHtml(state.message)}</small></span><button type="button" data-restart>重新开始</button></div>`);
  }
  const attr = (value) => escapeHtml(value).replaceAll('"', "&quot;");
  const uri = attr(state.uri);
  return htmlNode(`<div class="settings-device-wait">
    <div class="settings-device-head">${SPINNER_ICON}<span><strong>在 ChatGPT 页面输入验证码</strong><small>打开登录页并输入下面的验证码；授权完成后这里会自动继续。</small></span><button type="button" data-cancel>取消</button></div>
    <div class="settings-device-code"><strong>${escapeHtml(state.code)}</strong><button type="button" data-copy="${attr(state.code)}" aria-label="复制验证码">${COPY_ICON}<span>复制</span></button></div>
    <div class="settings-device-link"><a href="${uri}" target="_blank" rel="noreferrer" data-open>打开登录页面</a><code title="${uri}">${uri}</code><button type="button" data-copy="${uri}" aria-label="复制登录链接">${COPY_ICON}<span>复制链接</span></button></div>
    <small class="settings-device-note"><span data-remaining></span> · 在另一台设备上打开链接也可以</small>
  </div>`);
}

function formatClock(ms) {
  const seconds = Math.ceil(ms / 1000);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

// 远程以 http 访问时没有剪贴板 API，退回选中文本复制；按钮文字短暂改为“已复制”。
async function copyText(button) {
  const value = button.dataset.copy ?? "";
  try {
    await navigator.clipboard.writeText(value);
  } catch {
    const field = document.createElement("textarea");
    field.value = value;
    field.setAttribute("readonly", "");
    field.style.position = "fixed";
    field.style.opacity = "0";
    document.body.append(field);
    field.select();
    document.execCommand("copy");
    field.remove();
  }
  const label = button.querySelector("span");
  if (!label) return;
  const original = label.textContent;
  label.textContent = "已复制";
  window.setTimeout(() => { label.textContent = original; }, 1500);
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
    || typeof value.actions.cancelAuth !== "function" || typeof value.actions.sync !== "function" || typeof value.actions.selectModels !== "function"
    || typeof value.dirty !== "function" || typeof value.close !== "function" || typeof value.changed !== "function" || !value.state) {
    throw new Error("models.connection-types.v1 props 无效");
  }
  if (value.state.connection !== null && value.state.connection !== undefined
    && (typeof value.state.connection !== "object" || typeof value.state.connection.id !== "string")) {
    throw new Error("models.connection-types.v1 connection 无效");
  }
  return value;
}

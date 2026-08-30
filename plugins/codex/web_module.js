export function activate(ctx) {
  return ctx.ui.inject("models.connection-types.v1", (mount) => mount.register({
    id: "codex",
    label: "Codex",
    detail: "使用 ChatGPT 订阅登录",
    icon: "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Crect width='24' height='24' rx='5' fill='%237A9DFF'/%3E%3Cpath d='M8.5 8.2 6.8 12l1.7 3.8m7-7.6 1.7 3.8-1.7 3.8M11 16h4' fill='none' stroke='white' stroke-width='1.7' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E",
    order: 20,
    render(host, _view, rawProps) {
      const props = requireProps(rawProps);
      const content = document.createElement("section");
      content.className = "codex-provider-dialog";
      content.innerHTML = `<h2>${props.state.connection ? "重新连接" : "连接"} Codex</h2><p>授权 ChatGPT 订阅账号，完成后自动同步可用模型。</p><div data-challenge role="status"></div><p data-error role="alert"></p><footer><button type="button" data-cancel>取消</button><button type="button" data-start autofocus>开始登录</button></footer>`;
      host.replaceChildren(content);
      let timer = 0;
      let attemptId = "";
      let disposed = false;
      const error = content.querySelector("[data-error]");
      const start = content.querySelector("[data-start]");
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
        start.disabled = false;
      };
      start.addEventListener("click", () => {
        start.disabled = true;
        props.actions.startAuth({})
          .then((receipt) => {
            attemptId = receipt.attemptId;
            if (disposed) return props.actions.cancelAuth(attemptId);
            const challenge = content.querySelector("[data-challenge]");
            challenge.replaceChildren();
            const code = document.createElement("strong");
            code.textContent = receipt.challenge?.user_code ?? "";
            const link = document.createElement("a");
            link.href = receipt.challenge?.verification_uri ?? "#";
            link.target = "_blank";
            link.rel = "noreferrer";
            link.textContent = "打开登录页面";
            challenge.append("验证码：", code, link);
            timer = window.setTimeout(() => finish().catch(report), Number(receipt.challenge?.interval ?? 5) * 1000);
          })
          .catch(report);
      });
      content.querySelector("[data-cancel]").addEventListener("click", props.close);
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

function b(m) {
  return m.ui.inject("models.connection-types.v1", (u) => u.register({
    id: "gemini",
    label: "Gemini 原生 API",
    order: 35,
    detail: "连接 Google 或提供原生 Gemini 接口的网关",
    catalogSync: !0,
    templates: [{
      id: "gemini-native",
      label: "Gemini 原生 API",
      detail: "Google 与原生接口网关",
      order: 35,
      defaults: { name: "Gemini", endpoint: "https://generativelanguage.googleapis.com/v1beta" }
    }],
    render(n, y, f) {
      var d, c;
      const e = f;
      if (!(e != null && e.state) || typeof ((d = e.actions) == null ? void 0 : d.createManual) != "function")
        throw new Error("models.connection-types.v1 props 无效");
      const t = e.state.connection, l = ((c = e.state.template) == null ? void 0 : c.defaults) ?? {};
      n.innerHTML = `<header class="settings-dialog-header"><div class="settings-dialog-heading">
        <h2 class="settings-dialog-title">${t ? "编辑 Gemini 连接" : "连接 Gemini 原生 API"}</h2>
        <p class="settings-dialog-description">使用原生接口连接 Google 或你的网关。保存前会发送短消息验证模型。</p>
        </div><button type="button" class="settings-icon-button" data-close aria-label="关闭">×</button></header>
        <form class="settings-dialog-form"><div class="settings-dialog-body"><div class="settings-form-grid">
          <label class="is-wide"><span>连接名称</span><input name="name" required autocomplete="organization"></label>
          <label class="is-wide"><span>Base URL${t ? "（留空保持不变）" : ""}</span><input name="endpoint" type="url" ${t ? "" : "required"} placeholder="https://your-gateway.example/antigravity/v1beta"></label>
          <label class="is-wide"><span>API Key${t ? "（留空保持不变）" : ""}</span><input name="apiKey" type="password" ${t ? "" : "required"} autocomplete="off"></label>
          ${t ? "" : '<label class="is-wide"><span>模型名称</span><input name="model" required placeholder="gemini-3.8-flash-high"></label>'}
        </div><p class="settings-inline-error" data-error role="alert" hidden></p></div>
        <footer class="settings-dialog-footer"><div class="settings-dialog-actions"><button type="submit" class="settings-primary-button">保存连接</button></div></footer></form>`;
      const i = n.querySelector("form"), r = n.querySelector("[data-error]");
      return i.elements.name.value = (t == null ? void 0 : t.name) ?? l.name ?? "Gemini", i.elements.endpoint.value = t ? "" : l.endpoint ?? "", i.addEventListener("input", () => e.dirty(!0)), n.querySelector("[data-close]").addEventListener("click", e.close), i.addEventListener("submit", async (v) => {
        if (v.preventDefault(), !i.reportValidity()) return;
        const s = new FormData(i), o = String(s.get("apiKey")), p = String(s.get("endpoint")).trim(), g = i.querySelector("[type=submit]");
        g.disabled = !0, r.hidden = !0;
        try {
          const a = String(s.get("name")).trim();
          t ? await e.actions.update({
            name: a,
            endpoint: p || null,
            credential: o ? { driver: "api_key", access_token: o } : null,
            driverConfig: null
          }) : await e.actions.createManual({
            name: a,
            endpoint: p,
            credential: { driver: "api_key", access_token: o },
            driverConfig: { format_version: 1, catalog_provider_id: "gemini" },
            model: {
              kind: "chat",
              model: String(s.get("model")).trim(),
              capabilities: {},
              capability_sources: {},
              driver_config: { format_version: 1 }
            }
          }), e.changed("Gemini 原生连接已保存");
        } catch (a) {
          r.textContent = a instanceof Error ? a.message : String(a), r.hidden = !1;
        } finally {
          g.disabled = !1;
        }
      }), () => n.replaceChildren();
    }
  }));
}
export {
  b as activate
};

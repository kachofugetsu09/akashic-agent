const DEEPSEEK_ICON = svgData(`<svg height="1em" style="flex:none;line-height:1" viewBox="0 0 24 24" width="1em" xmlns="http://www.w3.org/2000/svg"><title>DeepSeek</title><path d="M23.748 4.482c-.254-.124-.364.113-.512.234-.051.039-.094.09-.137.136-.372.397-.806.657-1.373.626-.829-.046-1.537.214-2.163.848-.133-.782-.575-1.248-1.247-1.548-.352-.156-.708-.311-.955-.65-.172-.241-.219-.51-.305-.774-.055-.16-.11-.323-.293-.35-.2-.031-.278.136-.356.276-.313.572-.434 1.202-.422 1.84.027 1.436.633 2.58 1.838 3.393.137.093.172.187.129.323-.082.28-.18.552-.266.833-.055.179-.137.217-.329.14a5.526 5.526 0 01-1.736-1.18c-.857-.828-1.631-1.742-2.597-2.458a11.365 11.365 0 00-.689-.471c-.985-.957.13-1.743.388-1.836.27-.098.093-.432-.779-.428-.872.004-1.67.295-2.687.684a3.055 3.055 0 01-.465.137 9.597 9.597 0 00-2.883-.102c-1.885.21-3.39 1.102-4.497 2.623C.082 8.606-.231 10.684.152 12.85c.403 2.284 1.569 4.175 3.36 5.653 1.858 1.533 3.997 2.284 6.438 2.14 1.482-.085 3.133-.284 4.994-1.86.47.234.962.327 1.78.397.63.059 1.236-.03 1.705-.128.735-.156.684-.837.419-.961-2.155-1.004-1.682-.595-2.113-.926 1.096-1.296 2.746-2.642 3.392-7.003.05-.347.007-.565 0-.845-.004-.17.035-.237.23-.256a4.173 4.173 0 001.545-.475c1.396-.763 1.96-2.015 2.093-3.517.02-.23-.004-.467-.247-.588zM11.581 18c-2.089-1.642-3.102-2.183-3.52-2.16-.392.024-.321.471-.235.763.09.288.207.486.371.739.114.167.192.416-.113.603-.673.416-1.842-.14-1.897-.167-1.361-.802-2.5-1.86-3.301-3.307-.774-1.393-1.224-2.887-1.298-4.482-.02-.386.093-.522.477-.592a4.696 4.696 0 011.529-.039c2.132.312 3.946 1.265 5.468 2.774.868.86 1.525 1.887 2.202 2.891.72 1.066 1.494 2.082 2.48 2.914.348.292.625.514.891.677-.802.09-2.14.11-3.054-.614zm1-6.44a.306.306 0 01.415-.287.302.302 0 01.2.288.306.306 0 01-.31.307.303.303 0 01-.304-.308zm3.11 1.596c-.2.081-.399.151-.59.16a1.245 1.245 0 01-.798-.254c-.274-.23-.47-.358-.552-.758a1.73 1.73 0 01.016-.588c.07-.327-.008-.537-.239-.727-.187-.156-.426-.199-.688-.199a.559.559 0 01-.254-.078c-.11-.054-.2-.19-.114-.358.028-.054.16-.186.192-.21.356-.202.767-.136 1.146.016.352.144.618.408 1.001.782.391.451.462.576.685.914.176.265.336.537.445.848.067.195-.019.354-.25.452z" fill="#4D6BFE"></path></svg>`);
// 智谱与 Z.ai 图标来自 lobehub/icons（MIT），与 magpie 同源。
const ZHIPU_ICON = svgData(`<svg height="1em" style="flex:none;line-height:1" viewBox="0 0 24 24" width="1em" xmlns="http://www.w3.org/2000/svg"><title>Zhipu</title><path d="M11.991 23.503a.24.24 0 00-.244.248.24.24 0 00.244.249.24.24 0 00.245-.249.24.24 0 00-.22-.247l-.025-.001zM9.671 5.365a1.697 1.697 0 011.099 2.132l-.071.172-.016.04-.018.054c-.07.16-.104.32-.104.498-.035.71.47 1.279 1.186 1.314h.366c1.309.053 2.338 1.173 2.286 2.523-.052 1.332-1.152 2.38-2.478 2.327h-.174c-.715.018-1.274.64-1.239 1.368 0 .124.018.23.053.337.209.373.54.658.96.8.75.23 1.517-.125 1.9-.782l.018-.035c.402-.64 1.17-.96 1.92-.711.854.284 1.378 1.226 1.099 2.167a1.661 1.661 0 01-2.077 1.102 1.711 1.711 0 01-.907-.711l-.017-.035c-.2-.323-.463-.58-.851-.711l-.056-.018a1.646 1.646 0 00-1.954.746 1.66 1.66 0 01-1.065.764 1.677 1.677 0 01-1.989-1.279c-.209-.906.332-1.83 1.257-2.043a1.51 1.51 0 01.296-.035h.018c.68-.071 1.151-.622 1.116-1.333a1.307 1.307 0 00-.227-.693 2.515 2.515 0 01-.366-1.403 2.39 2.39 0 01.366-1.208c.14-.195.21-.444.227-.693.018-.71-.506-1.261-1.186-1.332l-.07-.018a1.43 1.43 0 01-.299-.07l-.05-.019a1.7 1.7 0 01-1.047-2.114 1.68 1.68 0 012.094-1.101zm-5.575 10.11c.26-.264.639-.367.994-.27.355.096.633.379.728.74.095.362-.007.748-.267 1.013-.402.41-1.053.41-1.455 0a1.062 1.062 0 010-1.482zm14.845-.294c.359-.09.738.024.992.297.254.274.344.665.237 1.025-.107.36-.396.634-.756.718-.551.128-1.1-.22-1.23-.781a1.05 1.05 0 01.757-1.26zm-.064-4.39c.314.32.49.753.49 1.206 0 .452-.176.886-.49 1.206-.315.32-.74.5-1.185.5-.444 0-.87-.18-1.184-.5a1.727 1.727 0 010-2.412 1.654 1.654 0 012.369 0zm-11.243.163c.364.484.447 1.128.218 1.691a1.665 1.665 0 01-2.188.923c-.855-.36-1.26-1.358-.907-2.228a1.68 1.68 0 011.33-1.038c.593-.08 1.183.169 1.547.652zm11.545-4.221c.368 0 .708.2.892.524.184.324.184.724 0 1.048a1.026 1.026 0 01-.892.524c-.568 0-1.03-.47-1.03-1.048 0-.579.462-1.048 1.03-1.048zm-14.358 0c.368 0 .707.2.891.524.184.324.184.724 0 1.048a1.026 1.026 0 01-.891.524c-.569 0-1.03-.47-1.03-1.048 0-.579.461-1.048 1.03-1.048zm10.031-1.475c.925 0 1.675.764 1.675 1.706s-.75 1.705-1.675 1.705-1.674-.763-1.674-1.705c0-.942.75-1.706 1.674-1.706zm-2.626-.684c.362-.082.653-.356.761-.718a1.062 1.062 0 00-.238-1.028 1.017 1.017 0 00-.996-.294c-.547.14-.881.7-.752 1.257.13.558.675.907 1.225.783zm0 16.876c.359-.087.644-.36.75-.72a1.062 1.062 0 00-.237-1.019 1.018 1.018 0 00-.985-.301 1.037 1.037 0 00-.762.717c-.108.361-.017.754.239 1.028.245.263.606.377.953.305l.043-.01zM17.19 3.5a.631.631 0 00.628-.64c0-.355-.279-.64-.628-.64a.631.631 0 00-.628.64c0 .355.28.64.628.64zm-10.38 0a.631.631 0 00.628-.64c0-.355-.28-.64-.628-.64a.631.631 0 00-.628.64c0 .355.279.64.628.64zm-5.182 7.852a.631.631 0 00-.628.64c0 .354.28.639.628.639a.63.63 0 00.627-.606l.001-.034a.62.62 0 00-.628-.64zm5.182 9.13a.631.631 0 00-.628.64c0 .355.279.64.628.64a.631.631 0 00.628-.64c0-.355-.28-.64-.628-.64zm10.38.018a.631.631 0 00-.628.64c0 .355.28.64.628.64a.631.631 0 00.628-.64c0-.355-.279-.64-.628-.64zm5.182-9.148a.631.631 0 00-.628.64c0 .354.279.639.628.639a.631.631 0 00.628-.64c0-.355-.28-.64-.628-.64zm-.384-4.992a.24.24 0 00.244-.249.24.24 0 00-.244-.249.24.24 0 00-.244.249c0 .142.122.249.244.249zM11.991.497a.24.24 0 00.245-.248A.24.24 0 0011.99 0a.24.24 0 00-.244.249c0 .133.108.236.223.247l.021.001zM2.011 6.36a.24.24 0 00.245-.249.24.24 0 00-.244-.249.24.24 0 00-.244.249.24.24 0 00.244.249zm0 11.263a.24.24 0 00-.243.248.24.24 0 00.244.249.24.24 0 00.244-.249.252.252 0 00-.244-.248zm19.995-.018a.24.24 0 00-.245.248.24.24 0 00.245.25.24.24 0 00.244-.25.252.252 0 00-.244-.248z" fill="#3859FF" fill-rule="nonzero"></path></svg>`);
const ZAI_ICON = svgData(`<svg fill="currentColor" fill-rule="evenodd" height="1em" style="flex:none;line-height:1" viewBox="0 0 24 24" width="1em" xmlns="http://www.w3.org/2000/svg"><title>Z.ai</title><path d="M12.105 2L9.927 4.953H.653L2.83 2h9.276zM23.254 19.048L21.078 22h-9.242l2.174-2.952h9.244zM24 2L9.264 22H0L14.736 2H24z"></path></svg>`);
const CLOSE_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M18 6 6 18"></path><path d="m6 6 12 12"></path></svg>`;
const EYE_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0"></path><circle cx="12" cy="12" r="3"></circle></svg>`;
const EYE_OFF_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m2 2 20 20"></path><path d="M6.71 6.71C4.7 8.1 3.24 10.06 2.06 11.65a1 1 0 0 0 0 .7c2.34 5.64 8.94 8.32 14.24 5.36"></path><path d="M10.73 5.08A10.66 10.66 0 0 1 21.94 11.65a1 1 0 0 1 0 .7 10.83 10.83 0 0 1-2.06 3.1"></path><path d="M14.12 14.12A3 3 0 0 1 9.88 9.88"></path></svg>`;
const SHIELD_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"></path><path d="m9 12 2 2 4-4"></path></svg>`;
const SPINNER_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="is-spinning" aria-hidden="true"><path d="M21 12a9 9 0 1 1-6.219-8.56"></path></svg>`;

export function activate(ctx) {
  return ctx.ui.inject("models.connection-types.v1", (mount) => mount.register({
    id: "openai-compatible",
    label: "OpenAI Compatible",
    detail: "API Key 与任意 OpenAI 格式服务",
    order: 30,
    embeddingApiKey: true,
    catalogSync: true,
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
          thinkingFormat: "deepseek",
          modelHint: "deepseek-chat",
        },
      },
      // GLM Coding Plan 走独立的 OpenAI 地址；按量付费的 key 把地址里的 /coding 去掉即可。
      {
        id: "zhipu",
        label: "智谱 GLM",
        detail: "GLM Coding Plan · 可读套餐额度",
        icon: ZHIPU_ICON,
        order: 31,
        defaults: {
          name: "智谱 GLM",
          endpoint: "https://open.bigmodel.cn/api/coding/paas/v4",
          provider: "zai",
          thinkingFormat: "deepseek",
          modelHint: "glm-5.1",
        },
      },
      {
        id: "zai",
        label: "Z.ai",
        detail: "GLM Coding Plan（海外）· 可读套餐额度",
        icon: ZAI_ICON,
        order: 32,
        defaults: {
          name: "Z.ai",
          endpoint: "https://api.z.ai/api/coding/paas/v4",
          provider: "zai",
          thinkingFormat: "deepseek",
          modelHint: "glm-5.1",
        },
      },
      {
        id: "custom-api",
        label: "自定义 API",
        detail: "连接任意兼容服务",
        group: "custom",
        order: 40,
        defaults: {name: "", endpoint: "", provider: "openai"},
      },
    ],
    render(host, _view, rawProps) {
      const props = requireProps(rawProps);
      const existing = props.state.connection;
      const defaults = props.state.template?.defaults ?? {};
      const provider = defaults.provider ?? "openai";
      // 预设（DeepSeek、智谱等）的标题与占位文字取自模板；自定义 API 用通用文案。
      const preset = !existing && props.state.template && props.state.template.id !== "custom-api" ? props.state.template : null;
      const title = existing ? `编辑 ${existing.name}` : preset ? `连接 ${preset.label}` : "添加 OpenAI 兼容服务";
      const description = existing
        ? "更新服务地址或密钥时会向已启用模型发送测试验证；若验证失败将保留原配置。仅修改名称时不调用模型。"
        : "输入接口地址与密钥，读取模型目录后保存连接。";
      host.innerHTML = `<header class="settings-dialog-header"><div class="settings-dialog-heading"><h2 class="settings-dialog-title">${escapeHtml(title)}</h2><p class="settings-dialog-description">${description}</p></div>
        <button type="button" class="settings-icon-button" aria-label="关闭" data-close>${CLOSE_ICON}</button></header>
        <form class="settings-dialog-form"><div class="settings-dialog-body"><div class="settings-form-grid">
          <label class="is-wide"><span>连接名称</span><input name="name" aria-label="连接名称" required autocomplete="organization" placeholder="${preset ? `例如：${escapeHtml(defaults.name ?? preset.label)}` : "例如：公司网关"}"></label>
          <label class="is-wide"><span>Base URL</span><input name="endpoint" aria-label="Base URL" ${existing ? "" : "required"} type="url" placeholder="${existing?.endpointHost ? `当前 ${existing.endpointHost.replace(/[^a-z0-9.:-]/giu, "")}，填写新地址才会更换` : "https://api.example.com/v1"}"></label>
          <label class="settings-secret is-wide"><span>API Key</span><input name="apiKey" aria-label="API Key" type="password" ${existing ? "" : "required"} autocomplete="off" placeholder="${existing ? "留空保留现有密钥" : "sk-…"}"><button type="button" data-show-key aria-label="显示 API Key">${EYE_ICON}</button></label>
        </div>
        ${existing ? "" : `<details class="settings-advanced"><summary>高级设置</summary><p>Provider ID 用于辅助识别模型功能；思考参数格式通常保持默认即可。</p><div class="settings-form-grid"><label class="is-wide"><span>Provider ID</span><input name="provider" aria-label="Provider ID" required placeholder="例如：openai"></label><label class="is-wide"><span>思考参数格式</span><select name="thinkingFormat" aria-label="思考参数格式"><option value="none">标准兼容格式（默认）</option><option value="deepseek">DeepSeek / GLM 格式</option></select></label></div></details>`}
        <p class="settings-credential-note">${SHIELD_ICON}<span>API Key 保存后不会在页面中明文回显</span></p>
        ${existing ? "" : `<section class="settings-model-discovery"><header><div><h3>可用模型</h3><p>读取服务的模型目录，先用其中一个验证连接；其余模型建好后在详情里勾选。</p></div></header>
          <div class="settings-discovery-empty" data-discovery-empty>
            <button type="button" class="settings-primary-button" data-discover>自动探测模型</button>
            <button type="button" class="settings-text-button" data-manual>手动输入模型名</button>
          </div>
          <p class="settings-discovery-status" data-picked role="status" hidden></p>
          <div class="settings-discovery-manual" data-discovery-manual hidden><label><span>模型名称</span><input name="manualModel" aria-label="模型名称" placeholder="例如：${escapeHtml(defaults.modelHint ?? "gpt-4o")}"></label><p>手动添加的模型将先作为通用对话模型，保存后可随时重新探测能力。</p><label class="settings-manual-confirm"><input type="checkbox" name="manualConfirm"><span>我确认将此模型用于对话，并在保存前发送测试请求。</span></label></div>
          <p class="settings-discovery-status" data-status role="status" aria-live="polite" hidden></p>
        </section>`}
        <p class="settings-inline-error" data-error role="alert" hidden></p></div>
        <footer class="settings-dialog-footer" data-footer ${existing ? "" : "hidden"}><span class="settings-dialog-footer-note" data-footer-note>${SHIELD_ICON}保存前会向所选模型发送一条短消息，验证对话用途</span><div class="settings-dialog-actions">${existing ? "" : '<button type="button" class="settings-secondary-button" data-rescan>重新探测</button>'}<button type="submit" class="settings-primary-button">保存连接</button></div></footer></form>`;
      const form = host.querySelector("form");
      const changed = () => props.dirty(true);
      form.addEventListener("input", changed);
      form.addEventListener("change", changed);
      form.elements.name.value = existing?.name ?? defaults.name ?? "";
      form.elements.endpoint.value = defaults.endpoint ?? "";
      if (form.elements.provider) form.elements.provider.value = provider;
      if (form.elements.thinkingFormat) form.elements.thinkingFormat.value = defaults.thinkingFormat ?? "none";
      const showKey = host.querySelector("[data-show-key]");
      showKey.addEventListener("click", () => {
        const visible = form.elements.apiKey.type === "text";
        form.elements.apiKey.type = visible ? "password" : "text";
        showKey.setAttribute("aria-label", visible ? "显示 API Key" : "隐藏 API Key");
        showKey.innerHTML = visible ? EYE_ICON : EYE_OFF_ICON;
      });
      host.querySelector("[data-close]").addEventListener("click", props.close);
      const stopDiscovery = existing ? () => {} : setupDiscovery(host, form, props);
      const pickedModels = () => host._pickedModels?.() ?? [];
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
        if (!form.reportValidity()) return;
        const manualVisible = !host.querySelector("[data-discovery-manual]").hidden;
        const selected = manualVisible
          ? [{kind: "chat", model: form.elements.manualModel.value.trim()}]
          : pickedModels();
        if (!selected.length || !selected[0].model) {
          const error = host.querySelector("[data-error]");
          error.textContent = "请先探测模型目录，或手动填写型号。";
          error.hidden = false;
          return;
        }
        // 连接带一个已验证模型建好后，宿主会展开它的详情，其余模型在那里勾选。
        submit(form, props.actions.createManual({
          name: String(data.get("name")),
          endpoint: String(data.get("endpoint")),
          credential: {driver: "api_key", access_token: String(data.get("apiKey"))},
          driverConfig: {format_version: 1, catalog_provider_id: String(data.get("provider")), thinking_format: String(data.get("thinkingFormat")), allow_unverified_manual: true},
          model: modelInput(selected[0], manualVisible),
        }).then(() => props.changed(`${String(data.get("name")).trim() || "连接"} 已保存`)));
      });
      queueMicrotask(() => form.elements.name.focus());
      return () => {
        stopDiscovery();
        host.replaceChildren();
      };
    },
  }));
}

function svgData(svg) {
  return `data:image/svg+xml,${encodeURIComponent(svg)}`;
}

function requireProps(value) {
  if (!value || typeof value !== "object" || typeof value.actions !== "object"
    || typeof value.actions.discover !== "function" || typeof value.actions.createManual !== "function"
    || typeof value.actions.update !== "function" || typeof value.actions.sync !== "function"
    || typeof value.actions.discoverSaved !== "function" || typeof value.actions.addModel !== "function"
    || typeof value.actions.verifyModel !== "function"
    || typeof value.dirty !== "function" || typeof value.close !== "function" || typeof value.changed !== "function" || !value.state) {
    throw new Error("models.connection-types.v1 props 无效");
  }
  if (value.state.connection !== null && value.state.connection !== undefined
    && (typeof value.state.connection !== "object" || typeof value.state.connection.id !== "string")) {
    throw new Error("models.connection-types.v1 connection 无效");
  }
  return value;
}

// 新建连接的探测：拿表单当前值读目录 → 宿主勾选层 → 保存时批量验证采纳。
function setupDiscovery(host, form, props) {
  const manual = host.querySelector("[data-discovery-manual]");
  const pickedLine = host.querySelector("[data-picked]");
  const status = host.querySelector("[data-status]");
  const footer = host.querySelector("[data-footer]");
  const footerNote = host.querySelector("[data-footer-note]");
  const manualInput = form.elements.manualModel;
  const manualConfirm = form.elements.manualConfirm;
  const discoverButton = host.querySelector("[data-discover]");
  const rescanButton = host.querySelector("[data-rescan]");
  const owner = createDiscoveryOwner();
  let picked = [];
  let detectedConnection = "";

  const showPicked = () => {
    if (!picked.length) {
      pickedLine.hidden = true;
      return;
    }
    pickedLine.hidden = false;
    pickedLine.textContent = `先用 ${picked[0].model} 验证连接；保存后在详情里勾选其他模型、设置参数。`;
  };
  const showManual = () => {
    owner.invalidate();
    picked = [];
    showPicked();
    manual.hidden = false;
    footer.hidden = false;
    manualInput.required = true;
    manualConfirm.required = true;
    footerNote.lastChild.textContent = "目录未读取；保存时仍会实际验证对话用途";
    status.hidden = true;
    manualInput.focus();
  };
  const detect = async (button) => {
    manualInput.required = false;
    manualConfirm.required = false;
    if (!form.reportValidity()) return;
    const data = new FormData(form);
    const fingerprint = connectionFingerprint(form);
    const attempt = owner.start(fingerprint);
    clearInlineError(host);
    status.hidden = false;
    status.textContent = "正在读取服务的模型目录…";
    try {
      const discovered = await runButton(button, "探测中", props.actions.discover({
        name: String(data.get("name")),
        endpoint: String(data.get("endpoint")),
        credential: {driver: "api_key", access_token: String(data.get("apiKey"))},
        driverConfig: {format_version: 1, catalog_provider_id: String(data.get("provider")), thinking_format: String(data.get("thinkingFormat")), allow_unverified_manual: true},
      }, attempt.signal), host.querySelector("[data-error]"), false);
      if (!attempt.isCurrent(connectionFingerprint(form))) return;
      const candidates = discovered.filter((model) => ["chat", null].includes(model.kind));
      if (candidates.length === 0) throw new Error("服务返回了模型目录，但没有可选择的模型");
      // 仿 magpie：建连接只验证一个模型，其余在建好后自动展开的详情里勾选。
      const hint = props.state.template?.defaults?.modelHint;
      picked = [candidates.find((model) => model.model === hint) ?? candidates[0]];
      detectedConnection = fingerprint;
      showPicked();
      manual.hidden = true;
      footer.hidden = false;
      footerNote.lastChild.textContent = "保存前会向这个模型发送一条短消息，验证连接可用";
      status.textContent = `读到 ${candidates.length} 个模型。`;
    } catch (reason) {
      if (!attempt.isCurrent(connectionFingerprint(form))) return;
      status.textContent = reason?.code === "forbidden_contract"
        ? "服务刚刚更新。请刷新页面后重新探测；连接信息尚未保存。"
        : "未能读取模型目录。请检查 Base URL 和 API Key，或改用手动填写。";
      const error = host.querySelector("[data-error]");
      error.textContent = reason instanceof Error ? reason.message : String(reason);
      error.hidden = false;
    }
  };

  discoverButton.addEventListener("click", () => detect(discoverButton));
  rescanButton.addEventListener("click", () => detect(rescanButton));
  host.querySelector("[data-manual]").addEventListener("click", showManual);
  for (const field of [form.elements.endpoint, form.elements.apiKey, form.elements.provider, form.elements.thinkingFormat]) {
    field.addEventListener("input", () => {
      const current = connectionFingerprint(form);
      if (detectedConnection === current) return;
      owner.invalidate();
      picked = [];
      showPicked();
      status.hidden = false;
      status.textContent = "连接信息已更改，请重新探测。";
      footer.hidden = manual.hidden;
    });
  }
  host._pickedModels = () => picked;
  return () => owner.close();
}

export function createDiscoveryOwner() {
  let generation = 0;
  let controller = null;
  let closed = false;

  const invalidate = () => {
    generation += 1;
    controller?.abort();
    controller = null;
  };
  return Object.freeze({
    start(fingerprint) {
      if (closed) throw new Error("模型检测面板已关闭");
      invalidate();
      controller = new AbortController();
      const currentGeneration = generation;
      const signal = controller.signal;
      return Object.freeze({
        signal,
        isCurrent(currentFingerprint) {
          return !closed && !signal.aborted && generation === currentGeneration
            && currentFingerprint === fingerprint;
        },
      });
    },
    invalidate,
    close() {
      closed = true;
      invalidate();
    },
  });
}

function connectionFingerprint(form) {
  return JSON.stringify([
    form.elements.endpoint.value,
    form.elements.apiKey.value,
    form.elements.provider.value,
    form.elements.thinkingFormat.value,
  ]);
}

function modelInput(model, manual = false) {
  const capabilities = model.capabilities ?? {};
  const sources = model.capabilitySources ?? {};
  return {
    kind: "chat",
    discovery_owned: !manual,
    model: model.model,
    capabilities: {
      context_window: capabilities.contextWindow ?? null,
      max_output_tokens: capabilities.maxOutputTokens ?? null,
      input_modalities: capabilities.inputModalities ?? ["text"],
      supports_tool_calls: capabilities.supportsToolCalls ?? null,
      supports_parallel_tool_calls: capabilities.supportsParallelToolCalls ?? null,
      supported_reasoning_efforts: capabilities.supportedReasoningEfforts ?? [],
      embedding_dimensions: capabilities.embeddingDimensions ?? null,
      embedding_normalization: capabilities.embeddingNormalization ?? null,
    },
    capability_sources: {
      context_window: sources.contextWindow ?? "unknown",
      max_output_tokens: sources.maxOutputTokens ?? "unknown",
      input_modalities: sources.inputModalities ?? "unknown",
      tool_calls: sources.toolCalls ?? "unknown",
      parallel_tool_calls: sources.parallelToolCalls ?? "unknown",
      reasoning_efforts: sources.reasoningEfforts ?? "unknown",
      embedding_dimensions: sources.embeddingDimensions ?? "unknown",
      embedding_normalization: sources.embeddingNormalization ?? "unknown",
    },
    default_reasoning_effort: model.defaultReasoningEffort ?? null,
    driver_config: model.driverConfig ?? {format_version: 1},
  };
}

function clearInlineError(host) {
  const error = host.querySelector("[data-error]");
  error.hidden = true;
  error.textContent = "";
}

async function runButton(button, busyText, work, error, reportError = true) {
  const label = button.textContent;
  button.disabled = true;
  button.replaceChildren(htmlNode(SPINNER_ICON), busyText);
  error.hidden = true;
  error.textContent = "";
  try {
    return await work;
  } catch (reason) {
    if (reportError) {
      error.textContent = reason instanceof Error ? reason.message : String(reason);
      error.hidden = false;
    }
    throw reason;
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
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

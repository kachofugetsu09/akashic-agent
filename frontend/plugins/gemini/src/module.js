const CLOSE_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M18 6 6 18"></path><path d="m6 6 12 12"></path></svg>`;
const EYE_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0"></path><circle cx="12" cy="12" r="3"></circle></svg>`;
const EYE_OFF_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m2 2 20 20"></path><path d="M6.71 6.71C4.7 8.1 3.24 10.06 2.06 11.65a1 1 0 0 0 0 .7c2.34 5.64 8.94 8.32 14.24 5.36"></path><path d="M10.73 5.08A10.66 10.66 0 0 1 21.94 11.65a1 1 0 0 1 0 .7 10.83 10.83 0 0 1-2.06 3.1"></path><path d="M14.12 14.12A3 3 0 0 1 9.88 9.88"></path></svg>`;
const SHIELD_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"></path><path d="m9 12 2 2 4-4"></path></svg>`;
const SPINNER_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="is-spinning" aria-hidden="true"><path d="M21 12a9 9 0 1 1-6.219-8.56"></path></svg>`;

export function activate(ctx) {
  return ctx.ui.inject("models.connection-types.v1", (types) => types.register({
    id: "gemini", label: "Gemini 原生 API", order: 35,
    detail: "连接 Google 或提供原生 Gemini 接口的网关", catalogSync: true,
    templates: [{id: "gemini-native", label: "Gemini 原生 API", detail: "Google 与原生接口网关", order: 35,
      defaults: {name: "Gemini", endpoint: "https://generativelanguage.googleapis.com/v1beta"}}],
    render(host, _view, props) {
      if (!props?.state || typeof props.actions?.discover !== "function"
        || typeof props.actions.createManual !== "function" || typeof props.actions.addModel !== "function"
        || typeof props.actions.update !== "function" || typeof props.ui?.pickModels !== "function") {
        throw new Error("models.connection-types.v1 props 无效");
      }
      const existing = props.state.connection;
      const defaults = props.state.template?.defaults ?? {};
      let closed = false;
      host.innerHTML = `<header class="settings-dialog-header"><div class="settings-dialog-heading">
        <h2 class="settings-dialog-title">${existing ? "编辑 Gemini 连接" : "连接 Google Gemini"}</h2>
        <p class="settings-dialog-description">${existing
          ? "更新地址或密钥前会向已启用模型发送测试验证；若验证失败保留原配置。仅修改名称时不调用模型。"
          : "输入 Google API Key 或兼容网关地址，探测可用模型后选择启用。"}</p>
        </div><button type="button" class="settings-icon-button" data-close aria-label="关闭">${CLOSE_ICON}</button></header>
        <form class="settings-dialog-form"><div class="settings-dialog-body"><div class="settings-form-grid">
          <label class="is-wide"><span>连接名称</span><input name="name" aria-label="连接名称" required autocomplete="organization"></label>
          <label class="is-wide"><span>Base URL${existing ? "（留空保持不变）" : ""}</span><input name="endpoint" aria-label="Base URL" type="url" ${existing ? "" : "required"} placeholder="https://generativelanguage.googleapis.com/v1beta"></label>
          <label class="settings-secret is-wide"><span>API Key</span><input name="apiKey" aria-label="API Key" type="password" ${existing ? "" : "required"} autocomplete="off" placeholder="${existing ? "留空保留现有密钥" : "AIzaSy…"}"><button type="button" data-show-key aria-label="显示 API Key">${EYE_ICON}</button></label>
        </div><p>支持官方地址或自定义代理网关（以 /v1 或 /v1beta 结尾）。</p>
        <p class="settings-credential-note">${SHIELD_ICON}<span>API Key 保存后不会在页面中明文回显</span></p>
        ${existing ? "" : `<section class="settings-model-discovery"><header><div><h3>可用模型</h3><p>探测结果仅供选择；勾选的模型将在保存前发送测试请求验证可用性。</p></div></header>
          <div class="settings-discovery-empty" data-discovery-empty><button type="button" class="settings-primary-button" data-discover>自动探测模型</button><button type="button" class="settings-text-button" data-manual>手动输入模型名</button></div>
          <p class="settings-discovery-status" data-picked role="status" hidden></p>
          <div class="settings-discovery-manual" data-discovery-manual hidden><label><span>模型名称</span><input name="manualModel" aria-label="模型名称" placeholder="例如：gemini-2.5-flash"></label><p>若网关暂不支持目录探测，可手动输入模型名称。</p><label class="settings-manual-confirm"><input type="checkbox" name="manualConfirm"><span>我确认将此模型用于对话，并在保存前发送测试请求。</span></label></div>
          <p class="settings-discovery-status" data-status role="status" aria-live="polite" hidden></p>
        </section>`}
        <p class="settings-inline-error" data-error role="alert" hidden></p></div>
        <footer class="settings-dialog-footer" data-footer ${existing ? "" : "hidden"}><span class="settings-dialog-footer-note" data-footer-note>${SHIELD_ICON}保存前会向所选模型发送一条测试消息以验证可用性</span><div class="settings-dialog-actions">${existing ? "" : '<button type="button" class="settings-secondary-button" data-rescan>重新探测</button>'}<button type="submit" class="settings-primary-button">保存连接</button></div></footer></form>`;
      const form = host.querySelector("form");
      form.elements.name.value = existing?.name ?? defaults.name ?? "Gemini";
      form.elements.endpoint.value = existing ? "" : defaults.endpoint ?? "";
      form.addEventListener("input", () => props.dirty(true));
      form.addEventListener("change", () => props.dirty(true));
      host.querySelector("[data-close]").addEventListener("click", props.close);
      const showKey = host.querySelector("[data-show-key]");
      showKey.addEventListener("click", () => {
        const visible = form.elements.apiKey.type === "text";
        form.elements.apiKey.type = visible ? "password" : "text";
        showKey.setAttribute("aria-label", visible ? "显示 API Key" : "隐藏 API Key");
        showKey.innerHTML = visible ? EYE_ICON : EYE_OFF_ICON;
      });
      const discovery = existing ? null : setupDiscovery(host, form, props);
      form.addEventListener("submit", (event) => {
        event.preventDefault();
        if (!form.reportValidity()) return;
        const data = new FormData(form);
        if (existing) {
          const apiKey = String(data.get("apiKey"));
          submit(form, props.actions.update({name: String(data.get("name")).trim(),
            endpoint: String(data.get("endpoint")).trim() || null,
            credential: apiKey ? {driver: "api_key", access_token: apiKey} : null,
            driverConfig: null}).then(() => { if (!closed) props.changed("Gemini 原生连接已更新"); }));
          return;
        }
        const manual = !host.querySelector("[data-discovery-manual]").hidden;
        const selected = manual ? [{kind: "chat", model: form.elements.manualModel.value.trim()}] : discovery.pickedModels();
        if (!selected.length || !selected[0].model) {
          const error = host.querySelector("[data-error]");
          error.textContent = "请先探测目录并勾选型号，或手动填写型号。";
          error.hidden = false;
          return;
        }
        submit(form, (async () => {
          await props.actions.createManual({name: String(data.get("name")).trim(),
            endpoint: String(data.get("endpoint")).trim(),
            credential: {driver: "api_key", access_token: String(data.get("apiKey"))},
            driverConfig: {format_version: 1, catalog_provider_id: "gemini"},
            model: modelInput(selected[0], manual)});
          // 第一项提交后不回滚连接；其余项逐个验证并报告部分完成。
          let opened = 1;
          const failures = [];
          const status = host.querySelector("[data-status]");
          for (const [index, model] of selected.slice(1).entries()) {
            if (closed) return;
            status.hidden = false;
            status.textContent = `正在验证并开放 ${index + 2}/${selected.length}：${model.model}`;
            try { await props.actions.addModel(modelInput(model)); opened += 1; }
            catch (reason) { failures.push(`${model.model}：${reason instanceof Error ? reason.message : String(reason)}`); }
          }
          if (!closed) props.changed(failures.length
            ? `连接已保存；已开放 ${opened}/${selected.length} 个模型。未开放：${failures.join("；")}`
            : `Gemini 原生连接已保存，已开放 ${opened} 个模型。`);
        })());
      });
      queueMicrotask(() => { if (!closed) form.elements.name.focus(); });
      return () => { closed = true; discovery?.close(); host.replaceChildren(); };
    },
  }));
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
    const names = picked.map((model) => model.model);
    pickedLine.hidden = false;
    pickedLine.textContent = `已选 ${names.length} 个型号：${names.slice(0, 4).join("、")}${names.length > 4 ? ` 等 ${names.length} 个` : ""}；保存时逐个验证。`;
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
        driverConfig: {format_version: 1, catalog_provider_id: "gemini"},
      }, attempt.signal), host.querySelector("[data-error]"), false);
      if (!attempt.isCurrent(connectionFingerprint(form))) return;
      const candidates = discovered.filter((model) => ["chat", null].includes(model.kind));
      if (candidates.length === 0) throw new Error("服务返回了模型目录，但没有可选择的模型");
      const selected = await props.ui.pickModels(candidates, {
        title: `目录 · ${String(data.get("name")).trim() || "新连接"}`,
        hint: "勾选的型号在保存时逐个验证对话用途。",
        checked: picked.map((model) => model.model),
        confirmLabel: "开放所选",
      });
      if (!attempt.isCurrent(connectionFingerprint(form))) return;
      if (selected === null) {
        status.textContent = picked.length ? "未改动当前选择。" : "未选择型号。";
        return;
      }
      if (!selected.length) {
        picked = [];
        showPicked();
        detectedConnection = fingerprint;
        status.textContent = "未勾选型号；保存前可重新探测或手动填写。";
        return;
      }
      picked = [...selected];
      detectedConnection = fingerprint;
      showPicked();
      manual.hidden = true;
      footer.hidden = false;
      footerNote.lastChild.textContent = "保存前会向所选模型发送一条短消息，验证对话用途";
      status.textContent = `已选 ${picked.length} 个型号。`;
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
  for (const field of [form.elements.endpoint, form.elements.apiKey]) {
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
  return {pickedModels: () => picked, close: () => owner.close()};
}

function createDiscoveryOwner() {
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

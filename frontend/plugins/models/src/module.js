const SEARCH_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-search" aria-hidden="true"><circle cx="11" cy="11" r="8"></circle><path d="m21 21-4.3-4.3"></path></svg>`;
const CHEVRON_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-chevron-right" aria-hidden="true"><path d="m9 18 6-6-6-6"></path></svg>`;
const KEY_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M2.586 17.414A2 2 0 0 0 2 18.828V21a1 1 0 0 0 1 1h3a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1h1a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1h.172a2 2 0 0 0 1.414-.586l.814-.814a6.5 6.5 0 1 0-4-4z"></path><circle cx="16.5" cy="7.5" r=".5" fill="currentColor"></circle></svg>`;
const CLOSE_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M18 6 6 18"></path><path d="m6 6 12 12"></path></svg>`;
const SPINNER_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="is-spinning" aria-hidden="true"><path d="M21 12a9 9 0 1 1-6.219-8.56"></path></svg>`;

const ROLE_LABELS = [
  ["default", "默认模型", "普通模型调用与系统默认"],
  ["agent", "Agent 模型", "被动对话与计划任务 ReAct"],
  ["fast", "轻量模型", "压缩、标签与后台提取"],
  ["vision", "视觉模型", "看图与包含图片的对话"],
];

export async function readJsonResponse(response) {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const code = body && typeof body.code === "string" ? body.code : null;
    const message = code === "forbidden_contract"
      ? "服务已更新，请刷新页面后重试。"
      : body && typeof body.detail === "string" ? body.detail
      : Array.isArray(body?.detail) ? "填写的信息格式不正确，请检查必填字段和地址后重新保存。"
      : `保存未完成（${response.status}），请稍后重试；持续失败时刷新页面重新加载配置。`;
    const error = new Error(message);
    error.status = response.status;
    error.code = code;
    throw error;
  }
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    throw new Error("服务返回了无效 JSON");
  }
  return body;
}

export function activate(ctx) {
  return ctx.ui.inject("shell.pages.v1", (mount) => mount.register({
    id: "models",
    label: "模型",
    iconSvg: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-sliders-horizontal" aria-hidden="true"><path d="M10 5H3"></path><path d="M12 19H3"></path><path d="M14 3v4"></path><path d="M16 17v4"></path><path d="M21 12h-9"></path><path d="M21 19h-5"></path><path d="M21 5h-7"></path><path d="M8 10v4"></path><path d="M8 12H3"></path></svg>',
    route: "models",
    order: 30,
    children: [{id: "models.connection-types.v1", cardinality: "list"}],
    render(host, view, props = {}) {
      const connectionTypes = view.child("models.connection-types.v1");
      const providerEntries = connectionTypes.entries.map(requireProviderEntry);
      const providerTemplates = providerEntries
        .flatMap((entry) => templatesFor(entry))
        .sort((left, right) => (left.order ?? left.owner.order ?? 0) - (right.order ?? right.owner.order ?? 0));
      const page = document.createElement("main");
      page.className = `settings-page ${props.embedded ? "settings-page--embedded" : ""}`;
      page.innerHTML = `<div class="settings-shell">
        <header class="settings-header">
          <div><h1 data-title>模型连接</h1><p data-description>每套账号或 API Key 都是独立连接；未知模型能力不会被猜测。</p></div>
          <div class="settings-header-actions"></div>
        </header>
        <label class="settings-search" data-search>${SEARCH_ICON}<span class="sr-only">搜索模型连接</span><input placeholder="搜索连接或模型"></label>
        <p class="settings-inline-error" data-error role="alert" hidden></p>
        <section class="settings-section" data-connected>
          <header><div><h2>已连接</h2><p>同一供应商可以添加多个账号，模型选择时按连接名称区分。</p></div><span data-count></span></header>
          <div class="settings-gallery" data-connections></div>
        </section>
        <section class="settings-section settings-section--templates" data-templates-section>
          <header><div><h2 data-templates-title>添加其他连接</h2><p data-templates-detail>可以继续添加另一个账号或服务。</p></div></header>
          <div class="settings-gallery" data-providers></div>
        </section>
        <section class="settings-section settings-roles" data-roles>
          <header><div><h2>系统模型</h2><p>修改后无需重启；固定模型的会话保持原选择。</p></div></header>
          <div class="settings-role-grid" data-bindings></div>
        </section>
      </div>
      <div class="settings-toast-region" aria-live="polite" aria-atomic="true" data-toast-region></div>`;
      host.replaceChildren(page);

      const shell = page.querySelector(".settings-shell");
      const title = page.querySelector("[data-title]");
      const description = page.querySelector("[data-description]");
      const search = page.querySelector("[data-search]");
      const searchInput = search.querySelector("input");
      const errorMessage = page.querySelector("[data-error]");
      const connectedSection = page.querySelector("[data-connected]");
      const connectionCount = page.querySelector("[data-count]");
      const connections = page.querySelector("[data-connections]");
      const templatesSection = page.querySelector("[data-templates-section]");
      const templatesTitle = page.querySelector("[data-templates-title]");
      const templatesDetail = page.querySelector("[data-templates-detail]");
      const providers = page.querySelector("[data-providers]");
      const roles = page.querySelector("[data-roles]");
      const bindings = page.querySelector("[data-bindings]");
      const toastRegion = page.querySelector("[data-toast-region]");
      let catalog = null;
      let query = "";
      let disposeDialog = () => {};
      let closed = false;
      let bindingSave = null;
      // 页面级浮层（候选勾选/角色挑选）；关闭宿主对话框或页面时统一释放。
      const openOverlays = new Set();
      const closeOverlays = () => { for (const close of [...openOverlays]) close(); };

      const request = async (path, init) => {
        const response = await ctx.http.request(path, init);
        return readJsonResponse(response);
      };
      const reads = createLatestCatalogRead(
        (signal) => request("/api/dashboard/models/catalog", {signal}),
        (nextCatalog) => {
          if (bindingSave?.saving) return;
          const settledBinding = bindingSave;
          bindingSave = null;
          catalog = nextCatalog;
          clearError();
          renderCatalog();
          if (settledBinding) {
            if (settledBinding.error) showError(`保存请求未成功确认，当前显示重新核对的实际设置。${settledBinding.error}`);
            else showNotice(`${settledBinding.label}保存请求已完成，当前显示最新设置；会话固定模型不受影响。`);
          }
          props.changed?.();
        },
      );
      const load = () => reads.run();
      // 返回设置只读核对目录；弹窗开启后，迟到的后台结果不能替换其依据。
      const canRefresh = () => !bindingSave?.saving && !page.querySelector("dialog[open]");
      const refreshVisible = () => {
        if (page.getClientRects().length && canRefresh()) report(reads.run(canRefresh));
      };
      const visibility = new IntersectionObserver(entries => {
        if (entries.some(entry => entry.isIntersecting)) refreshVisible();
      });
      visibility.observe(page);
      window.addEventListener("focus", refreshVisible);
      page.addEventListener("close", refreshVisible, true);
      const sendCommand = (payload) => request("/api/dashboard/models/command", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      const command = async (payload) => {
        const receipt = await sendCommand(payload);
        try {
          await load();
        } catch (error) {
          showError(new Error(`设置已保存，但刷新失败。请刷新页面查看最新配置。${error instanceof Error ? error.message : String(error)}`));
        }
        return receipt;
      };
      const report = (work) => work.catch(showError);

      function clearError() {
        errorMessage.hidden = true;
        errorMessage.textContent = "";
      }

      function showError(reason) {
        if (closed) return;
        errorMessage.hidden = false;
        errorMessage.textContent = reason instanceof Error ? reason.message : String(reason);
        if (bindingSave && !bindingSave.saving) {
          errorMessage.prepend("保存后的设置尚未核对，当前选择暂时保留。 ");
          const retry = document.createElement("button");
          retry.type = "button";
          retry.textContent = "重新核对";
          retry.addEventListener("click", () => report(load()));
          errorMessage.append(" ", retry);
        }
      }

      function showNotice(message) {
        const toast = document.createElement("div");
        toast.className = "settings-toast";
        toast.setAttribute("role", "status");
        toast.innerHTML = `<span><strong></strong></span><button type="button" aria-label="关闭通知">×</button>`;
        toast.querySelector("strong").textContent = message;
        toast.querySelector("button").addEventListener("click", () => toast.remove());
        toastRegion.replaceChildren(toast);
      }

      function renderCatalog() {
        const chatModels = catalog.models.filter((model) => model.kind === "chat");
        const chatConnections = catalog.connections;
        const hasConnections = chatConnections.length > 0;

        title.textContent = "模型连接";
        description.textContent = hasConnections
          ? "每套账号或 API Key 都是独立连接；探测目录后勾选开放的模型。"
          : "选择登录方式或 API 服务。";
        search.hidden = !hasConnections;
        connectedSection.hidden = !hasConnections;
        roles.hidden = false;

        templatesTitle.textContent = hasConnections ? "添加其他连接" : "选择连接方式";
        templatesDetail.textContent = hasConnections
          ? "可以继续添加另一个账号或服务。"
          : "登录或填写密钥后探测目录，勾选要开放的模型。";
        renderConnections(chatConnections);
        renderBindings(chatModels);
        for (const button of providers.querySelectorAll("button")) button.disabled = false;
      }

      function renderConnections(allConnections) {
        const normalizedQuery = query.trim().toLocaleLowerCase();
        const filtered = allConnections.filter((connection) => {
          if (!normalizedQuery) return true;
          const modelNames = catalog.models
            .filter((model) => model.connectionId === connection.id)
            .map((model) => model.model)
            .join(" ");
          return `${connection.name} ${connection.driverId} ${modelNames}`.toLocaleLowerCase().includes(normalizedQuery);
        });
        connectionCount.textContent = `${filtered.length} 个`;
        connections.replaceChildren();
        for (const connection of filtered) {
          const models = catalog.models.filter((model) => model.connectionId === connection.id);
          const entry = providerEntries.find((candidate) => candidate.id === connection.driverId);
          const item = document.createElement(entry ? "button" : "article");
          if (entry) item.type = "button";
          item.className = "settings-connection-card";
          item.appendChild(connectionMark(entry, connection.name));
          const copy = document.createElement("span");
          copy.className = "settings-card-copy";
          const name = document.createElement("strong");
          const detail = document.createElement("small");
          name.textContent = connection.name;
          const openCount = models.filter((model) => model.availability !== "disabled").length;
          detail.textContent = connection.availability === "disabled"
            ? `${entry?.label ?? connection.driverId} · ${models.length} 个模型`
            : `${entry?.label ?? connection.driverId} · ${openCount}/${models.length} 开放`;
          copy.append(name, detail);
          const meta = document.createElement("span");
          meta.className = "settings-card-meta";
          const available = document.createElement("i");
          available.innerHTML = "<span></span>";
          const AVAILABILITY_LABELS = { available: "已连接", disabled: "已停用", driver_unavailable: "驱动不可用" };
          if (connection.availability !== "available") available.classList.add("is-unavailable");
          available.append(AVAILABILITY_LABELS[connection.availability] ?? connection.availability);
          const count = document.createElement("small");
          count.textContent = capabilitySummary(models);
          meta.append(available, count);
          item.append(copy, meta);
          item.insertAdjacentHTML("beforeend", CHEVRON_ICON);
          if (connection.availability === "disabled") item.disabled = true;
          if (entry) {
            item.setAttribute("aria-label", `编辑连接 ${connection.name}`);
            item.addEventListener("click", () => openProvider(entry, item, connection, editTemplate(entry)));
          }
          connections.appendChild(item);
        }
      }

      function renderBindings(chatModels) {
        // 搜索或目录刷新不能撤销保存锁，也不能换掉尚未核对的选择。
        if (bindingSave) return;
        bindings.replaceChildren();
        for (const [role, label, detail] of ROLE_LABELS) {
          const bound = catalog.roleBindings[role] ?? "";
          const availableModels = modelsForRole(chatModels, role)
            .filter((model) => model.availability === "available" || model.id === bound);
          bindings.appendChild(bindingRow({
            label,
            detail,
            models: availableModels,
            value: bound,
            change(modelId) {
              return sendCommand({type: "set_default", expected_revision: catalog.revision, role, model_id: modelId});
            },
          }));
        }
        const embeddingBound = catalog.defaultEmbeddingModelId ?? "";
        const embeddingModels = catalog.models.filter((model) =>
          model.kind === "embedding" && (model.availability === "available" || model.id === embeddingBound));
        bindings.appendChild(bindingRow({
          label: "向量模型",
          detail: "记忆检索与向量化",
          models: embeddingModels,
          value: embeddingBound,
          change(modelId) {
            return sendCommand({type: "set_default", expected_revision: catalog.revision, role: null, model_id: modelId});
          },
        }));
        const addEmbedding = document.createElement("button");
        addEmbedding.type = "button";
        addEmbedding.className = "settings-add-embedding";
        addEmbedding.textContent = "添加向量模型";
        addEmbedding.addEventListener("click", () => openEmbedding(addEmbedding));
        bindings.appendChild(addEmbedding);
      }

      function openEmbedding(trigger) {
        if (bindingSave) { showNotice("请先等待模型选择保存或核对完成，再添加向量模型。"); return; }
        disposeDialog();
        const dialog = document.createElement("dialog");
        dialog.className = "settings-scrim";
        dialog.setAttribute("aria-label", "添加向量模型");
        dialog.innerHTML = `<form class="settings-dialog settings-dialog-form embedding-form">
          <header class="settings-dialog-header"><div><h2>添加向量模型</h2><p>用于情景记忆。读取型号，再用两段固定测试文本测量实际维度；不需要先配置聊天模型。</p></div></header>
          <div class="settings-dialog-body"><div class="settings-form-grid">
          <label class="is-wide"><span>使用连接</span><select name="connection" aria-label="使用连接" required></select></label>
          <div class="is-wide settings-form-grid" data-new-connection>
            <label class="is-wide"><span>连接名称</span><input name="name" aria-label="向量连接名称" value="向量服务" maxlength="128"></label>
            <label class="is-wide"><span>Base URL</span><input name="endpoint" aria-label="向量 Base URL" type="url" placeholder="https://api.example.com/v1"></label>
            <label class="is-wide"><span>API Key</span><input name="key" aria-label="向量 API Key" type="password" autocomplete="off"></label>
          </div>
          <button type="button" class="settings-secondary-button is-wide" data-directory>读取模型目录</button>
          <label class="is-wide" data-candidates hidden><span>模型型号（用途由试算核对）</span><select name="candidate" aria-label="向量模型型号"></select></label>
          <button type="button" class="settings-text-button is-wide" data-manual hidden>目录没有所需型号？手动填写</button>
          <label class="is-wide" data-manual-field><span>模型名称</span><input name="model" aria-label="向量模型名称" required maxlength="256" placeholder="从目录选择；目录不可用时填写服务提供的型号"></label>
          <button type="button" class="settings-secondary-button is-wide" data-probe>试算实际维度</button>
          <p class="is-wide" role="status" data-result>还未试算。目录中的型号不代表已经支持向量。</p>
          </div><p class="settings-inline-error" role="alert" hidden></p>
          <p>保存并设为默认只更改模型设置；记忆仍由你决定开启或关闭。已有记忆空间不匹配时会明确阻塞，不删除或自动重建。</p></div>
          <footer class="settings-dialog-footer"><button type="button" class="settings-secondary-button" data-cancel>取消</button><button type="submit" class="settings-primary-button" disabled>保存并设为默认</button></footer>
        </form>`;
        const form = dialog.querySelector("form"), select = form.elements.connection;
        for (const entry of providerEntries.filter(item => item.embeddingApiKey === true)) select.append(new Option(`新建 ${entry.label} 向量连接`, `new:${entry.id}`));
        for (const connection of catalog.connections.filter(item => item.availability === "available")) select.append(new Option(connection.name, `saved:${connection.id}`));
        const error = form.querySelector('[role="alert"]'), status = form.querySelector("[data-result]");
        const save = form.querySelector('button[type="submit"]'), directory = form.querySelector("[data-directory]"), probe = form.querySelector("[data-probe]");
        const candidatePanel = form.querySelector("[data-candidates]");
        const draftId = `embedding-${randomToken()}`, modelId = `${draftId}__model`;
        let dirty = false, busy = false, closed = false, preview = null, controller = null, sequence = 0;
        const fingerprint = () => JSON.stringify([select.value, ...["name", "endpoint", "key", "model"].map(name => form.elements[name].value)]);
        const newConnection = () => ({expected_revision:catalog.revision, connection_id:draftId, name:form.elements.name.value.trim(),
          driver_id:select.value.slice(4), endpoint:form.elements.endpoint.value.trim(), auth_identity:`api:${draftId}`,
          credential:{driver:"api_key", access_token:form.elements.key.value}, driver_config:{format_version:1, allow_unverified_manual:true}});
        const invalidate = () => { sequence += 1; controller?.abort(); controller = null; preview = null; save.disabled = true; status.textContent = "信息已更改，请重新试算实际维度。"; directory.disabled = false; probe.disabled = false; };
        const updateMode = () => {
          const isNew = select.value.startsWith("new:");
          form.querySelector("[data-new-connection]").hidden = !isNew;
          for (const name of ["name", "endpoint", "key"]) { form.elements[name].required = isNew; form.elements[name].disabled = !isNew; }
          candidatePanel.hidden = true; form.querySelector("[data-manual-field]").hidden = false; form.querySelector("[data-manual]").hidden = true; form.elements.candidate.replaceChildren(); form.elements.model.value = "";
        };
        updateMode();
        if (!select.options.length) { status.textContent = "没有可用的向量连接方式，请先安装支持向量的驱动。"; directory.disabled = true; probe.disabled = true; }
        form.querySelector("[data-manual]").addEventListener("click", () => { invalidate(); form.querySelector("[data-manual-field]").hidden = false; form.elements.model.focus(); });
        const stopGuard = guardDialog(dialog, () => ({dirty, busy}));
        const changed = event => {
          if (busy) return;
          dirty = true; props.dirty?.(true); invalidate();
          if (event.target === select) updateMode();
          if ([form.elements.endpoint, form.elements.key, select].includes(event.target)) candidatePanel.hidden = true;
        };
        form.addEventListener("input", changed); select.addEventListener("change", changed);
        form.elements.candidate.addEventListener("change", () => { form.elements.model.value = form.elements.candidate.value; changed({target:form.elements.model}); });
        form.querySelector("[data-cancel]").addEventListener("click", () => dialog.dispatchEvent(new Event("cancel", {cancelable:true})));
        const runPreview = async (operation) => {
          if (busy || controller) return;
          const isNew = select.value.startsWith("new:");
          const model = form.elements.model;
          model.required = operation === "probe";
          const valid = form.reportValidity(); model.required = true;
          if (!valid) return;
          const at = fingerprint(), revision = catalog.revision, attempt = ++sequence;
          controller = new AbortController(); const signal = controller.signal;
          directory.disabled = true; probe.disabled = true; save.disabled = true; error.hidden = true;
          status.textContent = operation === "probe" ? "正在向服务试算两段固定文本…" : "正在读取型号；不会保存连接…";
          try {
            const connection = isNew ? newConnection() : null, connectionId = select.value.slice(6);
            const body = operation === "probe"
              ? {expected_revision:revision, model:model.value.trim(), ...(connection ? {connection} : {connection_id:connectionId})}
              : connection ?? {expected_revision:revision, connection_id:connectionId};
            const path = operation === "probe" ? "probe_embedding" : connection ? "discover" : "discover_saved";
            const result = await request(`/api/dashboard/models/${path}`, {method:"POST", signal, headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
            if (closed || attempt !== sequence || fingerprint() !== at) return;
            if (operation === "probe") {
              preview = {at, revision, model:result.model};
              status.textContent = `试算成功：${result.model.model} 实际返回 ${result.model.capabilities.embeddingDimensions} 维。尚未保存；保存前会再核对一次。`;
              save.disabled = false;
            } else {
              const candidates = result.models.filter(item => item.kind === null || item.kind === "embedding");
              if (!candidates.length) throw new Error("目录没有向量候选，请检查服务或手动填写型号后试算。");
              form.elements.candidate.replaceChildren(...candidates.map(item => new Option(item.model, item.model)));
              candidatePanel.hidden = false; model.value = candidates[0].model;
              form.querySelector("[data-manual-field]").hidden = true; form.querySelector("[data-manual]").hidden = false;
              status.textContent = `读取到 ${candidates.length} 个候选型号。请选择，再点击“试算实际维度”。`;
            }
          } catch (reason) {
            if (closed || attempt !== sequence || signal.aborted) return;
            error.textContent = `${reason instanceof Error ? reason.message : String(reason)} ${operation === "probe" ? "请检查服务是否支持向量、型号或密钥，再重新试算。" : "目录不可用时可手动填写型号，再试算。"} 配置尚未保存。`; error.hidden = false;
          } finally {
            if (!closed && attempt === sequence) { controller = null; directory.disabled = false; probe.disabled = false; }
          }
        };
        directory.addEventListener("click", () => runPreview("directory"));
        probe.addEventListener("click", () => runPreview("probe"));
        form.addEventListener("submit", async event => {
          event.preventDefault(); if (busy || !preview || preview.at !== fingerprint()) return;
          busy = true; error.hidden = true;
          const controls = [...form.querySelectorAll("input,select,button")]; controls.forEach(item => { item.disabled = true; });
          save.textContent = "正在验证并保存…";
          const chosen = preview.model, isNew = select.value.startsWith("new:");
          const connectionId = isNew ? draftId : select.value.slice(6);
          const existing = catalog.models.find(item => item.connectionId === connectionId && item.kind === "embedding" && item.model === chosen.model);
          const targetModelId = existing?.id ?? modelId;
          try {
            if (existing) {
              if (isNew) throw new Error("此连接已经保存。请取消当前窗口，在已连接的卡片中核对或编辑；这里不会覆盖已保存的地址与密钥。");
              if (existing.capabilities.embeddingDimensions !== chosen.capabilities.embeddingDimensions) throw new Error("试算维度与已保存模型不同。请新建独立连接；不会改变已有记忆空间。");
              await command({type:"set_default", expected_revision:preview.revision, role:null, model_id:existing.id, verify_embedding:true});
            } else {
              const input = {expected_revision:preview.revision, model_id:modelId, connection_id:connectionId, kind:"embedding", model:chosen.model,
                capabilities:{embedding_dimensions:chosen.capabilities.embeddingDimensions, embedding_normalization:chosen.capabilities.embeddingNormalization},
                capability_sources:{embedding_dimensions:"probe", embedding_normalization:"driver"}, driver_config:chosen.driverConfig, make_default_embedding:true};
              if (isNew) await command({type:"create_connection_with_model", connection:{...newConnection(), expected_revision:preview.revision}, model:input});
              else await command({...input, type:"add_model"});
            }
            if (closed) return;
            dirty = false; dialog.close(); showNotice("向量模型已验证、保存并设为默认；记忆开关保持你的选择。");
          } catch (reason) {
            // HTTP 回执丢失时只读取权威目录，不重发可能已经提交的新增请求。
            let recovered = false, recoveryError = "";
            if (!reason?.status && !closed) {
              try {
                await load();
                const saved = catalog.models.find(item => item.id === targetModelId && item.connectionId === connectionId && item.kind === "embedding" && item.model === chosen.model && item.capabilities.embeddingDimensions === chosen.capabilities.embeddingDimensions);
                recovered = !!saved && catalog.defaultEmbeddingModelId === targetModelId;
                if (recovered && !closed) { dirty = false; dialog.close(); showNotice("已核对最新设置：向量模型已保存并设为默认。"); }
              } catch (readError) {
                recoveryError = `保存结果尚未确认，读取最新设置也失败。请恢复网络后重新加载模型页面，先核对结果再操作。${readError instanceof Error ? readError.message : String(readError)}`;
              }
            }
            if (!closed && !recovered) { error.textContent = recoveryError || `${reason instanceof Error ? reason.message : String(reason)} 保存结果以模型页面最新设置为准；请重新试算后再操作。`; error.hidden = false; invalidate(); }
          } finally {
            busy = false;
            if (!closed) { controls.forEach(item => { item.disabled = false; }); save.textContent = "保存并设为默认"; save.disabled = !preview; for (const name of ["name","endpoint","key"]) form.elements[name].disabled = !select.value.startsWith("new:"); }
          }
        });
        const close = () => disposeDialog(); dialog.addEventListener("close", close, {once:true});
        page.appendChild(dialog); dialog.showModal(); select.focus();
        disposeDialog = () => { closed = true; closeOverlays(); controller?.abort(); sequence += 1; stopGuard(); props.dirty?.(false); dialog.removeEventListener("close", close); dialog.close(); dialog.remove(); restoreFocus(trigger); disposeDialog = () => {}; };
      }

      function bindingRow({label, detail, models, value, change}) {
        const row = document.createElement("div");
        row.className = "settings-binding";
        const copy = document.createElement("span");
        const title = document.createElement("strong");
        const description = document.createElement("small");
        title.textContent = label;
        description.textContent = detail;
        const feedback = document.createElement("small");
        feedback.setAttribute("role", "status");
        feedback.setAttribute("aria-live", "polite");
        copy.append(title, description, feedback);
        const pick = document.createElement("button");
        pick.type = "button";
        pick.className = "settings-role-pick";
        const current = models.find((model) => model.id === value);
        const currentConnection = current && catalog.connections.find((item) => item.id === current.connectionId);
        const pickText = document.createElement("span");
        pickText.className = "settings-role-pick-value";
        pickText.textContent = current
          ? `${current.model} · ${currentConnection?.name ?? current.connectionId}`
          : "尚未配置";
        pick.append(pickText);
        pick.insertAdjacentHTML("beforeend", CHEVRON_ICON);
        pick.addEventListener("click", async () => {
          if (bindingSave || closed) return;
          if (!models.length) { feedback.textContent = "没有可用模型。"; return; }
          const picked = await modelPickSheet({title: label, models, currentId: value});
          if (closed || bindingSave || picked === null || picked === value) return;
          saveBinding({label, modelId: picked, change, feedback});
        });
        row.append(copy, pick);
        return row;
      }

      // 即使 POST 响应丢失，也只读回实际状态，不重发、不宣称已恢复旧选择。
      function saveBinding({label, modelId, change, feedback}) {
        const operation = {saving: true, label, error: ""};
        bindingSave = operation;
        reads.cancel();
        clearError();
        toastRegion.replaceChildren();
        for (const control of bindings.querySelectorAll("button")) control.disabled = true;
        feedback.textContent = "正在保存…";
        void (async () => {
          try {
            await change(modelId);
          } catch (error) {
            operation.error = error instanceof Error ? error.message : String(error);
          }
          if (closed || bindingSave !== operation) return;
          operation.saving = false;
          feedback.textContent = "正在核对保存结果…";
          await report(load());
        })();
      }

      function modelPickSheet({title, models, currentId}) {
        const groups = [];
        for (const connection of catalog.connections) {
          const rows = models
            .filter((model) => model.connectionId === connection.id)
            .map((model) => ({
              data: model.id,
              primary: model.model,
              badges: capabilityBadges(model.capabilities),
              current: model.id === currentId,
            }));
          if (rows.length) groups.push({label: connection.name, rows});
        }
        const orphans = models
          .filter((model) => !catalog.connections.some((item) => item.id === model.connectionId))
          .map((model) => ({data: model.id, primary: model.model, badges: capabilityBadges(model.capabilities), current: model.id === currentId}));
        if (orphans.length) groups.push({label: "其他", rows: orphans});
        return modelSheet({title, multiple: false, groups});
      }

      // 统一的小型浮层：multiple=true 是候选勾选（复选 + 底部确认），false 是单选列表（点击即选）。
      // 解析单选行 data、复选行 data 数组；取消一律解析 null。
      function modelSheet({title, hint, multiple = false, confirmLabel = "确定", groups}) {
        return new Promise((resolve) => {
          const scrim = document.createElement("dialog");
          scrim.className = "settings-scrim settings-sheet-scrim";
          scrim.setAttribute("aria-label", title ?? "选择模型");
          const sheet = document.createElement("section");
          sheet.className = "settings-sheet";
          const head = document.createElement("header");
          head.className = "settings-sheet-head";
          const headCopy = document.createElement("div");
          const heading = document.createElement("h3");
          heading.textContent = title ?? "选择模型";
          headCopy.append(heading);
          if (hint) {
            const hintLine = document.createElement("p");
            hintLine.textContent = hint;
            headCopy.append(hintLine);
          }
          const closeButton = document.createElement("button");
          closeButton.type = "button";
          closeButton.className = "settings-icon-button";
          closeButton.setAttribute("aria-label", "关闭");
          closeButton.innerHTML = CLOSE_ICON;
          head.append(headCopy, closeButton);
          sheet.append(head);
          const rows = [];
          if (multiple) {
            const tools = document.createElement("div");
            tools.className = "settings-sheet-tools";
            const all = document.createElement("button");
            all.type = "button"; all.className = "settings-text-button"; all.textContent = "全选";
            const none = document.createElement("button");
            none.type = "button"; none.className = "settings-text-button"; none.textContent = "全不选";
            tools.append(all, none);
            sheet.append(tools);
            all.addEventListener("click", () => { for (const row of rows) if (!row.box.disabled) row.box.checked = true; updateCount(); });
            none.addEventListener("click", () => { for (const row of rows) if (!row.box.disabled) row.box.checked = false; updateCount(); });
          }
          const list = document.createElement("div");
          list.className = "settings-sheet-list";
          for (const group of groups) {
            if (group.label) {
              const label = document.createElement("p");
              label.className = "settings-sheet-group";
              label.textContent = group.label;
              list.append(label);
            }
            for (const row of group.rows) {
              rows.push(row);
              const item = document.createElement(multiple ? "label" : "button");
              if (!multiple) item.type = "button";
              item.className = "settings-sheet-row";
              if (row.current) item.setAttribute("aria-current", "true");
              if (multiple) {
                const box = document.createElement("input");
                box.type = "checkbox";
                box.checked = row.checked ?? false;
                box.disabled = row.disabled ?? false;
                row.box = box;
                item.append(box);
              }
              const main = document.createElement("span");
              main.className = "settings-sheet-main";
              const primary = document.createElement("strong");
              primary.textContent = row.primary;
              main.append(primary);
              const marks = [...(row.marks ?? []), ...(row.badges ?? [])];
              if (marks.length) {
                const flags = document.createElement("span");
                flags.className = "settings-sheet-flags";
                for (const mark of marks) {
                  const chip = document.createElement("i");
                  chip.textContent = mark;
                  if ((row.marks ?? []).includes(mark)) chip.className = "is-mark";
                  flags.append(chip);
                }
                main.append(flags);
              }
              item.append(main);
              if (row.disabledNote || row.current) {
                const tail = document.createElement("span");
                tail.className = "settings-sheet-tail";
                tail.textContent = row.disabledNote ?? (row.current ? "当前" : "");
                item.append(tail);
              }
              if (!multiple) item.addEventListener("click", () => settle(row.data));
              list.append(item);
            }
          }
          sheet.append(list);
          let okButton = null;
          if (multiple) {
            const foot = document.createElement("footer");
            foot.className = "settings-sheet-foot";
            const cancelButton = document.createElement("button");
            cancelButton.type = "button";
            cancelButton.className = "settings-secondary-button";
            cancelButton.textContent = "取消";
            cancelButton.addEventListener("click", () => settle(null));
            okButton = document.createElement("button");
            okButton.type = "button";
            okButton.className = "settings-primary-button";
            okButton.addEventListener("click", () => settle(rows.filter((row) => row.box.checked).map((row) => row.data)));
            foot.append(cancelButton, okButton);
            sheet.append(foot);
          }
          function updateCount() {
            if (!okButton) return;
            const count = rows.filter((row) => row.box.checked).length;
            okButton.textContent = count ? `${confirmLabel} (${count})` : confirmLabel;
          }
          list.addEventListener("change", updateCount);
          updateCount();
          let settled = false;
          const release = () => settle(null);
          const close = () => {
            openOverlays.delete(release);
            scrim.removeEventListener("cancel", onCancel);
            window.removeEventListener("akashic:before-navigate", onNavigate);
            scrim.close();
            scrim.remove();
          };
          const settle = (value) => {
            if (settled) return;
            settled = true;
            close();
            resolve(value);
          };
          const onCancel = (event) => { event.preventDefault(); settle(null); };
          // Shell keeps inactive pages mounted. Wait for every leave guard before
          // releasing a sheet; a cancelled navigation must preserve the selection.
          const onNavigate = (event) => {
            queueMicrotask(() => { if (!event.defaultPrevented) settle(null); });
          };
          closeButton.addEventListener("click", () => settle(null));
          scrim.addEventListener("cancel", onCancel);
          window.addEventListener("akashic:before-navigate", onNavigate);
          scrim.append(sheet);
          openOverlays.add(release);
          page.appendChild(scrim);
          scrim.showModal();
        });
      }

      // 探测候选勾选层：present 标「已有」，locked 锁定勾选状态（在用模型不可取消）。
      function candidateSheet({title, hint, confirmLabel, candidates, checked, present, locked}) {
        const rows = candidates.map((candidate) => {
          const isPresent = present.has(candidate.model);
          const isLocked = locked.has(candidate.model);
          return {
            data: candidate,
            primary: candidate.model,
            marks: [isPresent ? "已有" : "新", ...(isLocked ? ["在用"] : [])],
            badges: capabilityBadges(candidate.capabilities),
            checked: isLocked || checked.has(candidate.model),
            disabled: isLocked,
            disabledNote: isLocked ? "在用不可关闭" : "",
          };
        });
        return modelSheet({title, hint, multiple: true, confirmLabel, groups: [{label: "", rows}]});
      }

      function usedByMap() {
        const map = new Map();
        for (const [role, label] of ROLE_LABELS) {
          const bound = catalog.roleBindings?.[role];
          if (bound) map.set(bound, label.replace("模型", "").trim());
        }
        if (catalog.defaultEmbeddingModelId) map.set(catalog.defaultEmbeddingModelId, "默认向量");
        return map;
      }

      // 连接对话框内的模型管理面：逐模型开放/停用、验证、探测差异采纳、手动添加、停用连接。
      function buildModelManager({connection, entry, actions, setEnabled, dialogClosed, finishDisable}) {
        const section = document.createElement("section");
        section.className = "settings-model-manage";
        section.innerHTML = `<header class="settings-model-manage-head"><div><h3>模型</h3><p data-manage-status role="status"></p></div>
          <div class="settings-model-manage-actions">
            <button type="button" class="settings-secondary-button" data-probe>探测目录</button>
            ${entry.catalogSync ? '<button type="button" class="settings-secondary-button" data-sync>同步目录</button>' : ""}
            <button type="button" class="settings-text-button" data-manual-toggle>手动添加</button>
          </div></header>
          <div class="settings-model-manual" data-manual hidden><input aria-label="模型型号" maxlength="256" placeholder="型号，例如 gpt-5"><button type="button" class="settings-primary-button" data-manual-add>验证并添加</button></div>
          <div class="settings-model-rows" data-rows></div>
          <p class="settings-inline-error" data-manage-error role="alert" hidden></p>
          <button type="button" class="settings-text-button" data-disable-conn>停用此连接（保留数据）</button>
          <p class="settings-inline-error" data-disable-error role="alert" hidden></p>`;
        const status = section.querySelector("[data-manage-status]");
        const error = section.querySelector("[data-manage-error]");
        const rowsElement = section.querySelector("[data-rows]");
        const manual = section.querySelector("[data-manual]");
        const manualInput = manual.querySelector("input");
        const manualAdd = section.querySelector("[data-manual-add]");
        const manualToggle = section.querySelector("[data-manual-toggle]");
        const probe = section.querySelector("[data-probe]");
        const sync = section.querySelector("[data-sync]");
        const disableConnection = section.querySelector("[data-disable-conn]");
        const disableError = section.querySelector("[data-disable-error]");
        const showManageError = (reason) => {
          error.textContent = reason instanceof Error ? reason.message : String(reason);
          error.hidden = false;
        };
        const clearManageError = () => { error.hidden = true; error.textContent = ""; };
        const connectionModels = () => catalog.models.filter((model) => model.connectionId === connection.id);

        const refreshRows = () => {
          if (dialogClosed()) return;
          const used = usedByMap();
          const models = connectionModels();
          const open = models.filter((model) => model.availability !== "disabled").length;
          status.textContent = `${open}/${models.length} 开放`;
          rowsElement.replaceChildren(...models.map((model) => modelRow(model, used.get(model.id) ?? "")));
        };

        function modelRow(model, inUse) {
          const row = document.createElement("div");
          row.className = `settings-model-row${model.availability === "disabled" ? " is-disabled" : ""}`;
          const toggleTarget = document.createElement("label");
          toggleTarget.className = "settings-model-toggle";
          const toggle = document.createElement("input");
          toggle.type = "checkbox";
          toggle.checked = model.availability !== "disabled";
          toggle.setAttribute("aria-label", `开放 ${model.model}`);
          toggle.addEventListener("change", () => {
            const target = toggle.checked;
            toggle.disabled = true;
            clearManageError();
            setEnabled(model.id, target).then(() => {
              if (dialogClosed()) return;
              status.textContent = target ? `${model.model} 已验证并开放` : `${model.model} 已停用，历史数据保留`;
              refreshRows();
            }).catch((reason) => {
              if (dialogClosed()) return;
              showManageError(reason);
              refreshRows();
            });
          });
          const main = document.createElement("span");
          main.className = "settings-model-main";
          const modelId = document.createElement("strong");
          modelId.textContent = model.model;
          const flags = document.createElement("span");
          flags.className = "settings-sheet-flags";
          if (model.kind === "embedding") {
            const chip = document.createElement("i");
            chip.className = "is-mark";
            chip.textContent = "向量";
            flags.append(chip);
          }
          for (const badge of capabilityBadges(model.capabilities)) {
            const chip = document.createElement("i");
            chip.textContent = badge;
            flags.append(chip);
          }
          main.append(modelId, flags);
          const use = document.createElement("span");
          use.className = "settings-model-use";
          use.textContent = inUse ? `在用 · ${inUse}` : "";
          toggleTarget.append(toggle);
          row.append(toggleTarget, main, use);
          if (model.availability !== "disabled") {
            const verify = document.createElement("button");
            verify.type = "button";
            verify.className = "settings-text-button";
            verify.textContent = "验证";
            verify.addEventListener("click", () => {
              verify.disabled = true;
              clearManageError();
              actions.verifyModel(model.id).then(() => {
                if (!dialogClosed()) status.textContent = `${model.model} 验证通过`;
              }).catch((reason) => {
                if (!dialogClosed()) showManageError(reason);
              }).finally(() => { verify.disabled = false; });
            });
            row.append(verify);
          }
          return row;
        }

        probe.addEventListener("click", () => {
          probe.disabled = true;
          clearManageError();
          status.textContent = "正在读取服务目录…";
          actions.discoverSaved().then(async (discovered) => {
            if (dialogClosed()) return;
            const candidates = discovered.filter((item) => item.kind === null || item.kind === "chat");
            if (!candidates.length) throw new Error("目录没有可开放的对话模型。");
            const used = usedByMap();
            const saved = connectionModels().filter((model) => model.kind === "chat");
            const picked = await candidateSheet({
              title: `目录 · ${connection.name}`,
              hint: "勾选的型号逐个验证后开放；取消勾选的开放型号会停用但保留数据。",
              confirmLabel: "开放所选",
              candidates,
              checked: new Set(saved.filter((model) => model.availability !== "disabled").map((model) => model.model)),
              present: new Set(saved.map((model) => model.model)),
              locked: new Set(saved.filter((model) => model.availability !== "disabled" && used.has(model.id)).map((model) => model.model)),
            });
            if (dialogClosed()) return;
            if (!picked) { status.textContent = "未改动。"; return; }
            // 勾选期间目录可能已变化，按最新状态重新计算差异。
            const chosen = new Set(picked.map((candidate) => candidate.model));
            const freshSaved = connectionModels().filter((model) => model.kind === "chat");
            const freshUsed = usedByMap();
            const failures = [];
            status.textContent = "正在按选择更新…";
            for (const candidate of picked) {
              const existing = freshSaved.find((model) => model.model === candidate.model);
              try {
                if (!existing) await actions.addModel(candidateModelInput(candidate));
                else if (existing.availability === "disabled") await setEnabled(existing.id, true);
              } catch (reason) {
                failures.push(`${candidate.model}：${reason instanceof Error ? reason.message : String(reason)}`);
              }
              if (dialogClosed()) return;
            }
            // 只停用本次目录实际返回却被取消勾选的型号；目录未覆盖的手动行保持原状。
            const discoveredNames = new Set(candidates.map((candidate) => candidate.model));
            for (const model of freshSaved.filter((item) => item.availability !== "disabled" && discoveredNames.has(item.model) && !chosen.has(item.model) && !freshUsed.has(item.id))) {
              try {
                await setEnabled(model.id, false);
              } catch (reason) {
                failures.push(`${model.model}：${reason instanceof Error ? reason.message : String(reason)}`);
              }
              if (dialogClosed()) return;
            }
            refreshRows();
            status.textContent = failures.length
              ? `部分完成；${failures.length} 项失败：${failures.join("；")}`
              : "已按选择更新开放状态。";
          }).catch((reason) => {
            if (dialogClosed()) return;
            showManageError(reason);
            status.textContent = "探测未完成。";
          }).finally(() => { probe.disabled = false; });
        });

        sync?.addEventListener("click", () => {
          sync.disabled = true;
          clearManageError();
          actions.sync().then(() => {
            if (dialogClosed()) return;
            status.textContent = "目录已同步。";
            refreshRows();
          }).catch((reason) => {
            if (!dialogClosed()) showManageError(reason);
          }).finally(() => { sync.disabled = false; });
        });

        manualToggle.addEventListener("click", () => {
          manual.hidden = !manual.hidden;
          if (!manual.hidden) manualInput.focus();
        });
        manualAdd.addEventListener("click", () => {
          const name = manualInput.value.trim();
          if (!name) { manualInput.focus(); return; }
          manualAdd.disabled = true;
          clearManageError();
          actions.addModel({kind: "chat", model: name, capabilities: {}, capability_sources: {}, driver_config: {}}).then(() => {
            if (dialogClosed()) return;
            manualInput.value = "";
            manual.hidden = true;
            status.textContent = `${name} 已验证并开放。`;
            refreshRows();
          }).catch((reason) => {
            if (!dialogClosed()) showManageError(reason);
          }).finally(() => { manualAdd.disabled = false; });
        });

        disableConnection.addEventListener("click", () => {
          if (!window.confirm(`停用 ${connection.name} 的全部模型？未保存的修改会放弃，历史对话和记忆保留。`)) return;
          disableConnection.disabled = true;
          actions.disableConnection().then(() => {
            if (dialogClosed()) return;
            finishDisable();
          }).catch((reason) => {
            if (dialogClosed()) return;
            disableError.textContent = `尚未确认停用结果。请关闭窗口后核对最新设置，再决定是否重试。${reason instanceof Error ? reason.message : String(reason)}`;
            disableError.hidden = false;
          }).finally(() => { disableConnection.disabled = false; });
        });

        refreshRows();
        return section;
      }

      function restoreFocus(trigger) {
        queueMicrotask(() => {
          if (trigger.isConnected && trigger.getClientRects().length) trigger.focus();
          else document.querySelector('.primary-band button[aria-current="page"]')?.focus();
        });
      }

      function guardDialog(dialog, state) {
        // 1. 原生关闭和路由离开共享一次判断，不读取表单来猜草稿。
        const leave = () => {
          const {dirty, busy} = state();
          if (busy) {
            let notice = dialog.querySelector("[data-leave-status]");
            if (!notice) { notice = document.createElement("p"); notice.dataset.leaveStatus = ""; notice.setAttribute("role", "status"); dialog.firstElementChild.append(notice); }
            notice.textContent = "请求正在执行，请等待结果后再离开。关闭页面不会撤销已提交的操作。";
            return false;
          }
          return !dirty || window.confirm("放弃尚未保存的修改？选择取消可继续填写。");
        };
        const cancel = event => { event.preventDefault(); if (leave()) dialog.close(); };
        const navigate = event => { if (!dialog.open || event.defaultPrevented) return; if (leave()) dialog.close(); else event.preventDefault(); };
        const unload = event => { const current = state(); if (current.dirty || current.busy) { event.preventDefault(); event.returnValue = ""; } };
        dialog.addEventListener("cancel", cancel);
        window.addEventListener("akashic:before-navigate", navigate);
        window.addEventListener("beforeunload", unload);
        // 2. 模块卸载释放监听；持久化和 auth 取消仍由原 owner 处理。
        return () => { dialog.removeEventListener("cancel", cancel); window.removeEventListener("akashic:before-navigate", navigate); window.removeEventListener("beforeunload", unload); };
      }

      function openProvider(entry, trigger, connection = null, template = null) {
        if (bindingSave) { showNotice("请先等待模型选择保存或核对完成，再修改连接。"); return; }
        disposeDialog();
        const connectionId = connection?.id ?? `${entry.id}-${randomToken()}`;
        const auth = createDialogAuthOwner((attemptId) => request(
          "/api/dashboard/models/command",
          {
            method: "POST",
            keepalive: true,
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({type: "cancel_auth", attempt_id: attemptId}),
          },
        ));
        const setDefaultIfMissing = async (revision, preferredModelId = "") => {
          if (auth.closed || catalog.roleBindings.default) return;
          const modelId = preferredModelId || catalog.models.find(
            (model) => model.connectionId === connectionId && model.kind === "chat" && model.availability === "available",
          )?.id;
          if (modelId) await command({type: "set_default", expected_revision: revision, role: "default", model_id: modelId});
        };
        let dirty = false, busy = false;
        // 手动创建或认证提交成功后，连接已真实存在，可继续探测、同步与添加模型。
        let created = false;
        const operations = {
          async discover(input, signal) {
            if (connection || created) throw new Error("已有连接请使用重新检测");
            const result = await request("/api/dashboard/models/discover", {
              method: "POST",
              signal,
              headers: {"Content-Type": "application/json"},
              body: JSON.stringify({
                expected_revision: catalog.revision,
                connection_id: connectionId,
                name: input.name,
                driver_id: entry.id,
                endpoint: input.endpoint,
                auth_identity: `api:${connectionId}`,
                credential: input.credential,
                driver_config: input.driverConfig,
              }),
            });
            return result.models;
          },
          async discoverSaved(signal) {
            if (!connection && !created) throw new Error("请先保存连接");
            const result = await request("/api/dashboard/models/discover_saved", {
              method: "POST", signal, headers: {"Content-Type": "application/json"},
              body: JSON.stringify({connection_id: connectionId, expected_revision: catalog.revision}),
            });
            return result.models;
          },
          async disableConnection() {
            if (!connection && !created) throw new Error("请选择已保存连接");
            await command({type:"disable_connection", expected_revision:catalog.revision, connection_id:connectionId});
          },
          async verifyModel(modelId) {
            if ((!connection && !created) || !catalog.models.some((model) => model.id === modelId && model.connectionId === connectionId)) throw new Error("请选择此连接的现有模型");
            await request("/api/dashboard/models/command", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({type:"verify_model", expected_revision:catalog.revision, model_id:modelId})});
          },
          async addModel(input) {
            if (!connection && !created) throw new Error("请先保存连接");
            const existing = catalog.models.find((model) => model.connectionId === connectionId && model.kind === input.kind && model.model === input.model);
            const modelId = existing?.id ?? `${connectionId}__${randomToken()}`;
            const receipt = existing
              ? await request("/api/dashboard/models/command", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({type:"verify_model", expected_revision:catalog.revision, model_id:modelId})})
              : await command({...input, type: "add_model", expected_revision: catalog.revision, model_id: modelId, connection_id: connectionId});
            await setDefaultIfMissing(receipt.revision, modelId);
          },
          async createManual(input) {
            if (connection || created) throw new Error("已有连接不能重复创建");
            const modelId = `${connectionId}__${randomToken()}`;
            const receipt = await command({
              type: "create_connection_with_model",
              connection: {
                expected_revision: catalog.revision,
                connection_id: connectionId,
                name: input.name,
                driver_id: entry.id,
                endpoint: input.endpoint,
                auth_identity: `api:${connectionId}`,
                credential: input.credential,
                driver_config: input.driverConfig,
              },
              model: {...input.model, expected_revision: catalog.revision, model_id: modelId, connection_id: connectionId},
            });
            created = true;
            await setDefaultIfMissing(receipt.revision, modelId);
          },
          async update(input) {
            if (!connection) throw new Error("新连接不能执行更新");
            await command({
              type: "update_connection",
              expected_revision: catalog.revision,
              connection_id: connectionId,
              name: input.name,
              endpoint: input.endpoint,
              auth_identity: connection.authIdentity,
              credential: input.credential,
              driver_config: input.driverConfig,
            });
          },
          async startAuth(input) {
            const receipt = await command({
              type: "start_auth",
              driver_id: entry.id,
              connection_id: connectionId,
              input: {...input, auth_identity: connection?.authIdentity ?? connectionId},
            });
            if (!receipt.attemptId) throw new Error(`${entry.label} 登录没有返回 attempt ID`);
            await auth.add(receipt.attemptId);
            return receipt;
          },
          async finishAuth(attemptId) {
            auth.checkFinish(attemptId);
            const receipt = await command({type: "finish_auth", expected_revision: catalog.revision, attempt_id: attemptId});
            if (receipt.status !== "pending") {
              auth.complete(attemptId);
              if (receipt.status === "committed") created = true;
            }
            return receipt;
          },
          async cancelAuth(attemptId) {
            await auth.cancel(attemptId);
            if (!auth.closed) await load();
          },
          async sync() {
            if (!connection && !created) throw new Error("请先保存连接");
            const receipt = await command({type: "sync_models", expected_revision: catalog.revision, connection_id: connectionId});
            await setDefaultIfMissing(receipt.revision);
          },
        };
        // 模型管理面的非 actions 操作（set_model_enabled）复用同一串行守卫。
        const runManaged = async (work) => {
          if (auth.closed) throw new Error("窗口已关闭；已提交请求以实际结果为准。");
          if (busy) throw new Error("请求仍在执行，请等待结果后再操作。");
          busy = true;
          try { return await work(); }
          finally { busy = false; }
        };
        const setEnabled = (modelId, enabled) => runManaged(() => command({
          type: "set_model_enabled",
          expected_revision: catalog.revision,
          model_id: modelId,
          enabled,
        }));
        const ui = Object.freeze({
          pickModels: (candidates, options = {}) => candidateSheet({
            title: options.title ?? "目录候选",
            hint: options.hint,
            confirmLabel: options.confirmLabel ?? "开放所选",
            candidates,
            checked: new Set(options.checked ?? []),
            present: new Set(options.present ?? []),
            locked: new Set(options.locked ?? []),
          }),
        });
        // 请求生命周期归宿主；表单只报告自己的未保存草稿。
        const actions = Object.freeze(Object.fromEntries(Object.entries(operations).map(([name, action]) => [name, async (...args) => {
          if (name === "cancelAuth") return action(...args);
          if (auth.closed) throw new Error("窗口已关闭；已提交请求以实际结果为准。");
          if (busy) throw new Error("请求仍在执行，请等待结果后再操作。");
          busy = true;
          try {
            const result = await action(...args);
            if (auth.closed) throw new Error("窗口已关闭；已提交请求以实际结果为准。");
            return result;
          }
          finally { busy = false; }
        }])));
        const scrim = document.createElement("dialog");
        scrim.className = "settings-scrim";
        scrim.setAttribute("aria-label", entry.label);
        const dialogHost = document.createElement("section");
        dialogHost.className = "settings-dialog";
        scrim.appendChild(dialogHost);
        page.appendChild(scrim);
        const stopGuard = guardDialog(scrim, () => ({dirty, busy}));
        let disposeEntry;
        try {
          disposeEntry = connectionTypes.render(entry.id, dialogHost, {
            get state() {
              return Object.freeze({
                connection,
                models: Object.freeze(catalog.models.filter((model) => model.connectionId === connectionId)),
                template,
              });
            },
            actions,
            ui,
            dirty(value) { if (!auth.closed) { dirty = value; props.dirty?.(value); } },
            close() { scrim.dispatchEvent(new Event("cancel", {cancelable:true})); },
            changed(message) {
              if (auth.closed) return;
              dirty = false; props.dirty?.(false);
              showNotice(message);
              scrim.close();
            },
          });
        } catch (error) {
          stopGuard();
          scrim.remove();
          showError(error);
          return;
        }
        if (connection) {
          const dialogBody = dialogHost.querySelector(".settings-dialog-body") ?? dialogHost;
          dialogBody.appendChild(buildModelManager({
            connection,
            entry,
            actions,
            setEnabled,
            dialogClosed: () => auth.closed,
            finishDisable: () => {
              dirty = false;
              props.dirty?.(false);
              showNotice("连接已停用，历史数据保留。请添加正确用途的新连接。");
              scrim.close();
            },
          }));
        }
        const leaveDocument = event => { if (!event.persisted) report(auth.close()); };
        window.addEventListener("pagehide", leaveDocument);
        const close = () => disposeDialog();
        scrim.addEventListener("close", close, {once: true});
        // 背景点击不关闭；所有显式离开复用同一草稿和请求判断。
        disposeDialog = () => {
          closeOverlays();
          stopGuard();
          window.removeEventListener("pagehide", leaveDocument);
          props.dirty?.(false);
          report(auth.close());
          scrim.removeEventListener("close", close);
          disposeEntry();
          scrim.close();
          scrim.remove();
          restoreFocus(trigger);
          disposeDialog = () => {};
        };
        scrim.showModal();
      }

      providers.replaceChildren();
      for (const template of providerTemplates) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "settings-connection-card";
        button.disabled = true;
        button.appendChild(providerMark(template));
        const copy = document.createElement("span");
        copy.className = "settings-card-copy";
        const title = document.createElement("strong");
        const detail = document.createElement("small");
        title.textContent = template.label;
        detail.textContent = template.detail;
        copy.append(title, detail);
        button.append(copy);
        button.insertAdjacentHTML("beforeend", CHEVRON_ICON);
        button.lastElementChild.classList.add("settings-template-action");
        button.addEventListener("click", () => openProvider(template.owner, button, null, template));
        providers.appendChild(button);
      }
      if (!providerTemplates.length) providers.textContent = "没有 Provider 插件提供连接方式。";
      searchInput.addEventListener("input", () => {
        query = searchInput.value;
        if (catalog) renderCatalog();
      });
      report(load());

      return () => {
        closed = true;
        visibility.disconnect();
        window.removeEventListener("focus", refreshVisible);
        page.removeEventListener("close", refreshVisible, true);
        reads.close();
        closeOverlays();
        disposeDialog();
        host.replaceChildren();
      };
    },
  }));
}

function requireProviderEntry(entry) {
  if (!entry || typeof entry.id !== "string" || typeof entry.label !== "string"
    || typeof entry.detail !== "string" || typeof entry.render !== "function") {
    throw new Error("models.connection-types.v1 entry 无效");
  }
  return entry;
}

function templatesFor(entry) {
  const templates = Array.isArray(entry.templates) && entry.templates.length ? entry.templates : [entry];
  return templates.map((template) => {
    if (!template || typeof template.label !== "string" || typeof template.detail !== "string") {
      throw new Error(`连接类型 ${entry.id} 的模板无效`);
    }
    return Object.freeze({...template, owner: entry});
  });
}

function editTemplate(entry) {
  const templates = templatesFor(entry);
  return templates.find((template) => template.id === entry.editTemplateId) ?? templates[0];
}

function connectionMark(entry, fallback) {
  return providerMark({icon: entry?.connectionIcon}, fallback);
}

function providerMark(entry, fallback) {
  const mark = document.createElement("span");
  mark.className = "settings-connection-mark";
  mark.setAttribute("aria-hidden", "true");
  if (typeof entry?.icon === "string" && entry.icon.startsWith("data:image/svg+xml,")) {
    const image = document.createElement("img");
    image.src = entry.icon;
    image.alt = "";
    mark.appendChild(image);
  } else if (fallback) {
    mark.textContent = String(fallback).slice(0, 1).toUpperCase();
  } else {
    mark.innerHTML = KEY_ICON;
  }
  return mark;
}

export function createLatestCatalogRead(read, apply) {
  let active = null;
  let closed = false;
  return {
    async run(canApply = () => true) {
      if (closed) return;
      active?.abort();
      const controller = new AbortController();
      active = controller;
      try {
        const value = await read(controller.signal);
        if (!closed && active === controller && canApply()) apply(value);
      } catch (error) {
        if (!controller.signal.aborted) throw error;
      } finally {
        if (active === controller) active = null;
      }
    },
    close() {
      closed = true;
      active?.abort();
    },
    cancel() {
      active?.abort();
      active = null;
    },
  };
}

export function createDialogAuthOwner(cancelAttempt) {
  const attempts = new Set();
  const cancellations = new Map();
  let closed = false;
  const cancel = async (attemptId) => {
    if (!attempts.has(attemptId)) return;
    let pending = cancellations.get(attemptId);
    if (!pending) {
      pending = cancelAttempt(attemptId)
        .then(() => { attempts.delete(attemptId); })
        .finally(() => { cancellations.delete(attemptId); });
      cancellations.set(attemptId, pending);
    }
    await pending;
  };
  return {
    get closed() { return closed; },
    async add(attemptId) {
      if (closed) {
        await cancelAttempt(attemptId);
        throw new Error("登录面板已关闭");
      }
      attempts.add(attemptId);
    },
    checkFinish(attemptId) {
      if (closed) throw new Error("登录面板已关闭");
      if (!attempts.has(attemptId)) throw new Error("登录 attempt 不属于当前 Provider 面板");
    },
    complete(attemptId) { attempts.delete(attemptId); },
    cancel,
    close() {
      closed = true;
      return Promise.all([...attempts].map(cancel));
    },
  };
}

export function capabilitySummary(models) {
  const visionCount = models.filter((model) => model.capabilities.inputModalities.includes("image")).length;
  const unknownCount = models.filter((model) => model.capabilitySources.inputModalities === "unknown").length;
  return [
    `${models.length} 个模型`,
    visionCount ? `${visionCount} 个可看图` : "",
    unknownCount ? `${unknownCount} 个待识别` : "",
  ].filter(Boolean).join(" · ");
}

export function modelsForRole(models, role) {
  if (role !== "vision") return models;
  return models.filter((model) => model.capabilities.inputModalities.includes("image"));
}

// 模型与探测候选共享同一组能力徽标：上下文、多模态、开放强度。
export function capabilityBadges(capabilities) {
  const caps = capabilities ?? {};
  const badges = [];
  if (caps.contextWindow) {
    badges.push(caps.contextWindow >= 1_000_000
      ? `${Number((caps.contextWindow / 1_000_000).toFixed(1))}M`
      : `${Math.round(caps.contextWindow / 1000)}K`);
  }
  if ((caps.inputModalities ?? []).includes("image")) badges.push("多模态");
  const efforts = caps.supportedReasoningEfforts ?? [];
  if (efforts.length) badges.push(efforts.map(effortShort).join("·"));
  return badges;
}

function effortShort(effort) {
  const labels = {minimal: "微", low: "低", medium: "中", high: "高"};
  return labels[effort] ?? effort;
}

// 探测候选 → add_model 所需负载；kind 为 null 的候选按对话用途验证。
export function candidateModelInput(candidate) {
  const caps = candidate.capabilities ?? {};
  const sources = candidate.capabilitySources ?? {};
  return {
    kind: candidate.kind === "embedding" ? "embedding" : "chat",
    model: candidate.model,
    capabilities: {
      context_window: caps.contextWindow ?? null,
      max_output_tokens: caps.maxOutputTokens ?? null,
      input_modalities: caps.inputModalities ?? ["text"],
      supports_tool_calls: caps.supportsToolCalls ?? null,
      supports_parallel_tool_calls: caps.supportsParallelToolCalls ?? null,
      supported_reasoning_efforts: caps.supportedReasoningEfforts ?? [],
      embedding_dimensions: caps.embeddingDimensions ?? null,
      embedding_normalization: caps.embeddingNormalization ?? null,
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
    default_reasoning_effort: candidate.defaultReasoningEffort ?? null,
    driver_config: candidate.driverConfig ?? {},
  };
}

function randomToken() {
  return [...crypto.getRandomValues(new Uint8Array(16))]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

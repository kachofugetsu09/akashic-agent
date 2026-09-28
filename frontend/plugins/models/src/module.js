const SEARCH_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-search" aria-hidden="true"><circle cx="11" cy="11" r="8"></circle><path d="m21 21-4.3-4.3"></path></svg>`;
const CHEVRON_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-chevron-right" aria-hidden="true"><path d="m9 18 6-6-6-6"></path></svg>`;
const KEY_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M2.586 17.414A2 2 0 0 0 2 18.828V21a1 1 0 0 0 1 1h3a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1h1a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1h.172a2 2 0 0 0 1.414-.586l.814-.814a6.5 6.5 0 1 0-4-4z"></path><circle cx="16.5" cy="7.5" r=".5" fill="currentColor"></circle></svg>`;

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
          <header><div><h2>系统模型</h2><p>修改后无需重启；当前回复继续使用原模型，之后使用新选择。固定模型的会话保持原选择。</p></div></header>
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

      const request = async (path, init) => {
        const response = await ctx.http.request(path, init);
        return readJsonResponse(response);
      };
      const reads = createLatestCatalogRead(
        (signal) => request("/api/dashboard/models/catalog", {signal}),
        (nextCatalog) => {
          catalog = nextCatalog;
          clearError();
          renderCatalog();
          props.changed?.();
        },
      );
      const load = () => reads.run();
      const command = async (payload) => {
        const receipt = await request("/api/dashboard/models/command", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload),
        });
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
        errorMessage.hidden = false;
        errorMessage.textContent = reason instanceof Error ? reason.message : String(reason);
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
          ? "每套账号或 API Key 都是独立连接；未知模型能力不会被猜测。"
          : "选择登录方式或 API 服务。连接后会自动同步模型并识别图片能力。";
        search.hidden = !hasConnections;
        connectedSection.hidden = !hasConnections;
        roles.hidden = false;

        templatesTitle.textContent = hasConnections ? "添加其他连接" : "选择连接方式";
        templatesDetail.textContent = hasConnections
          ? "可以继续添加另一个账号或服务。"
          : "登录后自动同步模型和已知能力；无法确认时会明确显示待识别。";
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
          detail.textContent = `${connection.driverId} · ${models.map((model) => model.model).join("、") || "尚未同步模型"}`;
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
        bindings.replaceChildren();
        for (const [role, label, detail] of ROLE_LABELS) {
          const availableModels = modelsForRole(chatModels, role);
          bindings.appendChild(bindingRow({
            label,
            detail,
            models: availableModels,
            value: catalog.roleBindings[role] ?? "",
            change(modelId) {
              return command({type: "set_default", expected_revision: catalog.revision, role, model_id: modelId});
            },
          }));
        }
        const embeddingModels = catalog.models.filter((model) => model.kind === "embedding");
        bindings.appendChild(bindingRow({
          label: "向量模型",
          detail: "记忆检索与向量化",
          models: embeddingModels,
          value: catalog.defaultEmbeddingModelId ?? "",
          change(modelId) {
            return command({type: "set_default", expected_revision: catalog.revision, role: null, model_id: modelId});
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
          <label class="is-wide"><span>模型名称</span><input name="model" aria-label="向量模型名称" required maxlength="256" placeholder="从目录选择；目录不可用时填写服务提供的型号"></label>
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
          candidatePanel.hidden = true; form.elements.candidate.replaceChildren(); form.elements.model.value = "";
        };
        updateMode();
        if (!select.options.length) { status.textContent = "没有可用的向量连接方式，请先安装支持向量的驱动。"; directory.disabled = true; probe.disabled = true; }
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
          try {
            const chosen = preview.model, isNew = select.value.startsWith("new:");
            const connectionId = isNew ? draftId : select.value.slice(6);
            const existing = catalog.models.find(item => item.connectionId === connectionId && item.kind === "embedding" && item.model === chosen.model);
            if (existing) {
              if (existing.capabilities.embeddingDimensions !== chosen.capabilities.embeddingDimensions) throw new Error("试算维度与已保存模型不同。请新建独立连接；不会改变已有记忆空间。");
              await command({type:"set_default", expected_revision:preview.revision, role:null, model_id:existing.id});
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
            if (!closed) { error.textContent = `${reason instanceof Error ? reason.message : String(reason)} 请重新试算后再保存。`; error.hidden = false; invalidate(); }
          } finally {
            busy = false;
            if (!closed) { controls.forEach(item => { item.disabled = false; }); save.textContent = "保存并设为默认"; save.disabled = !preview; for (const name of ["name","endpoint","key"]) form.elements[name].disabled = !select.value.startsWith("new:"); }
          }
        });
        const close = () => disposeDialog(); dialog.addEventListener("close", close, {once:true});
        page.appendChild(dialog); dialog.showModal(); select.focus();
        disposeDialog = () => { closed = true; controller?.abort(); sequence += 1; stopGuard(); props.dirty?.(false); dialog.removeEventListener("close", close); dialog.close(); dialog.remove(); restoreFocus(trigger); disposeDialog = () => {}; };
      }

      function bindingRow({label, detail, models, value, change}) {
        const row = document.createElement("label");
        const copy = document.createElement("span");
        const title = document.createElement("strong");
        const description = document.createElement("small");
        const select = document.createElement("select");
        title.textContent = label;
        description.textContent = detail;
        copy.append(title, description);
        const unconfigured = new Option("尚未配置", "");
        unconfigured.disabled = true;
        select.append(unconfigured);
        for (const model of models) {
          const connection = catalog.connections.find((item) => item.id === model.connectionId);
          select.append(new Option(`${model.model}：${connection?.name ?? model.connectionId}`, model.id));
        }
        select.value = value;
        const feedback = document.createElement("small");
        feedback.setAttribute("role", "status");
        feedback.setAttribute("aria-live", "polite");
        select.addEventListener("change", async () => {
          const selected = select.value;
          const controls = [...bindings.querySelectorAll("select")];
          controls.forEach((control) => { control.disabled = true; });
          feedback.textContent = "正在保存…";
          try {
            await change(selected);
            value = selected;
            feedback.textContent = "已保存";
            showNotice(`${label}已保存，下次使用系统绑定时生效；会话固定模型不受影响。`);
          } catch (error) {
            select.value = value;
            feedback.textContent = `未保存，已恢复原选择。${error instanceof Error ? error.message : String(error)} 请重新选择后重试。`;
          } finally {
            controls.forEach((control) => { control.disabled = false; });
          }
        });
        copy.append(feedback);
        row.append(copy, select);
        return row;
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
            (model) => model.connectionId === connectionId && model.kind === "chat",
          )?.id;
          if (modelId) await command({type: "set_default", expected_revision: revision, role: "default", model_id: modelId});
        };
        let dirty = false, busy = false;
        const operations = {
          async discover(input, signal) {
            if (connection) throw new Error("已有连接请使用重新检测");
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
            if (!connection) throw new Error("请先保存连接");
            const result = await request("/api/dashboard/models/discover_saved", {
              method: "POST", signal, headers: {"Content-Type": "application/json"},
              body: JSON.stringify({connection_id: connectionId, expected_revision: catalog.revision}),
            });
            return result.models;
          },
          async disableConnection() {
            if (!connection) throw new Error("请选择已保存连接");
            await command({type:"disable_connection", expected_revision:catalog.revision, connection_id:connectionId});
          },
          async verifyModel(modelId) {
            if (!connection || !catalog.models.some((model) => model.id === modelId && model.connectionId === connectionId)) throw new Error("请选择此连接的现有模型");
            await request("/api/dashboard/models/command", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({type:"verify_model", expected_revision:catalog.revision, model_id:modelId})});
          },
          async addModel(input) {
            if (!connection) throw new Error("请先保存连接");
            const existing = catalog.models.find((model) => model.connectionId === connectionId && model.kind === input.kind && model.model === input.model);
            const modelId = existing?.id ?? `${connectionId}__${randomToken()}`;
            const receipt = existing
              ? await request("/api/dashboard/models/command", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({type:"verify_model", expected_revision:catalog.revision, model_id:modelId})})
              : await command({...input, type: "add_model", expected_revision: catalog.revision, model_id: modelId, connection_id: connectionId});
            await setDefaultIfMissing(receipt.revision, modelId);
          },
          async createManual(input) {
            if (connection) throw new Error("已有连接不能重复创建");
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
            if (receipt.status !== "pending") auth.complete(attemptId);
            return receipt;
          },
          async cancelAuth(attemptId) {
            await auth.cancel(attemptId);
            if (!auth.closed) await load();
          },
          async sync() {
            const receipt = await command({type: "sync_models", expected_revision: catalog.revision, connection_id: connectionId});
            await setDefaultIfMissing(receipt.revision);
          },
        };
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
          const list = document.createElement("details");
          list.className = "settings-saved-models";
          const savedModels = catalog.models.filter((model) => model.connectionId === connectionId);
          const summary = document.createElement("summary");
          summary.textContent = `查看已保存模型（${savedModels.length}）`;
          const items = document.createElement("ul");
          for (const model of savedModels) {
            const item = document.createElement("li");
            item.textContent = `${model.model}${model.kind === "embedding" ? " · 向量模型" : ""}`;
            items.append(item);
          }
          const disable = document.createElement("button");
          disable.type = "button"; disable.className = "settings-text-button";
          disable.textContent = "停用此连接（保留数据）";
          disable.addEventListener("click", () => {
            if (busy || !window.confirm(`停用 ${connection.name} 的全部模型？未保存的修改会放弃，历史对话和记忆保留。`)) return;
            void actions.disableConnection().then(() => {
              if (auth.closed) return;
              dirty = false; props.dirty?.(false); showNotice("连接已停用，历史数据保留。请添加正确用途的新连接。"); scrim.close();
            }).catch(showError);
          });
          list.append(summary, items, disable);
          dialogHost.querySelector(".settings-dialog-body").append(list);
        }
        const leaveDocument = event => { if (!event.persisted) report(auth.close()); };
        window.addEventListener("pagehide", leaveDocument);
        const close = () => disposeDialog();
        scrim.addEventListener("close", close, {once: true});
        // 背景点击不关闭；所有显式离开复用同一草稿和请求判断。
        disposeDialog = () => {
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
        reads.close();
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
    async run() {
      if (closed) return;
      active?.abort();
      const controller = new AbortController();
      active = controller;
      try {
        const value = await read(controller.signal);
        if (!closed && active === controller) apply(value);
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

function randomToken() {
  return [...crypto.getRandomValues(new Uint8Array(16))]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

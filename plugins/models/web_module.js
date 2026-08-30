const SEARCH_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-search" aria-hidden="true"><circle cx="11" cy="11" r="8"></circle><path d="m21 21-4.3-4.3"></path></svg>`;
const CHEVRON_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-chevron-right" aria-hidden="true"><path d="m9 18 6-6-6-6"></path></svg>`;
const KEY_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M2.586 17.414A2 2 0 0 0 2 18.828V21a1 1 0 0 0 1 1h3a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1h1a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1h.172a2 2 0 0 0 1.414-.586l.814-.814a6.5 6.5 0 1 0-4-4z"></path><circle cx="16.5" cy="7.5" r=".5" fill="currentColor"></circle></svg>`;

const ROLE_LABELS = [
  ["default", "默认模型", "普通模型调用与系统默认"],
  ["agent", "Agent 模型", "被动对话与计划任务 ReAct"],
  ["fast", "轻量模型", "压缩、标签与后台提取"],
  ["vision", "视觉模型", "包含图片的输入"],
];

export function activate(ctx) {
  return ctx.ui.inject("shell.pages.v1", (mount) => mount.register({
    id: "models",
    label: "模型",
    iconSvg: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-sliders-horizontal" aria-hidden="true"><path d="M10 5H3"></path><path d="M12 19H3"></path><path d="M14 3v4"></path><path d="M16 17v4"></path><path d="M21 12h-9"></path><path d="M21 19h-5"></path><path d="M21 5h-7"></path><path d="M8 10v4"></path><path d="M8 12H3"></path></svg>',
    route: "models",
    order: 30,
    children: [{id: "models.connection-types.v1", cardinality: "list"}],
    render(host, view) {
      const connectionTypes = view.child("models.connection-types.v1");
      const providerEntries = connectionTypes.entries.map(requireProviderEntry);
      const providerTemplates = providerEntries
        .flatMap((entry) => templatesFor(entry))
        .sort((left, right) => (left.order ?? left.owner.order ?? 0) - (right.order ?? right.owner.order ?? 0));
      const page = document.createElement("main");
      page.className = "settings-page";
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
          <header><div><h2>系统模型</h2><p>修改后不重启进程；正在运行的完整 turn 保持旧快照，下一个执行读取最新绑定。</p></div></header>
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
        const body = await response.json();
        if (!response.ok) {
          throw new Error(typeof body.detail === "string" ? body.detail : `请求失败：${response.status}`);
        }
        return body;
      };
      const reads = createLatestCatalogRead(
        (signal) => request("/api/dashboard/models/catalog", {signal}),
        (nextCatalog) => {
          catalog = nextCatalog;
          clearError();
          renderCatalog();
        },
      );
      const load = () => reads.run();
      const command = async (payload) => {
        const receipt = await request("/api/dashboard/models/command", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload),
        });
        await load();
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
        const chatConnectionIds = new Set(chatModels.map((model) => model.connectionId));
        const chatConnections = catalog.connections.filter((connection) => chatConnectionIds.has(connection.id));
        const hasConnections = chatConnections.length > 0;
        shell.classList.toggle("settings-shell--first-run", !hasConnections);
        title.textContent = hasConnections ? "模型连接" : "连接你的第一个模型";
        description.textContent = hasConnections
          ? "每套账号或 API Key 都是独立连接；未知模型能力不会被猜测。"
          : "选择登录方式或 API 服务。连接成功后，再选择聊天和向量模型。";
        search.hidden = !hasConnections;
        connectedSection.hidden = !hasConnections;
        roles.hidden = !hasConnections;
        templatesSection.classList.toggle("is-first-run", !hasConnections);
        templatesTitle.textContent = hasConnections ? "添加其他连接" : "选择连接方式";
        templatesDetail.textContent = hasConnections
          ? "可以继续添加另一个账号或服务。"
          : "Codex 与 OpenCode 登录后自动同步模型；API 服务可手动填写模型。";
        renderConnections(chatConnections);
        renderBindings(chatModels);
        for (const button of providers.querySelectorAll("button")) button.disabled = false;
      }

      function renderConnections(allConnections) {
        const normalizedQuery = query.trim().toLocaleLowerCase();
        const filtered = allConnections.filter((connection) => {
          if (!normalizedQuery) return true;
          const modelNames = catalog.models
            .filter((model) => model.connectionId === connection.id && model.kind === "chat")
            .map((model) => model.model)
            .join(" ");
          return `${connection.name} ${connection.driverId} ${modelNames}`.toLocaleLowerCase().includes(normalizedQuery);
        });
        connectionCount.textContent = `${filtered.length} 个`;
        connections.replaceChildren();
        for (const connection of filtered) {
          const models = catalog.models.filter((model) => model.connectionId === connection.id && model.kind === "chat");
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
          available.append(connection.availability === "available" ? "已连接" : connection.availability);
          const count = document.createElement("small");
          count.textContent = `${models.length} 个模型`;
          meta.append(available, count);
          item.append(copy, meta);
          item.insertAdjacentHTML("beforeend", CHEVRON_ICON);
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
          bindings.appendChild(bindingRow({
            label,
            detail,
            models: chatModels,
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
        select.append(new Option("尚未配置", ""));
        for (const model of models) {
          const connection = catalog.connections.find((item) => item.id === model.connectionId);
          select.append(new Option(`${model.model}：${connection?.name ?? model.connectionId}`, model.id));
        }
        select.value = value;
        select.addEventListener("change", () => report(change(select.value)));
        row.append(copy, select);
        return row;
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
          if (catalog.roleBindings.default) return;
          const modelId = preferredModelId || catalog.models.find(
            (model) => model.connectionId === connectionId && model.kind === "chat",
          )?.id;
          if (modelId) await command({type: "set_default", expected_revision: revision, role: "default", model_id: modelId});
        };
        const actions = Object.freeze({
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
        });
        const scrim = document.createElement("dialog");
        scrim.className = "settings-scrim";
        scrim.setAttribute("aria-label", entry.label);
        const dialogHost = document.createElement("section");
        dialogHost.className = "settings-dialog";
        scrim.appendChild(dialogHost);
        page.appendChild(scrim);
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
            close() { scrim.close(); },
            changed(message) {
              showNotice(message);
              scrim.close();
            },
          });
        } catch (error) {
          scrim.remove();
          showError(error);
          return;
        }
        const close = () => disposeDialog();
        scrim.addEventListener("close", close, {once: true});
        scrim.addEventListener("click", (event) => {
          if (event.target === scrim) scrim.close();
        });
        disposeDialog = () => {
          report(auth.close());
          scrim.removeEventListener("close", close);
          disposeEntry();
          scrim.remove();
          if (trigger.isConnected) trigger.focus();
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

function randomToken() {
  return [...crypto.getRandomValues(new Uint8Array(16))]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

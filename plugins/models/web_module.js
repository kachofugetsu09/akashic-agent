export function activate(ctx) {
  return ctx.ui.inject("shell.pages.v1", (mount) => mount.register({
    id: "models",
    label: "模型",
    route: "models",
    order: 30,
    children: [{id: "models.connection-types.v1", cardinality: "list"}],
    render(host, view) {
      const connectionTypes = view.child("models.connection-types.v1");
      const page = document.createElement("div");
      page.className = "models-plugin-page";
      page.innerHTML = `
        <header class="models-plugin-header">
          <div><h1>模型</h1><p>连接、默认聊天模型和向量模型由当前 models 插件管理。</p></div>
          <button type="button" data-refresh>刷新</button>
        </header>
        <p class="models-plugin-notice" data-notice role="status" aria-live="polite"></p>
        <section><header><h2>已连接</h2></header><div class="models-plugin-list" data-connections></div></section>
        <section><header><h2>系统模型</h2></header><div class="models-plugin-bindings" data-bindings></div></section>
        <section><header><h2>添加连接</h2><p>下面每一项都由对应 Provider 插件注册。</p></header><div class="models-plugin-list" data-providers></div></section>`;
      host.replaceChildren(page);
      const notice = page.querySelector("[data-notice]");
      const connections = page.querySelector("[data-connections]");
      const bindings = page.querySelector("[data-bindings]");
      const providers = page.querySelector("[data-providers]");
      let catalog = null;
      let disposeDialog = () => {};

      const request = async (path, init) => {
        const response = await ctx.http.request(path, init);
        const body = await response.json();
        if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `请求失败：${response.status}`);
        return body;
      };
      const catalogReads = createLatestCatalogRead(
        (signal) => request("/api/dashboard/models/catalog", {signal}),
        (nextCatalog) => {
          catalog = nextCatalog;
          notice.textContent = "";
          renderCatalog();
        },
      );
      const load = () => catalogReads.run();
      const command = async (payload) => {
        const receipt = await request("/api/dashboard/models/command", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload),
        });
        await load();
        return receipt;
      };
      const report = (work) => work.catch((error) => {
        notice.setAttribute("role", "alert");
        notice.textContent = error instanceof Error ? error.message : String(error);
      });
      const openProvider = (entry, trigger, connection = null) => {
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
              model: {
                ...input.model,
                expected_revision: catalog.revision,
                model_id: modelId,
                connection_id: connectionId,
              },
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
              input: {
                ...input,
                auth_identity: connection?.authIdentity ?? connectionId,
              },
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
        const providerDialog = document.createElement("dialog");
        providerDialog.className = "models-provider-dialog";
        providerDialog.setAttribute("aria-label", entry.label);
        page.appendChild(providerDialog);
        let disposeEntry;
        try {
          disposeEntry = connectionTypes.render(entry.id, providerDialog, {
            get state() {
              return Object.freeze({
                connection,
                models: Object.freeze(catalog.models.filter((model) => model.connectionId === connectionId)),
              });
            },
            actions,
            close() { providerDialog.close(); },
            changed(message) {
              notice.setAttribute("role", "status");
              notice.textContent = message;
              providerDialog.close();
            },
          });
        } catch (error) {
          providerDialog.remove();
          report(Promise.reject(error));
          return;
        }
        const close = () => disposeDialog();
        providerDialog.addEventListener("close", close, {once: true});
        disposeDialog = () => {
          report(auth.close());
          providerDialog.removeEventListener("close", close);
          disposeEntry();
          providerDialog.remove();
          if (trigger.isConnected) trigger.focus();
          disposeDialog = () => {};
        };
        providerDialog.showModal();
      };
      const renderCatalog = () => {
        notice.setAttribute("role", "status");
        for (const button of providers.querySelectorAll("button")) button.disabled = false;
        connections.replaceChildren();
        for (const connection of catalog.connections) {
          const item = document.createElement("article");
          item.className = "models-plugin-card";
          item.innerHTML = `<strong></strong><small></small>`;
          item.querySelector("strong").textContent = connection.name;
          const models = catalog.models.filter((model) => model.connectionId === connection.id);
          item.querySelector("small").textContent = `${connection.driverId} · ${models.length} 个模型 · ${connection.availability}`;
          const entry = connectionTypes.entries.find((candidate) => candidate.id === connection.driverId);
          if (entry) {
            const edit = document.createElement("button");
            edit.type = "button";
            edit.textContent = "编辑连接";
            edit.addEventListener("click", () => openProvider(entry, edit, connection));
            item.appendChild(edit);
          }
          connections.appendChild(item);
        }
        if (!catalog.connections.length) connections.textContent = "尚未添加连接。";
        renderBindings();
      };
      const renderBindings = () => {
        bindings.replaceChildren();
        const options = catalog.models.filter((model) => model.kind === "chat");
        for (const [role, label] of [["default", "默认模型"], ["agent", "Agent 模型"], ["fast", "轻量模型"], ["vision", "视觉模型"]]) {
          const row = document.createElement("label");
          const title = document.createElement("span");
          const select = document.createElement("select");
          title.textContent = label;
          const empty = new Option("未选择", "");
          empty.disabled = true;
          select.append(empty);
          for (const model of options) select.append(new Option(model.model, model.id));
          select.value = catalog.roleBindings[role] ?? "";
          select.addEventListener("change", () => report(command({
            type: "set_default",
            expected_revision: catalog.revision,
            role,
            model_id: select.value,
          })));
          row.append(title, select);
          bindings.appendChild(row);
        }
        const embedding = document.createElement("label");
        const embeddingTitle = document.createElement("span");
        const embeddingSelect = document.createElement("select");
        embeddingTitle.textContent = "向量模型";
        const empty = new Option("未选择", "");
        empty.disabled = true;
        embeddingSelect.append(empty);
        for (const model of catalog.models.filter((item) => item.kind === "embedding")) {
          embeddingSelect.append(new Option(model.model, model.id));
        }
        embeddingSelect.value = catalog.defaultEmbeddingModelId ?? "";
        embeddingSelect.addEventListener("change", () => report(command({
          type: "set_default",
          expected_revision: catalog.revision,
          role: null,
          model_id: embeddingSelect.value,
        })));
        embedding.append(embeddingTitle, embeddingSelect);
        bindings.appendChild(embedding);
      };
      for (const entry of connectionTypes.entries) {
        if (typeof entry.label !== "string" || typeof entry.detail !== "string") {
          throw new Error(`连接类型 ${entry.id} 缺少 label 或 detail`);
        }
        const button = document.createElement("button");
        button.type = "button";
        button.className = "models-plugin-card models-plugin-card--button";
        button.disabled = true;
        const title = document.createElement("strong");
        const detail = document.createElement("small");
        title.textContent = entry.label;
        detail.textContent = entry.detail;
        button.append(title, detail);
        button.addEventListener("click", () => openProvider(entry, button));
        providers.appendChild(button);
      }
      if (!connectionTypes.entries.length) providers.textContent = "没有 Provider 插件提供连接方式。";
      page.querySelector("[data-refresh]").addEventListener("click", () => report(load()));
      report(load());
      return () => {
        catalogReads.close();
        disposeDialog();
        host.replaceChildren();
      };
    },
  }));
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

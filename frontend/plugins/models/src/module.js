import { usageSection } from "./usage-page.js";
import { mountModelPicker } from "./picker-mount";

const SEARCH_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-search" aria-hidden="true"><circle cx="11" cy="11" r="8"></circle><path d="m21 21-4.3-4.3"></path></svg>`;
const CHEVRON_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-chevron-right" aria-hidden="true"><path d="m9 18 6-6-6-6"></path></svg>`;
const TRASH_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 6h18"></path><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path></svg>`;
const REFRESH_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"></path><path d="M21 3v5h-5"></path><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"></path><path d="M8 16H3v5"></path></svg>`;
const SERVER_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect width="20" height="8" x="2" y="2" rx="2"></rect><rect width="20" height="8" x="2" y="14" rx="2"></rect><path d="M6 6h.01"></path><path d="M6 18h.01"></path></svg>`;
const CLOSE_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M18 6 6 18"></path><path d="m6 6 12 12"></path></svg>`;
const SPINNER_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="is-spinning" aria-hidden="true"><path d="M21 12a9 9 0 1 1-6.219-8.56"></path></svg>`;

// 聊天只保留全局默认；压缩、看图、标题等各自在插件设置里选模型，留空跟随这里。
const ROLE_LABELS = [
  ["default", "全局默认模型", "对话默认使用它；各插件未单独指定时也跟随它"],
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
  // 目录归模块所有：宿主空闲时预热，设置页挂载先画上次的目录，再在后台核对。
  const catalog = ctx.data.resource("catalog", async (signal) =>
    readJsonResponse(await ctx.http.request("/api/dashboard/models/catalog", {signal})), {warm: true});
  return ctx.ui.inject("shell.settings.v1", (mount) => {
    const releaseUsage = mount.register(usageSection(ctx, catalog));
    const releaseModels = registerModels(ctx, mount, catalog);
    return () => { releaseModels(); releaseUsage(); };
  });
}

function registerModels(ctx, mount, catalogResource) {
  return mount.register({
    id: "models",
    label: "模型",
    iconSvg: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/></svg>',
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
          <div><h1 data-title>模型连接</h1><p data-description>管理各模型服务商连接与密钥；未探测到的能力不会被默认开启。</p></div>
          <div class="settings-header-actions"></div>
        </header>
        <p class="settings-inline-error" data-error role="alert" hidden></p>
        <div class="settings-section-bar" data-list-bar>
          <h2 class="settings-section-label">服务商连接</h2>
          <label class="settings-find" data-search>${SEARCH_ICON}<span class="sr-only">查找连接或模型</span><input placeholder="查找连接或模型"></label>
        </div>
        <ul class="settings-provider-list" data-connections></ul>
        <div class="settings-add-provider">
          <button type="button" class="settings-add-button" data-add-toggle aria-expanded="false" disabled>+ 添加服务商</button>
          <section class="settings-add-panel" data-add-panel hidden>
            <div class="settings-section-bar">
              <h2 class="settings-section-label" data-add-title>添加服务商</h2>
              <label class="settings-find">${SEARCH_ICON}<span class="sr-only">查找服务商</span><input data-template-search placeholder="查找服务商…"></label>
              <button type="button" class="settings-text-button" data-add-close>关闭</button>
            </div>
            <div class="settings-add-list" data-add-list></div>
            <div class="settings-provider-editor" data-add-editor hidden></div>
          </section>
        </div>
        <section class="settings-section settings-roles" data-roles>
          <div class="settings-section-bar"><h2 class="settings-section-label">默认模型</h2><span class="settings-section-hint">压缩、看图、子任务等在各自的插件设置里可另选模型，未指定时跟随这里</span></div>
          <div class="settings-role-grid" data-bindings></div>
        </section>
      </div>
      <div class="settings-toast-region" aria-live="polite" aria-atomic="true" data-toast-region></div>`;
      host.replaceChildren(page);

      const title = page.querySelector("[data-title]");
      const description = page.querySelector("[data-description]");
      const search = page.querySelector("[data-search]");
      const searchInput = search.querySelector("input");
      const errorMessage = page.querySelector("[data-error]");
      const connections = page.querySelector("[data-connections]");
      const addToggle = page.querySelector("[data-add-toggle]");
      const addPanel = page.querySelector("[data-add-panel]");
      const addList = page.querySelector("[data-add-list]");
      const addEditor = page.querySelector("[data-add-editor]");
      const addTitle = page.querySelector("[data-add-title]");
      const addClose = page.querySelector("[data-add-close]");
      const templateSearch = page.querySelector("[data-template-search]");
      const listBar = page.querySelector("[data-list-bar]");
      const roles = page.querySelector("[data-roles]");
      const bindings = page.querySelector("[data-bindings]");
      const toastRegion = page.querySelector("[data-toast-region]");
      let catalog = null;
      let query = "";
      let quotas = new Map();
      let quotaKey = "";
      let quotaAt = 0;
      // 编辑器就地展开在连接卡片（编辑）或添加面板（新建）内，同一时刻只开一个。
      let activeEditor = null;
      // 新建成功后要展开的连接；目录读到它之前先记着。
      let openAfterCreate = "";
      const closeActiveEditor = (force = false) => {
        const editor = activeEditor;
        if (!editor) return true;
        if (!force) {
          if (editor.busy()) { showNotice("请求正在执行，请等待结果后再离开。关闭页面不会撤销已提交的操作。"); return false; }
          if (editor.dirty() && !window.confirm("放弃尚未保存的修改？选择取消可继续填写。")) return false;
        }
        editor.dispose();
        return true;
      };
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
      // 普通核对并入模块资源里进行中的读取；写入之后的核对另起一次，不读到写入前的目录。
      const reads = createLatestCatalogRead(
        (_signal, fresh) => catalogResource.refresh({fresh}),
        (nextCatalog) => {
          if (bindingSave?.saving) return;
          // 核对结果与正在显示的是同一份目录：不重绘，避免打断阅读与展开状态。
          if (nextCatalog === catalog && !bindingSave && errorMessage.hidden) return;
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
      const load = () => reads.run(undefined, true);
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

        title.textContent = "模型服务";
        description.textContent = hasConnections
          ? "已配置的服务商连接如下；点击展开卡片可管理可用模型。"
          : "请先添加模型连接，支持 ChatGPT 订阅、DeepSeek、Gemini 及兼容 API。";
        roles.hidden = false;
        addToggle.disabled = false;
        // 首次配置没有连接时直接进入添加目录且不能关闭；编辑器打开期间不抢面板状态。
        addTitle.textContent = hasConnections ? "添加服务商" : "添加第一个服务商";
        addClose.hidden = !hasConnections;
        if (!hasConnections && !activeEditor) setAddPanel(true);
        else if (!addPanel.hidden) listBar.hidden = connections.hidden = true;
        renderConnections(chatConnections);
        renderBindings(chatModels);
        loadQuotas().catch(() => {});
        openCreated();
      }

      // 仿 magpie：连接建好即打开它的详情，账号额度、模型勾选与参数都在那里完成。
      function openCreated() {
        if (!openAfterCreate || activeEditor) return;
        const created = catalog.connections.find((connection) => connection.id === openAfterCreate);
        if (!created) return;
        openAfterCreate = "";
        openExisting(created);
      }

      // 添加是页面的另一种状态：打开时连接列表让位给服务商目录，关闭后回到列表。
      function setAddPanel(open) {
        addPanel.hidden = !open;
        listBar.hidden = open;
        connections.hidden = open;
        addToggle.hidden = open;
        addToggle.setAttribute("aria-expanded", String(open));
        if (!open) return;
        templateSearch.value = "";
        renderTemplates();
        queueMicrotask(() => templateSearch.focus());
      }

      // 添加面板按接入方式分组：订阅登录、厂商 API、自定义；每组一格一种方式，已添加的单实例方式标出并直接打开它。
      function renderTemplates() {
        // 正在新建的编辑器挂在某一组下面，重绘目录前先收回来，避免随旧节点一起丢掉。
        addPanel.appendChild(addEditor);
        addList.replaceChildren();
        const needle = templateSearch.value.trim().toLocaleLowerCase();
        const matches = (template) => !needle || `${template.label} ${template.detail ?? ""}`.toLocaleLowerCase().includes(needle);
        for (const [group, label, hint] of TEMPLATE_GROUPS) {
          const items = providerTemplates.filter((template) => templateGroup(template) === group && matches(template));
          if (!items.length) continue;
          const section = document.createElement("section");
          section.className = "settings-template-group";
          section.innerHTML = "<h3><span></span><small></small></h3><div class=\"settings-template-grid\"></div>";
          section.querySelector("h3 span").textContent = label;
          section.querySelector("h3 small").textContent = hint;
          const grid = section.querySelector(".settings-template-grid");
          for (const template of items) grid.appendChild(templateTile(template));
          addList.appendChild(section);
        }
        if (!providerTemplates.length) addList.textContent = "没有 Provider 插件提供连接方式。";
        else if (!addList.childElementCount) addList.appendChild(noTemplateMatch(templateSearch.value.trim()));
      }

      // 查找无结果时给出去路：按自定义兼容地址添加。
      function noTemplateMatch(name) {
        const empty = document.createElement("p");
        empty.className = "settings-template-empty";
        empty.append(`没有叫“${name}”的服务商。`);
        const custom = providerTemplates.find((template) => templateGroup(template) === "custom");
        if (custom) {
          const link = document.createElement("button");
          link.type = "button";
          link.className = "settings-text-button";
          link.textContent = "用自定义 API 添加";
          link.addEventListener("click", () => {
            templateSearch.value = "";
            renderTemplates();
            addList.querySelector(`[data-template-id="${CSS.escape(custom.id)}"]`)?.click();
          });
          empty.append(link);
        }
        return empty;
      }

      function templateTile(template) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "settings-template-tile";
        button.dataset.templateId = template.id;
        button.appendChild(providerMark(template));
        // 格子只写名称；一句说明放进提示，组标题已经说明了接入方式。
        const title = document.createElement("span");
        title.className = "settings-template-name";
        title.textContent = template.label;
        if (template.detail) button.title = template.detail;
        button.append(title);
        // 只有一种模板的接入方式（Codex、OpenCode Go）一个账号一条连接：已连接时标出，再点打开它而不是新建。
        const existing = (template.owner.templates?.length ?? 0) < 2 && catalog
          ? catalog.connections.find((connection) => connection.driverId === template.owner.id)
          : null;
        if (existing) {
          const mark = document.createElement("i");
          mark.className = "settings-template-added";
          mark.title = "已连接";
          button.append(mark);
          button.setAttribute("aria-label", `${template.label}，已连接，打开 ${existing.name}`);
          button.addEventListener("click", () => openExisting(existing));
        } else {
          button.addEventListener("click", () => startCreate(template, button));
        }
        return button;
      }

      function openExisting(connection) {
        const head = connections.querySelector(`[data-connection-id="${CSS.escape(connection.id)}"] .settings-provider-head`);
        if (!head || head.disabled) { showNotice(`${connection.name} 已停用，请先在连接列表里启用。`); return; }
        setAddPanel(false);
        if (head.getAttribute("aria-expanded") !== "true") head.click();
        head.scrollIntoView({block: "start", behavior: "smooth"});
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
        // 编辑器就地展开在卡片内；目录刷新与搜索过滤都不能拆掉正在编辑的卡片，
        // 只移除其余卡片并用 insertBefore 重排，编辑卡片始终不脱离文档（保住焦点与输入状态）。
        const editing = activeEditor?.kind === "edit" ? activeEditor : null;
        if (editing && !filtered.some((connection) => connection.id === editing.connectionId)) {
          const current = allConnections.find((connection) => connection.id === editing.connectionId);
          if (current) filtered.unshift(current);
        }
        const editingCard = editing?.card ?? null;
        for (const child of [...connections.children]) {
          if (child !== editingCard) child.remove();
        }
        let next = connections.firstChild;
        for (const connection of filtered) {
          let element;
          if (editing && connection.id === editing.connectionId) {
            fillProviderHead(editing.head, providerEntries.find((candidate) => candidate.id === connection.driverId), connection);
            element = editing.card;
          } else {
            element = providerCard(connection);
          }
          if (element === next) { next = next.nextSibling; continue; }
          if (element === editingCard) {
            // 编辑卡片不移动，insertBefore 的重挂会丢失焦点；顺序在编辑器关闭后的重渲染自然恢复。
            next = element.nextSibling;
            continue;
          }
          connections.insertBefore(element, next);
        }
      }

      function providerCard(connection) {
        const entry = providerEntries.find((candidate) => candidate.id === connection.driverId);
        const card = document.createElement("li");
        card.className = "settings-provider-card";
        card.dataset.connectionId = connection.id;
        // 行内两个独立控件：展开按钮铺满整行，启用开关浮在其上；箭头只是装饰。
        const row = document.createElement("div");
        row.className = "settings-provider-row";
        const head = document.createElement("button");
        head.type = "button";
        head.className = "settings-provider-head";
        head.setAttribute("aria-expanded", "false");
        const copy = document.createElement("span");
        copy.className = "settings-card-copy";
        copy.innerHTML = "<strong data-name></strong><small data-detail></small>";
        const quota = document.createElement("span");
        quota.className = "settings-provider-quota";
        quota.dataset.quota = "";
        const status = document.createElement("span");
        status.className = "settings-provider-status";
        status.dataset.status = "";
        status.innerHTML = "<i aria-hidden=\"true\"></i><span></span>";
        head.append(connectionMark(entry), copy, quota, status);
        const toggle = document.createElement("label");
        toggle.className = "settings-provider-switch";
        toggle.innerHTML = "<input type=\"checkbox\" role=\"switch\">";
        const enabled = toggle.querySelector("input");
        enabled.checked = connection.availability !== "disabled";
        enabled.setAttribute("aria-label", `启用 ${connection.name}`);
        enabled.addEventListener("change", () => report(setConnectionEnabled(connection, enabled)));
        row.append(head, toggle);
        row.insertAdjacentHTML("beforeend", CHEVRON_ICON);
        fillProviderHead(head, entry, connection);
        if (!entry || connection.availability === "disabled") {
          head.disabled = true;
        } else {
          head.setAttribute("aria-label", `编辑连接 ${connection.name}`);
          head.addEventListener("click", () => toggleProviderEditor(entry, card, head, connection));
        }
        const editorRegion = document.createElement("div");
        editorRegion.className = "settings-provider-editor";
        editorRegion.hidden = true;
        card.append(row, editorRegion);
        return card;
      }

      function fillProviderHead(head, entry, connection) {
        const models = catalog.models.filter((model) => model.connectionId === connection.id);
        const openCount = models.filter((model) => model.availability !== "disabled").length;
        const reading = connection.availability === "available" ? quotas.get(connection.id) : undefined;
        head.querySelector("[data-name]").textContent = connection.name;
        // 副标题说明连的是哪个账号或地址、有几个模型；订阅连接先写套餐。
        const where = reading?.account || connection.endpointHost || entry?.label || connection.driverId;
        const who = [reading?.plan, where].filter(Boolean).join(" · ");
        const paused = models.length - openCount;
        head.querySelector("[data-detail]").textContent = connection.availability === "disabled" || !paused
          ? `${who} · ${models.length} 个模型`
          : `${who} · ${models.length} 个模型，${paused} 个已停用`;
        fillQuota(head.querySelector("[data-quota]"), reading);
        // 正常是常态，不占位；停用由开关表达，只有驱动异常才标出。
        const status = head.querySelector("[data-status]");
        status.hidden = connection.availability !== "driver_unavailable";
        status.classList.toggle("is-unavailable", connection.availability !== "available");
        const AVAILABILITY_LABELS = { available: "正常可用", disabled: "已停用", driver_unavailable: "驱动异常" };
        status.querySelector("span").textContent = AVAILABILITY_LABELS[connection.availability] ?? connection.availability;
      }

      // 行尾额度：最多两条最短窗口的已用比例，或余额；读不到时不占位，原因留在提示里。
      function fillQuota(slot, reading) {
        slot.replaceChildren();
        slot.removeAttribute("title");
        if (!reading) return;
        if (reading.error) slot.title = `额度读取失败：${reading.error}`;
        if (reading.windows.length) {
          for (const span of reading.windows.slice(0, 2)) {
            const used = Math.max(0, Math.min(100, Math.round(span.usedPercent)));
            const label = document.createElement("span");
            label.textContent = `${span.label} ${used}%`;
            const bar = document.createElement("span");
            bar.className = "settings-quota-bar";
            bar.innerHTML = "<i></i>";
            bar.firstElementChild.style.inlineSize = `${used}%`;
            bar.classList.toggle("is-high", used >= 80);
            slot.append(label, bar);
          }
        } else if (reading.balance) {
          slot.textContent = `余额 ${reading.balance}`;
        } else if (reading.error) {
          slot.textContent = "额度未读到";
        }
        slot.classList.toggle("is-stale", !!reading.error);
      }

      // 读取额度：连接集合变化或超过一分钟才重读；服务端另有缓存与最短间隔。
      function loadQuotas(fresh = false) {
        const ids = catalog.connections.filter((item) => item.availability === "available").map((item) => item.id).sort().join(",");
        if (!ids || (!fresh && ids === quotaKey && Date.now() - quotaAt < 60_000)) return Promise.resolve();
        quotaKey = ids;
        quotaAt = Date.now();
        return request(`/api/dashboard/models/quotas${fresh ? "?fresh=true" : ""}`).then((body) => {
          if (closed) return;
          quotas = new Map(body.quotas.map((item) => [item.connectionId, item]));
          for (const card of connections.children) {
            const connection = catalog.connections.find((item) => item.id === card.dataset.connectionId);
            const head = card.querySelector(".settings-provider-head");
            if (connection && head) fillProviderHead(head, providerEntries.find((item) => item.id === connection.driverId), connection);
          }
          activeEditor?.refresh?.();
        }).catch((reason) => { quotaAt = 0; throw reason; });
      }

      // 开关直接生效；停用承担默认模型的连接前先确认，失败时回到实际状态。
      async function setConnectionEnabled(connection, input) {
        const enable = input.checked;
        const roles = Object.values(catalog.roleBindings ?? {});
        const hostsDefault = catalog.models.some((model) => model.connectionId === connection.id
          && (roles.includes(model.id) || model.id === catalog.defaultEmbeddingModelId));
        if (!enable && hostsDefault
            && !window.confirm(`默认模型在 ${connection.name} 上，停用后用到它的对话会失败，需另选默认模型。仍然停用？`)) {
          input.checked = true;
          return;
        }
        if (activeEditor?.connectionId === connection.id && !closeActiveEditor()) {
          input.checked = !enable;
          return;
        }
        input.disabled = true;
        try {
          await command({type: enable ? "enable_connection" : "disable_connection",
            expected_revision: catalog.revision, connection_id: connection.id});
          showNotice(enable ? `${connection.name} 已启用。` : `${connection.name} 已停用，模型配置与历史保留。`);
        } catch (error) {
          input.checked = !enable;
          throw error;
        } finally {
          input.disabled = false;
        }
      }

      function toggleProviderEditor(entry, card, head, connection) {
        if (activeEditor?.kind === "edit" && activeEditor.card === card) { closeActiveEditor(); return; }
        if (bindingSave) { showNotice("请先等待模型选择保存或核对完成，再修改连接。"); return; }
        if (!closeActiveEditor()) return;
        openProvider(entry, head, connection, editTemplate(entry), {
          kind: "edit", card, head,
          host: card.querySelector(".settings-provider-editor"),
        });
      }

      function startCreate(template, trigger) {
        if (bindingSave) { showNotice("请先等待模型选择保存或核对完成，再添加连接。"); return; }
        if (!closeActiveEditor()) return;
        // 新建表单展开在所选格子那一组的下方，目录其余部分保持原位。
        trigger.closest(".settings-template-group")?.appendChild(addEditor);
        openProvider(template.owner, trigger, null, template, {
          kind: "create", card: addPanel, head: trigger, host: addEditor,
        });
      }

      function renderBindings(chatModels) {
        // 搜索或目录刷新不能撤销保存锁，也不能换掉尚未核对的选择。
        if (bindingSave) return;
        bindings.replaceChildren();
        for (const [role, label, detail] of ROLE_LABELS) {
          const bound = catalog.roleBindings[role] ?? "";
          const availableModels = chatModels
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
          label: "知识库嵌入向量模型",
          detail: "用于情景记忆检索与文本向量化",
          models: embeddingModels,
          value: embeddingBound,
          change(modelId) {
            return sendCommand({type: "set_default", expected_revision: catalog.revision, role: null, model_id: modelId});
          },
        }));
        const addEmbedding = document.createElement("button");
        addEmbedding.type = "button";
        addEmbedding.className = "settings-add-embedding";
        addEmbedding.textContent = "+ 添加向量模型";
        addEmbedding.addEventListener("click", () => openEmbedding(addEmbedding));
        bindings.appendChild(addEmbedding);
      }

      function openEmbedding(trigger) {
        if (bindingSave) { showNotice("请先等待模型选择保存或核对完成，再添加向量模型。"); return; }
        if (!closeActiveEditor()) return;
        disposeDialog();
        const dialog = document.createElement("dialog");
        dialog.className = "settings-scrim";
        dialog.setAttribute("aria-label", "添加向量模型");
        dialog.innerHTML = `<form class="settings-dialog settings-dialog-form embedding-form">
          <header class="settings-dialog-header"><div><h2>添加向量模型</h2><p>用于情景记忆检索。读取模型列表或手动指定，通过测试文本自动测量实际维度。</p></div></header>
          <div class="settings-dialog-body"><div class="settings-form-grid">
          <label class="is-wide"><span>所属服务连接</span><select name="connection" aria-label="所属服务连接" required></select></label>
          <div class="is-wide settings-form-grid" data-new-connection>
            <label class="is-wide"><span>连接名称</span><input name="name" aria-label="向量连接名称" value="向量服务" maxlength="128"></label>
            <label class="is-wide"><span>Base URL</span><input name="endpoint" aria-label="向量 Base URL" type="url" placeholder="https://api.example.com/v1"></label>
            <label class="is-wide"><span>API Key</span><input name="key" aria-label="向量 API Key" type="password" autocomplete="off"></label>
          </div>
          <button type="button" class="settings-secondary-button is-wide" data-directory>探测模型列表</button>
          <label class="is-wide" data-candidates hidden><span>选择模型</span><select name="candidate" aria-label="向量模型型号"></select></label>
          <button type="button" class="settings-text-button is-wide" data-manual hidden>列表中没有所需模型？手动填写</button>
          <label class="is-wide" data-manual-field><span>模型名称</span><input name="model" aria-label="向量模型名称" required maxlength="256" placeholder="例如：text-embedding-3-small"></label>
          <button type="button" class="settings-secondary-button is-wide" data-probe>测试并获取维度</button>
          <p class="is-wide" role="status" data-result>尚未测试。建议先完成测试以确保模型可用。</p>
          </div><p class="settings-inline-error" role="alert" hidden></p>
          <p>保存后将设为默认向量模型；已有记忆库若维度不匹配将暂停迁移，不会破坏历史数据。</p></div>
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

      // 场景选择与聊天胶囊共用选择器：搜索、来源栏、收藏与键盘操作一致。
      function modelPickSheet({title, models, currentId}) {
        const order = new Map(catalog.connections.map((connection, index) => [connection.id, index]));
        const runtimes = [...models]
          .sort((a, b) => (order.get(a.connectionId) ?? Infinity) - (order.get(b.connectionId) ?? Infinity))
          .map((model) => {
            const connection = catalog.connections.find((item) => item.id === model.connectionId);
            const entry = providerEntries.find((candidate) => candidate.id === connection?.driverId);
            const caps = model.capabilities ?? {};
            const icon = typeof entry?.connectionIcon === "string" && entry.connectionIcon.startsWith("data:image/svg+xml,") ? entry.connectionIcon : "";
            return {
              id: model.id, provider: connection?.driverId ?? "", model: model.model,
              sourceId: model.connectionId, sourceName: connection?.name ?? "其他", icon,
              reasoningEffort: "", supportedReasoningEfforts: caps.supportedReasoningEfforts ?? [], roles: [],
              contextWindow: caps.contextWindow ?? 0, inputModalities: caps.inputModalities ?? [],
            };
          });
        return modelSheet({
          title, groups: [],
          mount: (host, settle) => mountModelPicker(host, {runtimes, selectedId: currentId, follow: null, onChoose: settle}),
        });
      }

      // 统一的小型浮层：multiple=true 是候选勾选（复选 + 底部确认），false 是单选列表（点击即选）。
      // 解析单选行 data、复选行 data 数组；取消一律解析 null。
      // mount 给出时由它渲染列表区（返回卸载函数），用于单选的模型选择器。
      function modelSheet({title, hint, multiple = false, confirmLabel = "确定", groups, mount = null}) {
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
          let unmount = null;
          if (mount) {
            sheet.classList.add("settings-sheet--picker");
            list.className = "settings-sheet-picker";
          }
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
            unmount?.();
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
          if (mount) unmount = mount(list, settle);
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
          if (bound) map.set(bound, [map.get(bound), label.replace("模型", "").trim()].filter(Boolean).join("、"));
        }
        if (catalog.defaultEmbeddingModelId) map.set(catalog.defaultEmbeddingModelId, "默认向量");
        return map;
      }

      // 详情里的模型区：目录候选与已保存模型合成一组可勾选的标签，勾选、手填与参数都先暂存，保存时一并提交。
      function buildModelManager({connection, actions, readCatalog, dialogClosed, dirty}) {
        const section = document.createElement("section");
        section.className = "settings-model-manage";
        section.innerHTML = `<div class="settings-detail-row settings-account" data-account hidden></div>
          <div class="settings-detail-row"><span class="settings-detail-key">模型 <small data-manage-count></small></span><div class="settings-detail-value">
            <div class="settings-chip-toolbar">
              <label class="settings-model-filter">${SEARCH_ICON}<input aria-label="筛选模型" placeholder="筛选模型"></label>
              <button type="button" class="settings-text-button" data-all>全选</button>
              <button type="button" class="settings-text-button" data-none>全不选</button>
              <button type="button" class="settings-text-button" data-manual-toggle>添加模型 ID</button>
              <button type="button" class="settings-icon-button" data-reload aria-label="重新读取服务目录" title="重新读取服务目录">${REFRESH_ICON}</button>
              <button type="button" class="settings-text-button" data-params aria-expanded="false">参数</button>
            </div>
            <div class="settings-model-manual" data-manual hidden><input aria-label="模型 ID" maxlength="256" placeholder="型号，例如 gpt-5"><button type="button" class="settings-secondary-button" data-manual-add>加入</button></div>
            <div class="settings-chips" data-chips></div>
            <div class="settings-param-table" data-param-table hidden></div>
            <p class="settings-model-hint" data-manage-status role="status"></p>
            <p class="settings-inline-error" data-manage-error role="alert" hidden></p>
          </div></div>`;
        const $ = (selector) => section.querySelector(selector);
        const account = $("[data-account]");
        const count = $("[data-manage-count]");
        const filterInput = $(".settings-model-filter input");
        const chips = $("[data-chips]");
        const table = $("[data-param-table]");
        const params = $("[data-params]");
        const status = $("[data-manage-status]");
        const error = $("[data-manage-error]");
        const manual = $("[data-manual]");
        const manualInput = manual.querySelector("input");
        const showError = (reason) => { error.textContent = reason instanceof Error ? reason.message : String(reason); error.hidden = false; };
        const key = (kind, name) => kind === "embedding" ? `embedding:${name}` : name;
        const savedModels = () => catalog.models.filter((model) => model.connectionId === connection.id);
        // 目录候选与手填 ID 按 key 合并；picked 是暂存的勾选，参数草稿按模型 ID 存放。
        const candidates = new Map();
        let picked = new Set(savedModels().map((model) => key(model.kind, model.model)));
        const drafts = new Map();
        let loading = true;

        const isDirty = () => {
          const saved = new Set(savedModels().map((model) => key(model.kind, model.model)));
          return saved.size !== picked.size || [...picked].some((item) => !saved.has(item)) || drafts.size > 0;
        };
        const reportDirty = () => dirty(isDirty() || !!manualInput.value);
        for (const type of ["input", "change"]) section.addEventListener(type, (event) => { event.stopPropagation(); reportDirty(); });

        // 一个标签对应一个可选模型：已保存的带在用标记，新勾选的保存时才验证。
        function renderChips() {
          const used = usedByMap();
          const saved = new Map(savedModels().map((model) => [key(model.kind, model.model), model]));
          const all = new Map([...candidates, ...saved]);
          const needle = filterInput.value.trim().toLowerCase();
          chips.replaceChildren();
          for (const [id, item] of [...all].sort(([left], [right]) => left.localeCompare(right))) {
            const model = saved.get(id);
            const chip = document.createElement("button");
            chip.type = "button";
            chip.className = "settings-chip";
            chip.dataset.key = id;
            chip.hidden = !!needle && !item.model.toLowerCase().includes(needle);
            chip.setAttribute("aria-pressed", String(picked.has(id)));
            const name = document.createElement("span");
            name.textContent = item.model;
            const facts = document.createElement("small");
            facts.textContent = [item.kind === "embedding" ? "向量" : "", ...capabilityBadges(item.capabilities)].filter(Boolean).join(" · ");
            chip.append(name, facts);
            const use = model ? used.get(model.id) : "";
            if (use) chip.append(Object.assign(document.createElement("em"), {textContent: use}));
            if (!model) chip.classList.add("is-new");
            chips.append(chip);
          }
          if (!all.size) chips.textContent = loading ? "正在读取服务目录…" : "目录里没有模型；可以手动添加模型 ID。";
          else if (!loading && !savedModels().length && !status.textContent) status.textContent = "这个连接还没有启用的模型：勾选要用的，再点保存。";
          count.textContent = `${picked.size} / ${all.size}`;
          reportDirty();
        }

        // 参数表只列已保存的对话模型；改动暂存到 drafts，保存时逐个 update_model。
        function renderTable() {
          const rows = savedModels().filter((model) => model.kind === "chat");
          table.innerHTML = `<div class="settings-param-head"><span>模型</span><span>上下文</span><span>最大输出</span><span>看图</span><span></span></div>`;
          for (const model of rows) {
            const caps = model.capabilities ?? {};
            const draft = drafts.get(model.id);
            const row = document.createElement("div");
            row.className = "settings-param-row";
            row.innerHTML = `<strong></strong><input data-field="context" aria-label="上下文" placeholder="未知" autocomplete="off">
              <input data-field="output" aria-label="最大输出" placeholder="未知" autocomplete="off">
              <input type="checkbox" data-field="image" aria-label="可看图"><button type="button" class="settings-text-button" data-verify>验证</button>`;
            row.querySelector("strong").textContent = model.model;
            const [context, output, image] = row.querySelectorAll("[data-field]");
            const base = [formatTokenCount(caps.contextWindow), formatTokenCount(caps.maxOutputTokens), (caps.inputModalities ?? []).includes("image")];
            [context.value, output.value, image.checked] = draft ?? base;
            for (const input of [context, output, image]) input.addEventListener(input === image ? "change" : "input", () => {
              const next = [context.value, output.value, image.checked];
              if (next.every((value, index) => value === base[index])) drafts.delete(model.id);
              else drafts.set(model.id, next);
              input.removeAttribute("aria-invalid");
            });
            const verify = row.querySelector("[data-verify]");
            verify.addEventListener("click", () => {
              verify.disabled = true;
              actions.verifyModel(model.id).then(() => { status.textContent = `${model.model} 验证通过`; })
                .catch(showError).finally(() => { verify.disabled = false; });
            });
            table.append(row);
          }
          if (!rows.length) table.insertAdjacentHTML("beforeend", `<p class="settings-model-hint">保存后的对话模型才能设置参数。</p>`);
          table.insertAdjacentHTML("beforeend", `<p class="settings-model-hint">留空表示未知；K = 1,000，M = 1,000,000。勾选「看图」后，图片会直接发给这个模型。</p>`);
        }

        // 1. 先校验全部参数草稿，任何一格不合法都不开始写入。
        function parsedDrafts() {
          const patches = [];
          for (const [modelId, [context, output, image]] of drafts) {
            const contextWindow = parseTokenCount(context);
            const maxOutputTokens = parseTokenCount(output);
            if (contextWindow === false || maxOutputTokens === false) {
              params.setAttribute("aria-expanded", "true");
              table.hidden = false;
              renderTable();
              throw new Error("参数请输入正整数或带 K/M 后缀的数值；留空表示未知。");
            }
            patches.push([modelId, {context_window: contextWindow, max_output_tokens: maxOutputTokens, image_input: image}]);
          }
          return patches;
        }

        // 移除在用模型前要说明后果；保存按钮在开始写入前调用。
        function removalWarning() {
          const used = usedByMap();
          const leaving = savedModels().filter((model) => !picked.has(key(model.kind, model.model)) && used.get(model.id));
          if (!leaving.length) return "";
          const names = leaving.map((model) => `${model.model}（${used.get(model.id)}）`).join("、");
          // 后端只给对话角色挑替补；向量默认会被清空，需要用户重新指定。
          const leavingIds = new Set(leaving.map((model) => model.id));
          const spare = catalog.models.some((model) => model.kind === "chat" && model.availability === "available" && !leavingIds.has(model.id)
            && (model.connectionId !== connection.id || picked.has(key(model.kind, model.model))));
          const effects = [];
          if (leaving.some((model) => model.kind === "chat")) effects.push(spare ? "对话场景会改用剩下的可用模型" : "之后没有可用的对话模型，需要重新添加");
          if (leaving.some((model) => model.id === catalog.defaultEmbeddingModelId)) effects.push("默认向量模型会被清空，需要重新选择");
          return `将移除正在使用的 ${names}：${effects.join("；")}。继续保存？`;
        }

        // 2. 先验证并新增勾选的模型，全部成功才移除取消勾选的，最后写参数；保存期间锁住编辑。
        async function commit(operations) {
          const patches = parsedDrafts();
          section.inert = true;
          try { return await write(operations, patches); }
          finally { section.inert = false; refresh(); }
        }

        async function write(operations, patches) {
          const saved = savedModels();
          const savedKeys = new Set(saved.map((model) => key(model.kind, model.model)));
          const adding = [...picked].filter((item) => !savedKeys.has(item));
          const removing = saved.filter((model) => !picked.has(key(model.kind, model.model)));
          const failures = [];
          const attempt = async (label, work) => {
            try { await work(); } catch (reason) { failures.push(`${label}：${reason instanceof Error ? reason.message : String(reason)}`); }
            if (dialogClosed()) throw new Error("窗口已关闭；已提交的修改以实际结果为准。");
          };
          for (const [index, item] of adding.entries()) {
            const candidate = candidates.get(item);
            status.textContent = `正在验证 ${candidate.model}（${index + 1}/${adding.length}）…`;
            await attempt(candidate.model, () => operations.addModel(candidate.manual
              ? {kind: "chat", model: candidate.model, capabilities: {}, capability_sources: {}, driver_config: {}}
              : candidateModelInput(candidate)));
          }
          status.textContent = "";
          // 新增有失败时不删旧模型，免得替换不成反而清空可用配置；勾选保持原样可直接重试。
          if (failures.length) throw new Error(`新增未完成，已勾掉的模型先保留；${failures.join("；")}`);
          for (const model of removing) await attempt(model.model, () => operations.removeModel(model.id));
          for (const [modelId, patch] of patches) {
            const name = catalog.models.find((model) => model.id === modelId)?.model ?? modelId;
            await attempt(name, async () => { await operations.updateModel(modelId, patch); drafts.delete(modelId); });
          }
          if (failures.length) throw new Error(`部分完成；${failures.join("；")}`);
          return adding.length + removing.length + patches.length > 0;
        }

        function refresh() {
          if (dialogClosed()) return;
          renderAccount(account, connection);
          renderChips();
          if (!table.hidden && !table.contains(document.activeElement)) renderTable();
        }

        chips.addEventListener("click", (event) => {
          const chip = event.target instanceof Element ? event.target.closest(".settings-chip") : null;
          if (!chip) return;
          if (picked.has(chip.dataset.key)) picked.delete(chip.dataset.key);
          else picked.add(chip.dataset.key);
          chip.setAttribute("aria-pressed", String(picked.has(chip.dataset.key)));
          count.textContent = `${picked.size} / ${chips.querySelectorAll(".settings-chip").length}`;
          reportDirty();
        });
        const pickVisible = (on) => {
          for (const chip of chips.querySelectorAll(".settings-chip:not([hidden])")) {
            if (on) picked.add(chip.dataset.key); else picked.delete(chip.dataset.key);
          }
          renderChips();
        };
        $("[data-all]").addEventListener("click", () => pickVisible(true));
        $("[data-none]").addEventListener("click", () => pickVisible(false));
        filterInput.addEventListener("input", renderChips);
        params.addEventListener("click", () => {
          table.hidden = !table.hidden;
          params.setAttribute("aria-expanded", String(!table.hidden));
          if (!table.hidden) renderTable();
        });
        $("[data-manual-toggle]").addEventListener("click", () => {
          manual.hidden = !manual.hidden;
          if (!manual.hidden) manualInput.focus();
        });
        const addManual = () => {
          const name = manualInput.value.trim();
          if (!name) { manualInput.focus(); return; }
          if (!candidates.has(name) && !savedModels().some((model) => model.model === name)) {
            candidates.set(name, {model: name, kind: "chat", capabilities: {}, manual: true});
          }
          picked.add(name);
          manualInput.value = "";
          manual.hidden = true;
          renderChips();
        };
        $("[data-manual-add]").addEventListener("click", addManual);
        manualInput.addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); addManual(); } });

        // 打开详情即读一次服务目录；只读不占请求锁，读不到时仍可管理已保存的模型。
        const discover = () => {
          const revision = catalog.revision;
          loading = true;
          error.hidden = true;
          status.textContent = "正在读取服务目录…";
          renderChips();
          return readCatalog().then((found) => {
            if (dialogClosed()) return;
            for (const item of found) if (item.kind === null || item.kind === "chat") candidates.set(key("chat", item.model), item);
          }).catch((reason) => {
            // 读取期间设置已被保存改变：按新版本重读一次，不把版本冲突当错误展示。
            if (catalog.revision !== revision && !dialogClosed()) return discover();
            showError(reason);
          }).finally(() => {
            loading = false;
            if (status.textContent === "正在读取服务目录…") status.textContent = "";
            refresh();
          });
        };
        $("[data-reload]").addEventListener("click", () => { void discover(); });
        void discover();
        return {element: section, refresh, isDirty, removalWarning, commit, showError};
      }

      // 账号区：套餐、账号与各窗口的已用比例和重置时间；只有驱动提供额度的连接才显示。
      function renderAccount(slot, connection) {
        const reading = quotas.get(connection.id);
        slot.hidden = !reading;
        if (!reading) return;
        const read = reading.readAt ? `${relativeTime(reading.readAt)}读取` : "尚未读到";
        slot.innerHTML = `<span class="settings-detail-key">账号</span><div class="settings-detail-value settings-account-box">
          <div class="settings-account-top"><strong></strong><span class="settings-account-plan"></span><small></small>
          <button type="button" class="settings-text-button" data-quota-refresh>刷新额度</button></div>
          <div class="settings-account-meters"></div><p class="settings-model-hint" data-quota-note></p></div>`;
        slot.querySelector("strong").textContent = reading.account || connection.name;
        slot.querySelector(".settings-account-plan").textContent = reading.plan;
        slot.querySelector(".settings-account-plan").hidden = !reading.plan;
        slot.querySelector(".settings-account-top small").textContent = reading.balance ? `余额 ${reading.balance} · ${read}` : read;
        const meters = slot.querySelector(".settings-account-meters");
        for (const span of reading.windows) {
          const used = Math.max(0, Math.min(100, Math.round(span.usedPercent)));
          const meter = document.createElement("div");
          meter.className = "settings-account-meter";
          meter.innerHTML = `<span></span><span class="settings-quota-bar"><i></i></span><span></span>`;
          meter.children[0].textContent = span.label;
          meter.children[1].classList.toggle("is-high", used >= 80);
          meter.querySelector("i").style.inlineSize = `${used}%`;
          meter.children[2].textContent = span.resetsAt ? `已用 ${used}% · ${resetTime(span.resetsAt)}重置` : `已用 ${used}%`;
          meters.append(meter);
        }
        slot.querySelector("[data-quota-note]").textContent = [reading.note, reading.error && `最近一次读取失败：${reading.error}`].filter(Boolean).join("；");
        const refresh = slot.querySelector("[data-quota-refresh]");
        // 成功时账号区整体重绘；失败要留在原处说明原因，并让按钮可以再点。
        refresh.addEventListener("click", () => {
          refresh.disabled = true;
          loadQuotas(true).catch((reason) => {
            slot.querySelector("[data-quota-note]").textContent = `刷新失败：${reason instanceof Error ? reason.message : String(reason)}`;
          }).finally(() => { refresh.disabled = false; });
        });
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

      // 编辑器就地展开：session.host 是卡片（编辑）或添加面板（新建）内的挂载区。
      // 与旧弹窗共用同一套操作、草稿汇总与 auth 生命周期；离开守卫直接监听路由与卸载。
      function openProvider(entry, trigger, connection, template, session) {
        if (bindingSave) { showNotice("请先等待模型选择保存或核对完成，再修改连接。"); return; }
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
        let providerDirty = false, modelsDirty = false, busy = false;
        const reportDirty = () => props.dirty?.(providerDirty || modelsDirty);
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
          async removeModel(modelId) {
            if ((!connection && !created) || !catalog.models.some((model) => model.id === modelId && model.connectionId === connectionId)) throw new Error("请选择此连接的现有模型");
            await command({type: "remove_model", expected_revision: catalog.revision, model_id: modelId});
          },
          async updateModel(modelId, patch) {
            if ((!connection && !created) || !catalog.models.some((model) => model.id === modelId && model.connectionId === connectionId)) throw new Error("请选择此连接的现有模型");
            await sendCommand({type: "update_model", expected_revision: catalog.revision, model_id: modelId, ...patch});
            try {
              await load();
            } catch (reason) {
              throw new Error(`参数已提交，但读取最新状态失败。请关闭窗口后核对设置再操作。${reason instanceof Error ? reason.message : String(reason)}`);
            }
          },
          async selectModels() {
            // 详情里已有暂存的勾选：插件的「保存并选择模型」即提交它。
            if (modelManager) return modelManager.commit(operations);
            // 新建流程不在这里挑模型：创建成功后宿主展开新连接的详情，在模型标签里勾选。
            if (!created) throw new Error("请先完成连接");
            return false;
          },
          async addModel(input) {
            if (!connection && !created) throw new Error("请先保存连接");
            const existing = catalog.models.find((model) => model.connectionId === connectionId && model.kind === input.kind && model.model === input.model);
            const modelId = existing?.id ?? `${connectionId}__${randomToken()}`;
            const receipt = existing
              ? await request("/api/dashboard/models/command", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({type:"verify_model", expected_revision:catalog.revision, model_id:modelId})})
              : await command({...input, type: "add_model", expected_revision: catalog.revision, model_id: modelId, connection_id: connectionId});
            if (input.kind === "chat") await setDefaultIfMissing(receipt.revision, modelId);
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
            if (input.model.kind === "chat") await setDefaultIfMissing(receipt.revision, modelId);
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
              input,
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
        const ui = Object.freeze({
          pickModels: (candidates, options = {}) => candidateSheet({
            title: options.title ?? "目录候选",
            hint: options.hint,
            confirmLabel: options.confirmLabel ?? "添加所选",
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
          syncFooter();
          try {
            const result = await action(...args);
            if (auth.closed) throw new Error("窗口已关闭；已提交请求以实际结果为准。");
            return result;
          }
          finally { busy = false; syncFooter(); }
        }])));
        const dialogHost = document.createElement("section");
        dialogHost.className = "settings-dialog settings-dialog--inline";
        session.host.replaceChildren(dialogHost);
        session.host.hidden = false;
        session.head.setAttribute("aria-expanded", "true");
        // 就地编辑器没有原生 cancel 事件；路由离开与页面卸载共享一次草稿/请求判断，
        // 与 guardDialog 的语义保持一致，只是通知走页面 toast。
        const navigate = (event) => {
          if (event.defaultPrevented || auth.closed) return;
          if (busy) {
            event.preventDefault();
            showNotice("请求正在执行，请等待结果后再离开。关闭页面不会撤销已提交的操作。");
            return;
          }
          if ((providerDirty || modelsDirty) && !window.confirm("放弃尚未保存的修改？选择取消可继续填写。")) event.preventDefault();
        };
        const unload = (event) => {
          if (providerDirty || modelsDirty || busy) { event.preventDefault(); event.returnValue = ""; }
        };
        window.addEventListener("akashic:before-navigate", navigate);
        window.addEventListener("beforeunload", unload);
        const stopGuard = () => {
          window.removeEventListener("akashic:before-navigate", navigate);
          window.removeEventListener("beforeunload", unload);
        };
        let disposeEntry, modelManager, footer;
        let disposed = false;
        const syncFooter = () => {
          if (!footer) return;
          footer.querySelector("[data-save]").disabled = busy;
          footer.querySelector("[data-save]").textContent = busy ? "正在保存…" : "保存";
        };
        const editor = {
          kind: session.kind,
          connectionId,
          card: session.card,
          head: session.head,
          busy: () => busy,
          dirty: () => providerDirty || modelsDirty,
          refresh: () => modelManager?.refresh(),
          dispose() {
            if (disposed) return;
            disposed = true;
            if (activeEditor === editor) activeEditor = null;
            closeOverlays();
            stopGuard();
            window.removeEventListener("pagehide", leaveDocument);
            props.dirty?.(false);
            report(auth.close());
            disposeEntry();
            session.head.setAttribute("aria-expanded", "false");
            session.host.hidden = true;
            session.host.replaceChildren();
            if (session.kind !== "create" && catalog) {
              renderConnections(catalog.connections);
            }
            // 与旧弹窗 close 触发回读一致：关闭编辑器即核对权威目录，
            // 覆盖「写入已提交但读取失败」后内存旧值的不确定性。
            if (!closed) report(reads.run(canRefresh, true));
            restoreFocus(trigger);
          },
        };
        activeEditor = editor;
        const finish = (message) => {
          showNotice(message);
          closeActiveEditor(true);
          // 新建成功回到连接列表并展开新连接；取消新建则留在目录里继续挑。
          if (session.kind !== "create") return;
          setAddPanel(false);
          if (!created) return;
          openAfterCreate = connectionId;
          openCreated();
        };
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
            dirty(value) { if (!auth.closed) { providerDirty = value; reportDirty(); } },
            close() { closeActiveEditor(); },
            changed(message) {
              if (auth.closed) return;
              providerDirty = false;
              reportDirty();
              // 插件只保存了连接本身时，宿主接着提交暂存的模型改动再收起。
              if (modelManager?.isDirty()) {
                actions.selectModels().then(() => finish(message)).catch((reason) => modelManager.showError(reason));
                return;
              }
              finish(message);
            },
          });
        } catch (error) {
          activeEditor = null;
          stopGuard();
          session.head.setAttribute("aria-expanded", "false");
          session.host.hidden = true;
          session.host.replaceChildren();
          showError(error);
          return;
        }
        if (connection) {
          const dialogBody = dialogHost.querySelector(".settings-dialog-body") ?? dialogHost;
          modelManager = buildModelManager({
            connection,
            actions,
            readCatalog: () => operations.discoverSaved(),
            dialogClosed: () => auth.closed,
            dirty(value) { if (!auth.closed) { modelsDirty = value; reportDirty(); } },
          });
          dialogBody.appendChild(modelManager.element);
          const accountRow = modelManager.element.querySelector("[data-account]");
          dialogBody.prepend(accountRow);
          modelManager.refresh();
          // 编辑态由宿主统一底栏：插件自己的标题与底栏隐藏，保存时插件表单有改动才交给插件提交。
          dialogHost.classList.add("is-edit");
          footer = document.createElement("footer");
          footer.className = "settings-detail-footer";
          footer.innerHTML = `<button type="button" class="settings-secondary-button" data-cancel>取消</button><button type="button" class="settings-primary-button" data-save>保存</button>`;
          dialogHost.append(footer);
          syncFooter();
          footer.querySelector("[data-cancel]").addEventListener("click", () => { closeActiveEditor(); });
          footer.querySelector("[data-save]").addEventListener("click", () => {
            const warning = modelManager.removalWarning();
            if (warning && !window.confirm(warning)) return;
            const form = dialogHost.querySelector("form");
            if (providerDirty && form) { form.requestSubmit(); return; }
            actions.selectModels()
              .then((saved) => finish(saved ? `${connection.name} 的修改已保存` : "没有需要保存的修改"))
              .catch((reason) => modelManager.showError(reason));
          });
        }
        const leaveDocument = event => { if (!event.persisted) report(auth.close()); };
        window.addEventListener("pagehide", leaveDocument);
      }

      renderTemplates();
      if (!providerTemplates.length) addList.textContent = "没有 Provider 插件提供连接方式。";
      addToggle.addEventListener("click", () => {
        if (activeEditor && !closeActiveEditor()) return;
        setAddPanel(true);
      });
      addClose.addEventListener("click", () => {
        if (activeEditor?.kind === "create" && !closeActiveEditor()) return;
        setAddPanel(false);
      });
      templateSearch.addEventListener("input", () => {
        if (activeEditor?.kind === "create" && !closeActiveEditor()) return;
        renderTemplates();
      });
      searchInput.addEventListener("input", () => {
        query = searchInput.value;
        if (catalog) renderCatalog();
      });
      if (catalogResource.current) {
        catalog = catalogResource.current;
        renderCatalog();
      }
      report(reads.run());

      return () => {
        closed = true;
        visibility.disconnect();
        window.removeEventListener("focus", refreshVisible);
        page.removeEventListener("close", refreshVisible, true);
        reads.close();
        closeActiveEditor(true);
        closeOverlays();
        disposeDialog();
        host.replaceChildren();
      };
    },
  });
}

function requireProviderEntry(entry) {
  if (!entry || typeof entry.id !== "string" || typeof entry.label !== "string"
    || typeof entry.detail !== "string" || typeof entry.render !== "function") {
    throw new Error("models.connection-types.v1 entry 无效");
  }
  return entry;
}

// 接入方式分组；模板或连接类型用 group 声明，未声明的归入厂商 API。
const TEMPLATE_GROUPS = [
  ["subscription", "订阅与本机登录", "登录即可，无需密钥"],
  ["provider", "厂商 API", "填入官方密钥"],
  ["custom", "自定义", "任意 OpenAI 或 Gemini 兼容地址"],
];

function templateGroup(template) {
  const group = template.group ?? template.owner.group;
  return TEMPLATE_GROUPS.some(([id]) => id === group) ? group : "provider";
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

function connectionMark(entry) {
  return providerMark({icon: entry?.connectionIcon});
}

// 有插件声明的品牌图标就用它；没有时用中性的服务器图标，不拿首字母充当标识。
function providerMark(entry) {
  const mark = document.createElement("span");
  mark.className = "settings-connection-mark";
  mark.setAttribute("aria-hidden", "true");
  if (typeof entry?.icon === "string" && entry.icon.startsWith("data:image/svg+xml,")) {
    const image = document.createElement("img");
    image.src = entry.icon;
    image.alt = "";
    mark.appendChild(image);
  } else {
    mark.innerHTML = SERVER_ICON;
  }
  return mark;
}

export function createLatestCatalogRead(read, apply) {
  let active = null;
  let closed = false;
  return {
    async run(canApply = () => true, fresh = false) {
      if (closed) return;
      active?.abort();
      const controller = new AbortController();
      active = controller;
      try {
        const value = await read(controller.signal, fresh);
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
    discovery_owned: true,
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

// 接受裸数字与十进制 K/M 后缀；空串视为未知，非法输入返回 false。
export function parseTokenCount(text) {
  const raw = String(text ?? "").trim();
  if (!raw) return null;
  const match = /^(\d+(?:\.\d+)?)\s*([kKmM])?$/.exec(raw);
  if (!match) return false;
  const value = Number(match[1]) * (match[2] ? (match[2].toLowerCase() === "k" ? 1000 : 1_000_000) : 1);
  if (!Number.isSafeInteger(value) || value <= 0) return false;
  return value;
}

// 整千/整兆缩写成 K/M 便于阅读，其余保留原始数字。
export function formatTokenCount(value) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) return "";
  if (number % 1_000_000 === 0) return `${number / 1_000_000}M`;
  if (number % 1000 === 0) return `${number / 1000}K`;
  return String(number);
}

// 读取时间写成「刚刚 / N 分钟前 / N 小时前」，超过一天给日期。
function relativeTime(iso) {
  const minutes = Math.floor((Date.now() - Date.parse(iso)) / 60_000);
  if (!Number.isFinite(minutes) || minutes < 1) return "刚刚";
  if (minutes < 60) return `${minutes} 分钟前`;
  if (minutes < 24 * 60) return `${Math.floor(minutes / 60)} 小时前`;
  return `${resetTime(iso)} `;
}

// 重置时间：一天内只写钟点，否则带上月日。
function resetTime(iso) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  const time = date.toLocaleTimeString("zh-CN", {hour: "2-digit", minute: "2-digit", hour12: false});
  const sameDay = Math.abs(date.getTime() - Date.now()) < 24 * 3600_000 && date.getDate() === new Date().getDate();
  return sameDay ? `${time} ` : `${date.getMonth() + 1}月${date.getDate()}日 ${time} `;
}

function randomToken() {
  return [...crypto.getRandomValues(new Uint8Array(16))]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

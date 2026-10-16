import { currentTheme, subscribeTheme } from "@akashic/web-ui-v1";

const WIDTH_KEY = "akashic.conversation.tools.width";
const DEFAULT_WIDTH_RATIO = 0.42;
const MIN_PANEL_WIDTH = 360;
const MIN_CHAT_WIDTH = 420;

function syncFrameTheme(frame) {
  // iframe 隐藏后再返回时核对能力，不重建 iframe 或清空编辑器。
  const observer = new IntersectionObserver((entries) => {
    if (entries.some((entry) => entry.isIntersecting)) frame.contentWindow?.postMessage(
      { type: "akashic.models.changed" }, window.location.origin,
    );
  });
  observer.observe(frame);
  const send = () => frame.contentWindow?.postMessage(
    { type: "akashic.theme", themeId: currentTheme().id },
    window.location.origin,
  );
  frame.addEventListener("load", send);
  const unsubscribe = subscribeTheme(send);
  return () => {
    observer.disconnect();
    unsubscribe();
    frame.removeEventListener("load", send);
  };
}

function checkTabs(entries) {
  const tabs = entries.map((entry) => {
    if (
      typeof entry.id !== "string" || typeof entry.label !== "string"
      || (entry.icon !== undefined && (typeof entry.icon !== "string" || !entry.icon.startsWith("<svg")))
      || (entry.background !== undefined && typeof entry.background !== "boolean")
    ) {
      throw new Error(`对话工具标签合同无效: ${String(entry.id ?? "unknown")}`);
    }
    return entry;
  });
  if (new Set(tabs.map((entry) => entry.id)).size !== tabs.length) {
    throw new Error("对话工具标签 id 不能重复");
  }
  return [...tabs].sort((left, right) =>
    (left.order ?? 0) - (right.order ?? 0) || left.id.localeCompare(right.id));
}

// Shell 底栏动作：声明式投影跨 iframe 传给 chat 渲染，onActivate 留在宿主域。
function checkRailActions(value) {
  if (value === undefined || value === null) return [];
  if (!Array.isArray(value)) throw new Error("对话底栏动作必须是数组");
  const actions = value.map((entry) => {
    if (
      !entry || typeof entry !== "object"
      || typeof entry.id !== "string"
      || typeof entry.label !== "string"
      || typeof entry.iconSvg !== "string"
      || !entry.iconSvg.startsWith("<svg")
      || typeof entry.onActivate !== "function"
    ) {
      throw new Error(`对话底栏动作合同无效: ${String(entry?.id ?? "unknown")}`);
    }
    return entry;
  });
  if (new Set(actions.map((entry) => entry.id)).size !== actions.length) {
    throw new Error("对话底栏动作 id 不能重复");
  }
  return [...actions].sort((left, right) =>
    (left.order ?? 0) - (right.order ?? 0) || left.id.localeCompare(right.id));
}

function connectRailActions(frame, actions) {
  const projection = actions.map(({ id, order, label, iconSvg }) => ({ id, order, label, iconSvg }));
  const send = () => frame.contentWindow?.postMessage(
    { type: "akashic.rail-actions", actions: projection }, window.location.origin,
  );
  frame.addEventListener("load", send);
  const onMessage = (event) => {
    if (event.source !== frame.contentWindow || event.origin !== window.location.origin) return;
    const message = event.data;
    if (!message || typeof message !== "object" || message.type !== "akashic.rail-action-activate") return;
    if (typeof message.id !== "string") return;
    actions.find((entry) => entry.id === message.id)?.onActivate();
  };
  window.addEventListener("message", onMessage);
  return () => {
    frame.removeEventListener("load", send);
    window.removeEventListener("message", onMessage);
  };
}

function storedWidth() {
  try {
    const value = Number.parseInt(window.localStorage.getItem(WIDTH_KEY) ?? "", 10);
    return Number.isFinite(value) ? value : null;
  } catch (error) {
    if (!(error instanceof DOMException)) throw error;
    return null;
  }
}

function saveWidth(value) {
  try {
    window.localStorage.setItem(WIDTH_KEY, String(Math.round(value)));
  } catch (error) {
    if (!(error instanceof DOMException)) throw error;
  }
}

function toolIcon() {
  const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  icon.setAttribute("viewBox", "0 0 24 24");
  icon.setAttribute("aria-hidden", "true");
  icon.innerHTML = '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M15 3v18"/>';
  return icon;
}

// 告知 chat 宿主右上角有工具区开关，标题带据此让出开关所占的位置。
function announceTools(frame) {
  const send = () => frame.contentWindow?.postMessage(
    { type: "akashic.chat-tools", present: true }, window.location.origin,
  );
  frame.addEventListener("load", send);
  return () => frame.removeEventListener("load", send);
}

// 通知入口的 ?session= 只消费一次，避免刷新后反复跳回同一会话。
function takeRequestedSession() {
  const url = new URL(window.location.href);
  const sessionId = url.searchParams.get("session");
  if (!sessionId) return "";
  url.searchParams.delete("session");
  window.history.replaceState(window.history.state, "", url);
  return sessionId;
}

function frameSource(sessionId) {
  return sessionId ? `/chat?embedded=1&session=${encodeURIComponent(sessionId)}` : "/chat?embedded=1";
}

// chat iframe 报告当前会话；宿主只转交给标签，不保存、不推导。
function followChatSession(frame, onSession) {
  const onMessage = (event) => {
    if (event.source !== frame.contentWindow || event.origin !== window.location.origin) return;
    const message = event.data;
    if (!message || typeof message !== "object" || message.type !== "akashic.chat-session") return;
    if (message.sessionId !== null && typeof message.sessionId !== "string") return;
    onSession(message.sessionId || null);
  };
  window.addEventListener("message", onMessage);
  return () => window.removeEventListener("message", onMessage);
}

const NARROW_QUERY = "(max-width: 68rem)";
const LAYOUT_KEY = "akashic.conversation.tools.layout.";
const ICONS = {
  close: '<path d="M18 6 6 18M6 6l12 12"/>',
  add: '<path d="M12 5v14M5 12h14"/>',
  expand: '<path d="M15 4h5v5M9 20H4v-5M20 4l-6 6M4 20l6-6"/>',
  collapse: '<path d="M14 10h6M14 10V4M10 14H4M10 14v6M14 10l6-6M10 14l-6 6"/>',
};

function svgIcon(name) {
  return `<svg viewBox="0 0 24 24" aria-hidden="true">${ICONS[name]}</svg>`;
}

function chromeButton(className, label, icon) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `conversation-tools-button ${className}`;
  button.title = label;
  button.setAttribute("aria-label", label);
  button.innerHTML = svgIcon(icon);
  return button;
}

// 每个会话各记一份打开的标签、当前标签与展开状态；只是视图偏好，读写失败时退回当前布局。
function readLayout(session) {
  try {
    const value = JSON.parse(window.sessionStorage.getItem(LAYOUT_KEY + session) ?? "null");
    if (!value || typeof value !== "object" || !Array.isArray(value.opened)) return null;
    return value;
  } catch (error) {
    if (!(error instanceof DOMException) && !(error instanceof SyntaxError)) throw error;
    return null;
  }
}

function saveLayout(session, layout) {
  try {
    window.sessionStorage.setItem(LAYOUT_KEY + session, JSON.stringify(layout));
  } catch (error) {
    if (!(error instanceof DOMException)) throw error;
  }
}

function clearWidth() {
  try {
    window.localStorage.removeItem(WIDTH_KEY);
  } catch (error) {
    if (!(error instanceof DOMException)) throw error;
  }
}

// 工具区外框：标签条、启动器、铺满与关闭；内容区与浮动开关。只建 DOM，不含状态。
function buildChrome() {
  const panel = document.createElement("aside");
  panel.className = "conversation-tools";
  panel.id = "conversation-tools-panel";
  panel.setAttribute("aria-label", "对话工具");
  const splitter = document.createElement("div");
  splitter.className = "conversation-tools-splitter";
  splitter.tabIndex = 0;
  splitter.title = "拖动调整宽度，双击恢复默认";
  splitter.setAttribute("role", "separator");
  splitter.setAttribute("aria-orientation", "vertical");
  splitter.setAttribute("aria-label", "调整工具区宽度");
  const bar = document.createElement("div");
  bar.className = "conversation-tools-bar";
  const tabs = document.createElement("div");
  tabs.className = "conversation-tools-tabs";
  tabs.setAttribute("role", "tablist");
  tabs.setAttribute("aria-label", "工具标签");
  const launcher = chromeButton("conversation-tools-launcher", "打开工具", "add");
  const menu = document.createElement("div");
  menu.className = "conversation-tools-menu";
  menu.id = "conversation-tools-menu";
  menu.setAttribute("role", "menu");
  menu.hidden = true;
  launcher.setAttribute("aria-haspopup", "menu");
  launcher.setAttribute("aria-controls", menu.id);
  launcher.setAttribute("aria-expanded", "false");
  const maximize = chromeButton("conversation-tools-maximize", "铺满", "expand");
  const close = chromeButton("conversation-tools-close", "关闭工具区", "close");
  bar.append(tabs, launcher, maximize, close, menu);
  const content = document.createElement("div");
  content.className = "conversation-tools-content";
  panel.append(bar, content);
  const toggle = document.createElement("button");
  toggle.className = "conversation-tools-toggle";
  toggle.type = "button";
  toggle.setAttribute("aria-controls", panel.id);
  toggle.append(toolIcon(), document.createElement("span"));
  return { panel, splitter, bar, tabs, launcher, menu, maximize, close, content, toggle };
}

// 标签条里的一项：标签本身 + 关闭按钮；关闭按钮不进 Tab 序列，键盘用 Delete 关闭。
function buildTab(entry, index) {
  const item = document.createElement("div");
  item.className = "conversation-tools-tab-item";
  const button = document.createElement("button");
  button.type = "button";
  button.id = `conversation-tool-tab-${index}`;
  button.className = "conversation-tools-tab";
  button.setAttribute("role", "tab");
  button.setAttribute("aria-controls", `conversation-tool-panel-${index}`);
  button.textContent = entry.label;
  const close = chromeButton("conversation-tools-tab-close", `关闭 ${entry.label}`, "close");
  close.tabIndex = -1;
  item.append(button, close);
  const panel = document.createElement("section");
  panel.id = `conversation-tool-panel-${index}`;
  panel.className = "conversation-tool-panel";
  panel.setAttribute("role", "tabpanel");
  panel.setAttribute("aria-labelledby", button.id);
  return { item, button, close, panel };
}

function menuItem(entry) {
  const item = document.createElement("button");
  item.type = "button";
  item.className = "conversation-tools-menu-item";
  item.setAttribute("role", "menuitem");
  item.tabIndex = -1;
  item.dataset.id = entry.id;
  if (entry.icon) item.insertAdjacentHTML("afterbegin", entry.icon);
  const label = document.createElement("span");
  label.textContent = entry.label;
  item.append(label);
  return item;
}

// 启动器菜单：列出全部标签，选中即打开；方向键移动，Esc 与失焦关闭。
function connectLauncher(chrome, entries, onPick) {
  const { launcher, menu } = chrome;
  menu.replaceChildren(...entries.map(menuItem));
  const items = () => [...menu.querySelectorAll(".conversation-tools-menu-item")];
  const setOpen = (open, returnFocus) => {
    menu.hidden = !open;
    launcher.setAttribute("aria-expanded", String(open));
    if (open) items()[0]?.focus();
    else if (returnFocus) launcher.focus();
  };
  launcher.addEventListener("click", () => setOpen(menu.hidden, true));
  menu.addEventListener("click", (event) => {
    const item = event.target.closest(".conversation-tools-menu-item");
    if (!item) return;
    setOpen(false, false);
    onPick(item.dataset.id);
  });
  menu.addEventListener("keydown", (event) => {
    const list = items();
    const index = list.indexOf(document.activeElement);
    if (event.key === "Escape") setOpen(false, true);
    else if (event.key === "ArrowDown" || event.key === "ArrowUp")
      list[(index + (event.key === "ArrowDown" ? 1 : -1) + list.length) % list.length]?.focus();
    else return;
    event.preventDefault();
    event.stopPropagation();
  });
  menu.addEventListener("focusout", (event) => {
    if (!menu.contains(event.relatedTarget) && event.relatedTarget !== launcher) setOpen(false, false);
  });
}

// 分隔条：拖动与方向键调整宽度并记住，双击恢复默认比例。
function connectSplitter(root, splitter, sizing) {
  let dragStartX = 0;
  let dragStartWidth = 0;
  const stop = (event) => {
    if (!root.classList.contains("is-resizing")) return;
    root.classList.remove("is-resizing");
    if (splitter.hasPointerCapture(event.pointerId)) splitter.releasePointerCapture(event.pointerId);
    sizing.set(sizing.width(), true);
  };
  splitter.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    dragStartX = event.clientX;
    dragStartWidth = sizing.width();
    root.classList.add("is-resizing");
    splitter.setPointerCapture(event.pointerId);
  });
  splitter.addEventListener("pointermove", (event) => {
    if (!root.classList.contains("is-resizing")) return;
    const rtl = getComputedStyle(root).direction === "rtl";
    sizing.set(dragStartWidth + (dragStartX - event.clientX) * (rtl ? -1 : 1));
  });
  splitter.addEventListener("pointerup", stop);
  splitter.addEventListener("pointercancel", stop);
  splitter.addEventListener("dblclick", () => sizing.reset());
  splitter.addEventListener("keydown", (event) => {
    const rtl = getComputedStyle(root).direction === "rtl";
    let next = sizing.width();
    if (event.key === "Home") next = MIN_PANEL_WIDTH;
    else if (event.key === "End") next = sizing.maximum();
    else if (event.key === "ArrowLeft") next += rtl ? -24 : 24;
    else if (event.key === "ArrowRight") next += rtl ? 24 : -24;
    else return;
    event.preventDefault();
    sizing.set(next, true);
  });
}

// 宽度：全局一份；没有记录时按页面宽度的默认比例。
function createSizing(root, splitter) {
  const saved = storedWidth();
  const state = { width: saved ?? 0, preferred: saved ?? 0, useDefault: saved === null };
  const maximum = () => Math.max(MIN_PANEL_WIDTH, root.clientWidth - MIN_CHAT_WIDTH);
  const set = (value, persist = false) => {
    state.width = Math.min(maximum(), Math.max(MIN_PANEL_WIDTH, value));
    root.style.setProperty("--conversation-tools-width", `${state.width}px`);
    splitter.setAttribute("aria-valuemin", String(MIN_PANEL_WIDTH));
    splitter.setAttribute("aria-valuemax", String(Math.round(maximum())));
    splitter.setAttribute("aria-valuenow", String(Math.round(state.width)));
    if (!persist) return;
    state.useDefault = false;
    state.preferred = state.width;
    saveWidth(state.width);
  };
  const fit = () => set(state.useDefault ? root.clientWidth * DEFAULT_WIDTH_RATIO : state.preferred);
  const reset = () => {
    state.useDefault = true;
    clearWidth();
    fit();
  };
  return { width: () => state.width, maximum, set, fit, reset };
}

// 标签挂载与父视图的生命周期。每个标签只 render 一次：先占位再 render，render 里调用 view 方法引起的更新不会重入；
// 挂载前先记下当前可见值，订阅时收到的就是唯一一次首次通知。展开与当前标签只经 isActive 读取，不在这里保存。
function createTabLifecycle(tools, panelOf, isActive, currentSession, onAttention) {
  const mounted = new Map();
  const reported = new Map();
  const activeListeners = new Map();
  const sessionListeners = new Map();
  const listenersOf = (map, id) => {
    if (!map.has(id)) map.set(id, new Set());
    return map.get(id);
  };
  const subscribe = (map, id, listener, current, kind) => {
    if (typeof listener !== "function") throw new Error(`ConversationTab ${kind} listener 必须是函数`);
    const listeners = listenersOf(map, id);
    listeners.add(listener);
    listener(current);
    return () => listeners.delete(listener);
  };

  function view(entry) {
    return {
      get active() { return isActive(entry.id); },
      onActiveChange: (listener) => subscribe(activeListeners, entry.id, listener, isActive(entry.id), "active"),
      get session() { return currentSession(); },
      onSessionChange: (listener) => subscribe(sessionListeners, entry.id, listener, currentSession(), "session"),
      requestAttention: (noticeId) => onAttention(entry, noticeId),
    };
  }

  // render 失败时撤下它已登记的监听，面板里说明原因；占位保留，避免每次更新都重试。
  function mount(entry) {
    if (mounted.has(entry.id)) return;
    mounted.set(entry.id, () => {});
    reported.set(entry.id, isActive(entry.id));
    const panel = panelOf(entry.id);
    try {
      mounted.set(entry.id, tools.render(entry.id, panel, view(entry)));
    } catch (error) {
      activeListeners.delete(entry.id);
      sessionListeners.delete(entry.id);
      panel.textContent = `${entry.label}没能加载：${error instanceof Error ? error.message : String(error)}`;
    }
  }

  function report(entry) {
    const active = isActive(entry.id);
    if (reported.get(entry.id) === active) return;
    reported.set(entry.id, active);
    for (const listener of [...listenersOf(activeListeners, entry.id)]) listener(active);
  }

  function session(value) {
    for (const listeners of [...sessionListeners.values()])
      for (const listener of [...listeners]) listener(value);
  }

  function dispose() {
    activeListeners.clear();
    sessionListeners.clear();
    for (const stop of [...mounted.values()].reverse()) stop();
  }

  return { mount, report, session, dispose };
}

// 工具区状态的唯一 owner：打开的标签、当前标签、展开与铺满。标签只经 view 读到可见事实与当前会话。
function mountTools(root, frame, tools, entries) {
  const chrome = buildChrome();
  const sizing = createSizing(root, chrome.splitter);
  const narrow = window.matchMedia(NARROW_QUERY);
  const state = {
    open: false, maximized: false, activeId: entries[0].id, opened: new Set([entries[0].id]),
    // chat 的模态层（窄屏导航抽屉等）在 iframe 内，盖不住宿主的浮动开关；打开期间收起开关。
    chatOverlay: false, session: null,
    // 窄屏恢复布局时只是暂不展开；记下的仍是用户的展开偏好，用户自己开关后清除。
    held: false,
  };
  const handled = new Set();
  const attention = new Set();
  const tabs = new Map();
  const isActive = (id) => state.open && state.activeId === id;
  // 活动只提示；打开、关闭与切换由用户决定，包括移动端。
  const lifecycle = createTabLifecycle(tools, (id) => tabs.get(id).panel,
    isActive, () => state.session, (entry, noticeId) => {
      if (typeof noticeId !== "string" || !noticeId || handled.has(noticeId)) return;
      handled.add(noticeId);
      if (!isActive(entry.id)) attention.add(entry.id);
      update();
    });

  function update() {
    root.classList.toggle("tools-open", state.open);
    root.classList.toggle("tools-maximized", state.open && state.maximized);
    chrome.panel.hidden = !state.open;
    chrome.splitter.hidden = !state.open;
    chrome.toggle.hidden = state.open || state.chatOverlay;
    chrome.toggle.setAttribute("aria-expanded", String(state.open));
    const pending = entries.find((entry) => attention.has(entry.id));
    const target = entries.find((entry) => entry.id === state.activeId) ?? entries[0];
    const label = `打开 ${target.label}${pending ? " · 有新活动" : ""}`;
    chrome.toggle.classList.toggle("has-attention", Boolean(pending));
    chrome.toggle.title = label;
    chrome.toggle.setAttribute("aria-label", label);
    chrome.toggle.querySelector("span").textContent = target.label;
    const maximizeLabel = state.maximized ? "还原" : "铺满";
    chrome.maximize.title = maximizeLabel;
    chrome.maximize.setAttribute("aria-label", maximizeLabel);
    chrome.maximize.setAttribute("aria-pressed", String(state.maximized));
    chrome.maximize.innerHTML = svgIcon(state.maximized ? "collapse" : "expand");
    for (const item of chrome.menu.querySelectorAll(".conversation-tools-menu-item"))
      item.toggleAttribute("data-opened", state.opened.has(item.dataset.id));
    for (const entry of entries) updateTab(entry);
  }

  function updateTab(entry) {
    const active = state.open && entry.id === state.activeId;
    const tab = tabs.get(entry.id);
    tab.item.hidden = !state.opened.has(entry.id);
    tab.button.setAttribute("aria-selected", String(active));
    tab.button.classList.toggle("has-attention", attention.has(entry.id));
    tab.button.tabIndex = entry.id === state.activeId ? 0 : -1;
    tab.panel.hidden = entry.id !== state.activeId;
    lifecycle.report(entry);
    // 1. 首次被选中才渲染；声明 background 的标签挂载即渲染，以便在未打开时提示活动。
    if (active) lifecycle.mount(entry);
  }

  function remember() {
    if (state.session)
      saveLayout(state.session, { open: state.open || state.held, active: state.activeId, opened: [...state.opened] });
  }

  function openTab(id, focusTab = false) {
    state.held = false;
    attention.delete(id);
    state.opened.add(id);
    state.activeId = id;
    state.open = true;
    update();
    remember();
    if (focusTab) tabs.get(id)?.button.focus();
  }

  // 2. 关掉当前标签时转到相邻的已打开标签；最后一个关掉时收起工具区。
  function closeTab(id) {
    const order = entries.filter((entry) => state.opened.has(entry.id)).map((entry) => entry.id);
    state.opened.delete(id);
    if (state.opened.size === 0) {
      state.opened.add(id);
      closePanel();
      return;
    }
    if (state.activeId === id) {
      const index = order.indexOf(id);
      state.activeId = order[index + 1] ?? order[index - 1];
      tabs.get(state.activeId)?.button.focus();
    }
    update();
    remember();
  }

  function closePanel() {
    state.held = false;
    state.open = false;
    state.maximized = false;
    update();
    remember();
    chrome.toggle.focus();
  }

  // 3. 换会话时恢复那个会话记下的布局；没去过的会话继承当前布局并立即记下，之后不再跟随别的会话。
  function applySession(session) {
    if (session === state.session) return;
    state.session = session;
    const layout = session ? readLayout(session) : null;
    if (layout) adoptLayout(layout);
    else remember();
    update();
    lifecycle.session(session);
  }

  // 打开的标签与当前标签作为一个整体校正：已移除的标签丢掉，当前标签不在其中时取第一个；窄屏不自动展开。
  function adoptLayout(layout) {
    const known = entries.filter((entry) => layout.opened.includes(entry.id)).map((entry) => entry.id);
    state.opened = new Set(known.length ? known : [entries[0].id]);
    state.activeId = state.opened.has(layout.active) ? layout.active : [...state.opened][0];
    const open = layout.open === true && known.length > 0;
    state.held = open && narrow.matches;
    state.open = open && !narrow.matches;
    if (!state.open) state.maximized = false;
  }

  // 铺满时 Esc 在工具内容里也能还原；收起只从顶栏或分隔条触发。标签自己处理过的 Esc（defaultPrevented）不动。
  function onKeydown(event) {
    if (event.key !== "Escape" || !state.open || event.defaultPrevented) return;
    const inChrome = chrome.bar.contains(event.target) || event.target === chrome.splitter;
    if (state.maximized && (inChrome || chrome.content.contains(event.target))) {
      state.maximized = false;
      update();
    } else if (inChrome) closePanel();
    else return;
    event.preventDefault();
  }

  function followChatOverlay(event) {
    if (event.source !== frame.contentWindow || event.origin !== window.location.origin) return;
    const message = event.data;
    if (!message || typeof message !== "object" || message.type !== "akashic.chat-overlay") return;
    state.chatOverlay = message.open === true;
    update();
  }
  // iframe 重载时旧的打开状态作废。
  const resetChatOverlay = () => {
    state.chatOverlay = false;
    update();
  };

  for (const [index, entry] of entries.entries()) {
    const tab = buildTab(entry, index);
    tabs.set(entry.id, tab);
    tab.button.addEventListener("click", () => openTab(entry.id));
    tab.close.addEventListener("click", () => closeTab(entry.id));
    tab.button.addEventListener("keydown", (event) => {
      if (event.key === "Delete") {
        event.preventDefault();
        closeTab(entry.id);
        return;
      }
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const opened = entries.filter((item) => state.opened.has(item.id));
      const at = opened.findIndex((item) => item.id === entry.id);
      const next = opened[(at + (event.key === "ArrowRight" ? 1 : -1) + opened.length) % opened.length];
      openTab(next.id, true);
    });
    chrome.tabs.append(tab.item);
    chrome.content.append(tab.panel);
  }
  connectLauncher(chrome, entries, (id) => openTab(id, true));
  connectSplitter(root, chrome.splitter, sizing);
  chrome.close.addEventListener("click", closePanel);
  chrome.maximize.addEventListener("click", () => {
    state.maximized = !state.maximized;
    update();
  });
  chrome.toggle.addEventListener("click", () => {
    const pending = entries.find((entry) => attention.has(entry.id));
    openTab(pending?.id ?? state.activeId, true);
  });
  root.classList.add("has-tools");
  root.append(chrome.splitter, chrome.panel, chrome.toggle);
  root.addEventListener("keydown", onKeydown);
  window.addEventListener("message", followChatOverlay);
  frame.addEventListener("load", resetChatOverlay);
  window.addEventListener("resize", sizing.fit);
  const stopSession = followChatSession(frame, applySession);
  for (const entry of entries) if (entry.background) lifecycle.mount(entry);
  return {
    start() {
      sizing.fit();
      update();
    },
    dispose() {
      stopSession();
      root.removeEventListener("keydown", onKeydown);
      window.removeEventListener("message", followChatOverlay);
      frame.removeEventListener("load", resetChatOverlay);
      window.removeEventListener("resize", sizing.fit);
      lifecycle.dispose();
    },
  };
}

function renderConversation(host, view, props) {
  const tools = view.child("conversation.tools.v2");
  const entries = checkTabs(tools.entries);
  const railActions = checkRailActions(props?.railActions);
  const root = document.createElement("div");
  root.className = "conversation-page";
  const frame = document.createElement("iframe");
  frame.className = "conversation-page-frame";
  frame.title = "Akashic 对话";
  frame.src = frameSource(takeRequestedSession());
  root.appendChild(frame);
  const workbench = entries.length ? mountTools(root, frame, tools, entries) : null;
  host.replaceChildren(root);
  const stopThemeSync = syncFrameTheme(frame);
  const stopRailActions = connectRailActions(frame, railActions);
  const stopToolsNotice = workbench ? announceTools(frame) : () => {};
  workbench?.start();
  return () => {
    workbench?.dispose();
    stopToolsNotice();
    stopRailActions();
    stopThemeSync();
    host.replaceChildren();
  };
}

export function activate(ctx) {
  return ctx.ui.inject("shell.pages.v1", (mount) => mount.register({
    id: "conversation",
    label: "对话",
    iconSvg: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>',
    route: "",
    order: 10,
    children: [{ id: "conversation.tools.v2", cardinality: "list" }],
    render: renderConversation,
  }));
}

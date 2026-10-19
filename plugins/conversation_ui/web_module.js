import { mountChat } from "@akashic/chat-app-v1";

const WIDTH_KEY = "akashic.conversation.tools.width";
const DEFAULT_WIDTH_RATIO = 0.42;
const MIN_PANEL_WIDTH = 360;
const MIN_CHAT_WIDTH = 420;

// 聊天由 Host SDK 挂进同一文档（ADR 0126）。聊天发来的事件经 hooks 转给工作台，没有工作台时取默认值。
function connectChat(chatHost, railActions) {
  const hooks = { session() {}, overlay() {}, intent: async () => "unhandled" };
  const handle = mountChat(chatHost, {
    activateRailAction: (id) => railActions.find((entry) => entry.id === id)?.onActivate(),
    reportSession: (sessionId) => hooks.session(sessionId),
    reportOverlay: (open) => hooks.overlay(open),
    requestIntent: (name, session, payload) => hooks.intent(name, session, payload),
  });
  handle.setRailActions(railActions.map(({ id, order, label, iconSvg }) => ({ id, order, label, iconSvg })));
  return {
    handle,
    hooks,
    // 收起工具区后焦点回到输入框；聊天还没渲染出输入框时交给默认目标。
    focusTarget: () => chatHost.querySelector("textarea") ?? undefined,
  };
}

function checkTabs(entries) {
  const tabs = entries.map((entry) => {
    if (
      typeof entry.id !== "string" || typeof entry.label !== "string"
      || (entry.icon !== undefined && (typeof entry.icon !== "string" || !entry.icon.startsWith("<svg")))
      || (entry.background !== undefined && typeof entry.background !== "boolean")
      || (entry.multiple !== undefined && typeof entry.multiple !== "boolean")
      || (entry.hint !== undefined && (typeof entry.hint !== "string" || entry.hint.length > HINT_LIMIT))
      || (entry.shortcut !== undefined && (typeof entry.shortcut !== "string"
        || !ENTRY_SHORTCUT.test(entry.shortcut) || RESERVED_SHORTCUTS.has(entry.shortcut)))
      || (entry.accepts !== undefined && (!Array.isArray(entry.accepts)
        || entry.accepts.some((name) => typeof name !== "string" || !INTENT_NAME.test(name))))
    ) {
      throw new Error(`对话工具标签合同无效: ${String(entry.id ?? "unknown")}`);
    }
    return entry;
  });
  if (new Set(tabs.map((entry) => entry.id)).size !== tabs.length) {
    throw new Error("对话工具标签 id 不能重复");
  }
  const keys = tabs.flatMap((entry) => entry.shortcut ?? []);
  if (new Set(keys).size !== keys.length) throw new Error("对话工具快捷键不能重复");
  return [...tabs].sort((left, right) =>
    (left.order ?? 0) - (right.order ?? 0) || left.id.localeCompare(right.id));
}

// Shell 底栏动作：声明式投影交给聊天渲染，onActivate 留在宿主这边。
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

// 告知聊天右上角有工具区开关，标题带据此让出开关所占的位置；快捷键列给聊天的快捷键面板。
function toolShortcuts(entries) {
  return [
    { label: "打开或收起工作台", keys: shortcutKeys(HOST_SHORTCUTS.toggle) },
    { label: "工作台上下分屏", keys: shortcutKeys(HOST_SHORTCUTS.split) },
    ...entries.filter((entry) => entry.shortcut).map((entry) => ({ label: `打开${entry.label}`, keys: shortcutKeys(entry.shortcut) })),
  ];
}

const NARROW_QUERY = "(max-width: 68rem)";
const INTENT_NAME = /^[a-z][a-z0-9_-]*(\.[a-z0-9_-]+)*\.v[0-9]+$/;
const INTENT_LIMIT = 16 * 1024;
const INTENT_TIMEOUT_MS = 10_000;
const QUOTE_LIMIT = 8 * 1024;
const SOURCE_LIMIT = 200;
const LAYOUT_KEY = "akashic.conversation.tools.layout.v2.";
const TITLE_LIMIT = 80;
const DETAIL_LIMIT = 24;
const HINT_LIMIT = 40;
const LAUNCH_LABEL = "新标签页";
// 工具可声明的快捷键：Ctrl+` 或 Mod+Shift+字母（Mod 在 macOS 是 ⌘）；浏览器、对话页与宿主自己占用的组合不能用
// （含 macOS 的重做 ⌘⇧Z 与纯文本粘贴 Mod+Shift+V）。
const ENTRY_SHORTCUT = /^(?:Ctrl\+`|Mod\+Shift\+[A-Z])$/;
const RESERVED_SHORTCUTS = new Set([..."ABCIJNOPQRTVWZ"].map((key) => `Mod+Shift+${key}`));
const HOST_SHORTCUTS = { toggle: "Mod+J", split: "Mod+\\" };
const IS_MAC = /Mac|iPhone|iPad/.test(navigator.platform);
const OVERLAY_ROLES = "[role='dialog'], [role='alertdialog'], [role='menu'], [role='listbox'], [role='combobox'][aria-expanded='true']";
const FOCUS_FIRST = ["textarea, [contenteditable='true']", "input, select, button, a[href], [tabindex]:not([tabindex='-1'])"];
const ICONS = {
  close: '<path d="M18 6 6 18M6 6l12 12"/>',
  add: '<path d="M12 5v14M5 12h14"/>',
  expand: '<path d="M15 4h5v5M9 20H4v-5M20 4l-6 6M4 20l6-6"/>',
  collapse: '<path d="M14 10h6M14 10V4M10 14H4M10 14v6M14 10l6-6M10 14l-6 6"/>',
  launch: '<rect x="3" y="4" width="18" height="16" rx="2.5"/><path d="M3 9h18"/>',
  split: '<rect x="3" y="4" width="18" height="16" rx="2.5"/><path d="M3 12h18"/>',
};

// 按键换成快捷键写法（与 entry.shortcut 同一套）；输入法组合中、带 Alt 的不算。
function shortcutOf(event) {
  if (event.altKey || event.isComposing) return null;
  if (event.code === "Backquote" && event.ctrlKey && !event.metaKey && !event.shiftKey) return "Ctrl+`";
  if (!(IS_MAC ? event.metaKey && !event.ctrlKey : event.ctrlKey && !event.metaKey)) return null;
  const key = event.key.length === 1 ? event.key.toUpperCase() : "";
  if (!event.shiftKey && (key === "J" || key === "\\")) return `Mod+${key}`;
  return event.shiftKey && /^[A-Z]$/.test(key) ? `Mod+Shift+${key}` : null;
}

// 按键来自浮层（对话框、菜单、下拉列表、展开的组合框）内部，或页面上开着模态对话框、菜单时，快捷键让给浮层。
// 工作台面板里常驻的列表（如改动的文件列表）不算浮层：面板里的按键由标签自己 preventDefault 认领。
function overlayOwnsKey(event, panel) {
  const target = event.target;
  const overlay = typeof target?.closest === "function" ? target.closest(OVERLAY_ROLES) : null;
  if (overlay && !(overlay.getAttribute("role") === "listbox" && panel.contains(overlay))) return true;
  const page = target?.ownerDocument ?? document;
  return [...page.querySelectorAll("[aria-modal='true'], dialog[open], [role='menu']")]
    .some((node) => node.checkVisibility?.() ?? true);
}

// 给 chat 的快捷键面板用：Mod 换成本平台的写法。
function shortcutKeys(spec) {
  return spec.split("+").map((key) => (key === "Mod" ? (IS_MAC ? "⌘" : "Ctrl") : key));
}

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

function element(tag, className, text = "") {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = text;
  return node;
}

// 多开工具的实例 ID：只在本页与本会话的布局里用，不需要全局唯一。
function newInstanceId() {
  const bytes = window.crypto.getRandomValues(new Uint8Array(6));
  return [...bytes].map((value) => value.toString(16).padStart(2, "0")).join("");
}

// 告知聊天哪些意图有接受者，聊天据此决定是否显示“在侧栏打开”；不代表一定成功。
function intentNames(entries) {
  return [...new Set(entries.flatMap((entry) => entry.accepts ?? []))];
}

// 外层校验：只认当前会话的、大小合规的意图；领域 payload 由接收标签校验。
function readIntent(name, session, payload, current) {
  let size = Infinity;
  // 上限按 UTF-8 字节计（合同写的是 16 KiB）；不可序列化（循环引用、BigInt）按超限处理。
  try { size = new TextEncoder().encode(JSON.stringify(payload ?? null)).byteLength; } catch { /* 见上 */ }
  const valid = typeof name === "string" && INTENT_NAME.test(name)
    && typeof session === "string" && session === current && size <= INTENT_LIMIT;
  return valid ? { id: newInstanceId(), name, session, payload: payload ?? null } : null;
}

// 每个会话各记一份分组（上下最多两组，每组的标签与当前标签）与展开状态；只是视图偏好，读写失败时退回新标签页。
function readLayout(session) {
  try {
    const value = JSON.parse(window.sessionStorage.getItem(LAYOUT_KEY + session) ?? "null");
    if (!value || typeof value !== "object") return null;
    const groups = (Array.isArray(value.groups) ? value.groups : [value]).slice(0, 2).map(readGroup).filter(Boolean);
    if (!groups.length) return null;
    return { open: value.open === true, groups, focus: value.focus === 1 && groups.length > 1 ? 1 : 0 };
  } catch (error) {
    if (!(error instanceof DOMException) && !(error instanceof SyntaxError)) throw error;
    return null;
  }
}

function readGroup(value) {
  if (!value || typeof value !== "object" || !Array.isArray(value.tabs)) return null;
  const tabs = value.tabs.filter((tab) => tab && typeof tab === "object"
    && (tab.tool === null || typeof tab.tool === "string") && (tab.instance === null || typeof tab.instance === "string"));
  return { tabs, active: Number.isInteger(value.active) ? value.active : 0 };
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

// 工具区外框：浏览器式标签条（标签、新标签页、铺满与关闭）、内容区与浮动开关。只建 DOM，不含状态。
function buildChrome() {
  const panel = element("aside", "conversation-tools");
  panel.id = "conversation-tools-panel";
  panel.setAttribute("aria-label", "对话工具");
  const splitter = element("div", "conversation-tools-splitter");
  splitter.tabIndex = 0;
  splitter.title = "拖动调整宽度，双击恢复默认";
  splitter.setAttribute("role", "separator");
  splitter.setAttribute("aria-orientation", "vertical");
  splitter.setAttribute("aria-label", "调整工具区宽度");
  const groups = [buildGroup("工具标签"), buildGroup("下方工具标签")];
  const divider = element("div", "conversation-tools-divider");
  divider.tabIndex = 0;
  divider.title = "拖动调整上下高度，双击恢复一半";
  divider.setAttribute("role", "separator");
  divider.setAttribute("aria-orientation", "horizontal");
  divider.setAttribute("aria-label", "调整上下分屏");
  const end = element("div", "conversation-tools-end");
  const split = chromeButton("conversation-tools-split", "上下分屏", "split");
  const maximize = chromeButton("conversation-tools-maximize", "铺满", "expand");
  const close = chromeButton("conversation-tools-close", "收起工具区", "close");
  end.append(split, maximize, close);
  groups[0].bar.append(end);
  panel.append(groups[0].root, divider, groups[1].root);
  const toggle = element("button", "conversation-tools-toggle");
  toggle.type = "button";
  toggle.setAttribute("aria-controls", panel.id);
  toggle.append(toolIcon(), document.createElement("span"));
  return { panel, splitter, groups, divider, split, maximize, close, toggle };
}

// 一组标签：标签条（标签与“+”）与内容区；标签记录的面板挂在它所在那组的内容区里。
function buildGroup(label) {
  const root = element("div", "conversation-tools-group");
  const bar = element("div", "conversation-tools-bar");
  const tabs = element("div", "conversation-tools-tabs");
  tabs.setAttribute("role", "tablist");
  tabs.setAttribute("aria-label", label);
  const launch = chromeButton("conversation-tools-new", LAUNCH_LABEL, "add");
  bar.append(tabs, launch);
  const content = element("div", "conversation-tools-content");
  root.append(bar, content);
  return { root, bar, tabs, launch, content };
}

// 标签条里的一项：图标、标题、标签自报的简短状态、活动点与关闭按钮；关闭按钮不进 Tab 序列，键盘用 Delete。
function buildTabItem(record) {
  const item = element("div", "conversation-tools-tab-item");
  item.draggable = true;
  const button = element("button", "conversation-tools-tab");
  button.type = "button";
  button.id = `conversation-tool-tab-${record.key}`;
  button.setAttribute("role", "tab");
  button.setAttribute("aria-controls", record.panel.id);
  button.innerHTML = record.entry?.icon ?? svgIcon("launch");
  button.append(element("span", "conversation-tools-tab-label"), element("span", "conversation-tools-tab-dot"));
  const close = chromeButton("conversation-tools-tab-close", "关闭", "close");
  close.tabIndex = -1;
  item.append(button, close);
  record.panel.setAttribute("aria-labelledby", button.id);
  return { item, button, close };
}

function toolPanel(key) {
  const panel = element("section", "conversation-tool-panel");
  panel.id = `conversation-tool-panel-${key}`;
  panel.setAttribute("role", "tabpanel");
  panel.hidden = true;
  return panel;
}

// 面板换到另一组的内容区：支持 moveBefore 的浏览器原样搬过去（iframe 不重载，滚动与焦点保留），否则退回 append。
function movePanel(panel, content) {
  if (typeof content.moveBefore === "function" && content.isConnected && panel.isConnected) {
    try {
      content.moveBefore(panel, null);
      return;
    } catch (error) {
      if (!(error instanceof DOMException)) throw error;
    }
  }
  content.append(panel);
}

// 新标签页：列出全部工具（状态取自已打开的标签）；选中后这一页原地变成该工具。
// 暂不可用的工具（多开工具在会话发出第一条消息前）仍可聚焦读到原因，但标为禁用、点了不响应。
function renderLaunch(panel, tools, onTool) {
  const page = element("div", "conversation-tools-launch");
  const list = element("div", "conversation-tools-launch-tools");
  for (const tool of tools) {
    const item = element("button", "conversation-tools-launch-tool");
    item.type = "button";
    item.innerHTML = tool.entry.icon ?? svgIcon("launch");
    item.append(element("span", "conversation-tools-launch-name", tool.entry.label),
      element("span", "conversation-tools-launch-state", tool.state));
    if (tool.ready) item.addEventListener("click", () => onTool(tool.entry));
    else item.setAttribute("aria-disabled", "true");
    list.append(item);
  }
  page.append(element("h2", "conversation-tools-launch-heading", "工具"), list);
  panel.replaceChildren(page);
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

// 上下分屏的分隔条：拖动与上下方向键调整上方一组的高度比例（20%–80%），双击回到一半；不记住。
function connectDivider(panel, divider) {
  const set = (ratio) => {
    const value = Math.min(80, Math.max(20, ratio));
    panel.style.setProperty("--conversation-tools-upper", `${value}%`);
    divider.setAttribute("aria-valuenow", String(Math.round(value)));
  };
  const current = () => Number(divider.getAttribute("aria-valuenow") ?? 50);
  const move = (event) => {
    const box = panel.getBoundingClientRect();
    set(((event.clientY - box.top) / box.height) * 100);
  };
  const stop = (event) => {
    panel.classList.remove("is-dividing");
    divider.removeEventListener("pointermove", move);
    if (divider.hasPointerCapture(event.pointerId)) divider.releasePointerCapture(event.pointerId);
  };
  divider.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    panel.classList.add("is-dividing");
    divider.setPointerCapture(event.pointerId);
    divider.addEventListener("pointermove", move);
  });
  divider.addEventListener("pointerup", stop);
  divider.addEventListener("pointercancel", stop);
  divider.addEventListener("dblclick", () => set(50));
  divider.addEventListener("keydown", (event) => {
    const step = { ArrowUp: -5, ArrowDown: 5 }[event.key];
    if (!step) return;
    event.preventDefault();
    set(current() + step);
  });
  set(50);
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

// 标签挂载与父视图的生命周期，按标签记录（工具 + 实例）计。每个记录只 render 一次：先占位再 render，
// render 里调用 view 方法引起的更新不会重入；挂载前先记下当前可见值，订阅时收到的就是唯一一次首次通知。
function createTabLifecycle(tools, isActive, currentSession, hooks) {
  const mounted = new Map();
  const reported = new Map();
  const listeners = { active: new Map(), session: new Map(), intent: new Map(), close: new Map() };
  const listenersOf = (kind, key) => {
    if (!listeners[kind].has(key)) listeners[kind].set(key, new Set());
    return listeners[kind].get(key);
  };
  const subscribe = (kind, key, listener, initial) => {
    if (typeof listener !== "function") throw new Error(`ConversationTab ${kind} listener 必须是函数`);
    const set = listenersOf(kind, key);
    set.add(listener);
    if (initial) listener(initial.value);
    return () => set.delete(listener);
  };
  const sessionOf = (record) => (record.instance ? record.session : currentSession());

  function view(record) {
    return {
      get active() { return isActive(record.key); },
      onActiveChange: (listener) => subscribe("active", record.key, listener, { value: isActive(record.key) }),
      get session() { return sessionOf(record); },
      onSessionChange: (listener) => subscribe("session", record.key, listener, { value: sessionOf(record) }),
      requestAttention: (noticeId) => hooks.attention(record, noticeId),
      onIntent: (listener) => subscribe("intent", record.key, listener, null),
      quote: (request) => hooks.quote(request),
      instance: record.instance,
      setTitle: (title) => hooks.title(record, title),
      setIndicator: (indicator) => hooks.indicator(record, indicator),
      onClose: (listener) => subscribe("close", record.key, listener, null),
      get instances() { return hooks.instances(record); },
      openNew: () => hooks.openNew(record),
      cycle: (delta) => hooks.cycle(record, delta),
    };
  }

  function forget(key) {
    for (const kind of Object.keys(listeners)) listeners[kind].delete(key);
  }

  // render 失败时撤下它已登记的监听，面板里说明原因；占位保留，避免每次更新都重试。
  // 标签自身抛错由宿主经 onError 通知；宿主拒绝挂载则直接抛出，两条路径同样处理。
  function mount(record) {
    if (mounted.has(record.key)) return;
    mounted.set(record.key, () => {});
    reported.set(record.key, isActive(record.key));
    const fail = (error) => {
      forget(record.key);
      record.panel.textContent = `${record.entry.label}没能加载：${error instanceof Error ? error.message : String(error)}`;
    };
    try {
      mounted.set(record.key, tools.render(record.entry.id, record.panel, view(record), { onError: fail }));
    } catch (error) {
      fail(error);
    }
  }

  // 卸下一个实例：先告诉标签它被用户关闭（closed 为真时），再撤监听并调用 disposer。
  // 标签的监听或 disposer 抛错只记录，不打断其余监听、卸载与宿主后续的布局更新。
  function unmount(key, closed) {
    if (closed) for (const listener of [...listenersOf("close", key)]) {
      try { listener(); } catch (error) { console.error("[conversation-ui] onClose listener failed", error); }
    }
    const stop = mounted.get(key);
    mounted.delete(key);
    reported.delete(key);
    forget(key);
    try { stop?.(); } catch (error) { console.error("[conversation-ui] tab dispose failed", error); }
  }

  function report(record) {
    if (!mounted.has(record.key)) return;
    const active = isActive(record.key);
    if (reported.get(record.key) === active) return;
    reported.set(record.key, active);
    for (const listener of [...listenersOf("active", record.key)]) listener(active);
  }

  // 只有跟随当前会话的单开标签收到会话变化；实例属于创建它的会话。
  function session(records, value) {
    for (const record of records)
      for (const listener of [...listenersOf("session", record.key)]) listener(value);
  }

  // 意图只交给该标签最先登记的监听，不广播。
  const intentListener = (key) => [...(listeners.intent.get(key) ?? [])][0];

  function dispose() {
    for (const stop of [...mounted.values()].reverse()) stop();
    mounted.clear();
    for (const kind of Object.keys(listeners)) listeners[kind].clear();
  }

  return { mount, unmount, report, session, intentListener, isMounted: (key) => mounted.has(key), dispose };
}

// 标签记录：单开工具一份，跨会话复用；多开工具每个实例一份，属于创建它的会话；新标签页由宿主自己画。
function createRecords(entries, content) {
  const singles = new Map();
  let launchSeq = 0;
  const make = (key, entry, instance, session) => {
    const record = { key, entry, instance, session, title: null, indicator: null, attention: false, panel: toolPanel(key) };
    Object.assign(record, buildTabItem(record));
    content.append(record.panel);
    return record;
  };
  for (const entry of entries) if (!entry.multiple) singles.set(entry.id, make(entry.id, entry, null, null));
  return {
    single: (entry) => singles.get(entry.id),
    singles: () => [...singles.values()],
    instance: (entry, instance, session) => make(`${entry.id}--${instance}`, entry, instance, session),
    launch: () => make(`launch-${++launchSeq}`, null, null, null),
    drop: (record) => record.panel.remove(),
  };
}

function newGroup(tabs, activeKey = undefined) {
  return { tabs, activeKey: activeKey ?? tabs[0]?.key ?? null };
}

function titleOf(record, order) {
  if (!record.entry) return LAUNCH_LABEL;
  if (record.title) return record.title;
  return record.entry.multiple && order > 1 ? `${record.entry.label} ${order}` : record.entry.label;
}

// 标签自报的标题与状态：只收短的纯文本，超长截断；不合规的值直接拒绝，不改现有显示。
function readIndicator(value) {
  if (value === null) return null;
  if (!value || typeof value !== "object") throw new Error("setIndicator 需要对象或 null");
  if (value.busy !== undefined && typeof value.busy !== "boolean") throw new Error("indicator.busy 必须是布尔值");
  if (value.detail !== undefined && typeof value.detail !== "string") throw new Error("indicator.detail 必须是字符串");
  return { busy: value.busy === true, detail: value.detail ? value.detail.slice(0, DETAIL_LIMIT) : "" };
}

// 工具区状态的唯一 owner：本会话的分组（上下最多两组，各自的标签顺序与当前标签）、展开与铺满。
// 标签只经 view 读到可见事实与当前会话。
function mountTools(root, chat, tools, entries) {
  const chrome = buildChrome();
  const sizing = createSizing(root, chrome.splitter);
  const narrow = window.matchMedia(NARROW_QUERY);
  const records = createRecords(entries, chrome.groups[0].content);
  const state = {
    open: false, maximized: false, session: null, groups: [newGroup([records.launch()])], focus: 0,
    // 聊天的模态层（窄屏导航抽屉等）盖不住宿主的浮动开关；打开期间收起开关。
    chatOverlay: false,
    // 窄屏恢复布局时只是暂不展开；记下的仍是用户的展开偏好，用户自己开关后清除。
    held: false,
  };
  const handled = new Set();
  const shortcuts = new Map(entries.filter((entry) => entry.shortcut).map((entry) => [entry.shortcut, entry]));
  const allTabs = () => state.groups.flatMap((group) => group.tabs);
  const groupOf = (record) => state.groups.find((group) => group.tabs.includes(record)) ?? null;
  const focusedGroup = () => state.groups[state.focus] ?? state.groups[0];
  const activeOf = (group) => group.tabs.find((record) => record.key === group.activeKey) ?? null;
  const activeRecord = () => activeOf(focusedGroup());
  const isActive = (key) => state.open && state.groups.some((group) => group.activeKey === key);
  // 活动只提示；打开、关闭与切换由用户决定，包括移动端。
  const lifecycle = createTabLifecycle(tools, isActive, () => state.session, {
    attention(record, noticeId) {
      if (typeof noticeId !== "string" || !noticeId || handled.has(noticeId)) return;
      handled.add(noticeId);
      if (!isActive(record.key)) record.attention = true;
      update();
    },
    quote: (request) => forwardQuote(request),
    title(record, title) {
      if (title !== null && typeof title !== "string") throw new Error("setTitle 需要字符串或 null");
      record.title = title?.trim().slice(0, TITLE_LIMIT) || null;
      update();
    },
    indicator(record, value) {
      record.indicator = readIndicator(value);
      update();
    },
    instances: (record) => allTabs().filter((item) => item.entry === record.entry && item.instance).map((item) => item.instance),
    // 标签自己的“新开”：同一工具的新实例放在它后面，并把焦点移进去。
    openNew(record) {
      if (!record.entry?.multiple) throw new Error("openNew 只用于可多开的工具");
      const opened = openTool(record.entry, null, false, groupOf(record) ?? focusedGroup());
      if (opened) focusInto(opened);
    },
    // 在本组里切到相邻的标签（任何工具），焦点跟过去；只有一个标签时不动。
    cycle(record, delta) {
      const group = groupOf(record);
      if (!group || group.tabs.length < 2 || (delta !== 1 && delta !== -1)) return false;
      const next = group.tabs[(group.tabs.indexOf(record) + delta + group.tabs.length) % group.tabs.length];
      if (activate(next.key)) focusInto(next);
      return true;
    },
  });
  let disposed = false;
  let dragging = null;

  function update() {
    root.classList.toggle("tools-open", state.open);
    root.classList.toggle("tools-maximized", state.open && state.maximized);
    chrome.panel.hidden = !state.open;
    chrome.splitter.hidden = !state.open;
    chrome.toggle.hidden = state.open || state.chatOverlay;
    const maximizeLabel = state.maximized ? "还原" : "铺满";
    chrome.maximize.title = maximizeLabel;
    chrome.maximize.setAttribute("aria-label", maximizeLabel);
    chrome.maximize.setAttribute("aria-pressed", String(state.maximized));
    chrome.maximize.innerHTML = svgIcon(state.maximized ? "collapse" : "expand");
    const split = state.groups.length > 1;
    const splitLabel = `${split ? "取消分屏" : "上下分屏"}（${shortcutKeys(HOST_SHORTCUTS.split).join("+")}）`;
    // 先显示要用的组再搬面板，搬完再隐藏不用的组：面板只在可见的容器之间搬，滚动位置不丢。
    if (split) chrome.groups[1].root.hidden = false;
    for (const record of allTabs()) placePanel(record);
    chrome.panel.classList.toggle("is-split", split);
    chrome.groups[1].root.hidden = !split;
    chrome.divider.hidden = !split;
    chrome.split.title = splitLabel;
    chrome.split.setAttribute("aria-label", splitLabel);
    chrome.split.setAttribute("aria-pressed", String(split));
    paintStrips();
    updateToggle();
    for (const record of new Set([...records.singles(), ...allTabs()])) paintPanel(record);
  }

  // 1. 各组标签条按组内顺序排列，同一工具的多个实例按全局顺序编号；顺序没变时不动 DOM，免得夺走键盘焦点。
  function paintStrips() {
    const counts = new Map();
    chrome.groups.forEach((shell, index) => {
      const group = state.groups[index];
      for (const record of group?.tabs ?? []) {
        const order = record.entry ? (counts.get(record.entry.id) ?? 0) + 1 : 0;
        if (record.entry) counts.set(record.entry.id, order);
        paintTab(record, group, titleOf(record, order));
      }
      const items = group?.tabs.map((record) => record.item) ?? [];
      if (items.length !== shell.tabs.children.length || items.some((item, at) => shell.tabs.children[at] !== item))
        shell.tabs.replaceChildren(...items);
      shell.root.classList.toggle("is-focused", state.groups.length > 1 && index === state.focus);
      if (group) activeOf(group)?.item.scrollIntoView({ block: "nearest", inline: "nearest" });
    });
  }

  function paintTab(record, group, title) {
    const active = record.key === group.activeKey;
    const detail = record.indicator?.detail;
    record.label = title;
    record.item.classList.toggle("is-active", active);
    record.item.classList.toggle("is-busy", Boolean(record.indicator?.busy));
    record.item.classList.toggle("has-attention", record.attention);
    record.button.setAttribute("aria-selected", String(active && state.open));
    record.button.tabIndex = active ? 0 : -1;
    record.button.title = detail ? `${title} · ${detail}` : title;
    record.button.querySelector(".conversation-tools-tab-label").textContent = title;
    record.close.title = `关闭 ${title}`;
    record.close.setAttribute("aria-label", `关闭 ${title}`);
  }

  function placePanel(record) {
    const content = chrome.groups[state.groups.indexOf(groupOf(record))]?.content;
    if (content && record.panel.parentElement !== content) movePanel(record.panel, content);
  }

  // 2. 面板挂在所在那组的内容区（见 placePanel）；首次被选中才渲染，声明 background 的单开标签挂载即渲染，
  //    以便在未打开时提示活动。
  function paintPanel(record) {
    const group = groupOf(record);
    record.panel.hidden = group?.activeKey !== record.key;
    if (!record.entry) return paintLaunch(record);
    lifecycle.report(record);
    if (isActive(record.key)) lifecycle.mount(record);
  }

  // 新标签页只在刚变成当前标签、或工具状态变了时重画，其余更新不动 DOM；
  // 重画后焦点回到原来序号的那一项，键盘用户不会掉到页面开头。
  function paintLaunch(record) {
    if (!isActive(record.key)) {
      record.launchView = null;
      return;
    }
    const tools = entries.map((entry) => ({ entry, state: toolState(entry), ready: !(entry.multiple && !state.session) }));
    const view = JSON.stringify(tools.map((tool) => [tool.state, tool.ready]));
    if (record.launchView === view) return;
    record.launchView = view;
    const focused = [...record.panel.querySelectorAll("button")].indexOf(document.activeElement);
    renderLaunch(record.panel, tools, (entry) => openTool(entry, record, true));
    const buttons = record.panel.querySelectorAll("button");
    if (focused >= 0) buttons[Math.min(focused, buttons.length - 1)]?.focus();
  }

  function toolState(entry) {
    const opened = allTabs().filter((record) => record.entry === entry);
    if (entry.multiple && !state.session) return "发出第一条消息后可用";
    if (entry.multiple && opened.length) return `${opened.length} 个已打开`;
    const detail = opened[0]?.indicator?.detail;
    if (opened.length) return detail ? `已打开 · ${detail}` : "已打开";
    return entry.hint ?? "";
  }

  function updateToggle() {
    const pending = [...new Set([...records.singles(), ...allTabs()])].find((record) => record.attention || record.indicator?.busy);
    const keys = shortcutKeys(HOST_SHORTCUTS.toggle).join("+");
    const label = pending ? `打开工作台（${keys}）· ${pending.label ?? pending.entry.label}有新活动` : `打开工作台（${keys}）`;
    chrome.toggle.classList.toggle("has-attention", Boolean(pending));
    chrome.toggle.title = label;
    chrome.toggle.setAttribute("aria-label", label);
    chrome.toggle.setAttribute("aria-expanded", String(state.open));
    chrome.toggle.querySelector("span").textContent = "工作台";
  }

  function remember() {
    if (!state.session) return;
    saveLayout(state.session, {
      open: state.open || state.held, focus: state.focus,
      groups: state.groups.map((group) => ({
        active: group.tabs.findIndex((record) => record.key === group.activeKey),
        tabs: group.tabs.map((record) => ({ tool: record.entry?.id ?? null, instance: record.instance })),
      })),
    });
  }

  // 新标签放在目标组当前标签之后；replace 时原地替换（新标签页变成工具）。
  function insert(record, replace, target = focusedGroup()) {
    const group = (replace && groupOf(replace)) || target;
    const index = group.tabs.indexOf(replace ?? activeOf(group));
    if (replace && index >= 0) {
      group.tabs.splice(index, 1, record);
      if (!replace.entry) records.drop(replace);
    } else group.tabs.splice(index < 0 ? group.tabs.length : index + 1, 0, record);
    return record;
  }

  // 3. 打开工具：单开工具已在某组里就切过去，多开工具每次一个新实例。replace 只来自用户在新标签页里亲手点选：
  //    这一页原地变成该工具（工具已在别处时这一页让掉）。快捷键、意图等其它入口不删任何标签，只在新开工具时
  //    把目标组当前停着的新标签页原地变成它——组里标签数不变，分屏不会塌。
  function openTool(entry, replace = null, focusTab = false, target = focusedGroup()) {
    if (entry.multiple && !state.session) return null;
    const existing = entry.multiple ? null : allTabs().find((record) => record.entry === entry);
    if (existing && replace) removeTab(replace, false);
    const current = activeOf(target);
    if (!existing && !replace && current && !current.entry) replace = current;
    const record = existing ?? insert(entry.multiple
      ? records.instance(entry, newInstanceId(), state.session) : records.single(entry), replace, target);
    return activate(record.key, focusTab);
  }

  function openLaunch(target) {
    const record = target.tabs.find((item) => !item.entry) ?? insert(records.launch(), null, target);
    activate(record.key, true);
  }

  function activate(key, focusTab = false) {
    const group = state.groups.find((candidate) => candidate.tabs.some((record) => record.key === key));
    if (!group) return null;
    state.held = false;
    group.activeKey = key;
    state.focus = state.groups.indexOf(group);
    state.open = true;
    const record = activeOf(group);
    record.attention = false;
    update();
    remember();
    if (focusTab) record.button.focus();
    return record;
  }

  // 展开工具区：有活动的单开工具优先切过去，否则回到上次的当前标签。
  function openPanel(focusTab) {
    const pending = [...records.singles(), ...allTabs()].find((record) => record.attention);
    if (pending?.entry && !pending.instance) return openTool(pending.entry, null, focusTab);
    return activate((pending ?? activeRecord() ?? allTabs()[0]).key, focusTab);
  }

  // 从所在组拿掉一个标签：当前标签让给相邻的，下方一组空了就并掉；实例随之卸下（closed 时标签先收到 onClose）。
  function removeTab(record, closed) {
    const group = groupOf(record);
    if (!group) return;
    const index = group.tabs.indexOf(record);
    group.tabs.splice(index, 1);
    if (group.activeKey === record.key) group.activeKey = (group.tabs[index] ?? group.tabs[index - 1])?.key ?? null;
    if (!group.tabs.length && state.groups.length > 1) {
      state.groups.splice(state.groups.indexOf(group), 1);
      state.focus = 0;
    }
    if (record.instance) lifecycle.unmount(record.key, closed);
    if (record.instance || !record.entry) records.drop(record);
  }

  // 4. 关掉一个标签：实例随之卸下（标签先收到 onClose），单开工具保留挂载；关到没有标签时收起工具区。
  function closeTab(record) {
    if (!groupOf(record)) return;
    removeTab(record, true);
    if (!allTabs().length) {
      state.groups = [newGroup([records.launch()])];
      return closePanel();
    }
    update();
    remember();
    if (state.open) activeRecord()?.button.focus();
  }

  function closePanel(focusTarget = chrome.toggle) {
    state.held = false;
    state.open = false;
    state.maximized = false;
    update();
    remember();
    focusTarget.focus();
  }

  // 5. 分屏：当前标签移到下方一组（上方只有它一个时，下方开一个新标签页）；已分屏时把下方一组并回上方。
  //    并回时丢掉下方的空白新标签页；焦点在下方且它的当前标签是工具时沿用它，否则保留上方原来的当前标签。
  function toggleSplit() {
    const [upper, lower] = state.groups;
    if (lower) {
      const lowerActive = state.focus === 1 ? activeOf(lower) : null;
      for (const record of lower.tabs) if (!record.entry) records.drop(record);
      upper.tabs.push(...lower.tabs.filter((record) => record.entry));
      if (lowerActive?.entry) upper.activeKey = lowerActive.key;
      state.groups = [upper];
    } else {
      let moving = activeOf(upper);
      if (upper.tabs.length < 2 || !moving) moving = records.launch();
      else {
        const index = upper.tabs.indexOf(moving);
        upper.tabs.splice(index, 1);
        upper.activeKey = (upper.tabs[index] ?? upper.tabs[index - 1]).key;
      }
      state.groups.push(newGroup([moving]));
    }
    activate((state.groups[1] ?? upper).activeKey, true);
  }

  // 拖放：放到 before 之前（没有 before 时放到末尾），成为目标组的当前标签；拖空的那组并掉。
  function moveTab(record, target, before) {
    const from = groupOf(record);
    if (!from || record === before) return;
    const index = from.tabs.indexOf(record);
    from.tabs.splice(index, 1);
    if (from.activeKey === record.key) from.activeKey = (from.tabs[index] ?? from.tabs[index - 1])?.key ?? null;
    const at = before ? target.tabs.indexOf(before) : -1;
    target.tabs.splice(at < 0 ? target.tabs.length : at, 0, record);
    if (!from.tabs.length) state.groups.splice(state.groups.indexOf(from), 1);
    activate(record.key);
  }

  // 6. 换会话：上一个会话的实例与新标签页卸下，单开工具留着并跟随新会话；没去过的会话沿用当前打开的单开工具。
  function applySession(session) {
    if (session === state.session) return;
    for (const record of allTabs()) if (record.instance || !record.entry) removeTab(record, false);
    state.session = session;
    const layout = session ? readLayout(session) : null;
    if (layout) adoptLayout(layout);
    if (!allTabs().length) state.groups = [newGroup([records.launch()])];
    for (const group of state.groups) if (!activeOf(group)) group.activeKey = group.tabs[0]?.key ?? null;
    if (!layout) remember();
    update();
    lifecycle.session(records.singles(), session);
  }

  function adoptLayout(layout) {
    const used = new Set();
    const groups = layout.groups.map((spec) => {
      // 当前标签按存下时的下标取，再滤掉恢复不了的标签，下标不会错位。
      const restored = spec.tabs.map((item) => restoreTab(item, used));
      const tabs = restored.filter(Boolean);
      return newGroup(tabs, restored[spec.active]?.key);
    }).filter((group) => group.tabs.length);
    state.groups = groups.length ? groups : [newGroup([])];
    state.focus = Math.min(layout.focus, state.groups.length - 1);
    const open = layout.open && groups.length > 0;
    state.held = open && narrow.matches;
    state.open = open && !narrow.matches;
    if (!state.open) state.maximized = false;
  }

  function restoreTab(item, used) {
    const entry = entries.find((candidate) => candidate.id === item.tool);
    if (item.tool === null) return records.launch();
    if (entry?.multiple && item.instance) return records.instance(entry, item.instance, state.session);
    if (!entry || entry.multiple || used.has(entry)) return null;
    used.add(entry);
    return records.single(entry);
  }

  // 7. 意图交给第一个声明 accepts 的工具：先展开并激活（多开工具取本会话最近的实例），再投递；结果异步回给聊天。
  async function routeIntent(name, session, payload) {
    const intent = readIntent(name, session, payload, state.session);
    if (!intent) return "invalid";
    const target = entries.find((entry) => entry.accepts?.includes(intent.name));
    const existing = target?.multiple ? allTabs().findLast((record) => record.entry === target) : null;
    const record = existing ? activate(existing.key) : target ? openTool(target) : null;
    const listener = record ? lifecycle.intentListener(record.key) : null;
    if (!listener) return "unhandled";
    let timer = 0;
    const timeout = new Promise((resolve) => { timer = window.setTimeout(() => resolve("failed"), INTENT_TIMEOUT_MS); });
    const status = await Promise.race([Promise.resolve().then(() => listener(intent)), timeout])
      .catch((error) => { console.error("[conversation-ui] intent failed", error); return "failed"; });
    window.clearTimeout(timer);
    return ["handled", "invalid", "failed"].includes(status) ? status : "failed";
  }

  // 8. 引用只进发起时的那个会话：会话已切走、工具区已卸载或超长都拒绝，chat 收到后再核对一次。
  function forwardQuote(request) {
    if (disposed || !request || typeof request !== "object") return "rejected";
    const { session, text, source } = request;
    if (typeof text !== "string" || !text || text.length > QUOTE_LIMIT) return "rejected";
    if (typeof source !== "string" || !source || source.length > SOURCE_LIMIT) return "rejected";
    if (typeof session !== "string" || session !== state.session) return "stale";
    chat.handle.quote({ session, text, source });
    return "inserted";
  }

  // 9. 快捷键：Mod+J 开关工具区，Mod+\ 分屏，工具声明的组合切到（或打开）该工具并把焦点移进去。
  //    聊天与工具区同一文档，在 window 上监听；页面不可见、浮层占着键盘、标签自己处理过的按键（如终端里）都不接。
  //    按住不放的自动重复只吞掉，不反复开合。
  function onShortcut(event) {
    const spec = event.defaultPrevented || root.offsetParent === null ? null : shortcutOf(event);
    const entry = spec ? shortcuts.get(spec) : null;
    if (!entry && spec !== HOST_SHORTCUTS.toggle && spec !== HOST_SHORTCUTS.split) return;
    if (overlayOwnsKey(event, chrome.panel)) return;
    event.preventDefault();
    if (event.repeat) return;
    if (spec === HOST_SHORTCUTS.toggle) {
      if (state.open) closePanel(chat.focusTarget());
      else openPanel(true);
    } else if (spec === HOST_SHORTCUTS.split) toggleSplit();
    else focusTool(entry);
  }

  function focusTool(entry) {
    const existing = allTabs().findLast((record) => record.entry === entry);
    const record = existing ? activate(existing.key) : openTool(entry);
    if (record) focusInto(record);
  }

  // 焦点移进标签内容：先找输入区（终端的 textarea 等），再找第一个可聚焦元素，都没有就停在标签上。
  function focusInto(record) {
    requestAnimationFrame(() => {
      const target = FOCUS_FIRST.map((selector) => record.panel.querySelector(selector)).find(Boolean);
      (target ?? record.button).focus();
    });
  }

  // 铺满时 Esc 在工具内容里也能还原；收起只从顶栏或分隔条触发。标签自己处理过的 Esc（defaultPrevented）不动。
  function onKeydown(event) {
    if (event.key !== "Escape" || !state.open || event.defaultPrevented) return;
    const inChrome = chrome.groups.some((shell) => shell.bar.contains(event.target)) || event.target === chrome.splitter;
    if (state.maximized && (inChrome || chrome.panel.contains(event.target))) {
      state.maximized = false;
      update();
    } else if (inChrome) closePanel();
    else return;
    event.preventDefault();
  }

  function followChatOverlay(open) {
    state.chatOverlay = open === true;
    update();
  }

  // 标签条的指针与键盘：点选、中键关闭、关闭按钮；方向键与 Home/End 在本组里换标签，Delete 关闭。
  function onStripClick(event) {
    const record = allTabs().find((item) => item.item.contains(event.target));
    if (!record) return;
    if (record.close.contains(event.target) || event.button === 1) closeTab(record);
    else if (event.button === 0) activate(record.key);
  }

  function onStripKey(event) {
    const record = allTabs().find((item) => item.button === event.target);
    const group = record ? groupOf(record) : null;
    if (!group) return;
    const at = group.tabs.indexOf(record);
    const keys = { ArrowRight: at + 1, ArrowLeft: at - 1, Home: 0, End: group.tabs.length - 1 };
    if (event.key === "Delete") closeTab(record);
    else if (event.key in keys) activate(group.tabs[(keys[event.key] + group.tabs.length) % group.tabs.length].key, true);
    else return;
    event.preventDefault();
  }

  // 落点在标签的后半（按书写方向）时插到它后面，否则插到它前面；返回插入位置之后的那个标签，null 表示末尾。
  function dropTarget(event, group) {
    const over = allTabs().find((record) => record.item.contains(event.target)) ?? null;
    if (!over) return { over, before: null };
    const box = over.item.getBoundingClientRect();
    const after = (event.clientX > box.left + box.width / 2) !== (getComputedStyle(over.item).direction === "rtl");
    return { over, after, before: after ? group.tabs[group.tabs.indexOf(over) + 1] ?? null : over };
  }

  // 标签拖放：拖到另一个标签的前半插在它前面、后半插在它后面，拖到标签条空白处放到末尾；两组之间同样可以拖。
  // 数据用自定义类型，拖到对话输入框里不会变成文字。
  function connectDrag(shell, index) {
    const mark = (over, after = false) => {
      for (const record of allTabs()) {
        record.item.classList.toggle("is-drop-target", record === over);
        record.item.classList.toggle("is-drop-after", record === over && after === true);
      }
    };
    shell.bar.addEventListener("dragstart", (event) => {
      dragging = allTabs().find((record) => record.item === event.target) ?? null;
      if (!dragging) return;
      event.dataTransfer.effectAllowed = "move";
      event.dataTransfer.setData("application/x-akashic-tab", dragging.key);
    });
    shell.bar.addEventListener("dragover", (event) => {
      if (!dragging) return;
      event.preventDefault();
      const { over, after } = state.groups[index] ? dropTarget(event, state.groups[index]) : { over: null };
      mark(over, after);
    });
    shell.bar.addEventListener("dragleave", (event) => { if (!shell.bar.contains(event.relatedTarget)) mark(null); });
    shell.bar.addEventListener("drop", (event) => {
      const target = state.groups[index];
      if (!dragging || !target) return;
      event.preventDefault();
      mark(null);
      moveTab(dragging, target, dropTarget(event, target).before);
    });
    shell.bar.addEventListener("dragend", () => { dragging = null; mark(null); });
  }

  // 有两组时，点到或聚焦到哪组，“+”、快捷键打开的标签就进哪组。
  function focusGroup(index) {
    if (state.focus === index || !state.groups[index]) return;
    state.focus = index;
    chrome.groups.forEach((shell, at) => shell.root.classList.toggle("is-focused", state.groups.length > 1 && at === index));
  }

  chrome.groups.forEach((shell, index) => {
    shell.tabs.addEventListener("click", onStripClick);
    shell.tabs.addEventListener("auxclick", onStripClick);
    shell.tabs.addEventListener("keydown", onStripKey);
    shell.launch.addEventListener("click", () => openLaunch(state.groups[index] ?? focusedGroup()));
    shell.root.addEventListener("focusin", () => focusGroup(index));
    shell.root.addEventListener("pointerdown", () => focusGroup(index));
    connectDrag(shell, index);
  });
  connectSplitter(root, chrome.splitter, sizing);
  connectDivider(chrome.panel, chrome.divider);
  chrome.split.addEventListener("click", toggleSplit);
  chrome.close.addEventListener("click", () => closePanel());
  chrome.maximize.addEventListener("click", () => {
    state.maximized = !state.maximized;
    update();
  });
  chrome.toggle.addEventListener("click", () => openPanel(true));
  root.classList.add("has-tools");
  root.append(chrome.splitter, chrome.panel, chrome.toggle);
  root.addEventListener("keydown", onKeydown);
  window.addEventListener("keydown", onShortcut);
  window.addEventListener("resize", sizing.fit);
  Object.assign(chat.hooks, { session: applySession, overlay: followChatOverlay, intent: routeIntent });
  chat.handle.setTools(true, toolShortcuts(entries));
  chat.handle.setWorkbenchIntents(intentNames(entries));
  for (const record of records.singles()) if (record.entry.background) lifecycle.mount(record);
  return {
    start() {
      sizing.fit();
      update();
    },
    dispose() {
      disposed = true;
      root.removeEventListener("keydown", onKeydown);
      window.removeEventListener("keydown", onShortcut);
      window.removeEventListener("resize", sizing.fit);
      lifecycle.dispose();
    },
  };
}

function renderConversation(host, view, props) {
  const tools = view.child("conversation.tools.v2");
  const entries = checkTabs(tools.entries);
  const railActions = checkRailActions(props?.railActions);
  const root = element("div", "conversation-page");
  const chatHost = element("div", "conversation-page-chat");
  root.appendChild(chatHost);
  host.replaceChildren(root);
  const chat = connectChat(chatHost, railActions);
  const workbench = entries.length ? mountTools(root, chat, tools, entries) : null;
  // 页面隐藏后再返回时告诉聊天"重新显示了"，由聊天通知各插件重读；不重建聊天、不清空编辑器。
  const visibility = new IntersectionObserver((records) => {
    if (records.some((record) => record.isIntersecting)) chat.handle.shown();
  });
  visibility.observe(chatHost);
  workbench?.start();
  return () => {
    visibility.disconnect();
    workbench?.dispose();
    chat.handle.dispose();
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

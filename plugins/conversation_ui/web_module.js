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
      || (entry.multiple !== undefined && typeof entry.multiple !== "boolean")
      || (entry.hint !== undefined && (typeof entry.hint !== "string" || entry.hint.length > HINT_LIMIT))
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
const INTENT_NAME = /^[a-z][a-z0-9_-]*(\.[a-z0-9_-]+)*\.v[0-9]+$/;
const INTENT_LIMIT = 16 * 1024;
const INTENT_TIMEOUT_MS = 10_000;
const QUOTE_LIMIT = 8 * 1024;
const SOURCE_LIMIT = 200;
const LAYOUT_KEY = "akashic.conversation.tools.layout.v2.";
const TITLE_LIMIT = 80;
const DETAIL_LIMIT = 24;
const HINT_LIMIT = 40;
const CLOSED_LIMIT = 6;
const LAUNCH_LABEL = "新标签页";
const ICONS = {
  close: '<path d="M18 6 6 18M6 6l12 12"/>',
  add: '<path d="M12 5v14M5 12h14"/>',
  expand: '<path d="M15 4h5v5M9 20H4v-5M20 4l-6 6M4 20l6-6"/>',
  collapse: '<path d="M14 10h6M14 10V4M10 14H4M10 14v6M14 10l6-6M10 14l-6 6"/>',
  launch: '<rect x="3" y="4" width="18" height="16" rx="2.5"/><path d="M3 9h18"/>',
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

// 告知 chat 哪些意图有接受者，chat 据此决定是否显示“在侧栏打开”；不代表一定成功。
function announceIntents(frame, entries) {
  const names = [...new Set(entries.flatMap((entry) => entry.accepts ?? []))];
  const send = () => frame.contentWindow?.postMessage(
    { type: "akashic.workbench-intents", names }, window.location.origin,
  );
  frame.addEventListener("load", send);
  send();
  return () => frame.removeEventListener("load", send);
}

// 外层校验：只认自己 chat iframe 发来的、当前会话的、大小合规的意图；领域 payload 由接收标签校验。
function readIntent(event, frame, session) {
  if (event.source !== frame.contentWindow || event.origin !== window.location.origin) return null;
  const message = event.data;
  if (!message || typeof message !== "object" || message.type !== "akashic.workbench-intent") return null;
  if (typeof message.id !== "string" || !message.id || message.id.length > 64) return null;
  let size = Infinity;
  // 上限按 UTF-8 字节计（合同写的是 16 KiB）；不可序列化（循环引用、BigInt）按超限处理。
  try { size = new TextEncoder().encode(JSON.stringify(message.payload ?? null)).byteLength; } catch { /* 见上 */ }
  const valid = typeof message.name === "string" && INTENT_NAME.test(message.name)
    && typeof message.session === "string" && message.session === session && size <= INTENT_LIMIT;
  return { id: message.id, valid, intent: { id: message.id, name: message.name, session: message.session, payload: message.payload ?? null } };
}

function replyIntent(frame, id, status) {
  frame.contentWindow?.postMessage({ type: "akashic.workbench-intent-result", id, status }, window.location.origin);
}

// 每个会话各记一份标签（工具与实例）、当前标签与展开状态；只是视图偏好，读写失败时退回新标签页。
function readLayout(session) {
  try {
    const value = JSON.parse(window.sessionStorage.getItem(LAYOUT_KEY + session) ?? "null");
    if (!value || typeof value !== "object" || !Array.isArray(value.tabs)) return null;
    const tabs = value.tabs.filter((tab) => tab && typeof tab === "object"
      && (tab.tool === null || typeof tab.tool === "string") && (tab.instance === null || typeof tab.instance === "string"));
    return { open: value.open === true, tabs, active: Number.isInteger(value.active) ? value.active : 0 };
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
  const bar = element("div", "conversation-tools-bar");
  const tabs = element("div", "conversation-tools-tabs");
  tabs.setAttribute("role", "tablist");
  tabs.setAttribute("aria-label", "工具标签");
  const launch = chromeButton("conversation-tools-new", LAUNCH_LABEL, "add");
  const end = element("div", "conversation-tools-end");
  const maximize = chromeButton("conversation-tools-maximize", "铺满", "expand");
  const close = chromeButton("conversation-tools-close", "收起工具区", "close");
  end.append(maximize, close);
  bar.append(tabs, launch, end);
  const content = element("div", "conversation-tools-content");
  panel.append(bar, content);
  const toggle = element("button", "conversation-tools-toggle");
  toggle.type = "button";
  toggle.setAttribute("aria-controls", panel.id);
  toggle.append(toolIcon(), document.createElement("span"));
  return { panel, splitter, bar, tabs, launch, maximize, close, content, toggle };
}

// 标签条里的一项：图标、标题、标签自报的简短状态、活动点与关闭按钮；关闭按钮不进 Tab 序列，键盘用 Delete。
function buildTabItem(record) {
  const item = element("div", "conversation-tools-tab-item");
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

// 新标签页：列出全部工具（状态取自已打开的标签）与本会话最近关闭的标签；选中后这一页原地变成该工具。
function renderLaunch(panel, tools, closed, onTool, onClosed) {
  const page = element("div", "conversation-tools-launch");
  const list = element("div", "conversation-tools-launch-tools");
  for (const tool of tools) {
    const item = element("button", "conversation-tools-launch-tool");
    item.type = "button";
    item.innerHTML = tool.entry.icon ?? svgIcon("launch");
    item.append(element("span", "conversation-tools-launch-name", tool.entry.label),
      element("span", "conversation-tools-launch-state", tool.state));
    item.addEventListener("click", () => onTool(tool.entry));
    list.append(item);
  }
  page.append(element("h2", "conversation-tools-launch-heading", "工具"), list);
  if (closed.length) {
    const rows = element("div", "conversation-tools-launch-rows");
    for (const item of closed) {
      const row = element("button", "conversation-tools-launch-row");
      row.type = "button";
      row.innerHTML = item.entry.icon ?? svgIcon("launch");
      row.append(element("span", "conversation-tools-launch-name", item.title), element("span", "conversation-tools-launch-state", item.at));
      row.addEventListener("click", () => onClosed(item));
      rows.append(row);
    }
    page.append(element("h2", "conversation-tools-launch-heading", "最近关闭"), rows);
  }
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
  function unmount(key, closed) {
    if (closed) for (const listener of [...listenersOf("close", key)]) listener();
    const stop = mounted.get(key);
    mounted.delete(key);
    reported.delete(key);
    forget(key);
    stop?.();
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

// 工具区状态的唯一 owner：本会话的标签（顺序、当前标签）、展开与铺满。标签只经 view 读到可见事实与当前会话。
function mountTools(root, frame, tools, entries) {
  const chrome = buildChrome();
  const sizing = createSizing(root, chrome.splitter);
  const narrow = window.matchMedia(NARROW_QUERY);
  const records = createRecords(entries, chrome.content);
  const state = {
    open: false, maximized: false, session: null, tabs: [records.launch()], activeKey: null,
    // chat 的模态层（窄屏导航抽屉等）在 iframe 内，盖不住宿主的浮动开关；打开期间收起开关。
    chatOverlay: false,
    // 窄屏恢复布局时只是暂不展开；记下的仍是用户的展开偏好，用户自己开关后清除。
    held: false,
  };
  state.activeKey = state.tabs[0].key;
  const closedBySession = new Map();
  const handled = new Set();
  const isActive = (key) => state.open && state.activeKey === key;
  const activeRecord = () => state.tabs.find((record) => record.key === state.activeKey) ?? null;
  const closedList = () => closedBySession.get(state.session ?? "") ?? [];
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
  });
  let disposed = false;

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
    paintStrip();
    updateToggle();
    for (const record of new Set([...records.singles(), ...state.tabs])) paintPanel(record);
  }

  // 1. 标签条按 state.tabs 排列；顺序没变时不动 DOM，免得夺走键盘焦点。
  function paintStrip() {
    const counts = new Map();
    for (const record of state.tabs) {
      const order = record.entry ? (counts.get(record.entry.id) ?? 0) + 1 : 0;
      if (record.entry) counts.set(record.entry.id, order);
      paintTab(record, titleOf(record, order));
    }
    const items = state.tabs.map((record) => record.item);
    if (items.some((item, index) => chrome.tabs.children[index] !== item) || chrome.tabs.children.length !== items.length)
      chrome.tabs.replaceChildren(...items);
    activeRecord()?.item.scrollIntoView({ block: "nearest", inline: "nearest" });
  }

  function paintTab(record, title) {
    const active = record.key === state.activeKey;
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

  // 2. 首次被选中才渲染；声明 background 的单开标签挂载即渲染，以便在未打开时提示活动。新标签页每次显示时重画。
  function paintPanel(record) {
    record.panel.hidden = record.key !== state.activeKey;
    if (!record.entry) {
      if (isActive(record.key)) renderLaunch(record.panel, entries.map((entry) => ({ entry, state: toolState(entry) })),
        closedList(), (entry) => openTool(entry, record, true), (item) => reopen(item, record));
      return;
    }
    lifecycle.report(record);
    if (isActive(record.key)) lifecycle.mount(record);
  }

  function toolState(entry) {
    const opened = state.tabs.filter((record) => record.entry === entry);
    if (entry.multiple && !state.session) return "发出第一条消息后可用";
    if (entry.multiple && opened.length) return `${opened.length} 个已打开`;
    const detail = opened[0]?.indicator?.detail;
    if (opened.length) return detail ? `已打开 · ${detail}` : "已打开";
    return entry.hint ?? "";
  }

  function updateToggle() {
    const pending = [...new Set([...records.singles(), ...state.tabs])].find((record) => record.attention || record.indicator?.busy);
    const label = pending ? `打开工作台 · ${pending.label ?? pending.entry.label}有新活动` : "打开工作台";
    chrome.toggle.classList.toggle("has-attention", Boolean(pending));
    chrome.toggle.title = label;
    chrome.toggle.setAttribute("aria-label", label);
    chrome.toggle.setAttribute("aria-expanded", String(state.open));
    chrome.toggle.querySelector("span").textContent = "工作台";
  }

  function remember() {
    if (!state.session) return;
    saveLayout(state.session, {
      open: state.open || state.held, active: state.tabs.findIndex((record) => record.key === state.activeKey),
      tabs: state.tabs.map((record) => ({ tool: record.entry?.id ?? null, instance: record.instance })),
    });
  }

  function insert(record, replace) {
    const index = state.tabs.indexOf(replace ?? activeRecord());
    if (replace && index >= 0) {
      state.tabs.splice(index, 1, record);
      if (!replace.entry) records.drop(replace);
    } else state.tabs.splice(index < 0 ? state.tabs.length : index + 1, 0, record);
    return record;
  }

  // 3. 打开工具：单开工具已在标签条上就切过去；从新标签页打开时，这一页原地变成该工具。多开工具每次一个新实例。
  function openTool(entry, replace = null, focusTab = false) {
    if (entry.multiple && !state.session) return null;
    const existing = entry.multiple ? null : state.tabs.find((record) => record.entry === entry);
    if (existing && replace) removeTab(replace, false);
    const record = existing ?? insert(entry.multiple
      ? records.instance(entry, newInstanceId(), state.session) : records.single(entry), replace);
    activate(record.key, focusTab);
    return record;
  }

  function openLaunch() {
    const record = state.tabs.find((item) => !item.entry) ?? insert(records.launch(), null);
    activate(record.key, true);
  }

  function reopen(item, launch) {
    const list = closedList();
    list.splice(list.indexOf(item), 1);
    openTool(item.entry, launch, true);
  }

  function activate(key, focusTab = false) {
    state.held = false;
    state.activeKey = key;
    state.open = true;
    const record = activeRecord();
    if (record) record.attention = false;
    update();
    remember();
    if (focusTab) record?.button.focus();
  }

  function removeTab(record, closed) {
    state.tabs.splice(state.tabs.indexOf(record), 1);
    if (record.instance) lifecycle.unmount(record.key, closed);
    if (record.instance || !record.entry) records.drop(record);
  }

  // 4. 关掉一个标签：记进最近关闭；实例随之卸下（标签先收到 onClose），单开工具保留挂载；关到没有标签时收起工具区。
  function closeTab(record) {
    const index = state.tabs.indexOf(record);
    if (index < 0) return;
    if (record.entry) {
      const list = closedList();
      list.unshift({ entry: record.entry, title: record.label, at: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) });
      closedBySession.set(state.session ?? "", list.slice(0, CLOSED_LIMIT));
    }
    removeTab(record, true);
    if (!state.tabs.length) {
      state.activeKey = insert(records.launch(), null).key;
      return closePanel();
    }
    if (state.activeKey === record.key) state.activeKey = (state.tabs[index] ?? state.tabs[index - 1]).key;
    update();
    remember();
    if (state.open) activeRecord()?.button.focus();
  }

  function closePanel() {
    state.held = false;
    state.open = false;
    state.maximized = false;
    update();
    remember();
    chrome.toggle.focus();
  }

  // 5. 换会话：上一个会话的实例与新标签页卸下，单开工具留着并跟随新会话；没去过的会话沿用当前打开的单开工具。
  function applySession(session) {
    if (session === state.session) return;
    for (const record of [...state.tabs]) if (record.instance || !record.entry) removeTab(record, false);
    state.session = session;
    const layout = session ? readLayout(session) : null;
    if (layout) adoptLayout(layout);
    if (!state.tabs.length) state.tabs.push(records.launch());
    if (!activeRecord()) state.activeKey = state.tabs[0].key;
    if (!layout) remember();
    update();
    lifecycle.session(records.singles(), session);
  }

  function adoptLayout(layout) {
    const tabs = [];
    for (const item of layout.tabs) {
      const entry = entries.find((candidate) => candidate.id === item.tool);
      if (item.tool === null) tabs.push(records.launch());
      else if (entry?.multiple && item.instance) tabs.push(records.instance(entry, item.instance, state.session));
      else if (entry && !entry.multiple && !tabs.includes(records.single(entry))) tabs.push(records.single(entry));
    }
    state.tabs = tabs;
    state.activeKey = (tabs[layout.active] ?? tabs[0])?.key ?? null;
    const open = layout.open && tabs.length > 0;
    state.held = open && narrow.matches;
    state.open = open && !narrow.matches;
    if (!state.open) state.maximized = false;
  }

  // 6. 意图交给第一个声明 accepts 的工具：先展开并激活（多开工具取本会话最近的实例），再投递；结果异步回执给 chat。
  async function routeIntent(event) {
    const request = readIntent(event, frame, state.session);
    if (!request) return;
    if (!request.valid) return replyIntent(frame, request.id, "invalid");
    const target = entries.find((entry) => entry.accepts?.includes(request.intent.name));
    const existing = target?.multiple ? state.tabs.findLast((record) => record.entry === target) : null;
    const record = existing ?? (target ? openTool(target) : null);
    if (!record) return replyIntent(frame, request.id, "unhandled");
    if (existing) activate(existing.key);
    const listener = lifecycle.intentListener(record.key);
    if (!listener) return replyIntent(frame, request.id, "unhandled");
    let timer = 0;
    const timeout = new Promise((resolve) => { timer = window.setTimeout(() => resolve("failed"), INTENT_TIMEOUT_MS); });
    const status = await Promise.race([Promise.resolve().then(() => listener(request.intent)), timeout])
      .catch((error) => { console.error("[conversation-ui] intent failed", error); return "failed"; });
    window.clearTimeout(timer);
    replyIntent(frame, request.id, ["handled", "invalid", "failed"].includes(status) ? status : "failed");
  }

  // 7. 引用只进发起时的那个会话：会话已切走、工具区已卸载或超长都拒绝，chat 收到后再核对一次。
  function forwardQuote(request) {
    if (disposed || !request || typeof request !== "object") return "rejected";
    const { session, text, source } = request;
    if (typeof text !== "string" || !text || text.length > QUOTE_LIMIT) return "rejected";
    if (typeof source !== "string" || !source || source.length > SOURCE_LIMIT) return "rejected";
    if (typeof session !== "string" || session !== state.session) return "stale";
    frame.contentWindow?.postMessage({ type: "akashic.composer-quote", session, text, source }, window.location.origin);
    return "inserted";
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

  // 标签条的指针与键盘：点选、中键关闭、关闭按钮；方向键与 Home/End 换标签，Delete 关闭。
  function onStripClick(event) {
    const record = state.tabs.find((item) => item.item.contains(event.target));
    if (!record) return;
    if (record.close.contains(event.target) || event.button === 1) closeTab(record);
    else if (event.button === 0) activate(record.key);
  }

  function onStripKey(event) {
    const at = state.tabs.findIndex((record) => record.button === event.target);
    if (at < 0) return;
    const keys = { ArrowRight: at + 1, ArrowLeft: at - 1, Home: 0, End: state.tabs.length - 1 };
    if (event.key === "Delete") closeTab(state.tabs[at]);
    else if (event.key in keys) activate(state.tabs[(keys[event.key] + state.tabs.length) % state.tabs.length].key, true);
    else return;
    event.preventDefault();
  }

  chrome.tabs.addEventListener("click", onStripClick);
  chrome.tabs.addEventListener("auxclick", onStripClick);
  chrome.tabs.addEventListener("keydown", onStripKey);
  connectSplitter(root, chrome.splitter, sizing);
  chrome.launch.addEventListener("click", openLaunch);
  chrome.close.addEventListener("click", closePanel);
  chrome.maximize.addEventListener("click", () => {
    state.maximized = !state.maximized;
    update();
  });
  chrome.toggle.addEventListener("click", () => {
    const pending = [...records.singles(), ...state.tabs].find((record) => record.attention);
    if (pending?.entry && !pending.instance) openTool(pending.entry, null, true);
    else activate((pending ?? activeRecord() ?? state.tabs[0]).key, true);
  });
  root.classList.add("has-tools");
  root.append(chrome.splitter, chrome.panel, chrome.toggle);
  root.addEventListener("keydown", onKeydown);
  window.addEventListener("message", followChatOverlay);
  frame.addEventListener("load", resetChatOverlay);
  window.addEventListener("resize", sizing.fit);
  const stopSession = followChatSession(frame, applySession);
  const stopIntents = announceIntents(frame, entries);
  const onIntent = (event) => void routeIntent(event);
  window.addEventListener("message", onIntent);
  for (const record of records.singles()) if (record.entry.background) lifecycle.mount(record);
  return {
    start() {
      sizing.fit();
      update();
    },
    dispose() {
      disposed = true;
      stopSession();
      stopIntents();
      window.removeEventListener("message", onIntent);
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

import { currentTheme, subscribeTheme } from "@akashic/web-ui-v1";

const WIDTH_KEY = "akashic.conversation.tools.width";
const DEFAULT_WIDTH_RATIO = 0.42;
const MIN_PANEL_WIDTH = 360;
const MIN_CHAT_WIDTH = 420;

function syncFrameTheme(frame) {
  const send = () => frame.contentWindow?.postMessage(
    { type: "akashic.theme", themeId: currentTheme().id },
    window.location.origin,
  );
  frame.addEventListener("load", send);
  const unsubscribe = subscribeTheme(send);
  return () => {
    unsubscribe();
    frame.removeEventListener("load", send);
  };
}

function checkTabs(entries) {
  const tabs = entries.map((entry) => {
    if (typeof entry.id !== "string" || typeof entry.label !== "string") {
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

function renderConversation(host, view) {
  const tools = view.child("conversation.tools.v1");
  const entries = checkTabs(tools.entries);
  const root = document.createElement("div");
  root.className = "conversation-page";
  const frame = document.createElement("iframe");
  frame.className = "conversation-page-frame";
  frame.title = "Akashic 对话";
  frame.src = "/chat?embedded=1";
  root.appendChild(frame);

  if (entries.length === 0) {
    host.replaceChildren(root);
    const stopThemeSync = syncFrameTheme(frame);
    return () => {
      stopThemeSync();
      host.replaceChildren();
    };
  }

  root.classList.add("has-tools");
  const panel = document.createElement("aside");
  panel.className = "conversation-tools";
  panel.id = "conversation-tools-panel";
  panel.setAttribute("aria-label", "对话工具");
  const splitter = document.createElement("div");
  splitter.className = "conversation-tools-splitter";
  splitter.tabIndex = 0;
  splitter.setAttribute("role", "separator");
  splitter.setAttribute("aria-orientation", "vertical");
  splitter.setAttribute("aria-label", "调整工具区宽度");
  const bar = document.createElement("div");
  bar.className = "conversation-tools-bar";
  bar.setAttribute("role", "tablist");
  bar.setAttribute("aria-label", "工具标签");
  const close = document.createElement("button");
  close.className = "conversation-tools-close";
  close.type = "button";
  close.title = "关闭工具区";
  close.setAttribute("aria-label", "关闭工具区");
  close.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18 6 6 18M6 6l12 12"/></svg>';
  const content = document.createElement("div");
  content.className = "conversation-tools-content";
  const toggle = document.createElement("button");
  toggle.className = "conversation-tools-toggle";
  toggle.type = "button";
  toggle.setAttribute("aria-controls", panel.id);
  toggle.append(toolIcon(), document.createElement("span"));

  const savedWidth = storedWidth();
  const state = {
    open: false,
    activeId: entries[0].id,
    width: savedWidth ?? 0,
    preferredWidth: savedWidth ?? 0,
    useDefaultWidth: savedWidth === null,
  };
  const handled = new Set();
  const buttons = new Map();
  const panels = new Map();
  const activeValues = new Map();
  const activeListeners = new Map(entries.map((entry) => [entry.id, new Set()]));
  const disposers = [];

  function panelMaximum() {
    return Math.max(MIN_PANEL_WIDTH, root.clientWidth - MIN_CHAT_WIDTH);
  }

  function defaultWidth() {
    return root.clientWidth * DEFAULT_WIDTH_RATIO;
  }

  function setWidth(value, persist = false) {
    state.width = Math.min(panelMaximum(), Math.max(MIN_PANEL_WIDTH, value));
    root.style.setProperty("--conversation-tools-width", `${state.width}px`);
    splitter.setAttribute("aria-valuemin", String(MIN_PANEL_WIDTH));
    splitter.setAttribute("aria-valuemax", String(Math.round(panelMaximum())));
    splitter.setAttribute("aria-valuenow", String(Math.round(state.width)));
    if (persist) {
      state.useDefaultWidth = false;
      state.preferredWidth = state.width;
      saveWidth(state.width);
    }
  }

  function update() {
    root.classList.toggle("tools-open", state.open);
    panel.hidden = !state.open;
    splitter.hidden = !state.open;
    toggle.hidden = state.open;
    toggle.setAttribute("aria-expanded", String(state.open));
    const activeEntry = entries.find((entry) => entry.id === state.activeId) ?? entries[0];
    const toggleLabel = `打开 ${activeEntry.label}`;
    toggle.title = toggleLabel;
    toggle.setAttribute("aria-label", toggleLabel);
    toggle.querySelector("span").textContent = activeEntry.label;
    for (const entry of entries) {
      const active = state.open && entry.id === state.activeId;
      const button = buttons.get(entry.id);
      const child = panels.get(entry.id);
      button?.setAttribute("aria-selected", String(active));
      if (button) button.tabIndex = entry.id === state.activeId ? 0 : -1;
      if (child) child.hidden = entry.id !== state.activeId;
      if (activeValues.get(entry.id) === active) continue;
      activeValues.set(entry.id, active);
      for (const listener of activeListeners.get(entry.id) ?? []) listener(active);
    }
  }

  function openTab(id, focusTab = false) {
    state.activeId = id;
    state.open = true;
    update();
    if (focusTab) buttons.get(id)?.focus();
  }

  function closePanel() {
    state.open = false;
    update();
    toggle.focus();
  }

  function closePanelFromChrome(event) {
    if (event.key !== "Escape" || !state.open) return;
    if (!bar.contains(event.target) && event.target !== splitter) return;
    event.preventDefault();
    closePanel();
  }

  root.addEventListener("keydown", closePanelFromChrome);

  for (const [entryIndex, entry] of entries.entries()) {
    const button = document.createElement("button");
    const buttonId = `conversation-tool-tab-${entryIndex}`;
    const panelId = `conversation-tool-panel-${entryIndex}`;
    button.type = "button";
    button.id = buttonId;
    button.className = "conversation-tools-tab";
    button.setAttribute("role", "tab");
    button.setAttribute("aria-controls", panelId);
    button.textContent = entry.label;
    button.addEventListener("click", () => openTab(entry.id));
    button.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const index = entries.findIndex((item) => item.id === entry.id);
      const step = event.key === "ArrowRight" ? 1 : -1;
      const next = entries[(index + step + entries.length) % entries.length];
      openTab(next.id, true);
    });
    buttons.set(entry.id, button);
    bar.appendChild(button);

    const child = document.createElement("section");
    child.id = panelId;
    child.className = "conversation-tool-panel";
    child.setAttribute("role", "tabpanel");
    child.setAttribute("aria-labelledby", buttonId);
    panels.set(entry.id, child);
    content.appendChild(child);

    const listeners = activeListeners.get(entry.id);
    const tabView = {
      get active() { return state.open && state.activeId === entry.id; },
      onActiveChange(listener) {
        if (typeof listener !== "function") {
          throw new Error("ConversationTab active listener 必须是函数");
        }
        listeners.add(listener);
        listener(state.open && state.activeId === entry.id);
        return () => listeners.delete(listener);
      },
      requestAttention(noticeId) {
        if (typeof noticeId !== "string" || !noticeId || handled.has(noticeId)) return;
        handled.add(noticeId);
        openTab(entry.id);
      },
    };
    disposers.push(tools.render(entry.id, child, tabView));
  }

  close.addEventListener("click", closePanel);
  toggle.addEventListener("click", () => openTab(state.activeId, true));
  bar.appendChild(close);
  panel.append(bar, content);
  root.append(splitter, panel, toggle);
  host.replaceChildren(root);

  let dragStartX = 0;
  let dragStartWidth = 0;
  function moveSplitter(event) {
    if (!root.classList.contains("is-resizing")) return;
    const rtl = getComputedStyle(root).direction === "rtl";
    const delta = (dragStartX - event.clientX) * (rtl ? -1 : 1);
    setWidth(dragStartWidth + delta);
  }
  function stopSplitter(event) {
    if (!root.classList.contains("is-resizing")) return;
    root.classList.remove("is-resizing");
    if (splitter.hasPointerCapture(event.pointerId)) {
      splitter.releasePointerCapture(event.pointerId);
    }
    setWidth(state.width, true);
  }
  splitter.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    dragStartX = event.clientX;
    dragStartWidth = state.width;
    root.classList.add("is-resizing");
    splitter.setPointerCapture(event.pointerId);
  });
  splitter.addEventListener("pointermove", moveSplitter);
  splitter.addEventListener("pointerup", stopSplitter);
  splitter.addEventListener("pointercancel", stopSplitter);
  splitter.addEventListener("keydown", (event) => {
    const rtl = getComputedStyle(root).direction === "rtl";
    let next = state.width;
    if (event.key === "Home") next = MIN_PANEL_WIDTH;
    else if (event.key === "End") next = panelMaximum();
    else if (event.key === "ArrowLeft") next += rtl ? -24 : 24;
    else if (event.key === "ArrowRight") next += rtl ? 24 : -24;
    else return;
    event.preventDefault();
    setWidth(next, true);
  });

  const resize = () => setWidth(
    state.useDefaultWidth ? defaultWidth() : state.preferredWidth,
  );
  window.addEventListener("resize", resize);
  const stopThemeSync = syncFrameTheme(frame);
  resize();
  update();
  return () => {
    root.removeEventListener("keydown", closePanelFromChrome);
    window.removeEventListener("resize", resize);
    activeListeners.clear();
    for (const dispose of disposers.reverse()) dispose();
    stopThemeSync();
    host.replaceChildren();
  };
}

export function activate(ctx) {
  return ctx.ui.inject("shell.pages.v1", (mount) => mount.register({
    id: "conversation",
    label: "对话",
    iconSvg: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 8V4H8"></path><rect width="16" height="12" x="4" y="8" rx="2"></rect><path d="M2 14h2"></path><path d="M20 14h2"></path><path d="M15 13v2"></path><path d="M9 13v2"></path></svg>',
    route: "",
    order: 10,
    children: [{ id: "conversation.tools.v1", cardinality: "list" }],
    render: renderConversation,
  }));
}

import { currentTheme, subscribeTheme } from "@akashic/web-ui-v1";

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
  return [...tabs].sort((left, right) => (left.order ?? 0) - (right.order ?? 0) || left.id.localeCompare(right.id));
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

  const panel = document.createElement("aside");
  panel.className = "conversation-tools";
  panel.id = "conversation-tools-panel";
  panel.setAttribute("aria-label", "对话工具");
  const bar = document.createElement("div");
  bar.className = "conversation-tools-bar";
  bar.setAttribute("role", "tablist");
  bar.setAttribute("aria-label", "工具标签");
  const close = document.createElement("button");
  close.className = "conversation-tools-close";
  close.type = "button";
  close.title = "收起工具区";
  close.setAttribute("aria-label", "收起工具区");
  close.textContent = "›";
  const content = document.createElement("div");
  content.className = "conversation-tools-content";
  const toggle = document.createElement("button");
  toggle.className = "conversation-tools-toggle";
  toggle.type = "button";
  toggle.title = "展开工具区";
  toggle.setAttribute("aria-label", "展开工具区");
  toggle.setAttribute("aria-controls", panel.id);
  toggle.textContent = "‹";
  const state = { open: false, activeId: entries[0].id };
  const handled = new Set();
  const buttons = new Map();
  const panels = new Map();
  const disposers = [];

  function update() {
    root.classList.toggle("tools-open", state.open);
    panel.hidden = !state.open;
    toggle.hidden = state.open;
    toggle.setAttribute("aria-expanded", String(state.open));
    for (const entry of entries) {
      const active = entry.id === state.activeId;
      const button = buttons.get(entry.id);
      const child = panels.get(entry.id);
      button?.setAttribute("aria-selected", String(active));
      if (button) button.tabIndex = active ? 0 : -1;
      if (child) child.hidden = !active;
    }
  }

  function openTab(id) {
    state.activeId = id;
    state.open = true;
    update();
  }

  function closePanel() {
    state.open = false;
    update();
    toggle.focus();
  }

  function closePanelOnEscape(event) {
    if (event.key !== "Escape" || !state.open) return;
    event.preventDefault();
    closePanel();
  }

  root.addEventListener("keydown", closePanelOnEscape);

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
      openTab(next.id);
      buttons.get(next.id)?.focus();
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
    const tabView = {
      get active() { return state.open && state.activeId === entry.id; },
      requestAttention(noticeId) {
        if (typeof noticeId !== "string" || !noticeId || handled.has(noticeId)) return;
        handled.add(noticeId);
        openTab(entry.id);
      },
    };
    disposers.push(tools.render(entry.id, child, tabView));
  }

  close.addEventListener("click", closePanel);
  toggle.addEventListener("click", () => {
    openTab(state.activeId);
    buttons.get(state.activeId)?.focus();
  });
  bar.appendChild(close);
  panel.append(bar, content);
  root.append(panel, toggle);
  host.replaceChildren(root);
  const stopThemeSync = syncFrameTheme(frame);
  update();
  return () => {
    root.removeEventListener("keydown", closePanelOnEscape);
    for (const dispose of disposers.reverse()) dispose();
    stopThemeSync();
    host.replaceChildren();
  };
}

export function activate(ctx) {
  return ctx.ui.inject("shell.pages.v1", (mount) => mount.register({
    id: "conversation",
    label: "对话",
    iconSvg: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-bot" aria-hidden="true"><path d="M12 8V4H8"></path><rect width="16" height="12" x="4" y="8" rx="2"></rect><path d="M2 14h2"></path><path d="M20 14h2"></path><path d="M15 13v2"></path><path d="M9 13v2"></path></svg>',
    route: "",
    order: 10,
    children: [{ id: "conversation.tools.v1", cardinality: "list" }],
    render: renderConversation,
  }));
}

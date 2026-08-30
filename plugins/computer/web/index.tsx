import RFB from "@novnc/novnc";

import type { WebHostContextV1, WebUiDisposer } from "@akashic/web-ui-v1";

import { BACKGROUND_HOLD_MS, reconnectDelay } from "./connection.js";
import "./style.css";

interface ConversationTabView {
  readonly active: boolean;
  onActiveChange(listener: (active: boolean) => void): WebUiDisposer;
  requestAttention(noticeId: string): void;
}

interface ComputerActivity {
  readonly noticeId: number;
  readonly active: boolean;
}

function checkView(value: unknown): ConversationTabView {
  if (!value || typeof value !== "object") {
    throw new Error("Computer 缺少 conversation.tools.v1 view");
  }
  const view = value as Partial<ConversationTabView>;
  if (typeof view.active !== "boolean"
    || typeof view.onActiveChange !== "function"
    || typeof view.requestAttention !== "function") {
    throw new Error("Computer conversation.tools.v1 view 无效");
  }
  return view as ConversationTabView;
}

function iconButton(label: string, path: string): HTMLButtonElement {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "computer-icon-button";
  button.setAttribute("aria-label", label);
  button.title = label;
  button.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="${path}"/></svg>`;
  return button;
}

function checkedActivity(value: unknown): ComputerActivity {
  if (!value || typeof value !== "object") throw new Error("Computer activity 回执无效");
  const activity = value as Partial<ComputerActivity>;
  if (!Number.isInteger(activity.noticeId) || typeof activity.active !== "boolean") {
    throw new Error("Computer activity 回执无效");
  }
  return activity as ComputerActivity;
}

function renderComputer(
  ctx: WebHostContextV1,
  host: HTMLElement,
  rawView: unknown,
): WebUiDisposer {
  const view = checkView(rawView);
  const root = document.createElement("div");
  root.className = "computer-view";

  const desktop = document.createElement("div");
  desktop.className = "computer-desktop";
  const screen = document.createElement("div");
  screen.className = "computer-screen";
  screen.tabIndex = 0;
  screen.setAttribute("role", "group");
  screen.setAttribute(
    "aria-label",
    "Computer 远程桌面。点击后，鼠标和键盘会直接操作这台 Computer。",
  );

  const empty = document.createElement("div");
  empty.className = "computer-connection-state";
  empty.setAttribute("role", "status");
  const emptyTitle = document.createElement("strong");
  emptyTitle.textContent = "正在连接 Computer";
  const emptyDetail = document.createElement("span");
  emptyDetail.textContent = "桌面准备好后会自动出现";
  const retry = document.createElement("button");
  retry.type = "button";
  retry.textContent = "重新连接";
  retry.hidden = true;
  empty.append(emptyTitle, emptyDetail, retry);

  const status = document.createElement("div");
  status.className = "computer-status";
  status.setAttribute("role", "status");
  const statusDot = document.createElement("span");
  statusDot.className = "computer-status-dot";
  const statusText = document.createElement("span");
  statusText.textContent = "正在连接";
  status.append(statusDot, statusText);

  const actions = document.createElement("div");
  actions.className = "computer-actions";
  const clipboardButton = iconButton(
    "打开剪贴板",
    "M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2m1-2h6a1 1 0 0 1 1 1v2a1 1 0 0 1-1 1H9a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1Z",
  );
  clipboardButton.setAttribute("aria-expanded", "false");
  const fullscreenButton = iconButton(
    "全屏显示 Computer",
    "M8 3H5a2 2 0 0 0-2 2v3m13-5h3a2 2 0 0 1 2 2v3M8 21H5a2 2 0 0 1-2-2v-3m13 5h3a2 2 0 0 0 2-2v-3",
  );
  actions.append(clipboardButton, fullscreenButton);

  const clipboard = document.createElement("section");
  clipboard.className = "computer-clipboard";
  clipboard.id = "computer-clipboard-panel";
  clipboard.hidden = true;
  clipboard.setAttribute("aria-label", "Computer 剪贴板");
  clipboardButton.setAttribute("aria-controls", clipboard.id);
  const clipboardHeader = document.createElement("div");
  clipboardHeader.className = "computer-clipboard-header";
  const clipboardTitle = document.createElement("strong");
  clipboardTitle.textContent = "剪贴板";
  const clipboardClose = iconButton(
    "关闭剪贴板",
    "M18 6 6 18M6 6l12 12",
  );
  clipboardHeader.append(clipboardTitle, clipboardClose);
  const clipboardHelp = document.createElement("p");
  clipboardHelp.textContent = "在这里粘贴文字，可发送到 Computer；也可取回 Computer 中复制的文字。";
  const clipboardText = document.createElement("textarea");
  clipboardText.rows = 6;
  clipboardText.maxLength = 65_536;
  clipboardText.spellcheck = false;
  clipboardText.setAttribute("aria-label", "剪贴板文字");
  const clipboardFooter = document.createElement("div");
  clipboardFooter.className = "computer-clipboard-footer";
  const readClipboard = document.createElement("button");
  readClipboard.type = "button";
  readClipboard.textContent = "读取本机";
  const writeClipboard = document.createElement("button");
  writeClipboard.type = "button";
  writeClipboard.textContent = "复制到本机";
  const sendClipboard = document.createElement("button");
  sendClipboard.type = "button";
  sendClipboard.className = "is-primary";
  sendClipboard.textContent = "发送到 Computer";
  const ctrlAltDelete = document.createElement("button");
  ctrlAltDelete.type = "button";
  ctrlAltDelete.textContent = "发送 Ctrl Alt Delete";
  clipboardFooter.append(readClipboard, writeClipboard, sendClipboard, ctrlAltDelete);
  clipboard.append(clipboardHeader, clipboardHelp, clipboardText, clipboardFooter);

  const toolbar = document.createElement("div");
  toolbar.className = "computer-toolbar";
  toolbar.append(status, actions);
  desktop.append(screen, empty);
  root.append(toolbar, desktop, clipboard);
  host.replaceChildren(root);

  let rfb: RFB | null = null;
  let active = view.active;
  let disposed = false;
  let reconnectAttempt = 0;
  let reconnectTimer = 0;
  let backgroundTimer = 0;
  let lastNotice: number | null = null;
  let agentActive = false;
  let catalogStale = false;

  function setStatus(state: "connecting" | "connected" | "waiting" | "failed") {
    status.dataset.state = state;
    statusText.textContent = agentActive && state === "connected"
      ? "Agent 正在操作"
      : {
          connecting: "正在连接",
          connected: "已连接",
          waiting: "已暂停",
          failed: "连接中断",
        }[state];
  }

  function showConnection(title: string, detail: string, canRetry: boolean) {
    empty.hidden = false;
    emptyTitle.textContent = title;
    emptyDetail.textContent = detail;
    retry.hidden = !canRetry;
  }

  function clearTimers() {
    if (reconnectTimer) window.clearTimeout(reconnectTimer);
    if (backgroundTimer) window.clearTimeout(backgroundTimer);
    reconnectTimer = 0;
    backgroundTimer = 0;
  }

  function scheduleReconnect() {
    if (disposed || catalogStale || !active || reconnectTimer || rfb) return;
    const delay = reconnectDelay(reconnectAttempt++);
    reconnectTimer = window.setTimeout(() => {
      reconnectTimer = 0;
      connect();
    }, delay);
  }

  function connect() {
    if (disposed || catalogStale || !active || rfb) return;
    if (backgroundTimer) window.clearTimeout(backgroundTimer);
    backgroundTimer = 0;
    screen.replaceChildren();
    showConnection("正在连接 Computer", "正在建立安全的远程桌面会话", false);
    setStatus("connecting");

    let next: RFB;
    try {
      next = new RFB(
        screen,
        ctx.http.webSocketUrl("/api/dashboard/computer/display"),
        { shared: true, wsProtocols: ["binary"] },
      );
    } catch {
      showConnection("无法连接 Computer", "请检查插件是否正在运行", true);
      setStatus("failed");
      scheduleReconnect();
      return;
    }
    rfb = next;
    next.scaleViewport = true;
    next.resizeSession = false;
    next.clipViewport = false;
    next.focusOnClick = true;
    next.viewOnly = false;
    next.showDotCursor = true;
    next.qualityLevel = 7;
    next.compressionLevel = 2;

    next.addEventListener("connect", () => {
      if (rfb !== next) return;
      reconnectAttempt = 0;
      empty.hidden = true;
      setStatus("connected");
      if (active) next.focus({ preventScroll: true });
    });
    next.addEventListener("clipboard", (event) => {
      if (document.activeElement !== clipboardText) clipboardText.value = event.detail.text;
    });
    next.addEventListener("securityfailure", (event) => {
      showConnection(
        "Computer 拒绝了连接",
        event.detail.reason || "远程桌面安全握手失败",
        true,
      );
    });
    next.addEventListener("disconnect", () => {
      if (rfb !== next) return;
      rfb = null;
      if (disposed) return;
      if (!active) {
        showConnection("Computer 已暂停", "重新展开时会恢复连接", false);
        setStatus("waiting");
        return;
      }
      showConnection("连接已中断", "正在自动重新连接", true);
      setStatus("failed");
      void loadActivity();
      scheduleReconnect();
    });
  }

  function setActive(next: boolean) {
    active = next;
    if (reconnectTimer) window.clearTimeout(reconnectTimer);
    reconnectTimer = 0;
    if (active) {
      if (backgroundTimer) window.clearTimeout(backgroundTimer);
      backgroundTimer = 0;
      if (rfb) rfb.focus({ preventScroll: true });
      else connect();
      return;
    }
    rfb?.blur();
    if (backgroundTimer || !rfb) return;
    backgroundTimer = window.setTimeout(() => {
      backgroundTimer = 0;
      rfb?.disconnect();
    }, BACKGROUND_HOLD_MS);
  }

  async function loadActivity() {
    if (disposed || catalogStale) return;
    try {
      const response = await ctx.http.request("/api/dashboard/computer/activity", {
        cache: "no-store",
      });
      if (response.headers.get("X-Akashic-Web-Stale") === "1") {
        catalogStale = true;
        clearTimers();
        rfb?.disconnect();
        showConnection("界面已更新", "请刷新页面以使用新的 Computer", false);
        setStatus("failed");
        return;
      }
      if (!response.ok) throw new Error(`activity ${response.status}`);
      const activity = checkedActivity(await response.json());
      agentActive = activity.active;
      if (activity.active && activity.noticeId !== lastNotice) {
        view.requestAttention(`computer:${activity.noticeId}`);
      }
      lastNotice = activity.noticeId;
      if (rfb) setStatus("connected");
    } catch {
      if (!catalogStale && rfb === null) setStatus("failed");
    }
  }

  function setClipboardOpen(open: boolean, returnToDesktop = false) {
    clipboard.hidden = !open;
    clipboardButton.setAttribute("aria-expanded", String(open));
    if (open) clipboardText.focus();
    else if (returnToDesktop) rfb?.focus({ preventScroll: true });
    else clipboardButton.focus();
  }

  screen.addEventListener("focus", () => rfb?.focus({ preventScroll: true }));
  retry.addEventListener("click", () => {
    reconnectAttempt = 0;
    rfb?.disconnect();
    if (!rfb) connect();
  });
  clipboardButton.addEventListener("click", () => setClipboardOpen(clipboard.hidden));
  clipboardClose.addEventListener("click", () => setClipboardOpen(false));
  clipboard.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    event.preventDefault();
    setClipboardOpen(false);
  });
  readClipboard.addEventListener("click", () => {
    void navigator.clipboard.readText().then((text) => {
      clipboardText.value = text;
      clipboardText.focus();
    }).catch(() => {
      clipboardText.focus();
      clipboardText.select();
    });
  });
  writeClipboard.addEventListener("click", () => {
    void navigator.clipboard.writeText(clipboardText.value).catch(() => {
      clipboardText.focus();
      clipboardText.select();
    });
  });
  sendClipboard.addEventListener("click", () => {
    rfb?.clipboardPasteFrom(clipboardText.value);
    setClipboardOpen(false, true);
  });
  ctrlAltDelete.addEventListener("click", () => {
    rfb?.sendCtrlAltDel();
    setClipboardOpen(false, true);
  });
  fullscreenButton.addEventListener("click", () => {
    const operation = document.fullscreenElement
      ? document.exitFullscreen()
      : root.requestFullscreen();
    void operation.catch(() => setStatus(rfb ? "connected" : "failed"));
  });

  const stopActive = view.onActiveChange(setActive);
  const activityPoll = window.setInterval(() => void loadActivity(), 1_000);
  void loadActivity();

  return () => {
    disposed = true;
    stopActive();
    window.clearInterval(activityPoll);
    clearTimers();
    rfb?.disconnect();
    rfb = null;
    host.replaceChildren();
  };
}

export function activate(ctx: WebHostContextV1): WebUiDisposer {
  return ctx.ui.inject("conversation.tools.v1", (mount) => mount.register({
    id: "computer",
    label: "Computer",
    order: 10,
    render(host, _entryView, props) {
      return renderComputer(ctx, host, props);
    },
  }));
}

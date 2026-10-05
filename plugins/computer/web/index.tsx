import { ComputerDisplay } from "./display";

import type { WebHostContextV1, WebUiDisposer } from "@akashic/web-ui-v1";

import {
  BACKGROUND_HOLD_MS,
  reconnectDelay,
  shouldOpenForActivity,
} from "./connection.js";
import {
  clipboardShortcut,
  keysymForKey,
  pasteKeySequence,
} from "./remote-input.js";
import "./style.css";

interface ConversationTabView {
  readonly active: boolean;
  onActiveChange(listener: (active: boolean) => void): WebUiDisposer;
  requestAttention(noticeId: string): void;
}

interface ComputerActivity {
  readonly noticeId: number;
  readonly active: boolean;
  readonly browser: { readonly state: string; readonly error: string };
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
  if (!Number.isInteger(activity.noticeId) || typeof activity.active !== "boolean"
    || !activity.browser || typeof activity.browser.state !== "string"
    || typeof activity.browser.error !== "string") {
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
  const clipboardStatus = document.createElement("p");
  clipboardStatus.className = "computer-clipboard-status";
  clipboardStatus.setAttribute("role", "status");
  clipboardStatus.setAttribute("aria-live", "polite");
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
  clipboard.append(
    clipboardHeader,
    clipboardHelp,
    clipboardText,
    clipboardStatus,
    clipboardFooter,
  );

  const toolbar = document.createElement("div");
  toolbar.className = "computer-toolbar";
  toolbar.append(status, actions);
  desktop.append(screen, empty);
  root.append(toolbar, desktop, clipboard);
  host.replaceChildren(root);

  let display: ComputerDisplay | null = null;
  let displayConnected = false;
  let superseded = false;
  let active = view.active;
  let disposed = false;
  let reconnectAttempt = 0;
  let reconnectTimer = 0;
  let backgroundTimer = 0;
  let lastNotice: number | null = null;
  let agentActive = false;
  let catalogStale = false;
  let sleeping = false;
  let failed = false;
  let connectionFailed = false;
  let waking = false;
  let lastTouch = 0;
  const heldKeys = new Map<string, { keysym: number; code: string }>();
  let pasteAttempt: {
    readonly id: number;
    readonly code: string;
    finished: boolean;
    keyReleased: boolean;
  } | null = null;
  let pasteAttemptId = 0;
  let pasteFallbackTimer = 0;
  let clipboardWriteQueue = Promise.resolve(false);
  let latestRemoteClipboard: string | null = null;
  let remoteClipboardId = 0;

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
    if (disposed || catalogStale || superseded || sleeping || failed || connectionFailed || !active || reconnectTimer || display) return;
    const delay = reconnectDelay(reconnectAttempt++);
    reconnectTimer = window.setTimeout(() => {
      reconnectTimer = 0;
      connect();
    }, delay);
  }

  function connect() {
    if (disposed || catalogStale || superseded || sleeping || failed || connectionFailed || !active || reconnectTimer || display) return;
    if (backgroundTimer) window.clearTimeout(backgroundTimer);
    backgroundTimer = 0;
    screen.replaceChildren();
    showConnection("正在连接 Computer", "正在建立安全的远程桌面会话", false);
    setStatus("connecting");

    let next: ComputerDisplay;
    try {
      next = new ComputerDisplay(screen, ctx, {
        keydown: onRemoteKeyDown,
        keyup: onRemoteKeyUp,
        paste: onRemotePaste,
        touch,
        blur: onWindowBlur,
      });
    } catch (error) {
      connectionFailed = true;
      showConnection("无法连接 Computer", String(error), true);
      setStatus("failed");
      return;
    }
    display = next;
    displayConnected = false;

    next.addEventListener("superseded", () => { superseded = true; });
    next.addEventListener("stale", () => {
      if (display !== next) return;
      catalogStale = true;
      clearTimers();
      next.disconnect();
      showConnection("界面已更新", "请刷新页面以使用新的 Computer", false);
      setStatus("failed");
    });
    next.addEventListener("connect", () => {
      if (display !== next) return;
      displayConnected = true;
      reconnectAttempt = 0;
      empty.hidden = true;
      setStatus("connected");
      if (active) next.focus({ preventScroll: true });
    });
    next.addEventListener("clipboard", (event) => {
      void receiveRemoteClipboard((event as CustomEvent<{ text: string }>).detail.text);
    });
    next.addEventListener("error", (event) => {
      if (display !== next) return;
      connectionFailed = true;
      clearTimers();
      setStatus("failed");
      showConnection(
        "无法连接 Computer",
        (event as CustomEvent<{ reason: string }>).detail.reason || "远程显示连接失败",
        true,
      );
    });
    next.addEventListener("disconnect", () => {
      if (display !== next) return;
      display = null;
      displayConnected = false;
      if (disposed) return;
      if (catalogStale) {
        showConnection("界面已更新", "请刷新页面以使用新的 Computer", false);
        setStatus("failed");
        return;
      }
      if (superseded) {
        showConnection("Computer 已在另一窗口接管", "点击按钮可在此窗口继续操作", true);
        retry.textContent = "在此窗口接管";
        setStatus("waiting");
        return;
      }
      // 启动失败保留原因，轮询和断线事件不能改写成无限自动重连。
      if (connectionFailed) return;
      if (!active) {
        showConnection("Computer 已暂停", "重新展开时会恢复连接", false);
        setStatus("waiting");
        return;
      }
      showConnection("连接已中断", "正在自动重新连接", true);
      setStatus("failed");
      void loadActivity().then(scheduleReconnect);
    });
  }

  function setActive(next: boolean) {
    active = next;
    if (reconnectTimer) window.clearTimeout(reconnectTimer);
    reconnectTimer = 0;
    if (active) {
      if (backgroundTimer) window.clearTimeout(backgroundTimer);
      backgroundTimer = 0;
      if (display) display.focus({ preventScroll: true });
      else void wake();
      return;
    }
    releaseRemoteKeys();
    display?.blur();
    if (backgroundTimer || !display) return;
    backgroundTimer = window.setTimeout(() => {
      backgroundTimer = 0;
      display?.disconnect();
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
        display?.disconnect();
        showConnection("界面已更新", "请刷新页面以使用新的 Computer", false);
        setStatus("failed");
        return;
      }
      if (!response.ok) throw new Error(`activity ${response.status}`);
      const activity = checkedActivity(await response.json());
      sleeping = activity.browser.state === "sleeping" || activity.browser.state === "stopping";
      failed = activity.browser.state === "failed";
      if (sleeping && !waking) {
        clearTimers();
        display?.disconnect();
        showConnection("Computer 已休眠", "无操作满 10 分钟，桌面已释放；保存的登录身份仍在", true);
        retry.textContent = "唤醒 Computer";
        setStatus("waiting");
        statusText.textContent = "已休眠";
      } else if (activity.browser.state === "failed") {
        clearTimers();
        showConnection("Computer 启动或停止失败", activity.browser.error, true);
        setStatus("failed");
      } else if (!display && active && !waking && activity.browser.state === "ready") {
        scheduleReconnect();
      }
      agentActive = activity.active;
      if (shouldOpenForActivity(lastNotice, activity.noticeId, activity.active)) {
        view.requestAttention(`computer:${activity.noticeId}`);
      }
      lastNotice = activity.noticeId;
      if (displayConnected && activity.browser.state === "ready") setStatus("connected");
    } catch {
      if (!catalogStale && display === null) setStatus("failed");
    }
  }

  async function wake() {
    if (waking || disposed || catalogStale) return;
    waking = true;
    connectionFailed = false;
    superseded = false;
    showConnection("正在唤醒 Computer", "正在打开你的主浏览器", false);
    setStatus("connecting");
    try {
      const response = await ctx.http.request("/api/dashboard/computer/wake", { method: "POST" });
      if (!response.ok) throw new Error(`wake ${response.status}`);
      sleeping = false;
      failed = false;
      retry.textContent = "重新连接";
      connect();
    } catch (error) {
      showConnection("无法唤醒 Computer", String(error), true);
      setStatus("failed");
    } finally { waking = false; }
  }

  function touch(event: Event) {
    if (!event.isTrusted || !display || Date.now() - lastTouch < 1000) return;
    lastTouch = Date.now();
    void ctx.http.request("/api/dashboard/computer/touch", { method: "POST" }).then((response) => {
      if (!response.ok) throw new Error(`touch ${response.status}`);
    }).catch((error) => {
      if (disposed) return;
      console.warn("Computer activity update failed", error);
      statusText.textContent = "已连接 · 活动状态同步失败";
      void loadActivity();
    });
  }

  function setClipboardOpen(open: boolean, returnToDesktop = false) {
    if (open) {
      releaseRemoteKeys();
      display?.blur();
    }
    clipboard.hidden = !open;
    clipboardButton.setAttribute("aria-expanded", String(open));
    if (open) clipboardText.focus();
    else if (returnToDesktop) display?.focus({ preventScroll: true });
    else clipboardButton.focus();
  }

  function releaseRemoteKeys() {
    if (display) {
      for (const { keysym, code } of heldKeys.values()) {
        display.sendKey(keysym, code, false);
      }
    }
    heldKeys.clear();
  }

  function clearPasteAttempt() {
    if (pasteFallbackTimer) window.clearTimeout(pasteFallbackTimer);
    pasteFallbackTimer = 0;
    pasteAttempt = null;
  }

  function hasHeldControl() {
    return heldKeys.has("ControlLeft") || heldKeys.has("ControlRight");
  }

  function sendPasteKeys(controlHeld: boolean) {
    if (!display) return;
    const metaCodes = [...heldKeys.values()]
      .map((value) => value.code)
      .filter((code) => code === "MetaLeft" || code === "MetaRight");
    for (const event of pasteKeySequence(controlHeld, metaCodes)) {
      display.sendKey(event.keysym, event.code, event.down);
    }
  }

  async function finishRemotePaste(attemptId: number, rawText: string) {
    const attempt = pasteAttempt;
    if (!attempt || attempt.id !== attemptId || attempt.finished || !display) return;
    attempt.finished = true;
    clipboardText.value = rawText;
    const current = display;
    try {
      await current.clipboardPasteFrom(rawText);
      if (disposed || display !== current) return;
      sendPasteKeys(hasHeldControl());
      clipboardStatus.textContent = "已粘贴到 Computer。";
    } catch (error) {
      if (!disposed) clipboardStatus.textContent = `粘贴失败：${String(error)}`;
    }
    if (pasteAttempt === attempt && attempt.keyReleased) clearPasteAttempt();
  }

  function beginRemotePaste(event: KeyboardEvent) {
    event.stopPropagation();
    if (event.repeat) {
      event.preventDefault();
      return;
    }
    clearPasteAttempt();
    const attempt = {
      id: ++pasteAttemptId,
      code: event.code || "KeyV",
      finished: false,
      keyReleased: false,
    };
    pasteAttempt = attempt;

    // Keep the default action alive so the browser emits a trusted paste event.
    const clipboardApi = navigator.clipboard;
    if (clipboardApi?.readText) {
      void clipboardApi.readText()
        .then((text) => finishRemotePaste(attempt.id, text))
        .catch(() => undefined);
    }
    pasteFallbackTimer = window.setTimeout(() => {
      if (pasteAttempt?.id !== attempt.id || pasteAttempt.finished) return;
      clearPasteAttempt();
      setClipboardOpen(true);
      clipboardStatus.textContent = "浏览器未允许直接读取。请在文本框中按 Ctrl+V。";
    }, 1_000);
  }

  function onRemotePaste(event: ClipboardEvent) {
    if (!display) return;
    event.preventDefault();
    event.stopPropagation();
    const text = event.clipboardData?.getData("text/plain");
    if (text === undefined) return;
    if (!pasteAttempt) {
      pasteAttempt = {
        id: ++pasteAttemptId,
        code: "KeyV",
        finished: false,
        keyReleased: true,
      };
    }
    void finishRemotePaste(pasteAttempt.id, text);
  }

  async function writeTextToLocal(text: string): Promise<boolean> {
    const clipboardApi = navigator.clipboard;
    if (!clipboardApi?.writeText) return false;
    try {
      await clipboardApi.writeText(text);
      return true;
    } catch {
      return false;
    }
  }

  async function receiveRemoteClipboard(rawText: string) {
    const clipboardId = ++remoteClipboardId;
    latestRemoteClipboard = rawText;
    const editing = document.activeElement === clipboardText;
    if (!editing) clipboardText.value = rawText;
    const write = clipboardWriteQueue.then(() => writeTextToLocal(rawText));
    clipboardWriteQueue = write;
    const copied = await write;
    if (clipboardId !== remoteClipboardId) return;
    readClipboard.textContent = "读取本机";
    if (copied) {
      latestRemoteClipboard = null;
      clipboardStatus.textContent = "已复制到本机剪贴板。";
      return;
    }
    if (editing) {
      readClipboard.textContent = "载入 Computer 文字";
      clipboardStatus.textContent = "Computer 有新的复制内容，尚未覆盖你正在编辑的文字。";
      return;
    }
    latestRemoteClipboard = null;
    clipboardStatus.textContent = "Computer 已复制文字。点击“复制到本机”即可取回。";
  }

  function sendRemoteKey(event: KeyboardEvent, down: boolean) {
    event.preventDefault();
    event.stopPropagation();
    if (!display) return;
    const id = event.code || event.key;
    if (down && event.repeat) return;
    const held = heldKeys.get(id);
    if (!down && !held) return;
    const keysym = held?.keysym ?? keysymForKey(event.key, event.code);
    if (keysym === null) return;
    const code = held?.code ?? event.code;
    display.sendKey(keysym, code, down);
    if (down) heldKeys.set(id, { keysym, code });
    else heldKeys.delete(id);
  }

  function onRemoteKeyDown(event: KeyboardEvent) {
    if (clipboardShortcut(event.key, event.ctrlKey, event.metaKey, event.altKey) === "paste") {
      beginRemotePaste(event);
      return;
    }
    sendRemoteKey(event, true);
  }

  function onRemoteKeyUp(event: KeyboardEvent) {
    if (pasteAttempt && (event.code || event.key) === pasteAttempt.code) {
      event.preventDefault();
      event.stopPropagation();
      pasteAttempt.keyReleased = true;
      if (pasteAttempt.finished) clearPasteAttempt();
      return;
    }
    sendRemoteKey(event, false);
  }

  function onWindowBlur() {
    releaseRemoteKeys();
    display?.blur();
  }

  function onVisibilityChange() {
    if (document.visibilityState === "hidden") onWindowBlur();
  }

  function selectClipboardText() {
    clipboardText.focus();
    clipboardText.select();
  }

  async function readLocalClipboard() {
    const clipboardApi = navigator.clipboard;
    if (!clipboardApi?.readText) {
      selectClipboardText();
      clipboardStatus.textContent = "请在文本框中按 Ctrl+V 粘贴本机文字。";
      return;
    }
    try {
      clipboardText.value = await clipboardApi.readText();
      clipboardText.focus();
      clipboardStatus.textContent = "已读取本机剪贴板。";
    } catch {
      selectClipboardText();
      clipboardStatus.textContent = "浏览器未允许读取。请在文本框中按 Ctrl+V。";
    }
  }

  async function writeLocalClipboard() {
    if (await writeTextToLocal(clipboardText.value)) {
      clipboardStatus.textContent = "已复制到本机剪贴板。";
      return;
    }
    selectClipboardText();
    if (document.execCommand("copy")) {
      clipboardStatus.textContent = "已复制到本机剪贴板。";
      return;
    }
    clipboardStatus.textContent = "文字已选中。请按 Ctrl+C 复制。";
  }

  screen.addEventListener("focus", () => display?.focus({ preventScroll: true }));
  screen.addEventListener("keydown", onRemoteKeyDown, true);
  screen.addEventListener("keyup", onRemoteKeyUp, true);
  screen.addEventListener("paste", onRemotePaste, true);
  const inputEvents = ["pointerdown", "pointermove", "pointerup", "wheel", "keydown", "keyup", "paste"];
  for (const name of inputEvents) screen.addEventListener(name, touch, true);
  sendClipboard.addEventListener("click", touch);
  ctrlAltDelete.addEventListener("click", touch);
  window.addEventListener("blur", onWindowBlur);
  document.addEventListener("visibilitychange", onVisibilityChange);
  retry.addEventListener("click", () => {
    clearTimers();
    reconnectAttempt = 0;
    display?.disconnect();
    if (!display) void wake();
  });
  clipboardButton.addEventListener("click", () => setClipboardOpen(clipboard.hidden));
  clipboardClose.addEventListener("click", () => setClipboardOpen(false));
  clipboard.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    event.preventDefault();
    setClipboardOpen(false);
  });
  readClipboard.addEventListener("click", () => {
    if (latestRemoteClipboard !== null) {
      clipboardText.value = latestRemoteClipboard;
      latestRemoteClipboard = null;
      readClipboard.textContent = "读取本机";
      clipboardText.focus();
      clipboardStatus.textContent = "已载入 Computer 复制的文字。";
      return;
    }
    void readLocalClipboard();
  });
  writeClipboard.addEventListener("click", () => void writeLocalClipboard());
  sendClipboard.addEventListener("click", async () => {
    if (!display) {
      clipboardStatus.textContent = "Computer 尚未连接。";
      return;
    }
    const current = display;
    try {
      await current.clipboardPasteFrom(clipboardText.value);
      if (disposed || display !== current) return;
      clipboardStatus.textContent = "已发送到 Computer。关闭后在 Computer 中按 Ctrl+V 粘贴。";
    } catch (error) {
      if (!disposed) clipboardStatus.textContent = `发送失败：${String(error)}`;
    }
  });
  ctrlAltDelete.addEventListener("click", () => {
    display?.sendCtrlAltDel();
    setClipboardOpen(false, true);
  });
  function syncFullscreenLabel() {
    const fullscreen = document.fullscreenElement === root;
    const label = fullscreen ? "退出全屏" : "全屏显示 Computer";
    fullscreenButton.setAttribute("aria-label", label);
    fullscreenButton.title = label;
  }
  document.addEventListener("fullscreenchange", syncFullscreenLabel);
  fullscreenButton.addEventListener("click", () => {
    const operation = document.fullscreenElement
      ? document.exitFullscreen()
      : root.requestFullscreen();
    void operation.catch(() => {
      status.dataset.state = "failed";
      statusText.textContent = "无法进入全屏";
    });
  });

  const stopActive = view.onActiveChange(setActive);
  const activityPoll = window.setInterval(() => void loadActivity(), 1_000);
  void loadActivity();
  if (active) void wake();

  return () => {
    disposed = true;
    stopActive();
    window.clearInterval(activityPoll);
    document.removeEventListener("fullscreenchange", syncFullscreenLabel);
    screen.removeEventListener("keydown", onRemoteKeyDown, true);
    screen.removeEventListener("keyup", onRemoteKeyUp, true);
    screen.removeEventListener("paste", onRemotePaste, true);
    for (const name of inputEvents) screen.removeEventListener(name, touch, true);
    sendClipboard.removeEventListener("click", touch);
    ctrlAltDelete.removeEventListener("click", touch);
    window.removeEventListener("blur", onWindowBlur);
    document.removeEventListener("visibilitychange", onVisibilityChange);
    releaseRemoteKeys();
    clearPasteAttempt();
    clearTimers();
    display?.disconnect();
    display = null;
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

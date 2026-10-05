import type { WebHostContextV1, WebUiDisposer } from "@akashic/web-ui-v1";
import { ComputerDisplay } from "./display";
import { BrowserDisplay } from "./browser-display";
import { keysymForKey } from "./remote-input.js";
import "./style.css";

interface ConversationTabView {
  readonly active: boolean;
  onActiveChange(listener: (active: boolean) => void): WebUiDisposer;
  requestAttention(noticeId: string): void;
}
interface Target { id: string; label: string; kind: "desktop" | "browser"; url?: string }
interface Activity { noticeId: number; active: boolean; browser: { state: string; error: string } }

function button(label: string) {
  const node = document.createElement("button");
  node.type = "button";
  node.textContent = label;
  return node;
}

/** 目录、观看目标、输入权限与面板可见性各自拥有一个事实。 */
function renderComputer(ctx: WebHostContextV1, host: HTMLElement, rawView: unknown): WebUiDisposer {
  const view = rawView as ConversationTabView;
  if (!view || typeof view.active !== "boolean" || typeof view.onActiveChange !== "function"
    || typeof view.requestAttention !== "function") throw new Error("Computer 缺少 conversation.tools.v1 view");
  const root = document.createElement("div");
  root.className = "computer-view";
  const tabs = document.createElement("div");
  tabs.className = "computer-target-tabs";
  tabs.setAttribute("role", "tablist");
  tabs.setAttribute("aria-label", "Computer 观看目标");
  const toolbar = document.createElement("div");
  toolbar.className = "computer-toolbar";
  const status = document.createElement("span");
  status.className = "computer-status";
  status.setAttribute("role", "status");
  const controlButton = button("接管操作");
  const clipboardButton = button("剪贴板");
  const fullscreen = button("全屏");
  const actions = document.createElement("div");
  actions.className = "computer-actions";
  actions.append(controlButton, clipboardButton, fullscreen);
  toolbar.append(status, actions);
  const desktop = document.createElement("div");
  desktop.className = "computer-desktop";
  const screen = document.createElement("div");
  screen.className = "computer-screen";
  screen.tabIndex = 0;
  screen.setAttribute("role", "tabpanel");
  const empty = document.createElement("div");
  empty.className = "computer-connection-state";
  const detail = document.createElement("span");
  const retry = button("重新连接");
  empty.append(detail, retry);
  desktop.append(screen, empty);
  const clipboard = document.createElement("section");
  clipboard.className = "computer-clipboard";
  clipboard.hidden = true;
  clipboard.setAttribute("aria-label", "Computer 剪贴板");
  const text = document.createElement("textarea");
  text.maxLength = 65536;
  text.rows = 6;
  text.setAttribute("aria-label", "剪贴板文字");
  const send = button("粘贴到 Computer");
  const copy = button("复制到本机");
  const clipStatus = document.createElement("p");
  clipStatus.setAttribute("role", "status");
  clipboard.append(text, send, copy, clipStatus);
  root.append(tabs, toolbar, desktop, clipboard);
  host.replaceChildren(root);

  let targets: Target[] = [];
  let selected = "desktop";
  let control = "";
  let busy = false;
  let active = view.active;
  let ready = false;
  let connected = false;
  let disposed = false;
  let blocked = false;
  let lastNotice: number | null = null;
  let display: ComputerDisplay | BrowserDisplay | null = null;
  let timer = 0;
  let attempt = 0;
  let lastTouch = 0;
  const held = new Map<string, number>();

  function state(message: string, waiting = false) {
    status.textContent = message;
    detail.textContent = message;
    empty.hidden = !waiting;
    controlButton.textContent = control ? "释放操作" : busy ? "另一窗口正在操作" : "接管操作";
    controlButton.disabled = !ready || !connected || (busy && !control);
    send.disabled = !control || !connected;
  }

  function releaseKeys() {
    if (display instanceof ComputerDisplay)
      for (const [code, keysym] of held) display.sendKey(keysym, code, false);
    held.clear();
  }

  function disconnect() {
    attempt++;
    releaseKeys();
    display?.disconnect();
    display = null;
    control = "";
    connected = false;
  }

  function inputKey(event: KeyboardEvent) {
    if (!control || !connected) return;
    event.preventDefault();
    touch(event);
    if (display instanceof BrowserDisplay) { display.key(event); return; }
    if (event.type === "keydown" && (event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "v") {
      releaseKeys();
      void navigator.clipboard.readText().then(paste).catch(error => state(String(error)));
      return;
    }
    const keysym = keysymForKey(event.key, event.code);
    if (keysym == null) return;
    if (event.type === "keydown") held.set(event.code, keysym);
    else held.delete(event.code);
    display?.sendKey(keysym, event.code, event.type === "keydown");
  }

  function touch(event: Event) {
    if (!event.isTrusted || !control || Date.now() - lastTouch < 1000) return;
    lastTouch = Date.now();
    void ctx.http.request("/api/dashboard/computer/touch", { method: "POST" }).then(response => {
      if (!response.ok) state("操作活动同步失败，请重新连接");
    }).catch(() => state("操作活动同步失败，请重新连接"));
  }

  async function paste(value: string) {
    if (!control || !display || !connected) throw new Error("请先接管操作");
    await display.clipboardPasteFrom(value);
    if (display instanceof ComputerDisplay) {
      display.sendKey(0xffe3, "ControlLeft", true);
      display.sendKey(0x76, "KeyV", true);
      display.sendKey(0x76, "KeyV", false);
      display.sendKey(0xffe3, "ControlLeft", false);
    }
  }

  function connect(token = "") {
    if (!active || !ready || disposed || blocked) return;
    const target = targets.find(item => item.id === selected);
    if (!target) return;
    disconnect();
    control = token;
    const current = ++attempt;
    state(token ? "正在接管，等待 Agent 释放输入" : "正在连接 · 只读观看", true);
    const next = target.kind === "desktop"
      ? new ComputerDisplay(screen, ctx, { keydown: inputKey, keyup: inputKey,
        touch, blur: releaseKeys, paste: event => {
          event.preventDefault();
          void paste(event.clipboardData?.getData("text/plain") ?? "").catch(error => state(String(error)));
        } }, token)
      : new BrowserDisplay(screen, ctx, selected, token);
    display = next;
    next.addEventListener("connect", () => {
      if (attempt !== current) return;
      connected = true;
      state(control ? "你正在操作 · Agent 已暂停" : "只读观看");
    });
    next.addEventListener("clipboard", event => {
      text.value = (event as CustomEvent<{ text: string }>).detail.text;
    });
    next.addEventListener("error", event => {
      if (attempt !== current) return;
      blocked = true;
      state((event as CustomEvent<{ reason: string }>).detail.reason, true);
    });
    next.addEventListener("stale", () => {
      if (attempt !== current) return;
      blocked = true;
      state("界面已更新，请刷新页面", true);
    });
    next.addEventListener("disconnect", () => {
      if (attempt !== current) return;
      display = null;
      control = "";
      connected = false;
      state(blocked ? detail.textContent ?? "连接失败" : "连接中断，正在恢复只读观看", true);
    });
  }

  function select(id: string) {
    selected = id;
    blocked = false;
    disconnect();
    renderTabs();
    connect();
  }

  function renderTabs() {
    tabs.replaceChildren(...targets.map((target, index) => {
      const tab = button(target.label);
      tab.title = target.url || target.label;
      tab.id = `computer-target-${index}`;
      tab.setAttribute("role", "tab");
      tab.setAttribute("aria-selected", String(target.id === selected));
      tab.tabIndex = target.id === selected ? 0 : -1;
      if (target.id === selected) screen.setAttribute("aria-labelledby", tab.id);
      tab.onclick = () => select(target.id);
      tab.onkeydown = event => {
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
        event.preventDefault();
        const next = event.key === "Home" ? 0 : event.key === "End" ? targets.length - 1
          : (index + (event.key === "ArrowRight" ? 1 : -1) + targets.length) % targets.length;
        select(targets[next].id);
        (tabs.children[next] as HTMLElement).focus();
      };
      return tab;
    }));
  }

  /** 轮询只观察；休眠需明确唤醒，通知不修改面板和观看目标。 */
  async function poll() {
    try {
      const responses = await Promise.all([
        ctx.http.request("/api/dashboard/computer/activity", { cache: "no-store" }),
        ctx.http.request("/api/dashboard/computer/targets", { cache: "no-store" }),
      ]);
      if (disposed) return;
      if (responses.some(response => response.headers.get("X-Akashic-Web-Stale") === "1")) {
        blocked = true;
        disconnect();
        state("界面已更新，请刷新页面", true);
        return;
      }
      if (responses.some(response => !response.ok)) throw new Error("Computer 状态读取失败");
      const activity = await responses[0].json() as Activity;
      const catalog = await responses[1].json() as { targets: Target[]; control: boolean };
      if (!Number.isInteger(activity.noticeId) || typeof activity.active !== "boolean"
        || typeof activity.browser?.state !== "string" || !Array.isArray(catalog.targets)
        || typeof catalog.control !== "boolean" || catalog.targets.some(target => !target.id
          || !target.label || !["desktop", "browser"].includes(target.kind))) throw new Error("Computer 回执无效");
      ready = activity.browser.state === "ready";
      busy = catalog.control;
      if ((lastNotice !== null && lastNotice !== activity.noticeId) || (lastNotice === null && activity.active))
        view.requestAttention(`computer:${activity.noticeId}`);
      lastNotice = activity.noticeId;
      if (JSON.stringify(targets) !== JSON.stringify(catalog.targets)) {
        targets = catalog.targets;
        renderTabs();
      }
      if (!ready) {
        disconnect();
        retry.textContent = "唤醒 Computer";
        state(activity.browser.error || "Computer 已休眠，保存的身份仍在", true);
      } else if (!targets.some(target => target.id === selected)) {
        disconnect();
        blocked = true;
        state("观看目标已结束，请选择其他标签", true);
      } else {
        retry.textContent = "重新连接";
        if (active && !display && !blocked) connect();
        if (connected) state(control ? "你正在操作 · Agent 已暂停"
          : busy ? "只读观看 · 另一窗口正在操作" : activity.active ? "只读观看 · Agent 正在执行" : "只读观看");
      }
    } catch (error) { if (!disposed) state(String(error), !connected); }
    finally { if (!disposed) timer = window.setTimeout(() => void poll(), 1000); }
  }

  controlButton.onclick = () => {
    blocked = false;
    connect(control ? "" : crypto.randomUUID());
  };
  retry.onclick = async () => {
    blocked = false;
    if (!ready) {
      const response = await ctx.http.request("/api/dashboard/computer/wake", { method: "POST" });
      if (!response.ok) state("唤醒失败，请重试", true);
    } else connect();
  };
  clipboardButton.onclick = () => { clipboard.hidden = !clipboard.hidden; if (!clipboard.hidden) text.focus(); };
  fullscreen.onclick = () => void (document.fullscreenElement ? document.exitFullscreen() : root.requestFullscreen());
  send.onclick = () => { void paste(text.value).then(() => { clipStatus.textContent = "已粘贴"; },
    error => { clipStatus.textContent = String(error); }); };
  copy.onclick = () => { void navigator.clipboard.writeText(text.value).then(() => { clipStatus.textContent = "已复制"; },
    error => { clipStatus.textContent = String(error); }); };
  screen.addEventListener("keydown", inputKey, true);
  screen.addEventListener("keyup", inputKey, true);
  const stopActive = view.onActiveChange(value => {
    active = value;
    if (!active) disconnect();
    else connect();
  });
  state("正在读取 Computer 状态", true);
  void poll();
  return () => {
    disposed = true;
    window.clearTimeout(timer);
    stopActive();
    disconnect();
    host.replaceChildren();
  };
}

export function activate(ctx: WebHostContextV1): WebUiDisposer {
  return ctx.ui.inject("conversation.tools.v1", mount => mount.register({
    id: "computer", label: "Computer", order: 10,
    render(host, _entryView, props) { return renderComputer(ctx, host, props); },
  }));
}

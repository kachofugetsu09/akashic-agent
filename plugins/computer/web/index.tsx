import type { ConversationTabView } from "@akashic/conversation-tools-v2";
import type { WebHostContextV1, WebUiDisposer } from "@akashic/web-ui-v1";
import { ComputerDisplay } from "./display";
import { BrowserDisplay } from "./browser-display";
import { ClipboardPopover } from "./clipboard";
import { ComputerEvents, type ComputerState } from "./computer-events";
import { present, type ActionId, type ControlState, type LinkState, type PanelFacts, type RuntimeState } from "./panel-state";
import { PanelSurface, type Target, type ToolId } from "./panel-view";
import { keysymForKey } from "./remote-input.js";
import "./style.css";

// 这三种连接结果需要用户明确处理，轮询不自动重连。
const SETTLED_LINKS: readonly LinkState[] = ["failed", "stale", "ended"];

/** 控制器：采集事实、维持连接、执行动作。呈现规则在 panel-state，DOM 在 panel-view。 */
function renderComputer(ctx: WebHostContextV1, host: HTMLElement, rawView: unknown): WebUiDisposer {
  const view = rawView as ConversationTabView;
  if (!view || typeof view.active !== "boolean" || typeof view.onActiveChange !== "function"
    || typeof view.requestAttention !== "function") throw new Error("Computer 缺少 conversation.tools.v2 view");
  const surface = new PanelSurface(id => void perform(id), id => select(id), id => void tool(id));
  const screen = surface.screen;
  const clipboard = new ClipboardPopover(surface.clipboardToggle, {
    paste: text => paste(text),
    takeControl: () => void perform("take"),
  });
  surface.stage.append(clipboard.element);
  host.replaceChildren(surface.root);

  let targets: Target[] = [];
  let selected = "desktop";
  let runtime: RuntimeState = "loading";
  let runtimeError = "";
  let wakeError = "";
  let wakeRequested = false;
  let restartRequested = false;
  // 人工打开面板时还没收到首个状态，等状态到达后再决定是否唤醒。
  let wakeOnState = false;
  let statusError = "";
  let link: LinkState = "idle";
  let linkError = "";
  let control = "";
  let busy = false;
  let agentActive = false;
  let active = view.active;
  let disposed = false;
  let lastNotice: number | null = null;
  let display: ComputerDisplay | BrowserDisplay | null = null;
  let retry = 0;
  let attempt = 0;
  const events = new ComputerEvents(ctx);
  const held = new Map<string, number>();

  function facts(): PanelFacts {
    const target = targets.find(item => item.id === selected);
    const controlState: ControlState = control ? (link === "connected" ? "mine" : "taking") : busy ? "other" : "none";
    return { runtime, runtimeError: runtimeError || wakeError, link, linkError, control: controlState, agentActive,
      targetKind: target?.kind ?? "desktop", targetTitle: target?.kind === "browser" ? target.title || target.url || "" : "",
      wakeRequested, restartRequested, statusError };
  }

  function update() {
    if (disposed) return;
    surface.render(present(facts()));
    clipboard.setCanPaste(Boolean(control) && link === "connected");
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
  }

  function inputKey(event: KeyboardEvent) {
    if (!control || link !== "connected") return;
    event.preventDefault();
    if (display instanceof BrowserDisplay) { display.key(event); return; }
    if (event.type === "keydown" && (event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "v") {
      releaseKeys();
      void navigator.clipboard.readText().then(paste).catch(error => clipboard.report(String(error)));
      return;
    }
    const keysym = keysymForKey(event.key, event.code);
    if (keysym == null) return;
    if (event.type === "keydown") held.set(event.code, keysym);
    else held.delete(event.code);
    display?.sendKey(keysym, event.code, event.type === "keydown");
  }

  async function paste(value: string) {
    if (!control || !display || link !== "connected") throw new Error("请先接管操作");
    await display.clipboardPasteFrom(value);
    if (display instanceof ComputerDisplay) {
      display.sendKey(0xffe3, "ControlLeft", true);
      display.sendKey(0x76, "KeyV", true);
      display.sendKey(0x76, "KeyV", false);
      display.sendKey(0xffe3, "ControlLeft", false);
    }
  }

  function connect(token = "") {
    if (!active || runtime !== "ready" || disposed || SETTLED_LINKS.includes(link)) return;
    const target = targets.find(item => item.id === selected);
    if (!target) return;
    disconnect();
    control = token;
    link = "connecting";
    const current = ++attempt;
    const next = target.kind === "desktop"
      ? new ComputerDisplay(screen, ctx, { keydown: inputKey, keyup: inputKey,
        blur: releaseKeys, paste: event => {
          event.preventDefault();
          void paste(event.clipboardData?.getData("text/plain") ?? "").catch(error => clipboard.report(String(error)));
        } }, events, token)
      : new BrowserDisplay(screen, ctx, selected, token);
    display = next;
    next.addEventListener("connect", () => {
      if (attempt !== current) return;
      link = "connected";
      update();
    });
    next.addEventListener("clipboard", event => clipboard.receive((event as CustomEvent<{ text: string }>).detail.text));
    next.addEventListener("error", event => {
      if (attempt !== current) return;
      link = "failed";
      linkError = (event as CustomEvent<{ reason: string }>).detail.reason;
      update();
    });
    next.addEventListener("stale", () => {
      if (attempt !== current) return;
      link = "stale";
      update();
    });
    next.addEventListener("disconnect", () => {
      if (attempt !== current) return;
      display = null;
      control = "";
      if (!SETTLED_LINKS.includes(link)) link = "lost";
      update();
      // 画面断开不会带来新的状态事件，自己隔一秒重连。
      window.clearTimeout(retry);
      retry = window.setTimeout(() => { if (link === "lost") connect(); }, 1000);
    });
    update();
  }

  function select(id: string) {
    if (id === selected && link === "connected") return;
    selected = id;
    link = "idle";
    disconnect();
    surface.renderTargets(targets, selected);
    connect();
    update();
  }

  async function perform(id: ActionId) {
    // 1. 控制权：接管、交还和取消都以新的连接身份重连，旧连接的输入立即失效。
    if (id === "take" || id === "release" || id === "cancel") {
      if (id === "take" && SETTLED_LINKS.includes(link)) link = "idle";
      connect(id === "take" ? crypto.randomUUID() : "");
    } else if (id === "reconnect") {
      link = "idle";
      connect();
    } else if (id === "home") select("desktop");
    else if (id === "reload") location.reload();
    else if (id === "copy-error") await navigator.clipboard.writeText(runtimeError || wakeError);
    else if (id === "restart") {
      // 3. 重启同样是长请求；失败时后端回到 failed 并带上原因，由状态流呈现。
      restartRequested = true;
      wakeError = "";
      update();
      try {
        const response = await ctx.http.request("/api/dashboard/computer/restart", { method: "POST" });
        if (!response.ok) wakeError = `重启请求失败（${response.status}）`;
      } catch (error) { wakeError = String(error); }
      restartRequested = false;
      update();
    } else if (id === "wake") {
      // 2. 唤醒是长请求；期间轮询会看到 starting，失败原因留在本地直到下次尝试。
      wakeRequested = true;
      wakeError = "";
      update();
      try {
        const response = await ctx.http.request("/api/dashboard/computer/wake", { method: "POST" });
        if (!response.ok) wakeError = `唤醒请求失败（${response.status}）`;
      } catch (error) { wakeError = String(error); }
      wakeRequested = false;
      update();
    }
  }

  // 状态流的每个快照：只观察；休眠需明确唤醒，通知不修改面板和观看目标。
  function apply(state: ComputerState) {
    statusError = "";
    runtime = state.browser.state as RuntimeState;
    runtimeError = state.browser.error;
    if (wakeOnState) { wakeOnState = false; wakeIfSleeping(); }
    agentActive = state.active;
    busy = state.control;
    if ((lastNotice !== null && lastNotice !== state.noticeId) || (lastNotice === null && state.active))
      view.requestAttention(`computer:${state.noticeId}`);
    lastNotice = state.noticeId;
    if (JSON.stringify(targets) !== JSON.stringify(state.targets)) {
      targets = state.targets;
      surface.renderTargets(targets, selected);
    }
    if (runtime !== "ready") {
      if (display) disconnect();
      if (link !== "stale") link = "idle";
    } else if (!targets.some(target => target.id === selected)) {
      if (display) disconnect();
      link = "ended";
    } else if (active && !display) connect();
    update();
  }

  events.addEventListener("state", event => apply((event as CustomEvent<ComputerState>).detail));
  // 读取失败与画面连接无关：不改写 link，已确定的 stale 等终态保持不变。
  events.addEventListener("error", event => {
    statusError = (event as CustomEvent<string>).detail;
    update();
  });
  events.addEventListener("stale", () => {
    disconnect();
    link = "stale";
    update();
  });

  // 头部工具：都只读当前画面或地址，不改变控制权与连接。
  async function tool(id: ToolId) {
    const url = targets.find(item => item.id === selected)?.url ?? "";
    try {
      if (id === "fullscreen")
        await (document.fullscreenElement ? document.exitFullscreen() : surface.root.requestFullscreen());
      else if (id === "open-url") window.open(url, "_blank", "noopener");
      else if (id === "copy-url") { await navigator.clipboard.writeText(url); surface.notify("已复制链接"); }
      else if (id === "shot") {
        if (!display || link !== "connected") throw new Error("画面还没连上");
        save(await display.snapshot());
        surface.notify("截图已保存");
      }
    } catch (error) { surface.notify(`${id === "shot" ? "截图" : "操作"}失败：${error instanceof Error ? error.message : String(error)}`); }
  }

  function save(blob: Blob) {
    const link = document.createElement("a");
    const stamp = new Date().toISOString().slice(0, 19).replace(/[-:]/g, "").replace("T", "-");
    link.href = URL.createObjectURL(blob);
    link.download = `computer-${stamp}.${blob.type === "image/jpeg" ? "jpg" : blob.type === "image/webp" ? "webp" : "png"}`;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  }

  screen.addEventListener("keydown", inputKey, true);
  screen.addEventListener("keyup", inputKey, true);
  // 0084：人工打开面板唤醒一次。注册时的首次回调只报告现状，不算打开；
  // 之后保持打开不会再次唤醒，休眠后由用户点"唤醒电脑"。
  function wakeIfSleeping() {
    if (runtime === "loading") wakeOnState = true;
    else if (runtime === "sleeping" && !runtimeError && !wakeError && !wakeRequested) void perform("wake");
  }

  let registered = false;
  const stopActive = view.onActiveChange(value => {
    const opened = registered && value && !active;
    active = value;
    if (!active) { disconnect(); link = SETTLED_LINKS.includes(link) ? link : "idle"; wakeOnState = false; }
    else connect();
    update();
    if (opened) wakeIfSleeping();
  });
  registered = true;
  update();
  return () => {
    disposed = true;
    events.close();
    window.clearTimeout(retry);
    stopActive();
    disconnect();
    host.replaceChildren();
  };
}

export function activate(ctx: WebHostContextV1): WebUiDisposer {
  return ctx.ui.inject("conversation.tools.v2", mount => mount.register({
    id: "computer", label: "电脑", order: 10, background: true,
    icon: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8 20h8M12 16v4"/></svg>',
    render(host, _entryView, props) { return renderComputer(ctx, host, props); },
  }));
}

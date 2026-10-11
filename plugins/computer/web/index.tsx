import type { WebHostContextV1, WebUiDisposer } from "@akashic/web-ui-v1";
import { ComputerDisplay } from "./display";
import { BrowserDisplay } from "./browser-display";
import { ClipboardPopover } from "./clipboard";
import { present, type ActionId, type ControlState, type LinkState, type PanelFacts, type RuntimeState } from "./panel-state";
import { PanelSurface, type Target } from "./panel-view";
import { keysymForKey } from "./remote-input.js";
import "./style.css";

interface ConversationTabView {
  readonly active: boolean;
  onActiveChange(listener: (active: boolean) => void): WebUiDisposer;
  requestAttention(noticeId: string): void;
}
interface Activity { noticeId: number; active: boolean; browser: { state: string; error: string } }

const RUNTIME_STATES: readonly string[] = ["sleeping", "starting", "ready", "stopping", "failed"];
// 这三种连接结果需要用户明确处理，轮询不自动重连。
const SETTLED_LINKS: readonly LinkState[] = ["failed", "stale", "ended"];

/** 控制器：采集事实、维持连接、执行动作。呈现规则在 panel-state，DOM 在 panel-view。 */
function renderComputer(ctx: WebHostContextV1, host: HTMLElement, rawView: unknown): WebUiDisposer {
  const view = rawView as ConversationTabView;
  if (!view || typeof view.active !== "boolean" || typeof view.onActiveChange !== "function"
    || typeof view.requestAttention !== "function") throw new Error("Computer 缺少 conversation.tools.v1 view");
  const surface = new PanelSurface(id => void perform(id), id => select(id));
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
  let link: LinkState = "idle";
  let linkError = "";
  let control = "";
  let busy = false;
  let agentActive = false;
  let active = view.active;
  let disposed = false;
  let lastNotice: number | null = null;
  let display: ComputerDisplay | BrowserDisplay | null = null;
  let timer = 0;
  let attempt = 0;
  let lastTouch = 0;
  const held = new Map<string, number>();

  function facts(): PanelFacts {
    const target = targets.find(item => item.id === selected);
    const controlState: ControlState = control ? (link === "connected" ? "mine" : "taking") : busy ? "other" : "none";
    return { runtime, runtimeError: runtimeError || wakeError, link, linkError, control: controlState, agentActive,
      targetKind: target?.kind ?? "desktop", targetTitle: target?.kind === "browser" ? target.title || target.url || "" : "",
      wakeRequested };
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
    touch(event);
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

  function touch(event: Event) {
    if (!event.isTrusted || !control || Date.now() - lastTouch < 1000) return;
    lastTouch = Date.now();
    const lost = () => fail("操作活动同步失败，请重新连接");
    void ctx.http.request("/api/dashboard/computer/touch", { method: "POST" })
      .then(response => { if (!response.ok) lost(); }, lost);
  }

  function fail(reason: string) {
    disconnect();
    link = "failed";
    linkError = reason;
    update();
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
        touch, blur: releaseKeys, paste: event => {
          event.preventDefault();
          void paste(event.clipboardData?.getData("text/plain") ?? "").catch(error => clipboard.report(String(error)));
        } }, token)
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
    else if (id === "wake") {
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

  // 读取一轮活动和目标目录；格式不对就整体拒绝。
  async function read(): Promise<{ activity: Activity; catalog: { targets: Target[]; control: boolean } } | "stale"> {
    const responses = await Promise.all([
      ctx.http.request("/api/dashboard/computer/activity", { cache: "no-store" }),
      ctx.http.request("/api/dashboard/computer/targets", { cache: "no-store" }),
    ]);
    if (responses.some(response => response.headers.get("X-Akashic-Web-Stale") === "1")) return "stale";
    if (responses.some(response => !response.ok)) throw new Error("Computer 状态读取失败");
    const activity = await responses[0].json() as Activity;
    const catalog = await responses[1].json() as { targets: Target[]; control: boolean };
    if (!Number.isInteger(activity.noticeId) || typeof activity.active !== "boolean"
      || !RUNTIME_STATES.includes(activity.browser?.state) || typeof activity.browser.error !== "string"
      || !Array.isArray(catalog.targets) || typeof catalog.control !== "boolean"
      || catalog.targets.some(target => !target.id || !target.label || !["desktop", "browser"].includes(target.kind)))
      throw new Error("Computer 回执无效");
    return { activity, catalog };
  }

  /** 轮询只观察；休眠需明确唤醒，通知不修改面板和观看目标。 */
  async function poll() {
    try {
      const result = await read();
      if (disposed) return;
      if (result === "stale") {
        disconnect();
        link = "stale";
        return;
      }
      const { activity, catalog } = result;
      runtime = activity.browser.state as RuntimeState;
      runtimeError = activity.browser.error;
      agentActive = activity.active;
      busy = catalog.control;
      if ((lastNotice !== null && lastNotice !== activity.noticeId) || (lastNotice === null && activity.active))
        view.requestAttention(`computer:${activity.noticeId}`);
      lastNotice = activity.noticeId;
      if (JSON.stringify(targets) !== JSON.stringify(catalog.targets)) {
        targets = catalog.targets;
        surface.renderTargets(targets, selected);
      }
      if (runtime !== "ready") {
        if (display) disconnect();
        if (link !== "stale") link = "idle";
      } else if (!targets.some(target => target.id === selected)) {
        if (display) disconnect();
        link = "ended";
      } else if (active && !display) connect();
    } catch (error) {
      if (!disposed && link !== "connected") { link = "lost"; linkError = String(error); }
    } finally {
      update();
      if (!disposed) timer = window.setTimeout(() => void poll(), 1000);
    }
  }

  surface.fullscreenToggle.onclick = () =>
    void (document.fullscreenElement ? document.exitFullscreen() : surface.root.requestFullscreen());
  screen.addEventListener("keydown", inputKey, true);
  screen.addEventListener("keyup", inputKey, true);
  const stopActive = view.onActiveChange(value => {
    active = value;
    if (!active) { disconnect(); link = SETTLED_LINKS.includes(link) ? link : "idle"; }
    else connect();
    update();
  });
  update();
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
    id: "computer", label: "电脑", order: 10,
    render(host, _entryView, props) { return renderComputer(ctx, host, props); },
  }));
}

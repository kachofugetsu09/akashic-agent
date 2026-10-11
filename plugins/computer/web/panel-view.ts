// 面板的 DOM：只把 panel-state 给出的 PanelView 画出来，并把点击转成动作 ID。不做判断。
import type { ActionId, ActionView, Icon, PanelView } from "./panel-state";

export interface Target {
  readonly id: string;
  readonly label: string;
  readonly kind: "desktop" | "browser";
  readonly title?: string;
  readonly url?: string;
}

const ICONS: Record<Icon | "mask" | "hand" | "clipboard" | "expand", string> = {
  moon: `<path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/>`,
  spinner: `<path d="M12 3a9 9 0 1 0 9 9"/>`,
  alert: `<path d="M12 4l9 16H3z"/><path d="M12 10v4M12 17.5v.01"/>`,
  closed: `<rect x="3" y="4" width="18" height="14" rx="2"/><path d="M9 9l6 6M15 9l-6 6"/>`,
  refresh: `<path d="M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7"/>`,
  unplug: `<path d="M9 3v5M15 3v5M6 8h12v3a6 6 0 0 1-12 0zM12 17v4M4 4l16 16"/>`,
  mask: `<path d="M3 10c3-1.5 6-1.5 9 0 3-1.5 6-1.5 9 0-.5 4-2.5 6-5 6-2 0-3-1.5-4-3-1 1.5-2 3-4 3-2.5 0-4.5-2-5-6z"/><path d="M7.5 12h1.5M15 12h1.5"/>`,
  hand: `<path d="M8 13V6a1.5 1.5 0 0 1 3 0v5M11 11V4.5a1.5 1.5 0 0 1 3 0V11M14 11V6a1.5 1.5 0 0 1 3 0v7a7 7 0 0 1-7 7h-.5A6.5 6.5 0 0 1 4 15.5V12a1.5 1.5 0 0 1 3 0v1"/>`,
  clipboard: `<rect x="6" y="4" width="12" height="17" rx="2"/><path d="M9 4h6v3H9z"/>`,
  expand: `<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>`,
};

export function icon(name: keyof typeof ICONS): string {
  return `<svg class="computer-icon" viewBox="0 0 24 24" aria-hidden="true">${ICONS[name]}</svg>`;
}

function element<K extends keyof HTMLElementTagNameMap>(tag: K, className: string): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  node.className = className;
  return node;
}

function iconButton(label: string, name: keyof typeof ICONS): HTMLButtonElement {
  const node = element("button", "computer-icon-button");
  node.type = "button";
  node.setAttribute("aria-label", label);
  node.title = label;
  node.innerHTML = icon(name);
  return node;
}

export class PanelSurface {
  readonly root = element("div", "computer-view");
  readonly screen = element("div", "computer-screen");
  readonly stage = element("div", "computer-stage");
  readonly clipboardToggle = iconButton("剪贴板", "clipboard");
  readonly fullscreenToggle = iconButton("全屏", "expand");
  private readonly tabs = element("div", "computer-target-tabs");
  private readonly status = element("div", "computer-status");
  private readonly title = element("strong", "computer-status-title");
  private readonly detail = element("span", "computer-status-detail");
  private readonly action = element("button", "computer-action");
  private readonly actionHint = element("small", "computer-action-hint");
  private readonly actionWrap = element("div", "computer-action-wrap");
  private readonly frame = element("div", "computer-frame");
  private readonly legend = element("div", "computer-frame-legend");
  private readonly overlay = element("div", "computer-connection-state");
  private actionId: ActionId | null = null;
  private restoreAction = false;

  constructor(private readonly onAction: (id: ActionId) => void, private readonly onSelect: (id: string) => void) {
    // 1. 头部：观看目标标签 + 工具按钮。面板的开合属于宿主，这里不放关闭按钮。
    const head = element("div", "computer-head");
    this.tabs.setAttribute("role", "tablist");
    this.tabs.setAttribute("aria-label", "电脑画面");
    const tools = element("div", "computer-actions");
    tools.append(this.clipboardToggle, this.fullscreenToggle);
    head.append(this.tabs, tools);
    // 2. 状态条：一句话说清现状 + 唯一主操作；不可用时原因写在按钮下方。
    this.status.setAttribute("role", "status");
    const dot = element("span", "computer-status-dot");
    const text = element("div", "computer-status-text");
    text.append(this.title, this.detail);
    this.action.type = "button";
    this.actionHint.id = `computer-action-hint-${crypto.randomUUID().slice(0, 8)}`;
    this.actionWrap.append(this.action, this.actionHint);
    this.status.append(dot, text, this.actionWrap);
    this.action.addEventListener("click", () => { if (this.actionId) this.onAction(this.actionId); });
    // 3. 画面：外框承载身份与控制权，标签贴在边框上；遮罩只在没有画面时出现。
    this.screen.tabIndex = 0;
    this.screen.setAttribute("role", "tabpanel");
    this.legend.setAttribute("aria-hidden", "true");
    this.frame.append(this.legend, this.screen, this.overlay);
    this.stage.append(this.frame);
    this.root.append(head, this.status, this.stage);
  }

  render(view: PanelView) {
    this.status.dataset.tone = view.tone;
    this.title.textContent = view.title;
    this.detail.textContent = view.detail;
    this.detail.hidden = !view.detail;
    this.renderAction(view.action);
    this.frame.dataset.frame = view.frame.kind;
    this.legend.innerHTML = view.frame.kind === "none" ? ""
      : `${icon(view.frame.kind === "control" ? "hand" : "mask")}<span>${escape(view.frame.legend)}</span>`;
    this.overlay.hidden = !view.overlay;
    if (view.overlay) this.renderOverlay(view.overlay);
  }

  // 主操作在重连等过渡态会短暂消失：焦点先交给画面，按钮回来时再交还，避免掉到 body。
  private renderAction(action: ActionView | null) {
    const hiding = !action && !this.actionWrap.hidden && document.activeElement === this.action;
    this.actionWrap.hidden = !action;
    this.actionId = action?.id ?? null;
    if (hiding) { this.restoreAction = true; this.screen.focus(); }
    if (!action) return;
    if (this.restoreAction) {
      this.restoreAction = false;
      if (document.activeElement === this.screen) this.action.focus();
    }
    this.action.textContent = action.label;
    this.action.dataset.emphasis = action.emphasis;
    this.action.disabled = Boolean(action.blockedBy);
    this.actionHint.textContent = action.blockedBy ?? "";
    if (action.blockedBy) this.action.setAttribute("aria-describedby", this.actionHint.id);
    else this.action.removeAttribute("aria-describedby");
  }

  // 遮罩内容按 key 复用，轮询重绘不会打断正在读的详情或丢掉按钮焦点。
  private renderOverlay(overlay: NonNullable<PanelView["overlay"]>) {
    const key = JSON.stringify(overlay);
    if (this.overlay.dataset.key === key) return;
    const hadFocus = this.overlay.contains(document.activeElement);
    this.overlay.dataset.key = key;
    const body = element("div", "computer-connection-body");
    const mark = element("span", "computer-connection-icon");
    mark.dataset.icon = overlay.icon;
    mark.innerHTML = icon(overlay.icon);
    const text = element("p", "");
    text.textContent = overlay.text;
    body.append(mark, text);
    if (overlay.actions.length) {
      const row = element("div", "computer-connection-actions");
      for (const action of overlay.actions) {
        const node = element("button", "computer-action");
        node.type = "button";
        node.textContent = action.label;
        node.dataset.emphasis = action.emphasis;
        node.addEventListener("click", () => this.onAction(action.id));
        row.append(node);
      }
      body.append(row);
    }
    if (overlay.details) {
      const details = element("details", "computer-connection-details");
      const summary = element("summary", "");
      summary.textContent = "详情";
      const pre = element("pre", "");
      pre.textContent = overlay.details;
      details.append(summary, pre);
      body.append(details);
    }
    this.overlay.replaceChildren(body);
    if (hadFocus) body.querySelector<HTMLElement>("button")?.focus();
  }

  /** 标签页：roving tabindex，左右方向键切换（ARIA tabs 模式）。匿名浏览器用面具图标标出。 */
  renderTargets(targets: readonly Target[], selected: string) {
    const focused = this.tabs.contains(document.activeElement);
    const names = targets.map(target => target.kind === "browser"
      ? target.title || shortUrl(target.url) || "空白页" : target.label);
    this.tabs.replaceChildren(...targets.map((target, index) => {
      const tab = element("button", "computer-target");
      tab.type = "button";
      tab.id = `computer-target-${index}`;
      tab.title = target.url ? `${target.label}\n${target.url}` : target.label;
      tab.setAttribute("role", "tab");
      tab.setAttribute("aria-selected", String(target.id === selected));
      tab.tabIndex = target.id === selected ? 0 : -1;
      const anonymous = target.kind === "browser";
      if (anonymous) tab.dataset.anonymous = "";
      // 两个匿名浏览器停在同名页面时，补上浏览器编号（与 runtime label 中的编号一致）。
      const twin = anonymous && names.some((name, other) => other !== index && name === names[index]);
      const browser = twin ? `<span class="computer-target-tag">${escape(target.id.slice(0, 4))}</span>` : "";
      tab.innerHTML = `${anonymous ? icon("mask") : ""}<span class="computer-target-label">${escape(names[index])}</span>${browser}`
        + (anonymous ? `<span class="computer-visually-hidden">（匿名浏览器）</span>` : "");
      if (target.id === selected) this.screen.setAttribute("aria-labelledby", tab.id);
      tab.addEventListener("click", () => this.onSelect(target.id));
      tab.addEventListener("keydown", event => {
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
        event.preventDefault();
        const next = event.key === "Home" ? 0 : event.key === "End" ? targets.length - 1
          : (index + (event.key === "ArrowRight" ? 1 : -1) + targets.length) % targets.length;
        this.onSelect(targets[next].id);
      });
      return tab;
    }));
    if (focused) this.tabs.querySelector<HTMLElement>('[aria-selected="true"]')?.focus();
  }
}

// 没有标题的匿名页只显示主机和路径，协议头对辨认页面没有帮助。
function shortUrl(url = ""): string {
  return url.replace(/^[a-z]+:\/\//i, "").replace(/\/$/, "");
}

function escape(text: string): string {
  return text.replace(/[&<>"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;" })[char]!);
}

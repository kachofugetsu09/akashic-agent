// 面板的 DOM：只把 panel-state 给出的 PanelView 画出来，并把点击转成动作 ID。不做判断。
import type { ActionId, ActionView, Icon, PanelView } from "./panel-state";

export interface Target {
  readonly id: string;
  readonly label: string;
  readonly kind: "desktop" | "browser";
  readonly title?: string;
  readonly url?: string;
}

export type ToolId = "shot" | "fullscreen" | "copy-url" | "open-url";

const ICONS: Record<Icon | "mask" | "hand" | "clipboard" | "expand" | "camera" | "more" | "copy" | "external", string> = {
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
  camera: `<path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/>`,
  more: `<path d="M12 5v.01M12 12v.01M12 19v.01"/>`,
  copy: `<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/>`,
  external: `<path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/>`,
};

export function icon(name: keyof typeof ICONS): string {
  return `<svg class="computer-icon" viewBox="0 0 24 24" aria-hidden="true">${ICONS[name]}</svg>`;
}

function element<K extends keyof HTMLElementTagNameMap>(tag: K, className: string): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  node.className = className;
  return node;
}

function iconButton(label: string, name: keyof typeof ICONS, className = ""): HTMLButtonElement {
  const node = element("button", `computer-icon-button ${className}`.trim());
  node.type = "button";
  node.setAttribute("aria-label", label);
  node.title = label;
  node.innerHTML = icon(name);
  return node;
}

const MENU: readonly { readonly id: ToolId; readonly label: string; readonly anonymous?: boolean }[] = [
  { id: "shot", label: "保存截图" },
  { id: "fullscreen", label: "全屏" },
  { id: "copy-url", label: "复制链接", anonymous: true },
  { id: "open-url", label: "在你的浏览器打开", anonymous: true },
];

export class PanelSurface {
  readonly root = element("div", "computer-view");
  readonly screen = element("div", "computer-screen");
  readonly stage = element("div", "computer-stage");
  readonly clipboardToggle = iconButton("剪贴板", "clipboard");
  private readonly target = element("div", "computer-target");
  private readonly picker = element("select", "computer-target-select");
  private readonly address = element("div", "computer-address");
  private readonly addressText = element("span", "computer-address-text");
  private readonly moreToggle = iconButton("更多", "more", "computer-more");
  private readonly menu = element("div", "computer-menu");
  private readonly status = element("div", "computer-status");
  private readonly title = element("strong", "computer-status-title");
  private readonly detail = element("span", "computer-status-detail");
  private readonly action = element("button", "computer-action");
  private readonly frame = element("div", "computer-frame");
  private readonly overlay = element("div", "computer-connection-state");
  private readonly toast = element("div", "computer-toast");
  private actionId: ActionId | null = null;
  private restoreAction = false;
  private toastTimer = 0;

  constructor(private readonly onAction: (id: ActionId) => void, private readonly onSelect: (id: string) => void,
    private readonly onTool: (id: ToolId) => void) {
    // 1. 头部一行：观看目标、匿名页地址、工具按钮。面板开合属于宿主，这里不放关闭按钮。
    const head = element("div", "computer-head");
    this.picker.setAttribute("aria-label", "观看的画面");
    this.picker.addEventListener("change", () => this.onSelect(this.picker.value));
    this.target.append(this.picker);
    this.addressText.setAttribute("dir", "ltr");
    this.address.append(this.addressText, this.tool("copy-url", "复制链接", "copy"), this.tool("open-url", "在你的浏览器打开", "external"));
    const tools = element("div", "computer-actions");
    tools.append(this.tool("shot", "保存截图", "camera", "computer-wide"), this.clipboardToggle,
      this.tool("fullscreen", "全屏", "expand", "computer-wide"), this.moreToggle);
    head.append(this.target, this.address, tools, this.buildMenu());
    // 2. 画面：状态胶囊压在外框上沿，说明谁在操作并给出唯一主操作；遮罩只在没有画面时出现。
    this.status.setAttribute("role", "status");
    const dot = element("span", "computer-status-dot");
    this.action.type = "button";
    this.action.addEventListener("click", () => { if (this.actionId) this.onAction(this.actionId); });
    this.status.append(dot, this.title, this.detail, this.action);
    this.screen.tabIndex = 0;
    this.screen.setAttribute("role", "region");
    this.screen.setAttribute("aria-label", "电脑画面");
    this.toast.setAttribute("aria-live", "polite");
    this.frame.append(this.screen, this.overlay);
    this.stage.append(this.frame, this.status, this.toast);
    this.root.append(head, this.stage);
  }

  private tool(id: ToolId, label: string, name: keyof typeof ICONS, className = ""): HTMLButtonElement {
    const node = iconButton(label, name, className);
    node.addEventListener("click", () => this.onTool(id));
    return node;
  }

  // 窄宽度时收起的工具都在这里；菜单只做打开、关闭和方向键移动，动作仍交给 onTool。
  private buildMenu(): HTMLElement {
    this.menu.id = `computer-menu-${crypto.randomUUID().slice(0, 8)}`;
    this.menu.setAttribute("role", "menu");
    this.menu.hidden = true;
    for (const item of MENU) {
      const node = element("button", "computer-menu-item");
      node.type = "button";
      node.textContent = item.label;
      node.setAttribute("role", "menuitem");
      node.tabIndex = -1;
      if (item.anonymous) node.dataset.anonymous = "";
      node.addEventListener("click", () => { this.closeMenu(true); this.onTool(item.id); });
      this.menu.append(node);
    }
    this.moreToggle.setAttribute("aria-haspopup", "menu");
    this.moreToggle.setAttribute("aria-controls", this.menu.id);
    this.moreToggle.setAttribute("aria-expanded", "false");
    this.moreToggle.addEventListener("click", () => this.menu.hidden ? this.openMenu() : this.closeMenu(true));
    this.menu.addEventListener("keydown", event => {
      const items = [...this.menu.querySelectorAll<HTMLElement>(".computer-menu-item:not([hidden])")];
      const index = items.indexOf(document.activeElement as HTMLElement);
      if (event.key === "Escape") this.closeMenu(true);
      else if (event.key === "ArrowDown" || event.key === "ArrowUp")
        items[(index + (event.key === "ArrowDown" ? 1 : -1) + items.length) % items.length]?.focus();
      else return;
      event.preventDefault();
    });
    this.menu.addEventListener("focusout", event => {
      if (!this.menu.contains(event.relatedTarget as Node | null) && event.relatedTarget !== this.moreToggle) this.closeMenu(false);
    });
    return this.menu;
  }

  private openMenu() {
    this.menu.hidden = false;
    this.moreToggle.setAttribute("aria-expanded", "true");
    this.menu.querySelector<HTMLElement>(".computer-menu-item:not([hidden])")?.focus();
  }

  private closeMenu(returnFocus: boolean) {
    if (this.menu.hidden) return;
    this.menu.hidden = true;
    this.moreToggle.setAttribute("aria-expanded", "false");
    if (returnFocus) this.moreToggle.focus();
  }

  // 截图、复制等一次性结果的短提示；读屏经 aria-live 播报。
  notify(text: string) {
    this.toast.textContent = text;
    window.clearTimeout(this.toastTimer);
    this.toastTimer = window.setTimeout(() => { this.toast.textContent = ""; }, 2400);
  }

  render(view: PanelView) {
    this.status.dataset.tone = view.tone;
    this.status.classList.toggle("computer-visually-hidden", Boolean(view.overlay));
    this.title.textContent = view.title;
    this.detail.textContent = view.action?.blockedBy ?? view.detail;
    this.detail.hidden = !this.detail.textContent;
    this.status.title = this.detail.textContent;
    this.renderAction(view.action);
    this.frame.dataset.frame = view.frame.kind;
    this.frame.toggleAttribute("data-agent", view.tone === "busy" && !view.overlay);
    this.overlay.hidden = !view.overlay;
    if (view.overlay) this.renderOverlay(view.overlay);
  }

  // 主操作在重连等过渡态会短暂消失：焦点先交给画面，按钮回来时再交还，避免掉到 body。
  private renderAction(action: ActionView | null) {
    const hiding = !action && !this.action.hidden && document.activeElement === this.action;
    this.action.hidden = !action;
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

  // 观看目标用原生下拉：键盘、读屏与触屏都由平台处理。只有主桌面时显示为纯文字。
  renderTargets(targets: readonly Target[], selected: string) {
    const names = targets.map(target => target.kind === "browser"
      ? target.title || shortUrl(target.url) || "空白页" : target.label);
    this.picker.replaceChildren(...targets.map((target, index) => {
      // 两个匿名浏览器停在同名页面时，补上浏览器编号（与 runtime label 中的编号一致）。
      const twin = target.kind === "browser" && names.some((name, other) => other !== index && name === names[index]);
      const option = element("option", "");
      option.value = target.id;
      option.textContent = (target.kind === "browser" ? "匿名 · " : "") + names[index] + (twin ? ` · ${target.id.slice(0, 4)}` : "");
      option.selected = target.id === selected;
      return option;
    }));
    const current = targets.find(target => target.id === selected);
    const anonymous = current?.kind === "browser";
    this.picker.disabled = targets.length < 2;
    this.target.toggleAttribute("data-anonymous", anonymous);
    this.target.querySelector(".computer-icon")?.remove();
    if (anonymous) this.target.insertAdjacentHTML("afterbegin", icon("mask"));
    this.target.title = anonymous ? "匿名浏览器：不带你的登录" : "";
    this.address.hidden = !anonymous || !current.url;
    this.addressText.textContent = shortUrl(current?.url);
    this.address.title = current?.url ?? "";
    for (const item of this.menu.querySelectorAll<HTMLElement>("[data-anonymous]")) item.hidden = this.address.hidden;
  }
}

// 没有标题的匿名页只显示主机和路径，协议头对辨认页面没有帮助。
function shortUrl(url = ""): string {
  return url.replace(/^[a-z]+:\/\//i, "").replace(/\/$/, "");
}

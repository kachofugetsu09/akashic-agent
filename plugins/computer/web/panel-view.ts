// 面板的 DOM：只把 panel-state 给出的 PanelView 画出来，并把点击转成动作 ID。不做判断。
import type { ActionId, ActionView, Icon, PanelView } from "./panel-state";

/** 浏览器里的一个标签；front 是屏幕最前、正在画面里的那个。 */
export interface Target {
  readonly id: string;
  readonly title: string;
  readonly url: string;
  readonly anonymous: boolean;
  readonly agent: boolean;
  readonly turn: string | null;
  readonly front: boolean;
}

export type ToolId = "shot" | "fullscreen" | "copy-url" | "open-url";
/** 人工导航：地址栏打开、后退、前进、刷新，以及新开和关闭标签。 */
export type NavAction = "go" | "back" | "forward" | "reload" | "open" | "close";

const ICONS: Record<Icon | "mask" | "hand" | "clipboard" | "expand" | "camera" | "more" | "copy" | "external"
  | "back" | "forward" | "plus" | "close", string> = {
  moon: `<path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/>`,
  spinner: `<path d="M12 3a9 9 0 1 0 9 9"/>`,
  alert: `<path d="M12 4l9 16H3z"/><path d="M12 10v4M12 17.5v.01"/>`,
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
  back: `<path d="M19 12H5M11 6l-6 6 6 6"/>`,
  forward: `<path d="M5 12h14M13 6l6 6-6 6"/>`,
  plus: `<path d="M12 5v14M5 12h14"/>`,
  close: `<path d="M7 7l10 10M17 7L7 17"/>`,
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

const MENU: readonly { readonly id: ToolId; readonly label: string; readonly link?: boolean }[] = [
  { id: "shot", label: "保存截图" },
  { id: "fullscreen", label: "全屏" },
  { id: "copy-url", label: "复制链接", link: true },
  { id: "open-url", label: "在你的浏览器打开", link: true },
];

export class PanelSurface {
  readonly root = element("div", "computer-view");
  readonly screen = element("div", "computer-screen");
  readonly stage = element("div", "computer-stage");
  readonly clipboardToggle = iconButton("剪贴板", "clipboard");
  private readonly tabs = element("div", "computer-tabs");
  private readonly newTab = iconButton("新建标签页", "plus", "computer-new-tab");
  private readonly back = iconButton("后退（Alt+←）", "back");
  private readonly forward = iconButton("前进（Alt+→）", "forward");
  private readonly reload = iconButton("刷新", "refresh");
  private readonly address = element("form", "computer-address");
  private readonly addressInput = element("input", "computer-address-input");
  // 没在编辑时盖在输入框上：协议和路径淡、主机亮，一眼认出是哪个站（参考 Cursor）。
  private readonly addressShown = element("span", "computer-address-shown");
  private current: Target | undefined;
  private navEnabled = true;
  private spins = 0;
  private nudgeTimer = 0;
  private readonly moreToggle = iconButton("更多", "more", "computer-more");
  private readonly menu = element("div", "computer-menu");
  private readonly status = element("div", "computer-status");
  private readonly title = element("strong", "computer-status-title");
  private readonly detail = element("span", "computer-status-detail");
  private readonly action = element("button", "computer-action");
  private readonly frame = element("div", "computer-frame");
  private readonly overlay = element("div", "computer-connection-state");
  // 空白页上的引导：远端 about:blank 是一整块白，参考 Cursor 的欢迎页告诉人可以做什么。不挡点击。
  private readonly blank = element("div", "computer-blank");
  private readonly toast = element("div", "computer-toast");
  private actionId: ActionId | null = null;
  private restoreAction = false;
  private toastTimer = 0;

  constructor(private readonly onAction: (id: ActionId) => void, private readonly onSelect: (id: string) => void,
    private readonly onTool: (id: ToolId) => void, private readonly onNav: (action: NavAction, value?: string) => void) {
    // 1. 头部两行（参考 Cursor）：标签条与全屏；下面一行是导航、地址栏和工具。面板开合属于宿主，这里不放关闭按钮。
    const head = element("div", "computer-head");
    this.tabs.setAttribute("role", "tablist");
    this.tabs.setAttribute("aria-label", "浏览器标签");
    this.tabs.addEventListener("keydown", event => this.moveTab(event));
    this.newTab.addEventListener("click", () => this.onNav("open"));
    const fullscreen = this.tool("fullscreen", "全屏", "expand", "computer-wide computer-fullscreen");
    const tools = element("div", "computer-actions");
    tools.append(this.tool("shot", "保存截图", "camera", "computer-wide"), this.clipboardToggle, this.moreToggle);
    const bar = element("div", "computer-bar");
    this.back.addEventListener("click", () => this.onNav("back"));
    this.forward.addEventListener("click", () => this.onNav("forward"));
    this.reload.addEventListener("click", () => {
      // 刷新图标转一圈，确认按下去了；远端加载进度拿不到，不假装有进度条。
      const svg = this.reload.querySelector<SVGElement>(".computer-icon");
      if (svg) svg.style.rotate = `${++this.spins * 360}deg`;
      this.onNav("reload");
    });
    bar.append(this.back, this.forward, this.reload, this.buildAddress(), tools);
    head.append(this.tabs, this.newTab, fullscreen, this.buildMenu(), bar);
    // 2. 画面贴边铺满；状态胶囊浮在画面上沿，说明谁在操作并给出唯一主操作；遮罩只在没有画面时出现。
    this.status.setAttribute("role", "status");
    const dot = element("span", "computer-status-dot");
    this.action.type = "button";
    this.action.addEventListener("click", () => { if (this.actionId) this.onAction(this.actionId); });
    this.status.append(dot, this.title, this.detail, this.action);
    this.screen.tabIndex = 0;
    this.screen.setAttribute("role", "region");
    this.screen.setAttribute("aria-label", "浏览器画面");
    this.toast.setAttribute("aria-live", "polite");
    this.blank.setAttribute("aria-hidden", "true");
    this.blank.innerHTML = `<svg class="computer-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/>`
      + `<path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/></svg><strong>新标签页</strong>`
      + `<span>在上方输入网址或搜索，也可以让 Agent 帮你打开</span>`;
    this.frame.append(this.screen, this.blank, this.overlay);
    this.stage.append(this.frame, this.status, this.toast);
    this.root.append(head, this.stage);
  }

  private tool(id: ToolId, label: string, name: keyof typeof ICONS, className = ""): HTMLButtonElement {
    const node = iconButton(label, name, className);
    node.addEventListener("click", () => this.onTool(id));
    return node;
  }

  // 地址栏：显示完整地址；聚焦时全选，回车打开，Esc 放弃修改。没在编辑时随当前标签更新。
  private buildAddress(): HTMLFormElement {
    const input = this.addressInput;
    input.type = "text";
    input.dir = "ltr";
    input.spellcheck = false;
    input.autocomplete = "off";
    input.enterKeyHint = "go";
    input.placeholder = "搜索或输入网址";
    input.setAttribute("aria-label", "地址和搜索栏");
    // 点进来时全选；紧随其后的 mouseup 不能把选区清掉。
    let selecting = false;
    input.addEventListener("focus", () => { input.select(); selecting = true; });
    input.addEventListener("mouseup", event => { if (selecting) event.preventDefault(); selecting = false; });
    input.addEventListener("blur", () => { input.value = this.addressValue(); selecting = false; });
    input.addEventListener("keydown", event => {
      if (event.isComposing) {
        if (event.key === "Enter") event.preventDefault();
        return;
      }
      if (event.key !== "Escape") return;
      event.preventDefault();
      input.value = this.addressValue();
      this.screen.focus();
    });
    this.address.addEventListener("submit", event => {
      event.preventDefault();
      const value = input.value.trim();
      if (!value) return;
      this.onNav("go", value);
      this.screen.focus();
    });
    this.addressShown.setAttribute("aria-hidden", "true");
    const field = element("span", "computer-address-field");
    field.append(input, this.addressShown);
    this.address.append(field, this.tool("copy-url", "复制链接", "copy"), this.tool("open-url", "在你的浏览器打开", "external"));
    return this.address;
  }

  private addressValue(): string {
    const url = this.current?.url ?? "";
    return url === "about:blank" ? "" : url;
  }

  focusAddress() {
    this.addressInput.focus();
  }

  // Agent 正在用浏览器时导航也要先接管：按钮置灰，悬停说明原因。
  renderNav(enabled: boolean) {
    this.navEnabled = enabled;
    const reason = enabled ? "" : "Agent 正在使用浏览器，先接管";
    for (const node of [this.back, this.forward, this.reload, this.newTab, this.addressInput]) {
      node.disabled = !enabled;
      if (node !== this.addressInput) node.title = reason || node.getAttribute("aria-label") || "";
    }
    this.addressInput.title = reason;
    for (const close of this.tabs.querySelectorAll<HTMLButtonElement>(".computer-tab-close")) close.disabled = !enabled;
  }

  // 画面被挡住时有人点它：闪一下状态胶囊，告诉他要先接管（参考 Cursor 的锁定反馈）。
  nudge() {
    for (const node of [this.status, this.frame]) node.classList.add("computer-nudge");
    window.clearTimeout(this.nudgeTimer);
    this.nudgeTimer = window.setTimeout(() => {
      for (const node of [this.status, this.frame]) node.classList.remove("computer-nudge");
    }, 120);
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
      if (item.link) node.dataset.link = "";
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
    // 只隐藏胶囊、不改留白：画面尺寸一变，远端屏幕就跟着变，Agent 的截图坐标会失效。
    this.status.hidden = view.direct;
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

  // 标签条按 front 高亮；切换只是把那个标签提到屏幕最前，画面连接不断开，像本地切标签一样。
  renderTargets(targets: readonly Target[], selected: string) {
    const focused = this.tabs.contains(document.activeElement)
      ? (document.activeElement as HTMLElement).dataset.id : undefined;
    this.tabs.replaceChildren(...targets.map(target => {
      const tab = element("div", "computer-tab");
      tab.dataset.id = target.id;
      tab.setAttribute("role", "tab");
      tab.setAttribute("aria-selected", String(target.id === selected));
      tab.tabIndex = target.id === selected ? 0 : -1;
      tab.toggleAttribute("data-anonymous", target.anonymous);
      tab.toggleAttribute("data-agent", target.agent);
      const name = target.url === "about:blank" || !target.url ? "新标签页" : target.title || shortUrl(target.url);
      tab.title = `${target.anonymous ? "匿名 · 不带你的登录\n" : target.agent ? "Agent 打开的标签\n" : ""}${name}`;
      if (target.anonymous) tab.insertAdjacentHTML("afterbegin", icon("mask"));
      else if (target.agent) tab.append(element("span", "computer-tab-agent"));
      const label = element("span", "computer-tab-label");
      label.textContent = name;
      const close = iconButton(`关闭 ${name}`, "close", "computer-tab-close");
      close.tabIndex = -1;
      close.disabled = !this.navEnabled;
      close.addEventListener("click", event => { event.stopPropagation(); this.onNav("close", target.id); });
      tab.append(label, close);
      tab.addEventListener("click", () => this.onSelect(target.id));
      // 中键关闭，和本地浏览器一样。
      tab.addEventListener("auxclick", event => { if (event.button === 1 && this.navEnabled) this.onNav("close", target.id); });
      return tab;
    }));
    if (focused) (this.tabs.querySelector<HTMLElement>(`[data-id="${CSS.escape(focused)}"]`)
      ?? this.tabs.querySelector<HTMLElement>("[aria-selected=true]"))?.focus();
    this.tabs.querySelector<HTMLElement>("[aria-selected=true]")?.scrollIntoView({ block: "nearest", inline: "nearest" });
    this.current = targets.find(target => target.id === selected);
    const blank = !this.addressValue();
    this.blank.hidden = !this.current || !blank;
    this.address.toggleAttribute("data-anonymous", Boolean(this.current?.anonymous));
    if (document.activeElement !== this.addressInput) this.addressInput.value = this.addressValue();
    this.renderShownAddress();
    for (const node of this.address.querySelectorAll<HTMLElement>(".computer-icon-button")) node.hidden = blank;
    for (const item of this.menu.querySelectorAll<HTMLElement>("[data-link]")) item.hidden = blank;
  }

  // 地址拆成协议、主机、其余三段；解析不了的原样显示。
  private renderShownAddress() {
    const value = this.addressValue();
    const match = /^([a-z][a-z0-9+.-]*:\/\/)([^/?#]*)(.*)$/i.exec(value);
    const parts = match ? [["dim", match[1]], ["host", match[2]], ["dim", match[3]]] : [["host", value]];
    this.addressShown.replaceChildren(...parts.filter(([, text]) => text).map(([kind, text]) => {
      const part = element("span", `computer-address-${kind}`);
      part.textContent = text;
      return part;
    }));
  }

  // 标签条是一组 tab：左右键在标签间移动并直接切换，Home/End 到两端，Delete 关闭。
  private moveTab(event: KeyboardEvent) {
    const tabs = [...this.tabs.querySelectorAll<HTMLElement>(".computer-tab")];
    const index = tabs.indexOf(document.activeElement as HTMLElement);
    if (event.key === "Delete" && index >= 0 && this.navEnabled) {
      event.preventDefault();
      this.onNav("close", tabs[index].dataset.id!);
      return;
    }
    const next = { ArrowRight: index + 1, ArrowLeft: index - 1, Home: 0, End: tabs.length - 1 }[event.key];
    if (next === undefined || index < 0 || !tabs.length) return;
    event.preventDefault();
    const tab = tabs[(next + tabs.length) % tabs.length];
    tab.focus();
    this.onSelect(tab.dataset.id!);
  }
}

// 没有标题的匿名页只显示主机和路径，协议头对辨认页面没有帮助。
function shortUrl(url = ""): string {
  return url.replace(/^[a-z]+:\/\//i, "").replace(/\/$/, "");
}

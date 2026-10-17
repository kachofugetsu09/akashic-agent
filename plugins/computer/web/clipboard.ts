// 剪贴板浮层：只拥有自己的开合、焦点和回执。粘贴与接管由面板通过 ClipboardPort 提供。

export interface ClipboardPort {
  paste(text: string): Promise<void>;
  takeControl(): void;
}

function button(label: string, className = ""): HTMLButtonElement {
  const node = document.createElement("button");
  node.type = "button";
  node.textContent = label;
  if (className) node.className = className;
  return node;
}

export class ClipboardPopover {
  readonly element = document.createElement("section");
  private readonly locked = document.createElement("div");
  private readonly editor = document.createElement("div");
  private readonly text = document.createElement("textarea");
  private readonly receipt = document.createElement("p");
  private readonly take = button("接管操作", "is-primary");
  private canPaste = false;

  constructor(private readonly toggle: HTMLButtonElement, private readonly port: ClipboardPort) {
    this.element.className = "computer-clipboard";
    this.element.hidden = true;
    this.element.id = `computer-clipboard-${crypto.randomUUID().slice(0, 8)}`;
    this.element.setAttribute("role", "dialog");
    this.element.setAttribute("aria-label", "剪贴板");
    toggle.setAttribute("aria-controls", this.element.id);
    toggle.setAttribute("aria-expanded", "false");
    const header = document.createElement("div");
    header.className = "computer-clipboard-header";
    const title = document.createElement("strong");
    title.textContent = "剪贴板";
    const close = button("", "computer-icon-button");
    close.setAttribute("aria-label", "关闭剪贴板");
    close.innerHTML = `<svg class="computer-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>`;
    header.append(title, close);
    // 1. Agent 在用、不能输入：说明原因并直接给出接管入口，而不是一排禁用按钮。
    const why = document.createElement("p");
    why.textContent = "Agent 正在使用浏览器。接管后才能把文字粘贴进去，或把浏览器里复制的内容拿回本机。";
    this.locked.className = "computer-clipboard-locked";
    this.locked.append(why, this.take);
    // 2. 已接管：文字框 + 两个方向的动作 + 就地回执。
    this.text.maxLength = 65536;
    this.text.rows = 5;
    this.text.setAttribute("aria-label", "剪贴板文字");
    const send = button("粘贴到浏览器", "is-primary");
    const copy = button("复制到本机");
    this.receipt.className = "computer-clipboard-status";
    this.receipt.setAttribute("role", "status");
    const footer = document.createElement("div");
    footer.className = "computer-clipboard-footer";
    footer.append(this.receipt, copy, send);
    this.editor.append(this.text, footer);
    this.element.append(header, this.locked, this.editor);

    toggle.addEventListener("click", () => this.setOpen(this.element.hidden));
    close.addEventListener("click", () => this.setOpen(false));
    this.take.addEventListener("click", () => port.takeControl());
    send.addEventListener("click", () => void this.port.paste(this.text.value).then(
      () => this.say("已粘贴"), (error: unknown) => this.say(message(error))));
    copy.addEventListener("click", () => void navigator.clipboard.writeText(this.text.value).then(
      () => this.say("已复制到本机"), (error: unknown) => this.say(message(error))));
    this.element.addEventListener("keydown", event => {
      if (event.key !== "Escape") return;
      event.stopPropagation();
      this.setOpen(false);
    });
    this.editor.hidden = true;
    toggle.title = "剪贴板 · 接管后可用";
  }

  // 每轮轮询都会调用；只在权限切换、且原焦点所在的区域被隐藏时才移动焦点。
  setCanPaste(value: boolean) {
    if (value === this.canPaste) return;
    const lost = value ? this.locked : this.editor;
    const orphaned = lost.contains(document.activeElement);
    this.canPaste = value;
    this.locked.hidden = value;
    this.editor.hidden = !value;
    this.toggle.title = value ? "剪贴板" : "剪贴板 · 接管后可用";
    if (orphaned && !this.element.hidden) this.focusInside();
  }

  /** 浏览器里复制的新内容。 */
  receive(text: string) {
    this.text.value = text;
  }

  /** 面板外发生的剪贴板错误（例如在画面里按 Ctrl+V），在浮层里就地说明。 */
  report(text: string) {
    this.setOpen(true);
    this.say(text);
  }

  setOpen(open: boolean) {
    if (open === !this.element.hidden) return;
    this.element.hidden = !open;
    this.toggle.setAttribute("aria-expanded", String(open));
    this.say("");
    if (open) this.focusInside();
    else this.toggle.focus();
  }

  private focusInside() {
    (this.canPaste ? this.text : this.take).focus();
  }

  private say(text: string) {
    this.receipt.textContent = text;
  }
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

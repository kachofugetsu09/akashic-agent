// 剪贴板浮层：只拥有自己的开合、焦点和回执。粘贴由面板通过 ClipboardPort 提供。

export interface ClipboardPort {
  paste(text: string): Promise<void>;
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
  private readonly editor = document.createElement("div");
  private readonly text = document.createElement("textarea");
  private readonly receipt = document.createElement("p");

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
    // 文字框 + 两个方向的动作 + 就地回执。
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
    this.element.append(header, this.editor);

    toggle.addEventListener("click", () => this.setOpen(this.element.hidden));
    close.addEventListener("click", () => this.setOpen(false));
    send.addEventListener("click", () => void this.port.paste(this.text.value).then(
      () => this.say("已粘贴"), (error: unknown) => this.say(message(error))));
    copy.addEventListener("click", () => void navigator.clipboard.writeText(this.text.value).then(
      () => this.say("已复制到本机"), (error: unknown) => this.say(message(error))));
    this.element.addEventListener("keydown", event => {
      if (event.key !== "Escape") return;
      event.stopPropagation();
      this.setOpen(false);
    });
    toggle.title = "剪贴板";
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
    if (open) this.text.focus();
    else this.toggle.focus();
  }

  private say(text: string) {
    this.receipt.textContent = text;
  }
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

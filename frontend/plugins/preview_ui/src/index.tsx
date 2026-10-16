import type { ConversationIntent, ConversationIntentStatus, ConversationTabView } from "@akashic/conversation-tools-v2";
import type { WebHostContextV1, WebUiDisposer } from "@akashic/web-ui-v1";
import "./style.css";

/** artifact.open.v1 的 payload：{ message_id, artifact_id }，附件必须属于当前会话里的这条消息。 */
const OPEN_INTENT = "artifact.open.v1";
const LINE_LIMIT = 5000;
const ROW_LIMIT = 1000;

interface Preview { artifact_id: string; name: string; size_bytes: number; complete: boolean; text: string }
type View = "table" | "text";

function element<K extends keyof HTMLElementTagNameMap>(tag: K, className: string, text = ""): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = text;
  return node;
}

function button(className: string, text: string, onClick: () => void): HTMLButtonElement {
  const node = element("button", className, text);
  node.type = "button";
  node.addEventListener("click", onClick);
  return node;
}

function size(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function delimiterOf(name: string): string {
  return /\.csv$/iu.test(name) ? "," : /\.tsv$/iu.test(name) ? "\t" : "";
}

// 截断的前缀丢掉最后半行，避免把被切开的一行当成完整内容。
function linesOf(preview: Preview): string[] {
  const lines = preview.text.split("\n");
  if (!preview.complete && lines.length > 1) lines.pop();
  else if (lines.at(-1) === "") lines.pop();
  return lines;
}

// RFC 4180：引号字段里可以有分隔符、换行和成对的 ""；只取前 limit 条记录。
function readRows(text: string, delimiter: string, limit: number): string[][] {
  const rows: string[][] = [];
  let row: string[] = [], field = "", quoted = false, started = false;
  const endRow = () => {
    if (started) { row.push(field); rows.push(row); }
    row = []; field = ""; started = false;
  };
  for (let index = 0; index < text.length && rows.length < limit; index += 1) {
    const character = text[index];
    if (quoted) {
      if (character === '"' && text[index + 1] === '"') { field += '"'; index += 1; }
      else if (character === '"') quoted = false;
      else field += character;
    } else if (character === '"') quoted = started = true;
    else if (character === delimiter) { row.push(field); field = ""; started = true; }
    else if (character === "\n" || character === "\r") {
      if (character === "\r" && text[index + 1] === "\n") continue;
      endRow();
    } else { field += character; started = true; }
  }
  if (rows.length < limit) endRow();
  return rows;
}

function textBody(preview: Preview): HTMLElement {
  const lines = linesOf(preview);
  const body = element("pre", "preview-text");
  body.append(...lines.slice(0, LINE_LIMIT).map((line) => element("span", "preview-line", `${line}\n`)));
  return body;
}

function tableBody(preview: Preview, delimiter: string): HTMLElement {
  // 截断的前缀里最后一条记录可能不完整，不显示。
  const rows = readRows(preview.text, delimiter, ROW_LIMIT + 2);
  if (!preview.complete && rows.length > 1) rows.pop();
  const [head = [], ...records] = rows;
  const table = element("table", "preview-table");
  const headRow = table.createTHead().insertRow();
  for (const name of head) headRow.append(element("th", "", name));
  const tbody = table.createTBody();
  for (const record of records.slice(0, ROW_LIMIT)) {
    const tr = tbody.insertRow();
    for (const value of record) {
      // 单元格限宽省略时，悬停可看全文；原文视图能看到完整内容。
      const cell = element("td", "", value);
      if (value.length > 40) cell.title = value;
      tr.append(cell);
    }
  }
  const wrap = element("div", "preview-grid");
  wrap.append(table);
  return wrap;
}

function note(preview: Preview, view: View): string {
  const parts = [size(preview.size_bytes)];
  if (view === "table") {
    const rows = readRows(preview.text, delimiterOf(preview.name), Number.POSITIVE_INFINITY).length - 1;
    parts.push(preview.complete ? `${Math.max(0, rows)} 行数据` : "只显示开头");
    if (rows > ROW_LIMIT) parts.push(`表格只排前 ${ROW_LIMIT} 行`);
  } else {
    const lines = linesOf(preview).length;
    parts.push(preview.complete ? `${lines} 行` : "只显示开头");
    if (lines > LINE_LIMIT) parts.push(`只排前 ${LINE_LIMIT} 行`);
  }
  return parts.join(" · ");
}

class PreviewTab {
  readonly root = element("div", "preview-root");
  private readonly head = element("div", "preview-head");
  private readonly body = element("div", "preview-body");
  private preview: Preview | null = null;
  private view: View = "text";
  private loading = 0;

  constructor(private readonly ctx: WebHostContextV1, private session: string | null) {
    this.body.tabIndex = 0;
    this.body.setAttribute("aria-label", "附件内容");
    this.root.append(this.head, this.body);
    this.empty();
  }

  setSession(session: string | null): void {
    if (session === this.session) return;
    this.session = session;
    this.loading += 1;
    this.preview = null;
    this.empty();
  }

  // 意图在宿主激活本标签后投递；读不到时在标签里说明原因，只有请求本身失败才回执 failed 让聊天端回到弹窗。
  async open(intent: ConversationIntent): Promise<ConversationIntentStatus> {
    const payload = intent.payload as { message_id?: unknown; artifact_id?: unknown } | null;
    const message = payload?.message_id, artifact = payload?.artifact_id;
    if (typeof message !== "string" || !message || typeof artifact !== "string" || !artifact) return "invalid";
    this.session = intent.session;
    const ticket = ++this.loading;
    this.head.replaceChildren();
    this.status("正在读取附件…");
    try {
      const query = new URLSearchParams({ session: intent.session, message, artifact });
      const response = await this.ctx.http.request(`/api/dashboard/preview_ui/text?${query}`);
      if (ticket !== this.loading) return "handled";
      if (!response.ok) {
        const detail = await response.json().then((value: { detail?: unknown }) => value.detail, () => null);
        this.status(typeof detail === "string" ? detail : `读取附件失败（${response.status}）`, true);
        return "handled";
      }
      this.preview = await response.json() as Preview;
      this.view = delimiterOf(this.preview.name) ? "table" : "text";
      this.render();
      return "handled";
    } catch {
      if (ticket === this.loading) this.empty();
      return "failed";
    }
  }

  dispose(): void {
    this.loading += 1;
  }

  private render(): void {
    const preview = this.preview;
    if (!preview) return;
    const delimiter = delimiterOf(preview.name);
    const title = element("div", "preview-title");
    title.append(element("bdi", "preview-name", preview.name), element("span", "preview-meta", note(preview, this.view)));
    const actions = element("div", "preview-actions");
    if (delimiter) {
      const modes = element("div", "preview-modes");
      modes.setAttribute("role", "group");
      modes.setAttribute("aria-label", "显示方式");
      for (const [view, label] of [["table", "表格"], ["text", "原文"]] as const) {
        const mode = button("preview-action", label, () => { this.view = view; this.render(); });
        mode.setAttribute("aria-pressed", String(this.view === view));
        modes.append(mode);
      }
      actions.append(modes);
    }
    if (preview.complete) actions.append(this.copyButton(preview));
    const download = element("a", "preview-action", "下载");
    download.href = `/api/chat/artifacts/${encodeURIComponent(preview.artifact_id)}`;
    download.download = preview.name;
    actions.append(download);
    title.title = preview.name;
    this.head.replaceChildren(title, actions);
    this.body.replaceChildren(this.view === "table" && delimiter ? tableBody(preview, delimiter) : textBody(preview));
    this.body.scrollTo(0, 0);
  }

  private copyButton(preview: Preview): HTMLButtonElement {
    const copy = button("preview-action", "复制", () => {
      void navigator.clipboard.writeText(preview.text).then(() => { copy.textContent = "已复制"; }, () => { copy.textContent = "复制失败"; });
      window.setTimeout(() => { copy.textContent = "复制"; }, 1600);
    });
    return copy;
  }

  private empty(): void {
    this.head.replaceChildren();
    this.status(this.session ? "在对话里点开文本或表格附件，会在这里预览。" : "开始对话后，点开文本或表格附件会在这里预览。");
  }

  private status(text: string, failed = false): void {
    const node = element("p", failed ? "preview-message is-error" : "preview-message", text);
    node.setAttribute("role", failed ? "alert" : "status");
    this.body.replaceChildren(node);
  }
}

function renderPreview(ctx: WebHostContextV1, host: HTMLElement, rawView: unknown): WebUiDisposer {
  const view = rawView as ConversationTabView;
  if (!view || typeof view.onSessionChange !== "function" || typeof view.onIntent !== "function")
    throw new Error("预览标签缺少 conversation.tools.v2 view");
  const tab = new PreviewTab(ctx, view.session);
  host.replaceChildren(tab.root);
  const stops = [
    view.onSessionChange((session) => tab.setSession(session)),
    view.onIntent((intent) => intent.name === OPEN_INTENT ? tab.open(intent) : "invalid"),
  ];
  return () => {
    for (const stop of stops) stop();
    tab.dispose();
    host.replaceChildren();
  };
}

export function activate(ctx: WebHostContextV1): WebUiDisposer {
  return ctx.ui.inject("conversation.tools.v2", (mount) => mount.register({
    id: "preview", label: "预览", order: 40, accepts: [OPEN_INTENT],
    icon: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-5-5Z"/><path d="M14 3v5h5M9 13h6M9 17h6"/></svg>',
    render(host, _entryView, props) { return renderPreview(ctx, host, props); },
  }));
}

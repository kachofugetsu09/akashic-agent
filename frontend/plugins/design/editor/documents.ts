import { reactive } from "vue";
import { exportFigFile, parseFigFile, type Editor } from "@open-pencil/core";
import { SceneGraph } from "@open-pencil/scene-graph";
import type { DesignIndicator } from "./contract";

// 编辑器对外只需要这三件事：同源请求、改标签标题、在标签上亮“正在保存”。
export interface DocumentHost {
  request(path: string, init?: RequestInit): Promise<Response>;
  setTitle(title: string | null): void;
  setIndicator(indicator: DesignIndicator | null): void;
}

export interface DocumentMeta { id: string; name: string; revision: number; updated_at: string; size: number }
export type SaveState = "saved" | "dirty" | "saving" | "conflict" | "error";

const API = "/api/dashboard/design/documents";
const SAVE_DELAY_MS = 1200;
// 连续编辑时防抖会一直往后推；从第一次改动算起最多等这么久就先存一次。
const SAVE_MAX_WAIT_MS = 10_000;
// 与 open-pencil 应用判断“有未保存修改”用的是同一组事件；布局重排不算修改。
const CONTENT_EVENTS = ["node:created", "node:deleted", "node:reparented", "node:reordered", "history:changed"] as const;

// 当前会话的文档列表与正在编辑的那一份：打开时整份读入画布，修改后防抖导出 .fig 写回，修订号不一致时停下来让用户重新加载。
export class Documents {
  readonly state = reactive({
    session: null as string | null, list: [] as DocumentMeta[], current: null as DocumentMeta | null,
    loading: false, save: "saved" as SaveState, error: "",
  });
  private version = 0;
  private savedVersion = 0;
  private dirtySince = 0;
  private adopting = 0;
  private timer = 0;
  private saving: Promise<void> | null = null;
  private ticket = 0;
  private readonly stops: (() => void)[];

  constructor(private readonly editor: Editor, private readonly options: DocumentHost) {
    const changed = () => { if (!this.adopting && this.state.current) this.touch(); };
    this.stops = [
      ...CONTENT_EVENTS.map((name) => editor.onEditorEvent(name, changed)),
      editor.onEditorEvent("node:updated", () => { if (!editor.graph.isApplyingLayout) changed(); }),
    ];
  }

  async setSession(session: string | null): Promise<void> {
    await this.flush();
    const ticket = ++this.ticket;
    Object.assign(this.state, { session, list: [], current: null, error: "", save: "saved" });
    this.show(new SceneGraph());
    if (!session) return;
    const list = await this.json<{ documents: DocumentMeta[] }>(`${API}?${this.query()}`);
    if (ticket !== this.ticket || !list) return;
    this.state.list = list.documents;
    if (list.documents[0]) await this.open(list.documents[0].id);
  }

  async open(id: string): Promise<void> {
    await this.flush();
    const ticket = ++this.ticket;
    this.state.loading = true;
    try {
      const response = await this.options.request(`${API}/${id}?${this.query()}`);
      if (!response.ok) throw new Error(`读取设计文档失败（${response.status}）`);
      const data = await response.arrayBuffer();
      const graph = data.byteLength ? await parseFigFile(data) : new SceneGraph();
      const meta = this.state.list.find((item) => item.id === id) ?? null;
      if (ticket !== this.ticket) return;
      const etag = /^"(\d+)"$/.exec(response.headers.get("ETag") ?? "");
      if (meta && etag) meta.revision = Number(etag[1]);
      Object.assign(this.state, { current: meta, save: "saved", error: "" });
      this.show(graph);
      this.savedVersion = this.version;
    } catch (error) {
      if (ticket === this.ticket) this.fail(error);
    } finally {
      if (ticket === this.ticket) this.state.loading = false;
    }
  }

  async create(name = ""): Promise<DocumentMeta | null> {
    const created = await this.json<DocumentMeta>(`${API}?${this.query()}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }),
    });
    if (!created) return null;
    this.state.list = [created, ...this.state.list];
    await this.open(created.id);
    return this.state.current?.id === created.id ? created : null;
  }

  async flush(): Promise<void> {
    window.clearTimeout(this.timer);
    this.timer = 0;
    if (this.saving) await this.saving;
    if (this.version !== this.savedVersion && this.state.save === "dirty") await this.save();
  }

  dispose(): void {
    void this.flush();
    for (const stop of this.stops) stop();
  }

  // 换文档时不算用户修改：读入期间触发的事件都被吞掉，也不入撤销栈。
  private show(graph: SceneGraph): void {
    this.adopting++;
    try {
      this.editor.replaceGraph(graph);
      this.editor.undo.clear();
      this.editor.zoomToFit();
    } finally {
      this.adopting--;
    }
    this.options.setTitle(this.state.current ? `设计 · ${this.state.current.name}` : null);
  }

  touch(): void {
    this.version++;
    if (this.state.save === "conflict") return;
    this.state.save = "dirty";
    this.dirtySince ||= Date.now();
    window.clearTimeout(this.timer);
    const delay = Math.min(SAVE_DELAY_MS, Math.max(0, this.dirtySince + SAVE_MAX_WAIT_MS - Date.now()));
    this.timer = window.setTimeout(() => void this.save(), delay);
  }

  private save(): Promise<void> {
    this.saving ??= this.write().finally(() => { this.saving = null; });
    return this.saving;
  }

  private async write(): Promise<void> {
    const current = this.state.current;
    if (!current || this.version === this.savedVersion) return;
    const version = this.version;
    this.dirtySince = 0;
    this.state.save = "saving";
    this.options.setIndicator({ busy: true, detail: "正在保存" });
    try {
      const bytes = await exportFigFile(this.editor.graph);
      const response = await this.options.request(`${API}/${current.id}?${this.query()}&base=${current.revision}`, {
        method: "PUT", headers: { "Content-Type": "application/octet-stream" }, body: new Uint8Array(bytes),
      });
      if (response.status === 409) {
        this.state.save = "conflict";
        return;
      }
      if (!response.ok) throw new Error(`保存失败（${response.status}）`);
      Object.assign(current, await response.json() as DocumentMeta);
      this.savedVersion = version;
      this.state.save = this.version === version ? "saved" : "dirty";
      if (this.state.save === "dirty") this.touch();
    } catch (error) {
      this.fail(error);
    } finally {
      this.options.setIndicator(null);
    }
  }

  private async json<T>(path: string, init?: RequestInit): Promise<T | null> {
    try {
      const response = await this.options.request(path, init);
      if (!response.ok) throw new Error(`设计文档请求失败（${response.status}）`);
      return await response.json() as T;
    } catch (error) {
      this.fail(error);
      return null;
    }
  }

  private fail(error: unknown): void {
    this.state.save = this.state.save === "saving" ? "error" : this.state.save;
    this.state.error = error instanceof Error ? error.message : String(error);
  }

  private query(): string {
    return new URLSearchParams({ session: this.state.session ?? "" }).toString();
  }
}

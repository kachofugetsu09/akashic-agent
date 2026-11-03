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
    // 大于 0 时画布正在被替换（或 Agent 正在改）：外壳据此冻结用户输入
    busy: 0,
  });
  private version = 0;
  private savedVersion = 0;
  private dirtySince = 0;
  private adopting = 0;
  private timer = 0;
  private saving: Promise<void> | null = null;
  // 保存期间收到的最新修订通知：保存结束后再判断要不要跟进
  private behind: { id: string; revision: number } | null = null;
  private ticket = 0;
  private readonly stops: (() => void)[];

  constructor(private readonly editor: Editor, private readonly options: DocumentHost) {
    const changed = () => { if (!this.adopting && this.state.current) this.touch(); };
    this.stops = [
      ...CONTENT_EVENTS.map((name) => editor.onEditorEvent(name, changed)),
      editor.onEditorEvent("node:updated", () => { if (!editor.graph.isApplyingLayout) changed(); }),
    ];
  }

  // 替换画布期间冻结用户输入：先让正在编辑的输入框和进行中的拖动提交（settle），再挡住画布、面板、文字编辑与快捷键，免得修改落进已经离开的文档。
  async hold<T>(task: () => Promise<T>, settle = true): Promise<T> {
    if (settle) {
      (document.activeElement as HTMLElement | null)?.blur();
      window.dispatchEvent(new MouseEvent("mouseup"));
    }
    this.state.busy++;
    try {
      return await task();
    } finally {
      this.state.busy--;
    }
  }

  setSession(session: string | null): Promise<void> {
    return this.hold(() => this.switchTo(session));
  }

  open(id: string): Promise<void> {
    return this.hold(() => this.load(id));
  }

  private async switchTo(session: string | null): Promise<void> {
    const left = await this.leave();
    const ticket = ++this.ticket;
    Object.assign(this.state, { session, list: [], current: null, error: left.note, save: "saved" });
    this.show(new SceneGraph());
    if (!session) return;
    const list = await this.json<{ documents: DocumentMeta[] }>(`${API}?${this.query()}`);
    if (ticket !== this.ticket || !list) return;
    this.state.list = list.documents;
    if (list.documents[0]) await this.load(list.documents[0].id);
  }

  // 读取期间画布又有修改（版本变了）就不替换，留在原文档，修改照常保存。
  private async load(id: string): Promise<void> {
    const left = await this.leave();
    if (left.lost) return void (this.state.error = left.note);
    const ticket = ++this.ticket;
    const version = this.version;
    this.state.loading = true;
    try {
      const response = await this.options.request(`${API}/${id}?${this.query()}`);
      if (!response.ok) throw new Error(`读取设计文档失败（${response.status}）`);
      const data = await response.arrayBuffer();
      const graph = data.byteLength ? await parseFigFile(data) : new SceneGraph();
      const meta = this.state.list.find((item) => item.id === id) ?? null;
      if (ticket !== this.ticket) return;
      if (this.version !== version) return void (this.state.error = "读取期间画布又有修改，先留在当前文档。");
      const etag = /^"(\d+)"$/.exec(response.headers.get("ETag") ?? "");
      if (meta && etag) meta.revision = Number(etag[1]);
      Object.assign(this.state, { current: meta, save: "saved", error: left.note });
      this.show(graph);
      this.savedVersion = this.version;
    } catch (error) {
      if (ticket === this.ticket) this.fail(error);
    } finally {
      if (ticket === this.ticket) this.state.loading = false;
    }
  }

  // 冲突时用户点“重新加载”：明确放弃这块画布上没存下的修改，换成存储里的最新版本。
  reload(): Promise<void> {
    return this.hold(async () => {
      const current = this.state.current;
      if (!current) return;
      this.savedVersion = this.version;
      await this.load(current.id);
    });
  }

  create(name = ""): Promise<DocumentMeta | null> {
    return this.hold(() => this.make(name));
  }

  private async make(name: string): Promise<DocumentMeta | null> {
    const session = this.state.session;
    const created = await this.json<DocumentMeta>(`${API}?${this.query()}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }),
    });
    if (!created || this.state.session !== session) return null;
    this.state.list = [created, ...this.state.list];
    await this.load(created.id);
    return this.state.current?.id === created.id ? created : null;
  }

  async refresh(): Promise<void> {
    const session = this.state.session;
    const list = await this.json<{ documents: DocumentMeta[] }>(`${API}?${this.query()}`);
    if (list && this.state.session === session) this.state.list = list.documents;
  }

  // 实时连接（重新）连上后对一次账：断线期间别处保存的修订不会再广播。
  async resync(): Promise<void> {
    await this.refresh();
    const current = this.state.current;
    const latest = current && this.state.list.find((item) => item.id === current.id);
    if (current && latest && latest.revision > current.revision) await this.changed(current.id, latest.revision);
  }

  // 同一会话的另一块画布保存了文档：没有未保存修改就换成新版本，有修改时让下一次保存报冲突，由用户决定。
  async changed(id: string, revision: number): Promise<void> {
    if (!this.state.list.some((item) => item.id === id)) return await this.refresh();
    const current = this.state.current;
    if (current?.id !== id || revision <= current.revision) return;
    if (this.saving) this.behind = { id, revision: Math.max(revision, this.behind?.id === id ? this.behind.revision : 0) };
    else if (this.version === this.savedVersion) await this.hold(() => this.load(id), false);
  }

  async flush(): Promise<void> {
    window.clearTimeout(this.timer);
    this.timer = 0;
    if (this.saving) await this.saving;
    if (this.version !== this.savedVersion && this.state.save === "dirty") await this.save();
  }

  // 离开当前文档前先写回；冲突或保存失败时另存成一份新文档，普通切换不会悄悄丢掉修改。
  private async leave(): Promise<{ lost: boolean; note: string }> {
    await this.flush();
    const current = this.state.current;
    if (!current || this.version === this.savedVersion) return { lost: false, note: "" };
    const version = this.version;
    try {
      const copy = await this.json<DocumentMeta>(`${API}?${this.query()}`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: `${current.name}（未保存的修改）` }),
      });
      if (!copy) throw new Error("没能新建副本");
      const bytes = await exportFigFile(this.editor.graph);
      const response = await this.options.request(`${API}/${copy.id}?${this.query()}&base=0`, {
        method: "PUT", headers: { "Content-Type": "application/octet-stream" }, body: new Uint8Array(bytes),
      });
      if (!response.ok) throw new Error(`保存副本失败（${response.status}）`);
      this.state.list = [await response.json() as DocumentMeta, ...this.state.list];
      if (this.version !== version) return { lost: true, note: `“${current.name}”另存副本期间又有修改，先留在原文档。` };
      this.savedVersion = version;
      return { lost: false, note: `“${current.name}”的修改没能存回原文档，已另存为“${copy.name}”。` };
    } catch {
      return { lost: true, note: `“${current.name}”的修改没能保存，也没能另存副本。` };
    }
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
    this.saving ??= this.write().finally(() => {
      this.saving = null;
      const behind = this.behind;
      this.behind = null;
      if (behind) void this.changed(behind.id, behind.revision);
    });
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

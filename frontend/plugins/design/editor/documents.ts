import { reactive } from "vue";
import { exportFigFile, fontManager, parseFigFile, type Editor } from "@open-pencil/core";
import { SceneGraph } from "@open-pencil/scene-graph";
import type { DesignIndicator, DesignView } from "./contract";
import { rebase } from "./merge";

// 文字用到的字体（含中日韩回退）按需取回后重新排版、重绘：画框标签显示的是图层名，名字里的字一并取回；
// 回退字体是按用到的字取的子集，换来新的子集后已经排好的文字也要重排，不然缺字的地方一直是空的。
export async function loadDocumentFonts(editor: Editor): Promise<void> {
  const graph = editor.graph;
  const nodes = [...graph.nodes.values()];
  await ensureFallback(nodes.map((node) => `${node.name}${node.type === "TEXT" ? node.text : ""}`).join(""));
  await editor.loadFontsForNodes(graph.getPages().flatMap((page) => page.childIds));
  if (graph !== editor.graph) return;
  for (const node of nodes) if (node.type === "TEXT") node.textPicture = null;
  for (const page of graph.getPages()) editor.runLayoutForNode(page.id);
  editor.renderer?.invalidateAllPictures();
  editor.requestRender();
}

// 回退字体要在逐个节点加载字体之前按全部文字一次取齐：逐节点取时每段文字各得一份同名子集，后到的会盖掉先到的字形。
export async function ensureFallback(text: string): Promise<void> {
  const characters = [...new Set(text)].join("");
  if (/[^\u0000-\u024f]/u.test(characters)) await fontManager.ensureFallbackPack(["cjk"], characters);
}

// 编辑器对外只需要这三件事：同源请求、改标签标题、在标签上亮“正在保存”。
export interface DocumentHost {
  request(path: string, init?: RequestInit): Promise<Response>;
  setTitle(title: string | null): void;
  setIndicator(indicator: DesignIndicator | null): void;
}

export interface DocumentMeta { id: string; name: string; revision: number; updated_at: string; size: number; source: string }
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
    // 大于 0 时画布正在被替换（或正在导入网页）：外壳据此冻结用户输入
    busy: 0,
    // 由网页发布来的文档带一份预览：view 在画布与预览之间切换，page 是预览的 HTML
    view: "canvas" as "canvas" | "preview", page: "",
  });
  private version = 0;
  private savedVersion = 0;
  private dirtySince = 0;
  private adopting = 0;
  private timer = 0;
  private saving: Promise<void> | null = null;
  // 画布所基于的修订的原始字节：别处保存了新修订时，以它为共同基线做合并
  private base: ArrayBuffer | null = null;
  // 保存期间收到的最新修订通知：保存结束后再判断要不要跟进
  private behind: { id: string; revision: number } | null = null;
  private ticket = 0;
  // 列表请求的代次：A→B→A 或同会话乱序时，只有最后一次请求的响应生效
  private listing = 0;
  private queue: Promise<unknown> = Promise.resolve();
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

  // 会话切换、用户的文档操作、自动跟进与重连对账排在同一条队列里：前一件做完才轮到下一件，对账不会撞上还没加载完的文档。
  serial<T>(task: () => Promise<T>): Promise<T> {
    const next = this.queue.then(task);
    this.queue = next.catch(() => undefined);
    return next;
  }

  setSession(session: string | null): Promise<void> {
    return this.serial(() => this.hold(() => this.switchTo(session)));
  }

  open(id: string): Promise<void> {
    return this.serial(() => this.hold(() => this.load(id)));
  }

  // Agent 开始改某份文档：画布没有未保存的修改就切过去；有修改时留在原地，只刷新列表。
  focus(id: string, view: DesignView): Promise<void> {
    return this.serial(async () => {
      if (!this.state.list.some((item) => item.id === id)) await this.refresh();
      if (this.state.current?.id !== id) {
        if (this.saving || this.version !== this.savedVersion) return;
        await this.hold(() => this.load(id));
      }
      if (this.state.current?.id === id && (view === "canvas" || this.state.page)) this.state.view = view;
    });
  }

  private async switchTo(session: string | null): Promise<void> {
    const left = await this.leave();
    const ticket = ++this.ticket;
    const listing = ++this.listing;
    Object.assign(this.state, { session, list: [], current: null, error: left.note, save: "saved", view: "canvas", page: "" });
    this.base = null;
    this.show(new SceneGraph());
    if (!session) return;
    const list = await this.json<{ documents: DocumentMeta[] }>(`${API}?${this.query()}`);
    if (ticket !== this.ticket || !list) return;
    if (listing === this.listing) this.state.list = list.documents;
    if (this.state.list[0]) await this.load(this.state.list[0].id);
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
      const page = meta?.source ? await this.preview(id) : "";
      if (ticket !== this.ticket) return;
      if (this.version !== version) return void (this.state.error = "读取期间画布又有修改，先留在当前文档。");
      const etag = /^"(\d+)"$/.exec(response.headers.get("ETag") ?? "");
      if (meta && etag) meta.revision = Number(etag[1]);
      // 只有网页、还没有图层的文档直接看预览；重新加载同一份时保持原来的视图
      const empty = !graph.getPages().some((item) => item.childIds.length);
      const view = page && (empty || (this.state.current?.id === id && this.state.view === "preview")) ? "preview" : "canvas";
      Object.assign(this.state, { current: meta, save: "saved", error: left.note, page, view });
      this.base = data.byteLength ? data.slice(0) : null;
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
    return this.serial(() => this.hold(async () => {
      const current = this.state.current;
      if (!current) return;
      this.savedVersion = this.version;
      await this.load(current.id);
    }));
  }

  create(name = ""): Promise<DocumentMeta | null> {
    return this.serial(() => this.hold(() => this.make(name)));
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
    const listing = ++this.listing;
    const list = await this.json<{ documents: DocumentMeta[] }>(`${API}?${this.query()}`);
    if (list && listing === this.listing) this.state.list = list.documents;
  }

  // 实时连接（重新）连上后对一次账：断线期间别处保存的修订不会再广播。
  resync(): Promise<void> {
    return this.serial(() => this.reconcile());
  }

  private async reconcile(): Promise<void> {
    await this.refresh();
    const current = this.state.current;
    const latest = current && this.state.list.find((item) => item.id === current.id);
    if (current && latest && latest.revision > current.revision) await this.follow(current.id, latest.revision);
  }

  async preview(id: string): Promise<string> {
    const response = await this.options.request(`${API}/${id}/preview?${this.query()}`);
    if (!response.ok) throw new Error(`读取网页预览失败（${response.status}）`);
    return await response.text();
  }

  // 别处（Agent 或另一块画布）保存了文档：把新修订合进当前画布，本地没存下的修改保留并接着保存。
  changed(id: string, revision: number): Promise<void> {
    return this.serial(() => this.follow(id, revision));
  }

  private async follow(id: string, revision: number): Promise<void> {
    if (!this.state.list.some((item) => item.id === id)) return await this.refresh();
    const current = this.state.current;
    if (current?.id !== id || revision <= current.revision) return;
    if (this.saving) this.behind = { id, revision: Math.max(revision, this.behind?.id === id ? this.behind.revision : 0) };
    else if (!this.base && this.version === this.savedVersion) await this.hold(() => this.load(id), false);
    else await this.merge(current);
  }

  // 乐观并发：取回最新修订，以画布所基于的修订为基线，把对方的改动原地合进画布（视口、选择与本地修改都不动）；
  // 合并失败才退回冲突提示，由用户选择重新加载。
  private async merge(current: DocumentMeta): Promise<void> {
    try {
      const response = await this.options.request(`${API}/${current.id}?${this.query()}`);
      if (!response.ok) throw new Error(`读取设计文档失败（${response.status}）`);
      const data = await response.arrayBuffer();
      const revision = Number(/^"(\d+)"$/.exec(response.headers.get("ETag") ?? "")?.[1] ?? 0);
      if (this.state.current !== current || revision <= current.revision) return;
      const theirs = data.byteLength ? await parseFigFile(data.slice(0)) : new SceneGraph();
      const base = this.base ? await parseFigFile(this.base.slice(0)) : new SceneGraph();
      if (this.state.current !== current || revision <= current.revision) return;
      const editor = this.editor;
      const empty = !editor.graph.getPages().some((page) => page.childIds.length);
      // 合进来的改动在本地记成一步撤销（按页快照）：看着 Agent 改的用户撤销一步就回到合并之前
      const before = editor.graph.getPages().map((page) => editor.snapshotPage(page.id));
      this.adopting++;
      try {
        const merged = rebase(editor.graph, base, theirs);
        const after = editor.graph.getPages().map((page) => editor.snapshotPage(page.id));
        if (merged.added + merged.updated + merged.moved + merged.removed + merged.tables) editor.pushUndoStep({ label: "合并别处的修改",
          inverse: () => before.forEach((page) => editor.restorePageFromSnapshot(page)),
          forward: () => after.forEach((page) => editor.restorePageFromSnapshot(page)) });
      } finally {
        this.adopting--;
      }
      this.base = data.slice(0);
      current.revision = revision;
      if (this.state.save === "conflict") this.state.save = "dirty";
      if (empty) this.editor.zoomToFit();
      this.editor.requestRender();
      void loadDocumentFonts(this.editor);
      if (this.version !== this.savedVersion) this.touch();
    } catch (error) {
      this.state.save = "conflict";
      this.state.error = `没能合并别处的修改：${error instanceof Error ? error.message : String(error)}`;
    }
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
    void loadDocumentFonts(this.editor);
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
    // 基线修订在导出之前定下：导出期间合并进来的新修订不能借给这份旧内容，服务端会按旧基线拒绝，合并后重存
    const base = current.revision;
    try {
      const bytes = await exportFigFile(this.editor.graph);
      const response = await this.options.request(`${API}/${current.id}?${this.query()}&base=${base}`, {
        method: "PUT", headers: { "Content-Type": "application/octet-stream" }, body: new Uint8Array(bytes),
      });
      // 别处先存了：保存结束后取回最新修订合并，再带着新修订号重存
      if (response.status === 409) {
        this.state.save = "dirty";
        // 导出期间已经合并到更新的修订：按新基线重存；否则保存结束后取回最新修订，合并后重存
        if (current.revision > base) this.touch();
        else this.behind = { id: current.id, revision: base + 1 };
        return;
      }
      if (!response.ok) throw new Error(`保存失败（${response.status}）`);
      this.base = new Uint8Array(bytes).slice().buffer;
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

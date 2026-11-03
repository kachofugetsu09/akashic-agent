import { reactive } from "vue";
import { CORE_TOOLS, FigmaAPI, type Editor, type ToolDef } from "@open-pencil/core";
import { executeAtomicTool } from "@open-pencil/core/editor";
import { isAtomicTool, parseToolArgs } from "@open-pencil/core/tools";
import type { AgentCall, AgentReply } from "./contract";
import { ensureFallback, loadDocumentFonts, type Documents } from "./documents";

const TOOLS = new Map<string, ToolDef>(CORE_TOOLS.map((tool) => [tool.name, tool]));
const LABEL = "Agent";

// 与 open-pencil 应用的自动化入口一样：工具拿到的是绑定当前页面与用户选择的 FigmaAPI。
function figmaFor(editor: Editor): FigmaAPI {
  const figma = new FigmaAPI(editor.graph);
  figma.currentPage = figma.wrapNode(editor.state.currentPageId);
  figma.currentPage.selection = [...editor.state.selectedIds]
    .map((id) => figma.getNodeById(id)).filter((node): node is NonNullable<typeof node> => node !== null);
  return figma;
}

// 结果要能跨 postMessage 和 WebSocket：节点代理等对象只留可序列化的字段。
function plain(value: unknown): unknown {
  return value === undefined ? null : JSON.parse(JSON.stringify(value)) as unknown;
}

// Agent 的工具调用一次只跑一个，每次编辑是一步撤销；用户同时在画布上操作，两边改的是同一份场景图。
export class AgentBridge {
  readonly state = reactive({ running: 0 });

  constructor(private readonly editor: Editor, private readonly documents: Documents) {}

  run(call: AgentCall): Promise<AgentReply> {
    return this.documents.serial(() => this.execute(call));
  }

  private async execute(call: AgentCall): Promise<AgentReply> {
    // 调用属于发出它的会话：排到时画布已经换了会话就不执行
    if (call.session !== this.documents.state.session)
      return { id: call.id, ok: false, error: "用户已切换到别的会话，这次设计调用没有执行" };
    this.state.running += 1;
    try {
      const result = call.tool === "document" ? await this.document(call.arguments) : await this.tool(call);
      return { id: call.id, ok: true, result: plain(result) };
    } catch (error) {
      return { id: call.id, ok: false, error: error instanceof Error ? error.message : String(error) };
    } finally {
      this.state.running -= 1;
    }
  }

  // 1. 校验参数；2. 没有打开的文档就先新建一份；3. 只读工具直接跑，改文档的工具记一步撤销并写回存储
  private async tool(call: AgentCall): Promise<unknown> {
    const def = TOOLS.get(call.tool);
    if (!def) throw new Error(`未知的设计工具：${call.tool}`);
    const args = parseToolArgs(def.name, def.input, call.arguments) as Record<string, unknown>;
    if (!this.documents.state.current && !await this.documents.createNow())
      throw new Error(this.documents.state.error || "没能新建设计文档");
    const figma = figmaFor(this.editor);
    if (!def.mutates) return await def.execute(figma, args);
    await ensureFallback(JSON.stringify(call.arguments));
    const result = isAtomicTool(def)
      ? executeAtomicTool(this.editor, figma, def, args, { label: LABEL })
      : await this.recorded(def, figma, args);
    if (result && typeof result === "object" && "error" in result) throw new Error(String(result.error));
    this.documents.touch();
    void loadDocumentFonts(this.editor);
    return result;
  }

  // 异步工具（render 等）改动范围不固定：按页面整页快照记成一步撤销，字体加载完再排版。
  private async recorded(def: ToolDef, figma: FigmaAPI, args: Record<string, unknown>): Promise<unknown> {
    const editor = this.editor;
    const pageId = editor.state.currentPageId;
    return await this.exclusive(async () => {
      const before = editor.snapshotPage(pageId);
      const result = await editor.runMutationWithLayout(() => def.execute(figma, args), pageId, () => loadDocumentFonts(editor));
      const after = editor.snapshotPage(pageId);
      editor.pushUndoStep({
        label: `${LABEL}: ${def.name}`,
        inverse: () => editor.restorePageFromSnapshot(before),
        forward: () => editor.restorePageFromSnapshot(after),
      });
      // Agent 新画出的内容要让用户看见：视口缩放到整页内容
      editor.zoomToFit();
      editor.requestRender();
      return result;
    });
  }

  // 整页快照的前后之间不能混进用户的修改，否则撤销 Agent 这一步会连用户的一起撤掉：执行期间冻结画布、面板、文字编辑与快捷键。
  private exclusive<T>(task: () => Promise<T>): Promise<T> {
    return this.documents.hold(task);
  }

  private async document(args: Record<string, unknown>): Promise<unknown> {
    const action = String(args.action ?? "");
    const name = typeof args.name === "string" ? args.name.trim() : "";
    const list = () => this.documents.state.list.map(({ id, name: title, updated_at }) => ({ id, name: title, updated_at }));
    if (action === "create") await this.documents.createNow(name);
    else if (action === "open") {
      const wanted = String(args.document ?? "");
      const found = this.documents.state.list.find((item) => item.id === wanted || (name && item.name === name));
      if (!found) throw new Error(`没有这份设计文档：${wanted || name}；先用 action=list 查看`);
      await this.documents.openNow(found.id);
    } else if (action !== "list") throw new Error("action 只能是 list、open 或 create");
    const current = this.documents.state.current;
    return { documents: list(), open: current ? { id: current.id, name: current.name } : null };
  }
}

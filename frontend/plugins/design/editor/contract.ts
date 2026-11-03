// 标签模块与 iframe 里的编辑器之间的消息：标签只转交会话、可见性与主题，文档的读写都在编辑器里。
export const DESIGN_CHANNEL = "akashic.design.v1";

// iframe 不带 Web 模块身份，宿主会拒绝它直接访问 /api/dashboard：请求交给标签模块用 ctx.http.request 代发。
export const DESIGN_API = "/api/dashboard/design/";

export interface RelayedRequest { id: number; path: string; method: string; headers: [string, string][]; body: ArrayBuffer | null }
export type RelayedResponse =
  | { id: number; status: number; headers: [string, string][]; body: ArrayBuffer }
  | { id: number; error: string };

export type DesignToFrame =
  | { type: "init"; session: string | null; dark: boolean; active: boolean }
  | ({ type: "response" } & RelayedResponse)
  | { type: "session"; session: string | null }
  | { type: "active"; active: boolean }
  | { type: "theme"; dark: boolean }
  // Agent 的设计工具调用（经标签模块从实时连接转来）、别处保存后的新修订号，以及连接重新建立后的对账
  | ({ type: "call" } & AgentCall)
  | { type: "changed"; document: string; revision: number }
  | { type: "resync" };

export interface AgentCall { id: string; session: string; tool: string; arguments: Record<string, unknown> }
export type AgentReply = { id: string; ok: true; result: unknown } | { id: string; ok: false; error: string };

export interface DesignIndicator { busy?: boolean; detail?: string }

export type DesignFromFrame =
  | { type: "ready" }
  | ({ type: "request" } & RelayedRequest)
  | { type: "title"; title: string | null }
  | { type: "indicator"; indicator: DesignIndicator | null }
  | ({ type: "result" } & AgentReply);

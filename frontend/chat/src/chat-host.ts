// 聊天与所在宿主之间的接口。外壳同文档挂载时由 conversation_ui 直接实现；iframe 里由消息适配实现；独立页面没有宿主。
import type { WorkbenchIntentStatus } from "./shell-workbench";

/** 聊天到宿主：会话、遮罩、底栏动作激活与工具区意图。 */
export interface ChatHostLink {
  activateRailAction(id: string): void;
  reportSession(sessionId: string | null): void;
  reportOverlay(open: boolean): void;
  requestIntent(name: string, session: string, payload: unknown): Promise<WorkbenchIntentStatus>;
}

/** 宿主到聊天：外壳与工具区的状态只经这里进入聊天；入参来自宿主，聊天自行校验形状。 */
export interface ChatHandle {
  setRailActions(actions: unknown): void;
  setTools(present: boolean, shortcuts: unknown): void;
  setWorkbenchIntents(names: unknown): void;
  quote(quote: unknown): void;
  shown(): void;
}

let link: ChatHostLink | null = null;

export function hostLink(): ChatHostLink | null {
  return link;
}

// 同一文档只有一个聊天；后连上的宿主取代前一个，断开时只撤自己。
export function connectHost(next: ChatHostLink): () => void {
  link = next;
  return () => { if (link === next) link = null; };
}

// 聊天与所在宿主之间的接口：conversation_ui 在外壳页面里挂载聊天时实现（ADR 0126）；夹具页面给空实现。
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
let keyScope: HTMLElement | null = null;

export function hostLink(): ChatHostLink | null {
  return link;
}

// 页面上还有工作台等区域：window 级的按键与拖放只接发生在聊天里的事件；页面没有焦点（body）时，只在聊天可见时接。
export function inChat(target: EventTarget | null): boolean {
  if (keyScope === null) return true;
  if (!shown(keyScope)) return false;
  return !(target instanceof Node) || target === document.body || keyScope.contains(target);
}

// 外壳换页（display:none）与工作台铺满（visibility:hidden）都保留聊天挂载；visibility 会继承，读根节点即可。
function shown(scope: HTMLElement): boolean {
  return scope.isConnected && scope.getClientRects().length > 0 && getComputedStyle(scope).visibility !== "hidden";
}

// 聊天自己的对话框、菜单、选择器（都挂在聊天根节点里）；工作台等区域的常驻列表不算。
export const CHAT_LAYERS = "[role='dialog'], [role='menu'], [role='listbox'], [data-radix-popper-content-wrapper]";

export function chatLayerOpen(selector: string = CHAT_LAYERS): boolean {
  return (keyScope ?? document).querySelector(selector) !== null;
}

export function keyForChat(event: KeyboardEvent): boolean {
  return inChat(event.target);
}

// 聊天根节点即按键与浮层的范围；未挂载时不设范围。
export function setKeyScope(scope: HTMLElement | null): () => void {
  keyScope = scope;
  return () => { if (keyScope === scope) keyScope = null; };
}

// 同一文档只有一个聊天；后连上的宿主取代前一个，断开时只撤自己。
export function connectHost(next: ChatHostLink): () => void {
  link = next;
  return () => { if (link === next) link = null; };
}

import type { ComposerFile } from "./desktop-composer";
import type { ChatModelRuntime } from "./model-capsule-data";

export interface SessionRow {
  key: string;
  updated_at?: string;
  message_count?: number;
  first_message_content?: string;
}

export interface MessageRow {
  id: number | string;
  seq?: number;
  role: "user" | "assistant";
  content: string;
  timestamp?: string;
  media?: unknown;
  tool_chain?: unknown;
  reasoning_content?: unknown;
  turn_duration_ms?: unknown;
  extra?: Record<string, unknown>;
  reply_to_message_id?: string;
  reply_role?: "user" | "assistant";
  reply_preview?: string;
}

export interface ChatHistoryPage {
  items: MessageRow[];
  total: number;
  hasMore: boolean;
  beforeSeq: number | null;
}

export interface ChatModelState {
  generationId: number;
  defaultRuntime: string;
  sessionOverride: string;
  sessionSelection: { modelRef: string; reasoningEffort: string };
  runtimes: ChatModelRuntime[];
}

export interface WebShellState {
  status: "needs_setup" | "starting" | "ready";
  configured: boolean;
  chatReady: boolean;
  settingsPath: string;
}

export interface UploadedFile {
  filename: string;
  upload_path: string;
  upload_url?: string;
}

export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

export function errorMessage(error: unknown): string {
  if (error instanceof TypeError && error.message === "Failed to fetch") {
    return "无法连接 Akashic。请确认服务仍在运行，然后重试。";
  }
  return error instanceof Error ? error.message : String(error);
}

export async function fetchChatJson<T>(url: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(url, options);
  const text = await response.text();
  let payload: unknown = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      throw new Error(response.ok ? "服务器返回了无效 JSON" : `请求失败: ${response.status}`);
    }
  }
  if (!response.ok) {
    const body = recordValue(payload);
    const detail = typeof body?.detail === "string" ? body.detail : typeof body?.message === "string" ? body.message : "";
    throw new Error(detail || `请求失败: ${response.status}`);
  }
  if (payload === null) throw new Error("服务器返回空响应");
  return payload as T;
}

export function sessionRows(payload: unknown): SessionRow[] {
  const items = responseItems(payload, "/api/chat/sessions");
  if (items.some((item) => (
    typeof item.key !== "string"
    || !item.key.trim()
    || (item.first_message_content !== undefined && typeof item.first_message_content !== "string")
    || (item.updated_at !== undefined && typeof item.updated_at !== "string")
    || (item.message_count !== undefined && (typeof item.message_count !== "number" || !Number.isFinite(item.message_count)))
  ))) {
    throw new Error("/api/chat/sessions 返回了无效 session 行");
  }
  return items as unknown as SessionRow[];
}

export function messageRows(payload: unknown, endpoint: string): MessageRow[] {
  const items = responseItems(payload, endpoint);
  if (items.some((item) => (
    (typeof item.id !== "string" && (typeof item.id !== "number" || !Number.isFinite(item.id)))
    || (item.role !== "user" && item.role !== "assistant")
    || typeof item.content !== "string"
    || (item.reply_to_message_id !== undefined && typeof item.reply_to_message_id !== "string")
    || (item.reply_role !== undefined && item.reply_role !== "user" && item.reply_role !== "assistant")
    || (item.reply_preview !== undefined && typeof item.reply_preview !== "string")
    || ([item.reply_to_message_id, item.reply_role, item.reply_preview].filter((value) => value !== undefined).length % 3 !== 0)
  ))) {
    throw new Error(`${endpoint} 返回了无效 message 行`);
  }
  return items as unknown as MessageRow[];
}

export function chatHistoryPage(payload: unknown, endpoint: string): ChatHistoryPage {
  const body = recordValue(payload);
  if (!body
    || typeof body.total !== "number" || !Number.isInteger(body.total) || body.total < 0
    || typeof body.has_more !== "boolean"
    || (body.before_seq !== null && (!Number.isInteger(body.before_seq) || Number(body.before_seq) < 0))) {
    throw new Error(`${endpoint} 返回了无效历史页`);
  }
  const items = messageRows(payload, endpoint);
  if (items.some((item) => !Number.isInteger(item.seq) || Number(item.seq) < 0)) {
    throw new Error(`${endpoint} 返回了无效历史游标`);
  }
  if (body.has_more && (items.length === 0 || body.before_seq !== items[0].seq)) {
    throw new Error(`${endpoint} 返回了不一致的历史游标`);
  }
  return {
    items,
    total: body.total,
    hasMore: body.has_more,
    beforeSeq: body.before_seq as number | null,
  };
}

export function webShellState(payload: unknown): WebShellState {
  const body = recordValue(payload);
  if (!body
    || (body.status !== "needs_setup" && body.status !== "starting" && body.status !== "ready")
    || typeof body.configured !== "boolean"
    || typeof body.chatReady !== "boolean"
    || typeof body.settingsPath !== "string") {
    throw new Error("/api/shell/state 返回了无效状态");
  }
  return body as unknown as WebShellState;
}

export function chatModelState(payload: unknown): ChatModelState {
  const body = recordValue(payload);
  if (!body || !Number.isInteger(body.generationId)
    || typeof body.defaultRuntime !== "string"
    || typeof body.sessionOverride !== "string"
    || !recordValue(body.sessionSelection)
    || !Array.isArray(body.runtimes)) {
    throw new Error("/api/chat/models 返回了无效模型注册表");
  }
  const runtimes = body.runtimes.map((value) => {
    const item = recordValue(value);
    if (!item || typeof item.id !== "string" || typeof item.provider !== "string"
      || typeof item.model !== "string" || typeof item.sourceId !== "string"
      || typeof item.sourceName !== "string" || typeof item.reasoningEffort !== "string"
      || !Array.isArray(item.supportedReasoningEfforts)
      || !item.supportedReasoningEfforts.every((effort) => typeof effort === "string")
      || !Array.isArray(item.roles)
      || !item.roles.every((role) => typeof role === "string")) {
      throw new Error("/api/chat/models 返回了无效 runtime");
    }
    return {
      id: item.id,
      provider: item.provider,
      model: item.model,
      sourceId: item.sourceId,
      sourceName: item.sourceName,
      reasoningEffort: item.reasoningEffort,
      supportedReasoningEfforts: item.supportedReasoningEfforts as string[],
      roles: item.roles as string[],
    };
  });
  const selection = recordValue(body.sessionSelection);
  if (!selection || typeof selection.modelRef !== "string" || typeof selection.reasoningEffort !== "string") {
    throw new Error("/api/chat/models 返回了无效会话模型选择");
  }
  return {
    generationId: Number(body.generationId),
    defaultRuntime: body.defaultRuntime,
    sessionOverride: body.sessionOverride,
    sessionSelection: { modelRef: selection.modelRef, reasoningEffort: selection.reasoningEffort },
    runtimes,
  };
}

export async function uploadFiles(files: ComposerFile[], signal: AbortSignal): Promise<UploadedFile[]> {
  const result: UploadedFile[] = [];
  for (const file of files) {
    if (!file.url) throw new Error(`附件 ${file.filename || "未命名"} 缺少内容 URL`);
    const sourceResponse = await fetch(file.url, { signal });
    if (!sourceResponse.ok) throw new Error(`读取附件失败: ${sourceResponse.status}`);
    const blob = await sourceResponse.blob();
    const filename = file.filename || "upload.bin";
    const payload = await fetchChatJson<unknown>(`/api/chat/uploads?filename=${encodeURIComponent(filename)}`, {
      method: "POST",
      body: blob,
      signal,
    });
    result.push(uploadedFileResponse(payload));
  }
  return result;
}

function responseItems(payload: unknown, endpoint: string): Record<string, unknown>[] {
  const body = recordValue(payload);
  if (!body || !Array.isArray(body.items) || body.items.some((item) => !recordValue(item))) {
    throw new Error(`${endpoint} 返回格式无效`);
  }
  return body.items as Record<string, unknown>[];
}

function uploadedFileResponse(payload: unknown): UploadedFile {
  const body = recordValue(payload);
  if (!body || typeof body.filename !== "string" || typeof body.upload_path !== "string" || !body.upload_path) {
    throw new Error("上传接口返回格式无效");
  }
  if (body.upload_url !== undefined && typeof body.upload_url !== "string") {
    throw new Error("上传接口返回了无效 URL");
  }
  return body as unknown as UploadedFile;
}

function recordValue(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

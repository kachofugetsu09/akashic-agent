import type { PageResult } from "./types";

export interface InteractionDeleteRequirement {
  code: "interaction_delete_required";
  message_id: string;
  control_turn_id: string;
}

export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(status: number, detail: unknown, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export async function api<T = unknown>(url: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers ?? {}),
    },
    ...options,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({})) as { detail?: unknown };
    const detail = payload.detail;
    const message = typeof detail === "string"
      ? detail
      : typeof detail === "object" && detail !== null && typeof (detail as { message?: unknown }).message === "string"
        ? String((detail as { message: string }).message)
        : `请求失败: ${response.status}`;
    throw new ApiError(response.status, detail, message);
  }
  if (response.status === 204) {
    return null as T;
  }
  return response.json() as Promise<T>;
}

export function interactionDeleteRequirement(error: unknown): InteractionDeleteRequirement | null {
  if (!(error instanceof ApiError) || error.status !== 409) return null;
  const detail = error.detail;
  if (typeof detail !== "object" || detail === null || Array.isArray(detail)) return null;
  const candidate = detail as Partial<InteractionDeleteRequirement>;
  if (
    candidate.code !== "interaction_delete_required"
    || typeof candidate.message_id !== "string"
    || !candidate.message_id
    || typeof candidate.control_turn_id !== "string"
    || !candidate.control_turn_id
  ) return null;
  return candidate as InteractionDeleteRequirement;
}

export function pageCount(total: number, pageSize: number): number {
  return Math.max(1, Math.ceil(total / pageSize));
}

export function asPageResult<T>(payload: unknown): PageResult<T> {
  if (
    typeof payload !== "object"
    || payload === null
    || Array.isArray(payload)
    || !Array.isArray((payload as { items?: unknown }).items)
    || typeof (payload as { total?: unknown }).total !== "number"
    || !Number.isFinite((payload as { total: number }).total)
    || (payload as { total: number }).total < 0
  ) {
    throw new Error("分页接口返回格式无效");
  }

  const page = payload as PageResult<T>;
  return {
    items: page.items,
    total: page.total,
    page: page.page,
    page_size: page.page_size,
  };
}

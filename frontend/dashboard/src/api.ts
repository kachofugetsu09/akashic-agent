import type { PageResult } from "./types";

export async function api<T = unknown>(url: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers ?? {}),
    },
    ...options,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({})) as { detail?: string };
    throw new Error(payload.detail || `请求失败: ${response.status}`);
  }
  if (response.status === 204) {
    return null as T;
  }
  return response.json() as Promise<T>;
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

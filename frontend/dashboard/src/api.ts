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

export function asPageResult<T>(payload: PageResult<T>): PageResult<T> {
  return {
    items: payload.items ?? [],
    total: payload.total ?? 0,
    page: payload.page,
    page_size: payload.page_size,
  };
}

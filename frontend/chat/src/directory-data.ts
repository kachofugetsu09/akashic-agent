/** 执行主机上的路径状态；缺失和离线保留原路径供诊断。 */
export interface DirectoryState {
  path: string | null;
  status: string;
  error?: string;
}

export interface DirectoryPage extends DirectoryState {
  items: { name: string; path: string }[];
  parent: string | null;
  after: string | null;
}

export function directoryState(row: Record<string, unknown>): DirectoryState {
  if ((row.path !== null && typeof row.path !== "string") || typeof row.status !== "string"
    || (row.error !== undefined && typeof row.error !== "string")) throw new Error("目录状态响应无效");
  return { path: row.path, status: row.status, error: row.error as string | undefined };
}

export function directoryPage(row: Record<string, unknown>): DirectoryPage {
  const state = directoryState(row);
  if (state.status !== "available") throw new Error(`${directoryStatus(state.status)}：${state.path ?? ""}${state.error ? ` · ${state.error}` : ""}`);
  if (!Array.isArray(row.items) || (row.parent !== undefined && row.parent !== null && typeof row.parent !== "string")
    || (row.after !== undefined && row.after !== null && typeof row.after !== "string")) throw new Error("目录列表响应无效");
  const items = row.items.map((item: unknown) => {
    if (typeof item !== "object" || item === null || !("name" in item) || typeof item.name !== "string"
      || !("path" in item) || typeof item.path !== "string") throw new Error("目录条目无效");
    return { name: item.name, path: item.path };
  });
  return { ...state, items, parent: row.parent as string | null | undefined ?? null,
    after: row.after as string | null | undefined ?? null };
}

export function directoryStatus(status: string): string {
  const labels: Record<string, string> = { available: "可用", unset: "未设置", not_found: "目录已缺失",
    not_directory: "不是目录", permission_denied: "没有访问权限", offline: "执行主机离线", io_error: "读取失败" };
  return labels[status] ?? status;
}

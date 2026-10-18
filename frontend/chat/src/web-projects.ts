import { queryHostPlugin } from "./plugin-ui-runtime";
import { createUuid } from "./browser-uuid";

/** project 插件拥有项目记录；其他插件经 scope.fields 为新项目保存自己的设置。 */
export const PROJECTS_PLUGIN = "projects";
export const PROJECT_DIMENSION = "project";

export interface ProjectRow {
  id: string;
  name: string;
  archived: boolean;
  createdAt: string;
  directory: string | null;
}

interface PendingProject {
  id: string;
  name: string;
  directory?: string | null;
}

const PENDING_PREFIX = "akashic.project-create.";

export interface PendingProjectRow {
  key: string;
  id?: string;
  name: string;
  directory?: string | null;
  invalid: boolean;
}

function parsePending(key: string, raw: string | null): PendingProject {
  const value: unknown = JSON.parse(raw ?? "null");
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error("待创建项目记录无效");
  const row = value as Record<string, unknown>;
  // 旧版本的请求带记忆选择，继续创建会丢掉它；只能停止后重新新建。
  if (typeof row.id !== "string" || key !== PENDING_PREFIX + row.id
    || typeof row.name !== "string" || !row.name || "memory" in row
    || (row.directory !== undefined && row.directory !== null
      && (typeof row.directory !== "string" || !row.directory.startsWith("/")))) throw new Error("待创建项目记录无效");
  return row as unknown as PendingProject;
}

function readPending(key: string): PendingProject {
  return parsePending(key, localStorage.getItem(key));
}

/** 本地请求只供用户手动恢复；损坏的单条记录不阻断其他项目。 */
export function listPendingProjects(): { items: PendingProjectRow[]; error: string } {
  try {
    const items: PendingProjectRow[] = [];
    for (const key of Object.keys(localStorage).filter((key) => key.startsWith(PENDING_PREFIX))) {
      const raw = localStorage.getItem(key);
      try {
        const pending = parsePending(key, raw);
        items.push({ key, id: pending.id, name: pending.name, directory: pending.directory, invalid: false });
      } catch {
        items.push({ key, name: "无法读取的本地请求", invalid: true });
      }
    }
    return { items, error: "" };
  } catch {
    return { items: [], error: "本地未确认请求暂不可读；已提交项目仍可使用。" };
  }
}

export function stopProject(key: string): void {
  if (!key.startsWith(PENDING_PREFIX)) throw new Error("待创建项目标识无效");
  localStorage.removeItem(key);
}

export async function continueProject(key: string): Promise<ProjectRow> {
  if (!key.startsWith(PENDING_PREFIX)) throw new Error("待创建项目标识无效");
  return finishProject(readPending(key));
}

// 同一 ID 创建一次；只有项目记录提交后才开放 Session 入口。
async function finishProject(pending: PendingProject, signal?: AbortSignal): Promise<ProjectRow> {
  const project = projectRow(await queryHostPlugin(PROJECTS_PLUGIN, "project.create", {
    project_id: pending.id, name: pending.name, ...(pending.directory ? { directory: pending.directory } : {}),
  }, signal));
  if (project.id !== pending.id) throw new Error("项目创建响应 ID 不一致");
  localStorage.removeItem(PENDING_PREFIX + pending.id);
  return project;
}

export async function loadProjects(signal?: AbortSignal): Promise<ProjectRow[]> {
  const result = await queryHostPlugin(PROJECTS_PLUGIN, "project.list", {}, signal);
  if (!Array.isArray(result.items)) throw new Error("项目列表无效");
  return result.items.map(projectRow).filter((project) => !project.archived);
}

/** 先让其他插件为新 ID 保存设置，再幂等建项目；建项目失败的请求只由用户显式继续。 */
export async function createProject(
  name: string,
  directory: string | null,
  commitFields: (projectId: string) => Promise<void>,
): Promise<ProjectRow> {
  const snapshot = listPendingProjects();
  if (snapshot.error) throw new Error(snapshot.error);
  if (snapshot.items.some((item) => item.name === name)) {
    throw new Error("同名项目存在未确认请求，请在项目栏继续创建或停止尝试");
  }
  const pending = { id: `p_${createUuid().replaceAll("-", "")}`, name, directory };
  // 1. 附加设置（例如记忆策略）必须在项目出现第一个会话前保存；失败时什么都没创建，可直接重试。
  await commitFields(pending.id);
  // 2. 之后只差项目记录本身：先记下请求，响应丢失时可在项目栏继续。
  localStorage.setItem(PENDING_PREFIX + pending.id, JSON.stringify(pending));
  return finishProject(pending);
}

function projectRow(value: unknown): ProjectRow {
  const row = typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
  if (!row || typeof row.id !== "string" || !row.id || typeof row.name !== "string"
    || typeof row.archived !== "boolean" || typeof row.created_at !== "string") {
    throw new Error("项目记录无效");
  }
  if (row.directory !== undefined && row.directory !== null && typeof row.directory !== "string") {
    throw new Error("项目目录无效");
  }
  return { id: row.id, name: row.name, archived: row.archived, createdAt: row.created_at,
    directory: typeof row.directory === "string" ? row.directory : null };
}

/** 首次绑定由插件原子提交；UI 不提供重绑或清空请求。 */
export async function bindProjectDirectory(projectId: string, path: string): Promise<ProjectRow> {
  return projectRow(await queryHostPlugin(PROJECTS_PLUGIN, "project.bind_directory", { project_id: projectId, path }));
}

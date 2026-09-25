import { queryHostPlugin } from "./mobile-plugin-runtime";
import { createUuid } from "./browser-uuid";

/** project 插件拥有项目记录；记忆插件只按 (维度, 取值) 拥有学习策略。 */
export const PROJECTS_PLUGIN = "projects";
export const MEMORY_PLUGIN = "akasha";
export const PROJECT_DIMENSION = "project";

export type ProjectMemory = "global" | "isolated" | "off";

export const PROJECT_MEMORY_CHOICES: readonly { value: ProjectMemory; label: string; description: string }[] = [
  { value: "global", label: "共享全局记忆", description: "对话参与全局学习，也能回忆其他对话。" },
  { value: "isolated", label: "独立记忆", description: "只在本项目内学习和回忆，不影响全局。" },
  { value: "off", label: "不学习", description: "对话不写入记忆，仍可回忆全局记忆。" },
];

export interface ProjectRow {
  id: string;
  name: string;
  archived: boolean;
  createdAt: string;
  memory?: ProjectMemory;
  memoryUnreadable?: boolean;
}

interface PendingProject {
  id: string;
  name: string;
  memory: ProjectMemory;
  memoryInstalled: boolean;
}

const PENDING_PREFIX = "akashic.project-create.";

export interface PendingProjectRow {
  key: string;
  id?: string;
  name: string;
  memory?: ProjectMemory;
  invalid: boolean;
}

function parsePending(key: string, raw: string | null): PendingProject {
  const value: unknown = JSON.parse(raw ?? "null");
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error("待创建项目记录无效");
  const row = value as Record<string, unknown>;
  if (typeof row.id !== "string" || key !== PENDING_PREFIX + row.id
    || typeof row.name !== "string" || !row.name
    || (row.memory !== "global" && row.memory !== "isolated" && row.memory !== "off")
    || typeof row.memoryInstalled !== "boolean") throw new Error("待创建项目记录无效");
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
        items.push({ key, id: pending.id, name: pending.name, memory: pending.memory, invalid: false });
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

export async function continueProject(key: string, memoryInstalled: boolean): Promise<ProjectRow> {
  if (!key.startsWith(PENDING_PREFIX)) throw new Error("待创建项目标识无效");
  return finishProject(readPending(key), memoryInstalled);
}

async function finishProject(pending: PendingProject, memoryInstalled: boolean, signal?: AbortSignal): Promise<ProjectRow> {
  // 1. 策略先提交；请求或响应丢失时，同值 set 可以安全重放。
  if (pending.memoryInstalled) {
    if (!memoryInstalled) throw new Error("记忆插件暂不可用，项目创建等待重试");
    const result = await queryHostPlugin(MEMORY_PLUGIN, "scope.policy.set", {
      dimension: PROJECT_DIMENSION, value: pending.id, learn: pending.memory,
    }, signal);
    if (memoryValue(result.learn) !== pending.memory) throw new Error("项目记忆策略与创建请求不一致");
  }
  // 2. 同一 ID 创建一次；只有项目记录提交后才开放 Session 入口。
  const project = projectRow(await queryHostPlugin(PROJECTS_PLUGIN, "project.create", {
    project_id: pending.id, name: pending.name,
  }, signal));
  if (project.id !== pending.id) throw new Error("项目创建响应 ID 不一致");
  localStorage.removeItem(PENDING_PREFIX + pending.id);
  return pending.memoryInstalled ? { ...project, memory: pending.memory } : project;
}

export async function loadProjects(memoryInstalled: boolean, signal?: AbortSignal): Promise<ProjectRow[]> {
  const result = await queryHostPlugin(PROJECTS_PLUGIN, "project.list", {}, signal);
  if (!Array.isArray(result.items)) throw new Error("项目列表无效");
  const projects = result.items.map(projectRow).filter((project) => !project.archived);
  if (!memoryInstalled) return projects;
  return Promise.all(projects.map(async (project) => {
    try {
      return { ...project, memory: await readProjectMemory(project.id, signal) };
    } catch (error) {
      if (signal?.aborted) throw error;
      return { ...project, memoryUnreadable: true };
    }
  }));
}

/** 固定策略后幂等建项目；失败请求只由用户显式继续。 */
export async function createProject(
  name: string,
  memory: ProjectMemory,
  memoryInstalled: boolean,
): Promise<ProjectRow> {
  if (!memoryInstalled && memory !== "global") throw new Error("记忆插件暂不可用，不能创建非全局项目");
  const snapshot = listPendingProjects();
  if (snapshot.error) throw new Error(snapshot.error);
  if (snapshot.items.some((item) => item.name === name)) {
    throw new Error("同名项目存在未确认请求，请在项目栏继续创建或停止尝试");
  }
  const pending = { id: `p_${createUuid().replaceAll("-", "")}`, name, memory, memoryInstalled };
  localStorage.setItem(PENDING_PREFIX + pending.id, JSON.stringify(pending));
  return finishProject(pending, memoryInstalled);
}

async function readProjectMemory(projectId: string, signal?: AbortSignal): Promise<ProjectMemory> {
  const result = await queryHostPlugin(MEMORY_PLUGIN, "scope.policy.get", {
    dimension: PROJECT_DIMENSION, value: projectId,
  }, signal);
  return memoryValue(result.learn);
}

function memoryValue(value: unknown): ProjectMemory {
  if (value === "global" || value === "isolated" || value === "off") return value;
  throw new Error("记忆策略无效");
}

function projectRow(value: unknown): ProjectRow {
  const row = typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
  if (!row || typeof row.id !== "string" || !row.id || typeof row.name !== "string"
    || typeof row.archived !== "boolean" || typeof row.created_at !== "string") {
    throw new Error("项目记录无效");
  }
  return { id: row.id, name: row.name, archived: row.archived, createdAt: row.created_at };
}

export function projectMemoryLabel(memory: ProjectMemory | undefined): string {
  return PROJECT_MEMORY_CHOICES.find((choice) => choice.value === memory)?.label ?? "";
}

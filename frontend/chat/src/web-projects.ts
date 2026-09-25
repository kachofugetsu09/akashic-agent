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
}

interface PendingProject {
  id: string;
  name: string;
  memory: ProjectMemory;
  memoryInstalled: boolean;
}

const PENDING_PREFIX = "akashic.project-create.";

/** 浏览器只保存未完成请求；Projects 与 Akasha 各自保存提交后的事实。 */
function pendingProjects(): PendingProject[] {
  return Object.keys(localStorage).filter((key) => key.startsWith(PENDING_PREFIX)).map((key) => {
    const value: unknown = JSON.parse(localStorage.getItem(key) ?? "null");
    if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error("待创建项目记录无效");
    const row = value as Record<string, unknown>;
    if (typeof row.id !== "string" || key !== PENDING_PREFIX + row.id
      || typeof row.name !== "string" || !row.name
      || (row.memory !== "global" && row.memory !== "isolated" && row.memory !== "off")
      || typeof row.memoryInstalled !== "boolean") throw new Error("待创建项目记录无效");
    return row as unknown as PendingProject;
  });
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
  for (const pending of pendingProjects()) {
    await finishProject(pending, memoryInstalled, signal);
  }
  const result = await queryHostPlugin(PROJECTS_PLUGIN, "project.list", {}, signal);
  if (!Array.isArray(result.items)) throw new Error("项目列表无效");
  const projects = result.items.map(projectRow).filter((project) => !project.archived);
  if (!memoryInstalled) return projects;
  return Promise.all(projects.map(async (project) => ({
    ...project,
    memory: await readProjectMemory(project.id, signal),
  })));
}

/** 固定策略后幂等建项目；失败请求保留到刷新、重开或手动重试。 */
export async function createProject(
  name: string,
  memory: ProjectMemory,
  memoryInstalled: boolean,
): Promise<ProjectRow> {
  if (!memoryInstalled && memory !== "global") throw new Error("记忆插件暂不可用，不能创建非全局项目");
  const previous = pendingProjects().find((item) => item.name === name);
  if (previous && (previous.memory !== memory || previous.memoryInstalled !== memoryInstalled)) {
    throw new Error("同名项目仍在等待原记忆策略完成，请恢复原请求");
  }
  const pending = previous ?? { id: `p_${createUuid().replaceAll("-", "")}`, name, memory, memoryInstalled };
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

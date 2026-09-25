import { queryHostPlugin } from "./mobile-plugin-runtime";

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

export async function loadProjects(memoryInstalled: boolean, signal?: AbortSignal): Promise<ProjectRow[]> {
  const result = await queryHostPlugin(PROJECTS_PLUGIN, "project.list", {}, signal);
  if (!Array.isArray(result.items)) throw new Error("项目列表无效");
  const projects = result.items.map(projectRow).filter((project) => !project.archived);
  if (!memoryInstalled) return projects;
  return Promise.all(projects.map(async (project) => ({
    ...project,
    memory: await readProjectMemory(project.id, signal).catch(() => undefined),
  })));
}

/** 先建项目、再固定记忆策略；此时尚无对话，策略写入不会与已有路由冲突。 */
export async function createProject(
  name: string,
  memory: ProjectMemory,
  memoryInstalled: boolean,
): Promise<ProjectRow> {
  const project = projectRow(await queryHostPlugin(PROJECTS_PLUGIN, "project.create", { name }));
  if (!memoryInstalled) return project;
  const result = await queryHostPlugin(MEMORY_PLUGIN, "scope.policy.set", {
    dimension: PROJECT_DIMENSION, value: project.id, learn: memory,
  });
  return { ...project, memory: memoryValue(result.learn) };
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

import { ChevronRight, FolderPlus, Plus } from "lucide-react";
import { useState, type FormEvent } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  PROJECT_MEMORY_CHOICES,
  projectMemoryLabel,
  type ProjectMemory,
  type ProjectRow,
} from "./web-projects";

export interface ProjectSessionItem {
  id: string;
  title: string;
  updatedLabel?: string;
  active: boolean;
}

/** 侧栏项目分组：只按 Session 的 project 维度归类，不另存对话与项目的关系。 */
export function ProjectNavigation({
  projects,
  sessionsByProject,
  activeProjectId,
  pendingSessionId,
  memoryInstalled,
  onSelectSession,
  onNewProjectChat,
  onCreateProject,
}: {
  projects: ProjectRow[];
  sessionsByProject: ReadonlyMap<string, ProjectSessionItem[]>;
  activeProjectId: string;
  pendingSessionId: string;
  memoryInstalled: boolean;
  onSelectSession: (sessionId: string) => void;
  onNewProjectChat: (projectId: string) => void;
  onCreateProject: (name: string, memory: ProjectMemory) => Promise<void>;
}) {
  const [collapsed, setCollapsed] = useState<ReadonlySet<string>>(() => new Set());
  const [dialogOpen, setDialogOpen] = useState(false);
  const toggle = (projectId: string) => setCollapsed((current) => {
    const next = new Set(current);
    if (next.has(projectId)) next.delete(projectId);
    else next.add(projectId);
    return next;
  });

  return (
    <section className="project-navigation" aria-label="项目">
      <header className="project-navigation__header">
        <span>项目</span>
        <button type="button" className="project-navigation__icon" aria-label="新建项目" title="新建项目"
          onClick={() => setDialogOpen(true)}>
          <Plus size={15} aria-hidden="true" />
        </button>
      </header>
      {projects.length === 0 ? (
        <button type="button" className="project-navigation__empty" onClick={() => setDialogOpen(true)}>
          <FolderPlus size={16} aria-hidden="true" />
          <span>新建项目</span>
        </button>
      ) : null}
      {projects.map((project) => {
        const open = !collapsed.has(project.id);
        const items = sessionsByProject.get(project.id) ?? [];
        return (
          <div key={project.id} className={`project-group ${activeProjectId === project.id ? "active" : ""}`}>
            <div className="project-group__row">
              <button type="button" className="project-group__toggle" aria-expanded={open} onClick={() => toggle(project.id)}>
                <ChevronRight size={14} aria-hidden="true" className="project-group__chevron" />
                <span className="project-group__name">{project.name}</span>
                {project.memory && project.memory !== "global" ? (
                  <small className="project-group__memory">{projectMemoryLabel(project.memory)}</small>
                ) : null}
              </button>
              <button type="button" className="project-navigation__icon" aria-label={`在 ${project.name} 中新建对话`}
                title="新建对话" onClick={() => onNewProjectChat(project.id)}>
                <Plus size={14} aria-hidden="true" />
              </button>
            </div>
            {open ? (
              <nav className="project-group__sessions" aria-label={`${project.name} 的对话`}>
                {items.length === 0 ? <small className="project-group__hint">还没有对话</small> : null}
                {items.map((session) => (
                  <button key={session.id} type="button"
                    className={`project-session ${session.active ? "active" : ""}`}
                    aria-current={session.active ? "true" : undefined}
                    aria-busy={pendingSessionId === session.id || undefined}
                    title={session.title}
                    onClick={() => onSelectSession(session.id)}>
                    <span>{session.title}</span>
                    {session.updatedLabel ? <time>{session.updatedLabel}</time> : null}
                  </button>
                ))}
              </nav>
            ) : null}
          </div>
        );
      })}
      <NewProjectDialog
        open={dialogOpen}
        memoryInstalled={memoryInstalled}
        onOpenChange={setDialogOpen}
        onCreate={onCreateProject}
      />
    </section>
  );
}

/** 记忆策略在项目还没有对话时一次选定；之后改变需要显式重建学习图。 */
function NewProjectDialog({
  open,
  memoryInstalled,
  onOpenChange,
  onCreate,
}: {
  open: boolean;
  memoryInstalled: boolean;
  onOpenChange: (open: boolean) => void;
  onCreate: (name: string, memory: ProjectMemory) => Promise<void>;
}) {
  const [name, setName] = useState("");
  const [memory, setMemory] = useState<ProjectMemory>("global");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  const reset = (nextOpen: boolean) => {
    if (!nextOpen && submitting) return;
    if (!nextOpen) {
      setName("");
      setMemory("global");
      setError("");
    }
    onOpenChange(nextOpen);
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!name.trim() || submitting) return;
    setSubmitting(true);
    setError("");
    try {
      await onCreate(name.trim(), memory);
      setSubmitting(false);
      reset(false);
    } catch (reason: unknown) {
      setSubmitting(false);
      setError(reason instanceof Error ? reason.message : "创建项目失败");
    }
  };

  return (
    <Dialog open={open} onOpenChange={reset}>
      <DialogContent className="project-dialog">
        <form onSubmit={(event) => void submit(event)}>
          <DialogHeader className="project-dialog__header">
            <DialogTitle>新建项目</DialogTitle>
            <DialogDescription>项目里的对话会带上同一个项目标识。</DialogDescription>
          </DialogHeader>
          <label className="project-dialog__field">
            <span>名称</span>
            <Input autoFocus value={name} maxLength={80} placeholder="例如：论文写作"
              onChange={(event) => setName(event.target.value)} />
          </label>
          {memoryInstalled ? (
            <fieldset className="project-dialog__memory">
              <legend>记忆</legend>
              {PROJECT_MEMORY_CHOICES.map((choice) => (
                <label key={choice.value} className={`project-dialog__choice ${memory === choice.value ? "selected" : ""}`}>
                  <input type="radio" name="project-memory" value={choice.value}
                    checked={memory === choice.value} onChange={() => setMemory(choice.value)} />
                  <span>
                    <strong>{choice.label}</strong>
                    <small>{choice.description}</small>
                  </span>
                </label>
              ))}
              <small className="project-dialog__note">有对话之后，记忆选项就不能再修改。</small>
            </fieldset>
          ) : null}
          {error ? <p className="project-dialog__error" role="alert">{error}</p> : null}
          <DialogFooter className="project-dialog__footer">
            <button type="button" className="project-dialog__button" onClick={() => reset(false)} disabled={submitting}>取消</button>
            <button type="submit" className="project-dialog__button primary" disabled={!name.trim() || submitting}>
              {submitting ? "正在创建…" : "创建"}
            </button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

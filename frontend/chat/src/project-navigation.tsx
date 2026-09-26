import { ChevronDown, ChevronRight, Folder, FolderPlus, Lightbulb, Plus } from "lucide-react";
import { useRef, useState, type FormEvent } from "react";
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
  DropdownMenu, DropdownMenuContent, DropdownMenuRadioGroup,
  DropdownMenuRadioItem, DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  PROJECT_MEMORY_CHOICES,
  projectMemoryLabel,
  type PendingProjectRow,
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
  pending,
  pendingError,
  sessionsByProject,
  activeProjectId,
  pendingSessionId,
  memoryInstalled,
  onSelectSession,
  onNewProjectChat,
  onCreateProject,
  onContinueProject,
  onStopProject,
  onOpenCreateProject,
}: {
  projects: ProjectRow[];
  pending: PendingProjectRow[];
  pendingError: string;
  sessionsByProject: ReadonlyMap<string, ProjectSessionItem[]>;
  activeProjectId: string;
  pendingSessionId: string;
  memoryInstalled: boolean;
  onSelectSession: (sessionId: string) => void;
  onNewProjectChat: (projectId: string) => void;
  onCreateProject: (name: string, memory: ProjectMemory) => Promise<void>;
  onContinueProject: (key: string) => Promise<void>;
  onStopProject: (key: string) => void;
  onOpenCreateProject?: () => void;
}) {
  const [collapsed, setCollapsed] = useState<ReadonlySet<string>>(() => new Set());
  const [dialogOpen, setDialogOpen] = useState(false);
  const [busyKey, setBusyKey] = useState("");
  const [actionError, setActionError] = useState("");
  const createButtonRef = useRef<HTMLButtonElement>(null);
  const openCreate = onOpenCreateProject ?? (() => setDialogOpen(true));
  const toggle = (projectId: string) => setCollapsed((current) => {
    const next = new Set(current);
    if (next.has(projectId)) next.delete(projectId);
    else next.add(projectId);
    return next;
  });
  const continuePending = async (key: string) => {
    setBusyKey(key);
    setActionError("");
    try {
      await onContinueProject(key);
      createButtonRef.current?.focus();
    } catch (reason) {
      setActionError(reason instanceof Error ? reason.message : "继续创建失败");
    } finally {
      setBusyKey("");
    }
  };
  const stopPending = (key: string) => {
    setActionError("");
    try {
      onStopProject(key);
      createButtonRef.current?.focus();
    } catch (reason) {
      setActionError(reason instanceof Error ? reason.message : "停止尝试失败");
    }
  };

  return (
    <section className="project-navigation" aria-label="项目">
      <header className="project-navigation__header">
        <span>项目</span>
        <button ref={createButtonRef} type="button" className="project-navigation__icon" aria-label="新建项目" title="新建项目"
          onClick={openCreate}>
          <Plus size={15} aria-hidden="true" />
        </button>
      </header>
      {pendingError ? <p className="project-pending__error" role="alert">{pendingError}</p> : null}
      {pending.length ? <div className="project-pending" aria-label="未确认的项目创建">
        <strong>未确认的项目创建</strong>
        <p>仅在点“继续创建”时重试。停止尝试只清除本地请求，不撤销可能已提交的策略或项目。</p>
        {pending.map((item) => <div key={item.key} className="project-pending__item">
          <span className="project-pending__name" title={item.name}>{item.name}</span>
          <small>{item.invalid ? "本地请求损坏，无法继续" : projects.some((project) => item.id === project.id)
            ? "项目已在列表中，本地请求仍未确认" : `创建结果未确认 · ${projectMemoryLabel(item.memory)}`}</small>
          <div className="project-pending__actions">
            {!item.invalid ? <button type="button" disabled={Boolean(busyKey)} onClick={() => void continuePending(item.key)}>
              {busyKey === item.key ? "正在继续…" : "继续创建"}
            </button> : null}
            <button type="button" disabled={Boolean(busyKey)} onClick={() => stopPending(item.key)}>停止尝试</button>
          </div>
        </div>)}
        {actionError ? <p className="project-pending__error" role="alert">{actionError}</p> : null}
      </div> : null}
      {projects.length === 0 ? (
        <button type="button" className="project-navigation__empty" onClick={openCreate}>
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
              <button type="button" className="project-group__toggle" aria-label={`${open ? "收起" : "展开"} ${project.name} 的对话`} aria-expanded={open} onClick={() => toggle(project.id)}>
                <ChevronRight size={14} aria-hidden="true" className="project-group__chevron" />
              </button>
              <button type="button" className="project-group__open" onClick={() => onNewProjectChat(project.id)}
                aria-current={activeProjectId === project.id ? "page" : undefined} title={`打开 ${project.name}`}>
                <Folder size={18} strokeWidth={1.75} aria-hidden="true" />
                <span className="project-group__name">{project.name}</span>
                {project.memory && project.memory !== "global" ? (
                  <small className="project-group__memory">{projectMemoryLabel(project.memory)}</small>
                ) : project.memoryUnreadable ? <small className="project-group__memory">当前策略未能读取</small> : null}
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
      {!onOpenCreateProject ? <NewProjectDialog
        open={dialogOpen}
        memoryInstalled={memoryInstalled}
        onOpenChange={setDialogOpen}
        onCreate={onCreateProject}
      /> : null}
    </section>
  );
}

/** 记忆策略在项目还没有对话时一次选定；之后改变需要显式重建学习图。 */
export function NewProjectDialog({
  open,
  memoryInstalled,
  onOpenChange,
  onCreate,
  onCloseFocus,
}: {
  open: boolean;
  memoryInstalled: boolean;
  onOpenChange: (open: boolean) => void;
  onCreate: (name: string, memory: ProjectMemory) => Promise<void>;
  onCloseFocus?: () => void;
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
      <DialogContent className="project-dialog" overlayClassName="project-dialog-overlay" onCloseAutoFocus={onCloseFocus ? (event) => {
        event.preventDefault();
        onCloseFocus();
      } : undefined}>
        <form onSubmit={(event) => void submit(event)}>
          <DialogHeader className="project-dialog__header">
            <DialogTitle>创建项目</DialogTitle>
          </DialogHeader>
          <label className="project-dialog__field">
            <span>项目名称</span>
            <span className="project-dialog__name-input">
              <Folder size={20} strokeWidth={1.75} aria-hidden="true" />
              <Input autoFocus value={name} maxLength={80} placeholder="例如：论文写作" disabled={submitting}
                onChange={(event) => setName(event.target.value)} />
            </span>
          </label>
          <DialogDescription className="project-dialog__description">
            <Lightbulb size={22} strokeWidth={1.5} aria-hidden="true" />
            <span>把相关对话放在一起，方便持续开展同一项工作。创建时可以选择记忆范围。</span>
          </DialogDescription>
          {error ? <p className="project-dialog__error" role="alert">{error}。若项目栏出现未确认请求，可在那里继续创建或停止尝试。</p> : null}
          <DialogFooter className="project-dialog__footer">
            {memoryInstalled ? <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button type="button" className="project-dialog__memory-trigger" disabled={submitting}
                  aria-label={`记忆设置：${projectMemoryLabel(memory)}`}>
                  {projectMemoryLabel(memory)} <ChevronDown size={15} aria-hidden="true" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="project-memory-menu" align="start" sideOffset={8} collisionPadding={16}>
                <DropdownMenuRadioGroup value={memory} aria-label="项目记忆">
                  {PROJECT_MEMORY_CHOICES.map((choice) => (
                    <DropdownMenuRadioItem key={choice.value} value={choice.value}
                      className="project-memory-menu__choice" onSelect={() => setMemory(choice.value)}>
                      <span><strong>{choice.label}</strong><small>{choice.description}</small></span>
                    </DropdownMenuRadioItem>
                  ))}
                </DropdownMenuRadioGroup>
                <p className="project-memory-menu__note">开始对话后，记忆选项将固定。</p>
              </DropdownMenuContent>
            </DropdownMenu> : <span className="project-dialog__no-memory">记忆功能未启用</span>}
            <button type="submit" className="project-dialog__button primary" disabled={!name.trim() || submitting}>
              {submitting ? "正在创建…" : "创建项目"}
            </button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

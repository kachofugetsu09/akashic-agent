import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { queryHostPlugin } from "./plugin-ui-runtime";
import { PROJECTS_PLUGIN, type ProjectRow } from "./web-projects";
import { directoryPage, directoryState, directoryStatus, type DirectoryPage, type DirectoryState } from "./directory-data";

/** 一个选择入口；固定后同一入口只读，不提供清空或重绑。 */
export function ProjectDirectoryDialog({ project, onClose, onBind, onCloseFocus }: {
  project: ProjectRow;
  onClose: () => void;
  onBind: (projectId: string, path: string) => Promise<void>;
  onCloseFocus?: () => void;
}) {
  const [path, setPath] = useState("~");
  const [page, setPage] = useState<DirectoryPage | null>(null);
  const [state, setState] = useState<DirectoryState | null>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const request = useRef<AbortController | null>(null);
  const heading = useRef<HTMLParagraphElement>(null);

  const browse = useCallback(async (target: string, after?: string) => {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setLoading(true);
    setError("");
    try {
      const next = directoryPage(await queryHostPlugin(PROJECTS_PLUGIN, "directory.browse", { path: target, ...(after ? { after } : {}) }, controller.signal));
      if (controller.signal.aborted) return;
      setPage(next);
      setPath(next.path!);
      heading.current?.focus();
    } catch (reason) {
      if (!controller.signal.aborted) {
        setPage(null);
        setError(reason instanceof Error ? reason.message : "目录读取失败");
      }
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!project.directory) { void browse("~"); return () => request.current?.abort(); }
    const controller = new AbortController();
    request.current = controller;
    void queryHostPlugin(PROJECTS_PLUGIN, "project.directory", { project_id: project.id }, controller.signal)
      .then((row) => { if (!controller.signal.aborted) setState(directoryState(row)); })
      .catch((reason: unknown) => { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "目录状态读取失败"); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [browse, project.id, project.directory]);

  const submit = async () => {
    if (!page?.path || loading || submitting) return;
    setSubmitting(true);
    setError("");
    try {
      await onBind(project.id, page.path);
      onClose();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "目录绑定失败");
      setSubmitting(false);
    }
  };
  const openPath = (event: FormEvent) => { event.preventDefault(); if (!loading) void browse(path); };

  return <Dialog open onOpenChange={(open) => { if (!open && !submitting) onClose(); }}>
    <DialogContent className="directory-dialog" overlayClassName="project-dialog-overlay"
      onCloseAutoFocus={onCloseFocus ? (event) => { event.preventDefault(); onCloseFocus(); } : undefined}>
      <DialogHeader><DialogTitle>{project.directory ? "固定目录" : "选择项目目录"}</DialogTitle></DialogHeader>
      <DialogDescription>{project.directory ? "这个项目的目录已固定。已有对话的工作目录各自独立。"
        : "目录可不设置。首次绑定后不可更改或清空；只作为之后新建对话的默认目录，已有对话不变。"}</DialogDescription>
      {project.directory ? <div>
        <p className="directory-path">{project.directory}</p>
        <p role="status">{loading ? "正在检查执行主机…" : state ? directoryStatus(state.status) : "状态未能读取"}</p>
        {state?.error ? <p className="directory-error">{state.error}</p> : null}
      </div> : <>
        <form className="directory-open" onSubmit={openPath}>
          <label htmlFor="project-directory-path">执行主机路径</label>
          <div><Input id="project-directory-path" value={path} onChange={(event) => setPath(event.target.value)} disabled={submitting} />
            <button type="submit" disabled={!path.trim() || loading || submitting}>打开</button></div>
        </form>
        <p ref={heading} tabIndex={-1} className="directory-path" aria-live="polite">{page?.path ?? "尚未选择目录"}</p>
        {loading ? <p role="status">正在读取执行主机目录…</p> : null}
        {page && !loading ? <div className="directory-browser" aria-label="子目录">
          {page.parent ? <button type="button" disabled={submitting} onClick={() => void browse(page.parent!)}>↑ 上一级</button> : null}
          {page.items.map((item) => <button type="button" key={item.name} disabled={submitting} onClick={() => void browse(item.path)}>{item.name} /</button>)}
          {!page.items.length ? <p>没有子目录，可以选择当前目录。</p> : null}
          {page.after ? <button type="button" disabled={submitting} onClick={() => void browse(page.path!, page.after!)}>下一页</button> : null}
        </div> : null}
        {page?.path ? <p className="directory-notice">将永久固定为：<strong className="directory-path">{page.path}</strong></p> : null}
      </>}
      {error ? <p className="directory-error" role="alert">{error}</p> : null}
      <DialogFooter className="directory-actions">
        <button type="button" disabled={submitting} onClick={onClose}>{project.directory ? "关闭" : "取消"}</button>
        {!project.directory ? <button type="button" disabled={!page || loading || submitting} onClick={() => void submit()}>
          {submitting ? "正在绑定…" : "绑定此目录"}</button> : null}
      </DialogFooter>
    </DialogContent>
  </Dialog>;
}

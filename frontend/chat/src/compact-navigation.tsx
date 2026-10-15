import { Menu } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Dialog, DialogContent, DialogTitle } from "./components/ui/dialog";
import { DesktopSidebar, type DesktopSidebarProps } from "./desktop-sidebar";
import { NewProjectDialog } from "./project-navigation";
import { reportShellOverlay } from "./shell-rail-actions";

// 与 styles.css 紧凑外壳媒体块同一条件：不再满足时常驻侧栏回来，抽屉没有存在的理由。
const COMPACT_SHELL = "(max-width: 820px), (max-height: 30rem) and (pointer: coarse)";

/** Expose the desktop navigation contract as a modal drawer on narrow viewports. */
export function CompactNavigation(props: DesktopSidebarProps) {
  const [open, setOpen] = useState(false);
  const [projectDialogOpen, setProjectDialogOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const openProjectAfterClose = useRef(false);
  const lastProjects = useRef(props.projects);
  if (props.projects) lastProjects.current = props.projects;
  const navigationProjects = props.projects;
  const closeThen = useCallback((action: () => void) => {
    setOpen(false);
    action();
  }, []);
  const overlayOpen = open || projectDialogOpen;
  const searchOnOpen = useRef(false);
  // 必要 effect：常驻侧栏被隐藏时 Ctrl/⌘+K 没有可见的搜索框，改为打开抽屉并聚焦其中的搜索（平板竖屏接键盘）。
  useEffect(() => {
    const compact = window.matchMedia(COMPACT_SHELL);
    const onKeyDown = (event: KeyboardEvent) => {
      if (!compact.matches || !(event.metaKey || event.ctrlKey) || event.altKey || event.shiftKey || event.isComposing) return;
      if (event.key.toLowerCase() !== "k") return;
      if (document.querySelector("[role='dialog'], [role='menu'], [role='listbox'], [data-radix-popper-content-wrapper]")) return;
      event.preventDefault();
      searchOnOpen.current = true;
      setOpen(true);
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, []);
  // 必要 effect：平板旋转、分屏拖宽时视口离开紧凑断点，抽屉藏了但模态遮罩还压着整页，必须跟着关掉。
  useEffect(() => {
    if (!open) return;
    const compact = window.matchMedia(COMPACT_SHELL);
    const onChange = () => { if (!compact.matches) setOpen(false); };
    compact.addEventListener("change", onChange);
    return () => compact.removeEventListener("change", onChange);
  }, [open]);
  useEffect(() => {
    if (!overlayOpen) return;
    reportShellOverlay(true);
    return () => reportShellOverlay(false);
  }, [overlayOpen]);

  return <>
    <button ref={triggerRef} className="compact-navigation-trigger" type="button" aria-label="打开导航" onClick={() => setOpen(true)}>
      <Menu aria-hidden="true" size={20} />
    </button>
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent
        className="compact-navigation-dialog"
        overlayClassName="compact-navigation-overlay"
        showCloseButton={false}
        onOpenAutoFocus={(event) => {
          if (!searchOnOpen.current) return;
          searchOnOpen.current = false;
          const search = event.currentTarget instanceof HTMLElement ? event.currentTarget.querySelector<HTMLInputElement>(".chat-sidebar__search input") : null;
          if (!search) return;
          event.preventDefault();
          search.focus();
        }}
        onCloseAutoFocus={(event) => {
          event.preventDefault();
          if (openProjectAfterClose.current) {
            openProjectAfterClose.current = false;
            setProjectDialogOpen(true);
          } else {
            triggerRef.current?.focus();
          }
        }}
      >
        <DialogTitle className="sr-only">Akashic 导航</DialogTitle>
        <DesktopSidebar
          {...props}
          projects={navigationProjects ? {
            ...navigationProjects,
            onNewChat: (projectId) => {
              closeThen(() => navigationProjects.onNewChat(projectId));
            },
            onContinue: async (key) => {
              await navigationProjects.onContinue(key);
              setOpen(false);
            },
            onStop: (key) => {
              navigationProjects.onStop(key);
              setOpen(false);
            },
            onOpenCreate: () => {
              openProjectAfterClose.current = true;
              setOpen(false);
            },
          } : undefined}
          onSelectSession={(sessionId) => closeThen(() => props.onSelectSession(sessionId))}
          onCycleTheme={() => closeThen(props.onCycleTheme)}
          onNewChat={() => closeThen(props.onNewChat)}
        />
      </DialogContent>
    </Dialog>
    {lastProjects.current ? <NewProjectDialog
      open={projectDialogOpen}
      memoryInstalled={lastProjects.current.memoryInstalled}
      onOpenChange={setProjectDialogOpen}
      onCreate={lastProjects.current.onCreate}
      onCloseFocus={() => {
        if (lastProjects.current?.pending.length) setOpen(true);
        else triggerRef.current?.focus();
      }}
    /> : null}
  </>;
}

import { Menu } from "lucide-react";
import { useCallback, useRef, useState } from "react";
import { Dialog, DialogContent, DialogTitle } from "./components/ui/dialog";
import { DesktopSidebar, type DesktopSidebarProps } from "./desktop-sidebar";
import { NewProjectDialog } from "./project-navigation";

/** Expose the desktop navigation contract as a modal drawer on narrow viewports. */
export function DesktopMobileNavigation(props: DesktopSidebarProps) {
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

  return <>
    <button ref={triggerRef} className="desktop-mobile-navigation-trigger" type="button" aria-label="打开导航" onClick={() => setOpen(true)}>
      <Menu aria-hidden="true" size={20} />
    </button>
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent
        className="desktop-mobile-navigation-dialog"
        overlayClassName="desktop-mobile-navigation-overlay"
        showCloseButton={false}
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
          onOpenPairing={() => closeThen(props.onOpenPairing)}
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

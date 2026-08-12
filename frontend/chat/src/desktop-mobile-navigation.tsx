import { Menu } from "lucide-react";
import { useCallback, useRef, useState } from "react";
import { Dialog, DialogContent, DialogTitle } from "./components/ui/dialog";
import { DesktopSidebar, type DesktopSidebarProps } from "./desktop-sidebar";

/** Expose the desktop navigation contract as a modal drawer on narrow viewports. */
export function DesktopMobileNavigation(props: DesktopSidebarProps) {
  const [open, setOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const closeThen = useCallback((action: () => void) => {
    setOpen(false);
    action();
  }, []);

  return <>
    <button ref={triggerRef} className="desktop-mobile-navigation-trigger" type="button" aria-label="打开导航" onClick={() => setOpen(true)}>
      <Menu aria-hidden="true" size={22} />
    </button>
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent
        className="desktop-mobile-navigation-dialog"
        overlayClassName="desktop-mobile-navigation-overlay"
        onCloseAutoFocus={(event) => {
          event.preventDefault();
          triggerRef.current?.focus();
        }}
      >
        <DialogTitle className="sr-only">Akashic 导航</DialogTitle>
        <DesktopSidebar
          {...props}
          onSelectSession={(sessionId) => closeThen(() => props.onSelectSession(sessionId))}
          onOpenRuntime={() => closeThen(props.onOpenRuntime)}
          onCycleTheme={() => closeThen(props.onCycleTheme)}
          onOpenPairing={() => closeThen(props.onOpenPairing)}
          onNewChat={() => closeThen(props.onNewChat)}
        />
      </DialogContent>
    </Dialog>
  </>;
}

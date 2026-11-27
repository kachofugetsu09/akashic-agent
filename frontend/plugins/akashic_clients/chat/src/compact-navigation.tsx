import { Menu } from "lucide-react";
import { memo, useCallback, useEffect, useRef, useState } from "react";
import { Dialog, DialogContent, DialogTitle } from "./components/ui/dialog";
import { useExitPresence } from "./presence";
import { DesktopSidebar, type DesktopSidebarProps } from "./desktop-sidebar";
import { GroupFormDialog, type GroupFormRequest } from "./navigation-groups";
import { DestinationFormDialog } from "./navigation-destinations";
import type { DestinationForm } from "@akashic/chat-ui-v1";
import { reportShellOverlay } from "./shell-rail-actions";
import { chatLayerOpen, keyForChat } from "./chat-host";
import { chatLayout, subscribeChatLayout } from "./chat-layout";
import { useSidebarPanelAlert } from "./sidebar-panels";

// 与 styles.css 紧凑外壳媒体块同一条件：不再满足时常驻侧栏回来，抽屉没有存在的理由。

const shown = (element: HTMLElement | null): element is HTMLElement => element !== null && element.isConnected && element.getClientRects().length > 0;

// 窄屏抽屉首帧只同步挂载首屏可见范围及少量余量，其余会话由宏任务分批补齐（#2239）。
const COMPACT_INITIAL_SESSION_ROWS = 28;

/** Expose the desktop navigation contract as a modal drawer on narrow viewports. */
export const CompactNavigation = memo(function CompactNavigation(props: DesktopSidebarProps) {
  const [open, setOpen] = useState(false);
  const drawer = useExitPresence(open, 260);
  const [formRequest, setFormRequest] = useState<GroupFormRequest | null>(null);
  const [destinationForm, setDestinationForm] = useState<DestinationForm | null>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  // 分组对话框不叠在抽屉上：先关抽屉，关完再打开。
  const formAfterClose = useRef<GroupFormRequest | null>(null);
  // 插件面板同理：底部抽屉等导航抽屉关完、焦点处理完再弹出。
  const actionAfterClose = useRef<(() => void) | null>(null);
  const panelAlert = useSidebarPanelAlert();
  const navigationGroups = props.groups;
  const closeThen = useCallback((action: () => void) => {
    setOpen(false);
    action();
  }, []);
  // 抽屉关完（焦点回到打开按钮）再执行：插件面板、功能设置这类压在页面上的层要等模态遮罩撤掉才能点。
  const afterClose = useCallback((action: () => void) => {
    actionAfterClose.current = action;
    setOpen(false);
  }, []);
  const overlayOpen = open || formRequest !== null || destinationForm !== null;
  const searchOnOpen = useRef(false);
  // 关闭后回到打开前的位置：快捷键打开时是当时的焦点（输入框里写到一半的草稿，光标位置由 textarea 自己保留），点按钮打开时是按钮。
  const returnFocus = useRef<HTMLElement | null>(null);
  // 必要 effect：常驻侧栏被隐藏时 Ctrl/⌘+K 没有可见的搜索框，改为打开抽屉并聚焦其中的搜索（平板竖屏接键盘）。
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!keyForChat(event) || !chatLayout().compact || !(event.metaKey || event.ctrlKey) || event.altKey || event.shiftKey || event.isComposing) return;
      if (event.key.toLowerCase() !== "k") return;
      if (chatLayerOpen()) return;
      event.preventDefault();
      searchOnOpen.current = true;
      returnFocus.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
      setOpen(true);
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, []);
  // 必要 effect：平板旋转、分屏拖宽时视口离开紧凑断点，抽屉藏了但模态遮罩还压着整页，必须跟着关掉。
  useEffect(() => {
    if (!open) return;
    return subscribeChatLayout(() => { if (!chatLayout().compact) setOpen(false); });
  }, [open]);
  useEffect(() => {
    if (!overlayOpen) return;
    reportShellOverlay(true);
    return () => reportShellOverlay(false);
  }, [overlayOpen]);

  return <>
    <button ref={triggerRef} className="compact-navigation-trigger" type="button" data-alert={panelAlert || undefined}
      aria-label={panelAlert ? "打开导航，插件面板有提醒" : "打开导航"} onClick={() => {
      returnFocus.current = triggerRef.current;
      setOpen(true);
    }}>
      <Menu aria-hidden="true" size={20} />
    </button>
    <Dialog open={open} onOpenChange={setOpen}>
      {drawer.present ? <DialogContent
        forceMount
        animated={false}
        className="compact-navigation-dialog"
        overlayClassName="compact-navigation-overlay"
        showCloseButton={false}
        inert={!open ? true : undefined}
        onTransitionEnd={drawer.onTransitionEnd}
        onOpenAutoFocus={(event) => {
          event.preventDefault();
          const dialog = event.currentTarget instanceof HTMLElement ? event.currentTarget : null;
          const focusSearch = searchOnOpen.current;
          searchOnOpen.current = false;
          // 同步 focus() 会在 Radix 刚改完 DOM/aria-hidden 时强制全页样式与布局重算；
          // 延后到下一帧（布局已自然完成）聚焦，既保留无障碍焦点又不阻塞点击当帧（#2239）。
          window.requestAnimationFrame(() => {
            if (!dialog?.isConnected) return;
            if (focusSearch) {
              const search = dialog.querySelector<HTMLInputElement>(".chat-sidebar__search input");
              if (search) {
                search.focus();
                return;
              }
            }
            (dialog.querySelector<HTMLElement>(".chat-sidebar__new") ?? dialog).focus({ preventScroll: true });
          });
        }}
        onCloseAutoFocus={(event) => {
          event.preventDefault();
          if (formAfterClose.current) {
            setFormRequest(formAfterClose.current);
            formAfterClose.current = null;
          } else if (actionAfterClose.current) {
            triggerRef.current?.focus({ preventScroll: true });
            actionAfterClose.current();
            actionAfterClose.current = null;
          } else {
            // 打开期间拉宽离开紧凑断点时原目标可能已隐藏：依次退到打开按钮、输入框。
            const composer = document.querySelector<HTMLElement>(".chat-main .composer textarea");
            [returnFocus.current, triggerRef.current, composer].find(shown)?.focus({ preventScroll: true });
          }
          returnFocus.current = null;
        }}
      >
        <DialogTitle className="sr-only">Akashic 导航</DialogTitle>
        <DesktopSidebar
          {...props}
          initialSessionRows={COMPACT_INITIAL_SESSION_ROWS}
          groups={navigationGroups ? {
            ...navigationGroups,
            startChat: (ref) => closeThen(() => navigationGroups.startChat(ref)),
            onOpenForm: (request) => {
              formAfterClose.current = request;
              setOpen(false);
            },
          } : undefined}
          destination={props.destination ? { ...props.destination, open: (sessionId) => closeThen(() => props.destination?.open(sessionId)) } : undefined}
          onOpenDestinationForm={(form) => afterClose(() => setDestinationForm(form))}
          onSelectSession={(sessionId) => closeThen(() => props.onSelectSession(sessionId))}
          onNewChat={() => closeThen(props.onNewChat)}
          onOpenPanels={afterClose}
          onRailAction={afterClose}
        />
      </DialogContent> : null}
    </Dialog>
    {destinationForm && props.destination?.source ? <DestinationFormDialog source={props.destination.source} form={destinationForm}
      controls={{ reload: props.destination.reload, open: props.destination.open, openForm: setDestinationForm }}
      onClose={() => { setDestinationForm(null); props.destination?.reload(); }} /> : null}
    {formRequest && navigationGroups ? <GroupFormDialog request={formRequest} groups={navigationGroups}
      onClose={() => setFormRequest(null)} onCloseFocus={() => triggerRef.current?.focus()} /> : null}
  </>;
});

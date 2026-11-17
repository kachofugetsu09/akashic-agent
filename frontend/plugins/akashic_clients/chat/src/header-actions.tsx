import { MoreHorizontal } from "lucide-react";
import { useRef, useState, useSyncExternalStore, type ReactNode } from "react";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { hostLink } from "./chat-host";
import { chatLayout, subscribeChatLayout } from "./chat-layout";

// Codex 16×16 原生像素网格放大镜，保持软角与轻细线重。
function FindIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <path
        fillRule="evenodd"
        clipRule="evenodd"
        d="M7.328 1.91C10.288 1.91 12.687 4.309 12.687 7.269C12.687 8.574 12.219 9.769 11.444 10.698L14.043 13.298C14.248 13.503 14.248 13.835 14.043 14.04C13.838 14.245 13.506 14.245 13.301 14.04L10.696 11.435C9.775 12.179 8.605 12.627 7.328 12.627C4.369 12.627 1.97 10.228 1.97 7.269C1.97 4.309 4.369 1.91 7.328 1.91ZM7.328 2.961C4.949 2.961 3.021 4.889 3.021 7.269C3.021 9.648 4.949 11.577 7.328 11.577C9.708 11.577 11.637 9.648 11.637 7.269C11.637 4.889 9.708 2.961 7.328 2.961Z"
        fill="currentColor"
      />
    </svg>
  );
}

// 宿主下发的标题行动作（例如工作台开关）：处理函数留在宿主，聊天只画按钮并回传 id。
export interface HeaderActionItem {
  id: string;
  label: string;
  iconSvg: string;
  /** 提示文字；默认用 label。 */
  hint?: string;
  attention?: boolean;
  active?: boolean;
}

let current: readonly HeaderActionItem[] = [];
const listeners = new Set<() => void>();

function isHeaderAction(entry: unknown): entry is HeaderActionItem {
  if (typeof entry !== "object" || entry === null) return false;
  const row = entry as Record<string, unknown>;
  return typeof row.id === "string" && typeof row.label === "string"
    && typeof row.iconSvg === "string" && row.iconSvg.startsWith("<svg")
    && (row.hint === undefined || typeof row.hint === "string")
    && (row.attention === undefined || typeof row.attention === "boolean")
    && (row.active === undefined || typeof row.active === "boolean");
}

export function setHeaderActions(value: unknown): void {
  if (!Array.isArray(value)) return;
  current = value.filter(isHeaderAction);
  for (const listener of listeners) listener();
}

function useHeaderActions(): readonly HeaderActionItem[] {
  return useSyncExternalStore((listener) => {
    listeners.add(listener);
    return () => { listeners.delete(listener); };
  }, () => current);
}

function HostIcon({ svg }: { svg: string }) {
  return <span className="conversation-heading__icon" aria-hidden="true" dangerouslySetInnerHTML={{ __html: svg }} />;
}

interface HeadingAction { id: string; label: string; hint: string; icon: ReactNode; attention?: boolean; active?: boolean; keys?: string; run(): void }

// 标题行右侧的动作区：宽屏逐个画图标按钮；紧凑外壳只留新会话，其余收进「⋯」，标题因此能拿到整行宽度。
export function HeaderActions({ onFind, newChat }: { onFind?: () => void; newChat: ReactNode }) {
  const compact = useSyncExternalStore(subscribeChatLayout, () => chatLayout().compact);
  const host = useHeaderActions();
  const [open, setOpen] = useState(false);
  // 菜单项打开的查找条、工作台要拿焦点：菜单的焦点约束解除（关菜单）之后才执行，也不再把焦点还给「⋯」。
  const pending = useRef<(() => void) | null>(null);
  const actions: HeadingAction[] = [
    ...onFind ? [{ id: "find", label: "在对话中查找", hint: "在对话中查找 (Ctrl/⌘F)", keys: "Control+F Meta+F",
      icon: <FindIcon />, run: onFind }] : [],
    ...host.map((item) => ({ id: item.id, label: item.label, hint: item.hint ?? item.label, attention: item.attention, active: item.active,
      icon: <HostIcon svg={item.iconSvg} />, run: () => hostLink()?.activateHeaderAction(item.id) })),
  ];
  if (!compact) {
    return <>{actions.map((action) => <button key={action.id} type="button" className="conversation-heading__action"
      data-header-action={action.id} data-attention={action.attention || undefined}
      aria-pressed={action.active !== undefined ? action.active : undefined} aria-label={action.hint}
      data-tip={action.hint} aria-keyshortcuts={action.keys} onClick={action.run}>{action.icon}</button>)}{newChat}</>;
  }
  return <>
    {newChat}
    {actions.length ? <DropdownMenu open={open} onOpenChange={setOpen}>
      <DropdownMenuTrigger asChild>
        <button type="button" className="conversation-heading__action conversation-heading__more" aria-label="更多操作"
          data-attention={actions.some((action) => action.attention) || undefined}>
          <MoreHorizontal size={20} strokeWidth={1.75} aria-hidden="true" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="conversation-heading__menu" onCloseAutoFocus={(event) => {
        const run = pending.current;
        pending.current = null;
        if (!run) return;
        event.preventDefault();
        run();
      }}>
        {actions.map((action) => <DropdownMenuItem key={action.id} data-header-action={action.id}
          data-attention={action.attention || undefined} onSelect={() => { pending.current = action.run; }}>
          {action.icon}<span>{action.label}</span>
        </DropdownMenuItem>)}
      </DropdownMenuContent>
    </DropdownMenu> : null}
  </>;
}

import { MoreHorizontal, Search } from "lucide-react";
import { useRef, useState, useSyncExternalStore, type ReactNode } from "react";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { hostLink } from "./chat-host";
import { chatLayout, subscribeChatLayout } from "./chat-layout";

// 宿主下发的标题行动作（例如工作台开关）：处理函数留在宿主，聊天只画按钮并回传 id。
export interface HeaderActionItem {
  id: string;
  label: string;
  iconSvg: string;
  /** 提示文字；默认用 label。 */
  hint?: string;
  attention?: boolean;
}

let current: readonly HeaderActionItem[] = [];
const listeners = new Set<() => void>();

function isHeaderAction(entry: unknown): entry is HeaderActionItem {
  if (typeof entry !== "object" || entry === null) return false;
  const row = entry as Record<string, unknown>;
  return typeof row.id === "string" && typeof row.label === "string"
    && typeof row.iconSvg === "string" && row.iconSvg.startsWith("<svg")
    && (row.hint === undefined || typeof row.hint === "string")
    && (row.attention === undefined || typeof row.attention === "boolean");
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

interface HeadingAction { id: string; label: string; hint: string; icon: ReactNode; attention?: boolean; keys?: string; run(): void }

// 标题行右侧的动作区：宽屏逐个画图标按钮；紧凑外壳只留新会话，其余收进「⋯」，标题因此能拿到整行宽度。
export function HeaderActions({ onFind, newChat }: { onFind?: () => void; newChat: ReactNode }) {
  const compact = useSyncExternalStore(subscribeChatLayout, () => chatLayout().compact);
  const host = useHeaderActions();
  const [open, setOpen] = useState(false);
  // 菜单项打开的查找条、工作台要拿焦点：菜单的焦点约束解除（关菜单）之后才执行，也不再把焦点还给「⋯」。
  const pending = useRef<(() => void) | null>(null);
  const actions: HeadingAction[] = [
    ...onFind ? [{ id: "find", label: "在对话中查找", hint: "在对话中查找 (Ctrl/⌘F)", keys: "Control+F Meta+F",
      icon: <Search size={20} strokeWidth={1.75} aria-hidden="true" />, run: onFind }] : [],
    ...host.map((item) => ({ id: item.id, label: item.label, hint: item.hint ?? item.label, attention: item.attention,
      icon: <HostIcon svg={item.iconSvg} />, run: () => hostLink()?.activateHeaderAction(item.id) })),
  ];
  if (!compact) {
    return <>{actions.map((action) => <button key={action.id} type="button" className="conversation-heading__action"
      data-header-action={action.id} data-attention={action.attention || undefined} aria-label={action.hint}
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

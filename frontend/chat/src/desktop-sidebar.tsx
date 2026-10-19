import {
  MessageSquarePlus,
  Search,
  Pin,
  PinOff,
  ArrowUpDown,
  Keyboard,
  SunMoon,
  X,
} from "lucide-react";
import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type RefObject } from "react";
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuRadioGroup,
  DropdownMenuRadioItem, DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { ConversationNavigation, ConversationSessionRow, type ConversationRowDrag, type ConversationSession } from "./conversation-navigation";
import akashicBrandIcon from "../../theme/src/akashic-brand.svg";
import { activateShellRailAction, useShellRailActions } from "./shell-rail-actions";
import {
  GroupFormDialog, GroupNavigation, GroupNavigationRow, groupFormActions, groupKey,
  type GroupFormRequest, type GroupSessionRenderer, type GroupSessionItem, type NavigationGroupsState,
} from "./navigation-groups";
import { sessionLabel } from "./web-chat-message-data";
import type { NavigationPin, NavigationPinsState } from "./use-navigation-pins";
import {
  SIDEBAR_RAIL_MAX_REM, SIDEBAR_RAIL_MIN_REM, SIDEBAR_RAIL_STEP_REM,
  type SidebarRailControl,
} from "./use-sidebar-rail";
import type { NavigationRowAction } from "./navigation-row-menu";
import { getHostShortcuts } from "./shell-workbench";
import { PluginUiSlot } from "./plugin-ui-runtime";
import { chatLayerOpen, keyForChat } from "./chat-host";

export interface DesktopSidebarSession extends Omit<ConversationSession, "active" | "state"> {
  active: boolean;
  /** 会话的维度取值；属于已知分组的会话列在该分组下。 */
  scope?: Readonly<Record<string, string>>;
  /** 属于某个会话维度（如项目）的会话归在该分组下，不单独置顶。 */
  scoped?: boolean;
  updatedAt?: string;
  createdAt?: string;
}

/** 会话排序是客户端展示投影：只作用于最近会话区与项目内子列表，置顶区保持服务端顺序。 */
type SessionSortId = "activity" | "created" | "title" | "manual";

const SESSION_SORT_CHOICES: readonly { id: SessionSortId; label: string }[] = [
  { id: "activity", label: "最近活动" },
  { id: "created", label: "最近创建" },
  { id: "title", label: "标题" },
  { id: "manual", label: "手动排序" },
];
const SESSION_SORT_KEY = "akashic.chat.session-sort";
const SESSION_ORDER_KEY = "akashic.chat.session-order";

function readSessionSort(): SessionSortId {
  try {
    const value = localStorage.getItem(SESSION_SORT_KEY);
    return SESSION_SORT_CHOICES.some((choice) => choice.id === value) ? value as SessionSortId : "activity";
  } catch {
    return "activity";
  }
}

function readSessionOrder(): string[] {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(SESSION_ORDER_KEY) ?? "[]");
    return Array.isArray(value) ? value.filter((id): id is string => typeof id === "string") : [];
  } catch {
    return [];
  }
}

/** 把持久化顺序与当前全集合并：新出现（未排序过）的会话按现有展示序排到最前。 */
function mergeSessionOrder(stored: readonly string[], ids: readonly string[]): string[] {
  const display = new Set(ids);
  const kept = stored.filter((id) => display.has(id));
  const ordered = new Set(kept);
  const fresh = ids.filter((id) => !ordered.has(id));
  return [...fresh, ...kept];
}

/** 时间倒序；缺失时间戳的行（如置顶解析补齐的目录外会话）稳定排到末尾。 */
function compareSessionTime(a: string | undefined, b: string | undefined): number {
  if (!a && !b) return 0;
  if (!a) return 1;
  if (!b) return -1;
  return Date.parse(b) - Date.parse(a);
}

// 按时间排序时的分组：自然日边界按本地时区计算，跨过 7 天统一收进"更早"。
function sessionDayGroup(iso: string | undefined, now: Date): string | undefined {
  const time = iso ? Date.parse(iso) : Number.NaN;
  if (Number.isNaN(time)) return undefined;
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const days = Math.floor((startOfToday - time) / 86_400_000) + 1;
  if (time >= startOfToday) return "今天";
  if (days <= 1) return "昨天";
  if (days <= 7) return "7 天内";
  return "更早";
}

const SHORTCUT_MOD = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform) ? "⌘" : "Ctrl";

// 常驻侧栏的两个全局快捷键：Mod+K 聚焦会话搜索，Mod+Shift+O 新会话。
// 抽屉里的侧栏副本与被隐藏的常驻侧栏不挂监听，避免同一按键触发两次。
const SEARCH_HIGHLIGHT = "sidebar-search";

const NAVIGATION_ROW = "button.conversation-session, button.group-session, button.navigation-group__open";

// 侧栏里可见、可用的导航行主按钮，按文档顺序：置顶、插件分组（含组内会话）、最近会话。
function sessionRows(root: HTMLElement | null): HTMLButtonElement[] {
  return Array.from(root?.querySelector(".chat-sidebar__navigation")?.querySelectorAll<HTMLButtonElement>(NAVIGATION_ROW) ?? [])
    .filter((row) => row.offsetParent !== null && !row.disabled);
}

// 会话列表只占一个 Tab 停靠点：键盘上次停过的行；它被搜索过滤或项目折叠卸掉后回退到当前会话，再回退到首个可用行。
// 停靠点按 DOM 统一维护（行组件不写 tabIndex，免得和重渲染互相覆盖）；切换当前会话后停靠点跟到新会话。
function useRovingTabStop(sidebarRef: RefObject<HTMLElement | null>) {
  // 必要 effect：列表增删、当前会话变化与焦点移动都在 DOM 上发生，只能在这里观测。
  useEffect(() => {
    const navigation = sidebarRef.current?.querySelector<HTMLElement>(".chat-sidebar__navigation");
    if (!navigation) return;
    let last: HTMLButtonElement | null = null;
    const sync = () => {
      const rows = [...navigation.querySelectorAll<HTMLButtonElement>(NAVIGATION_ROW)];
      const usable = rows.filter((row) => !row.disabled);
      const stop = (last && usable.includes(last) ? last : null)
        ?? usable.find((row) => row.getAttribute("aria-current") === "true") ?? usable[0];
      // 显式写属性：没写 tabindex 的按钮本身就是 0，按属性比较才不会漏掉。
      for (const row of rows) {
        const value = row === stop ? "0" : "-1";
        if (row.getAttribute("tabindex") !== value) row.setAttribute("tabindex", value);
      }
    };
    const observer = new MutationObserver((records) => {
      if (records.some((record) => record.attributeName === "aria-current")) last = null;
      sync();
    });
    const onFocus = (event: FocusEvent) => {
      if (!(event.target instanceof HTMLButtonElement) || !event.target.matches(NAVIGATION_ROW)) return;
      last = event.target;
      sync();
    };
    observer.observe(navigation, { childList: true, subtree: true, attributes: true, attributeFilter: ["aria-current", "disabled"] });
    navigation.addEventListener("focusin", onFocus);
    sync();
    return () => {
      observer.disconnect();
      navigation.removeEventListener("focusin", onFocus);
    };
  }, [sidebarRef]);
}

// 焦点在行上时 ↑↓ 逐行、Home/End 到首末行，从第一行再按 ↑ 回到搜索框。
function moveSessionFocus(event: React.KeyboardEvent<HTMLElement>) {
  if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return;
  const target = event.target;
  if (!(target instanceof HTMLButtonElement) || !target.matches(NAVIGATION_ROW)) return;
  const sidebar = target.closest<HTMLElement>(".chat-sidebar");
  const rows = sessionRows(sidebar);
  const index = rows.indexOf(target);
  event.preventDefault();
  if (event.key === "Home") rows[0]?.focus();
  else if (event.key === "End") rows.at(-1)?.focus();
  else if (event.key === "ArrowDown") rows[Math.min(index + 1, rows.length - 1)]?.focus();
  else if (index > 0) rows[index - 1].focus();
  else sidebar?.querySelector<HTMLInputElement>(".chat-sidebar__search input")?.focus();
}

// 搜索时用 CSS Custom Highlight 标出标题里的命中片段：不改行组件的 DOM，不支持的浏览器只是不标。
// 抽屉与常驻侧栏各有一份实例，只清理自己登记的那一份。
function useSearchHighlight(rootRef: RefObject<HTMLElement | null>, needle: string) {
  useEffect(() => {
    const root = rootRef.current;
    if (!needle || !root || typeof CSS === "undefined" || !("highlights" in CSS)) return;
    const registry = CSS.highlights;
    const mine = new Highlight();
    // 1. 列表随会话推送重渲染，标题文本节点会被替换；观察子树变化后重新取范围。
    const collect = () => {
      mine.clear();
      for (const title of root.querySelectorAll(".conversation-session__title, .group-session__title, .navigation-group__name")) {
        const walker = document.createTreeWalker(title, NodeFilter.SHOW_TEXT);
        for (let node = walker.nextNode(); node; node = walker.nextNode()) {
          const text = (node.textContent ?? "").toLowerCase();
          for (let at = text.indexOf(needle); at >= 0; at = text.indexOf(needle, at + needle.length)) {
            const range = new Range();
            range.setStart(node, at);
            range.setEnd(node, at + needle.length);
            mine.add(range);
          }
        }
      }
    };
    collect();
    registry.set(SEARCH_HIGHLIGHT, mine);
    const observer = new MutationObserver(collect);
    observer.observe(root, { subtree: true, childList: true, characterData: true });
    return () => {
      observer.disconnect();
      if (registry.get(SEARCH_HIGHLIGHT) === mine) registry.delete(SEARCH_HIGHLIGHT);
    };
  }, [needle, rootRef]);
}

function useSidebarShortcuts(sidebarRef: RefObject<HTMLElement | null>, searchRef: RefObject<HTMLInputElement | null>,
  onNewChat: () => void, onShortcuts: () => void) {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      // 抽屉里那份侧栏不接（抽屉本身是对话框）；常驻侧栏在紧凑断点被隐藏时仍接新会话与快捷键面板，
      // 搜索要先有可见的搜索框，交给抽屉（compact-navigation.tsx）打开后聚焦。
      const sidebar = sidebarRef.current;
      if (!sidebar || sidebar.closest("[role='dialog']")) return;
      if (!(event.metaKey || event.ctrlKey) || event.altKey || event.isComposing || !keyForChat(event)) return;
      // 对话框、菜单、选择器打开时按键属于它们，不把焦点抢到背后或在背后开新会话。
      if (chatLayerOpen()) return;
      const key = event.key.toLowerCase();
      if (key === "k" && !event.shiftKey) {
        if (sidebar.offsetParent === null) return;
        event.preventDefault();
        searchRef.current?.focus();
        searchRef.current?.select();
      } else if (key === "o" && event.shiftKey) {
        event.preventDefault();
        onNewChat();
      } else if (key === "/" && !event.shiftKey) {
        event.preventDefault();
        onShortcuts();
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [onNewChat, onShortcuts, searchRef, sidebarRef]);
}

// 快捷键散在各处（输入框、侧栏、阅读区），只有 title 提示时几乎没人发现；这里集中列一份。
// "?" 会被阅读时直接打字进入输入框接走，所以入口是 Ctrl+/（Slack、Linear 的习惯）和侧栏底部的键盘图标。
function ShortcutsDialog({ open, onOpenChange, returnFocus }: { open: boolean; onOpenChange: (open: boolean) => void; returnFocus: () => void }) {
  const mod = SHORTCUT_MOD === "⌘" ? "⌘" : "Ctrl";
  const groups: { title: string; items: [string, string[]][] }[] = [
    { title: "输入", items: [
      ["发送；生成中排到本轮之后", ["Enter"]],
      ["换行", ["Shift", "Enter"]],
      ["停止生成", ["Esc"]],
      ["取回上一条提问（输入框为空时）", ["↑"]],
      ["在输入框里翻看对话", ["PageUp", "PageDown"]],
      ["复制对话里最后一个代码块", [mod, "Shift", ";"]],
      ["阅读时直接打字，进入输入框", ["任意字符"]],
    ] },
    { title: "导航", items: [
      ["搜索会话", [mod, "K"]],
      ["新会话", [mod, "Shift", "O"]],
      ["在会话列表里上下移动", ["↑", "↓"]],
      ["会话的操作菜单", ["Shift", "F10"]],
      ["看图时切到同一条消息里的上一个、下一个", ["←", "→"]],
      ["快捷键", [mod, "/"]],
    ] },
  ];
  const host = getHostShortcuts();
  if (host.length) groups.push({ title: "工作台", items: host.map((item) => [item.label, [...item.keys]]) });
  return <Dialog open={open} onOpenChange={onOpenChange}>
    <DialogContent className="shortcuts-dialog" onCloseAutoFocus={(event) => { event.preventDefault(); returnFocus(); }}>
      <DialogHeader>
        <DialogTitle>键盘快捷键</DialogTitle>
        <DialogDescription className="sr-only">聊天界面可用的键盘快捷键</DialogDescription>
      </DialogHeader>
      {groups.map((group) => <section key={group.title} aria-label={group.title}>
        <h3>{group.title}</h3>
        <dl>{group.items.map(([label, keys]) => <div key={label}>
          <dt>{label}</dt>
          <dd>{keys.map((key) => <kbd key={key}>{key}</kbd>)}</dd>
        </div>)}</dl>
      </section>)}
    </DialogContent>
  </Dialog>;
}

function sortSessions(rows: DesktopSidebarSession[], sort: SessionSortId, order: readonly string[]): DesktopSidebarSession[] {
  if (sort === "title") return [...rows].sort((a, b) => a.title.localeCompare(b.title, "zh-Hans-CN"));
  if (sort === "created") return [...rows].sort((a, b) => compareSessionTime(a.createdAt, b.createdAt));
  if (sort === "manual") {
    const rank = new Map(order.map((id, index) => [id, index]));
    return [...rows].sort((a, b) => (rank.get(a.id) ?? -1) - (rank.get(b.id) ?? -1));
  }
  return [...rows].sort((a, b) => compareSessionTime(a.updatedAt, b.updatedAt));
}

/** 插件分组；抽屉里先关抽屉再由抽屉打开对话框，常驻侧栏自己打开。 */
export interface DesktopSidebarGroups extends NavigationGroupsState {
  onOpenForm?: (request: GroupFormRequest) => void;
}

export interface DesktopSidebarProps {
  surface: "chat" | "runtime";
  sessions: DesktopSidebarSession[];
  activeSessionId: string;
  pendingSessionId: string;
  chatReady: boolean;
  themeLabel: string;
  groups?: DesktopSidebarGroups;
  navigationPins: NavigationPinsState;
  onSelectSession: (sessionId: string) => void;
  onPrefetchSession?: (sessionId: string) => void;
  onCycleTheme: () => void;
  onNewChat: () => void;
  /** 会话软删：行内两步确认后调用；Promise 拒绝时行保持原位。 */
  onDeleteSession?: (sessionId: string, title: string) => Promise<void>;
  /** 会话重命名：双击或菜单进入行内编辑；Promise 拒绝时编辑态保留。 */
  onRenameSession?: (sessionId: string, title: string) => Promise<void>;
  /** 最近一次删除的撤销窗口；null 时不在导航区展示提示行。 */
  deletedNotice?: { key: string; title: string } | null;
  onRestoreSession?: (sessionId: string) => void;
  onDismissDeletedNotice?: () => void;
  /** 宽屏侧栏的宽度控制；抽屉（compact）不带此 prop，因此不渲染拖拽柄。 */
  rail?: SidebarRailControl;
}

/** 会话导航竖栏；底部动作由宿主 Shell 经 shell.rail-actions.v1 桥下发，主题行是 chat 域原生。 */

// 品牌剪影跟随主题墨色。
const brandMask = { WebkitMaskImage: `url(${akashicBrandIcon})`, maskImage: `url(${akashicBrandIcon})` };

export const DesktopSidebar = memo(function DesktopSidebar({
  surface,
  sessions,
  activeSessionId,
  pendingSessionId,
  themeLabel,
  groups,
  navigationPins,
  onSelectSession,
  onPrefetchSession,
  onCycleTheme,
  onNewChat,
  onDeleteSession,
  onRenameSession,
  deletedNotice,
  onRestoreSession,
  onDismissDeletedNotice,
  rail,
}: DesktopSidebarProps) {
  const [query, setQuery] = useState("");
  const sidebarRef = useRef<HTMLElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  // 面板没有固定触发按钮（输入框里按快捷键也能打开）：打开前记下焦点与光标，关闭后放回原处接着写。
  const shortcutsReturn = useRef<{ element: HTMLElement; selection: [number, number, "forward" | "backward" | "none"] | null } | null>(null);
  const openShortcuts = useCallback(() => {
    const element = document.activeElement instanceof HTMLElement && document.activeElement !== document.body ? document.activeElement : null;
    const field = element instanceof HTMLTextAreaElement || element instanceof HTMLInputElement ? element : null;
    shortcutsReturn.current = element ? { element, selection: field && field.selectionStart !== null && field.selectionEnd !== null ? [field.selectionStart, field.selectionEnd, field.selectionDirection ?? "none"] : null } : null;
    setShortcutsOpen(true);
  }, []);
  const returnShortcutsFocus = useCallback(() => {
    const target = shortcutsReturn.current;
    shortcutsReturn.current = null;
    if (!target?.element.isConnected) return;
    target.element.focus({ preventScroll: true });
    if (target.selection && (target.element instanceof HTMLTextAreaElement || target.element instanceof HTMLInputElement)) {
      target.element.setSelectionRange(...target.selection);
    }
  }, []);
  // 新会话继承当前所在的分组；要不归属任何分组，在新会话的分组条里改。
  const newChat = useCallback(() => groups?.active ? groups.startChat(groups.active) : onNewChat(), [groups, onNewChat]);
  useSidebarShortcuts(sidebarRef, searchRef, newChat, openShortcuts);
  const [formRequest, setFormRequest] = useState<GroupFormRequest | null>(null);
  const [sessionSort, setSessionSort] = useState(readSessionSort);
  const [sessionOrder, setSessionOrder] = useState(readSessionOrder);
  // 行拖拽的瞬态：被拖源 id + 当前插入指示（目标行 id 与上/下半）。提交时写入
  // sessionOrder 并把排序切到「手动排序」——拖拽本身就是选择手动模式的意图。
  const [drag, setDrag] = useState<{ id: string; over: { id: string; half: "before" | "after" } | null } | null>(null);
  const openForm = groups?.onOpenForm ?? setFormRequest;
  const needle = query.trim().toLowerCase();
  const searching = Boolean(needle);
  useSearchHighlight(sidebarRef, needle);
  useRovingTabStop(sidebarRef);
  const allSessions = useMemo(() => {
    const rows = new Map<string, DesktopSidebarSession>(navigationPins.sessions.map((session) => [session.key, {
      id: session.key, title: sessionLabel(session), preview: session.message_count === undefined ? "" : `${session.message_count} 条消息`,
      active: activeSessionId === session.key,
      scope: session.scope, scoped: Object.keys(session.scope ?? {}).length > 0,
    }]));
    // 目录里的新鲜 metadata 优先；置顶解析补齐不在最近分页内的会话。
    for (const session of sessions) rows.set(session.id, session);
    return [...rows.values()];
  }, [navigationPins.sessions, sessions, activeSessionId]);
  const filteredSessions = useMemo(() => needle
    ? allSessions.filter((session) => `${session.title} ${session.preview}`.toLowerCase().includes(needle))
    : allSessions, [allSessions, needle]);
  // 排序只重排展示顺序：最近会话区与项目内子列表共享同一份投影，置顶区不经过它。
  const mergedOrder = useMemo(
    () => mergeSessionOrder(sessionOrder, allSessions.map((session) => session.id)),
    [sessionOrder, allSessions],
  );
  const sortedSessions = useMemo(() => sortSessions(filteredSessions, sessionSort, mergedOrder), [filteredSessions, sessionSort, mergedOrder]);
  useEffect(() => {
    try { localStorage.setItem(SESSION_SORT_KEY, sessionSort); }
    catch { /* 禁用存储时仍允许本次浏览的排序选择。 */ }
  }, [sessionSort]);
  useEffect(() => {
    try { localStorage.setItem(SESSION_ORDER_KEY, JSON.stringify(sessionOrder)); }
    catch { /* 禁用存储时仍允许本次浏览的手动排序。 */ }
  }, [sessionOrder]);
  // 行拖拽期间在 document 层接受 dragover/drop：行外不显示"禁止"光标，
  // dragend 的提交仍由行自己的 onDrop 完成。
  useEffect(() => {
    if (!drag) return;
    const acceptDrag = (event: DragEvent) => {
      event.preventDefault();
      if (event.dataTransfer !== null) event.dataTransfer.dropEffect = "move";
    };
    const acceptDrop = (event: DragEvent) => { event.preventDefault(); };
    document.addEventListener("dragover", acceptDrag);
    document.addEventListener("drop", acceptDrop);
    return () => {
      document.removeEventListener("dragover", acceptDrag);
      document.removeEventListener("drop", acceptDrop);
    };
  }, [drag]);
  const commitManualDrop = (targetId: string, half: "before" | "after") => {
    if (!drag || drag.id === targetId) return;
    const order = mergedOrder.filter((id) => id !== drag.id);
    const targetIndex = order.indexOf(targetId);
    if (targetIndex === -1) return;
    order.splice(targetIndex + (half === "after" ? 1 : 0), 0, drag.id);
    // 当前分页外的会话保留顺序尾巴，不因为本次可见提交丢掉位置。
    const offscreen = sessionOrder.filter((id) => !mergedOrder.includes(id));
    setSessionOrder([...order, ...offscreen]);
    setSessionSort("manual");
  };
  const sessionDrag = searching ? undefined : (session: ConversationSession): ConversationRowDrag => ({
    active: drag !== null,
    source: drag?.id === session.id,
    marker: drag?.over?.id === session.id ? drag.over.half : undefined,
    start: () => setDrag({ id: session.id, over: null }),
    end: () => setDrag(null),
    hover: (half) => setDrag((current) => current
      && (current.over?.id !== session.id || current.over.half !== half)
      ? { ...current, over: { id: session.id, half } }
      : current),
    drop: (half) => {
      commitManualDrop(session.id, half);
      setDrag(null);
    },
  });
  // 目录未解析/归档不改变 Session.scope；这类会话回退到最近会话，仍可在全局置顶。
  const { expandGroup } = navigationPins;
  const activeGroupKey = groups?.active ? groupKey(groups.active) : "";
  useEffect(() => { if (activeGroupKey) expandGroup(activeGroupKey); }, [activeGroupKey, expandGroup]);
  const sections = useMemo(() => groups?.sections ?? [], [groups?.sections]);
  const knownGroups = useMemo(() => new Map(sections.flatMap((section) => section.groups.map((group) =>
    [groupKey({ dimension: section.source.provider.dimension, id: group.id }), { section, group }] as const))), [sections]);
  // 会话归入第一个认得它取值的分区；都不认得时留在最近会话。
  const groupOf = useCallback((session: DesktopSidebarSession) => sections
    .map(({ source }) => session.scope?.[source.provider.dimension] ? groupKey({ dimension: source.provider.dimension, id: session.scope[source.provider.dimension] }) : "")
    .find((key) => key && knownGroups.has(key)) ?? "", [sections, knownGroups]);
  const pinnedSessions = useMemo(() => new Set(navigationPins.pins.filter((pin) => pin.kind === "session").map((pin) => pin.id)),
    [navigationPins.pins]);
  // 分组内置顶的会话排在所在文件夹最前面，其余保持当前排序。
  const sessionsByGroup = useMemo(() => {
    const byGroup = new Map<string, GroupSessionItem[]>();
    for (const session of sortedSessions) {
      const key = groupOf(session);
      if (!key) continue;
      byGroup.set(key, [...byGroup.get(key) ?? [], {
        id: session.id, title: session.title, active: surface === "chat" && session.active, status: session.status,
        pinned: pinnedSessions.has(session.id),
      }]);
    }
    for (const [key, items] of byGroup) byGroup.set(key, [...items.filter((item) => item.pinned), ...items.filter((item) => !item.pinned)]);
    return byGroup;
  }, [sortedSessions, groupOf, surface, pinnedSessions]);
  const pinnedGroups = new Set(navigationPins.pins.filter((pin) => pin.kind !== "session").map((pin) => groupKey({ dimension: pin.kind, id: pin.id })));
  const recentSessions = sortedSessions.filter((session) => !pinnedSessions.has(session.id) && !groupOf(session));
  const visibleGroup = (key: string, label: string) => !needle || label.toLowerCase().includes(needle)
    || Boolean(sessionsByGroup.get(key)?.length);
  const sectionGroups = sections.map((section) => section.groups.filter((group) => {
    const key = groupKey({ dimension: section.source.provider.dimension, id: group.id });
    return !pinnedGroups.has(key) && visibleGroup(key, group.label);
  }));
  // 搜索时只留有匹配行的置顶分组，空分组标题一起收起。
  const visiblePins = navigationPins.pins.filter((pin) => {
    if (pin.kind !== "session") {
      const key = groupKey({ dimension: pin.kind, id: pin.id });
      const known = knownGroups.get(key);
      if (known) return visibleGroup(key, known.group.label);
    } else {
      const session = allSessions.find((item) => item.id === pin.id);
      // 分组里的会话置顶在自己的文件夹内，不进全局置顶区；分组已不在时仍按普通置顶显示。
      if (session && groupOf(session)) return false;
      if (session) {
        return `${session.title} ${session.preview}`.toLowerCase().includes(needle);
      }
    }
    return pin.id.toLowerCase().includes(needle);
  });
  const hasMatches = recentSessions.length > 0 || sectionGroups.some((items) => items.length > 0) || visiblePins.length > 0;
  const pinAction = (pin: NavigationPin, pinned: boolean): NavigationRowAction[] => [{
    label: pinned ? "取消置顶" : "置顶",
    icon: pinned ? <PinOff size={18} aria-hidden="true" /> : <Pin size={18} aria-hidden="true" />,
    disabled: !navigationPins.ready || navigationPins.pending,
    onSelect: () => { void navigationPins.setPinned(pin, !pinned); },
  }];
  const sessionById = useMemo(() => new Map(allSessions.map((session) => [session.id, session])), [allSessions]);
  // 分组内的会话行：与最近会话同一套行菜单，置顶只在所在分组内生效。
  const groupSessionRow: GroupSessionRenderer = (item, label) => <ConversationSessionRow
    session={{ id: item.id, title: item.title, preview: "", active: item.active, status: item.status, pinned: item.pinned }}
    pendingSessionId={pendingSessionId} onActivate={onSelectSession} onPrefetch={onPrefetchSession}
    actions={[{ ...pinAction({ kind: "session", id: item.id }, Boolean(item.pinned))[0],
      label: item.pinned ? `取消${label}内置顶` : `在${label}内置顶` }]}
    onDelete={onDeleteSession ? () => onDeleteSession(item.id, item.title) : undefined}
    onRename={onRenameSession ? (title: string) => onRenameSession(item.id, title) : undefined} />;
  // 分组只按渲染时刻的日期，跨午夜后随下一次列表更新重算即可。
  const groupNow = new Date();
  const sessionView = (session: DesktopSidebarSession): ConversationSession => ({
    ...session, active: surface === "chat" && session.active,
  });
  const deleteHandler = onDeleteSession
    ? (session: ConversationSession) => onDeleteSession(session.id, session.title)
    : undefined;
  const railActions = useShellRailActions();

  return (
    <aside ref={sidebarRef} className="chat-sidebar chat-sidebar--entry">
      <div className="chat-sidebar__toolbar">
        <div className="chat-sidebar__brand-row">
          <span className="chat-sidebar__brand">
            <span className="chat-sidebar__mark" style={brandMask} aria-hidden="true" />
            <strong>Akashic</strong>
          </span>
          <button type="button" className="chat-sidebar__new" aria-label="新会话" title={`新会话（${SHORTCUT_MOD}+Shift+O）`} onClick={newChat}>
            <MessageSquarePlus size={18} strokeWidth={1.75} aria-hidden="true" />
          </button>
        </div>
        <label className="chat-sidebar__search">
          <Search size={14} aria-hidden="true" />
          <input
            ref={searchRef}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.nativeEvent.isComposing || event.nativeEvent.keyCode === 229) return;
              // 1. Esc 先清空，再按一次把焦点还给页面。
              if (event.key === "Escape") {
                if (query) setQuery("");
                else event.currentTarget.blur();
                return;
              }
              // 2. 回车打开第一条结果，↓ 进入结果列表；与命令面板同一手感。
              if (event.key !== "Enter" && event.key !== "ArrowDown") return;
              // 回车只打开会话，跳过项目标题行；↓ 落到列表首行（可能是项目）。
              const rows = sessionRows(sidebarRef.current);
              const first = event.key === "Enter" ? rows.find((row) => !row.matches(".navigation-group__open")) : rows[0];
              if (!first || (event.key === "Enter" && !searching)) return;
              event.preventDefault();
              if (event.key !== "Enter") { first.focus(); return; }
              // 回车是"跳过去"而不是"边搜边看"：打开后清掉筛选，宽屏把焦点交给输入框接着写；
              // 抽屉里不抢焦点，免得软键盘再弹出来。
              first.click();
              setQuery("");
              if (!sidebarRef.current?.closest("[role='dialog']")) {
                window.requestAnimationFrame(() => document.querySelector<HTMLTextAreaElement>(".composer-wrap textarea")?.focus({ preventScroll: true }));
              }
            }}
            placeholder="搜索会话"
            aria-label="搜索会话"
            aria-keyshortcuts={`${SHORTCUT_MOD === "⌘" ? "Meta" : "Control"}+K`}
          />
          <kbd className="chat-sidebar__search-key" aria-hidden="true">{SHORTCUT_MOD === "⌘" ? "⌘K" : "Ctrl K"}</kbd>
          {/* 触屏没有 Esc：有输入时给一个清除钮，清完焦点留在框里接着搜。 */}
          {query ? <button type="button" className="chat-sidebar__search-clear" aria-label="清除搜索" data-tip="清除搜索"
            onClick={() => { setQuery(""); searchRef.current?.focus(); }}>
            <X size={14} aria-hidden="true" />
          </button> : null}
        </label>
      </div>

      <div className="chat-sidebar__navigation" onKeyDown={moveSessionFocus}>
      {searching && !hasMatches ? <p className="navigation-search-empty" role="status">没有匹配的会话或分组</p> : null}
      {navigationPins.error ? <div className="navigation-pins-error" role="alert">
        <span>{navigationPins.error}</span>
        <button type="button" disabled={navigationPins.pending} onClick={() => { void navigationPins.reload(); }}>刷新置顶列表</button>
      </div> : null}
      {deletedNotice ? <div className="session-delete-notice" role="status">
        <span title={deletedNotice.title}>已删除「{deletedNotice.title}」</span>
        <button type="button" onClick={() => onRestoreSession?.(deletedNotice.key)}>撤销</button>
        <button type="button" className="session-delete-notice__dismiss" aria-label="关闭提示" onClick={onDismissDeletedNotice}>
          <X size={14} aria-hidden="true" />
        </button>
      </div> : null}
      {visiblePins.length ? <section className="pinned-navigation" aria-label="置顶">
        <header className="navigation-section__header"><span>置顶</span></header>
        {visiblePins.map((pin) => {
          if (pin.kind !== "session") {
            const ref = { dimension: pin.kind, id: pin.id };
            const known = knownGroups.get(groupKey(ref));
            if (known && groups) return visibleGroup(groupKey(ref), known.group.label) ? <GroupNavigationRow
              key={groupKey(ref)} groupRef={ref} group={known.group}
              items={sessionsByGroup.get(groupKey(ref)) ?? []}
              open={searching || navigationPins.expandedGroups.has(groupKey(ref))}
              active={activeGroupKey === groupKey(ref)} pendingSessionId={pendingSessionId}
              onToggle={() => navigationPins.toggleGroup(groupKey(ref))}
              onNewChat={() => groups.startChat(ref)}
              onSelectSession={onSelectSession} onPrefetchSession={onPrefetchSession}
              actions={[...groupFormActions(known.section.source, known.group, openForm), ...pinAction(pin, true)]} searching={searching}
              renderSession={(item) => groupSessionRow(item, known.section.source.provider.label)}
            /> : null;
          } else {
            const session = allSessions.find((item) => item.id === pin.id);
            if (session && !groupOf(session)) return !needle
              || `${session.title} ${session.preview}`.toLowerCase().includes(needle) ? <ConversationSessionRow
                key={`session:${pin.id}`} session={sessionView(session)} pendingSessionId={pendingSessionId}
                onActivate={onSelectSession} onPrefetch={onPrefetchSession}
                actions={pinAction(pin, true)}
                onDelete={onDeleteSession ? () => onDeleteSession(session.id, session.title) : undefined}
                onRename={onRenameSession ? (title: string) => onRenameSession(session.id, title) : undefined}
              /> : null;
          }
          if (needle && !pin.id.toLowerCase().includes(needle)) return null;
          // 分组取值已不存在或其插件不在：有该维度的分区时用它的名称，否则只说暂不可用。
          const label = pin.kind === "session" ? "会话" : sections.find((section) => section.source.provider.dimension === pin.kind)?.source.provider.label ?? "";
          const title = `${label}暂不可用`;
          return <ConversationSessionRow key={`${pin.kind}:${pin.id}`}
            session={{ id: pin.id, title, preview: pin.id, active: false, unavailable: true }}
            onActivate={onSelectSession} actions={pinAction(pin, true)} />;
        })}
      </section> : null}
      {groups ? sections.map((section, index) => {
        const { dimension, label } = section.source.provider;
        return <GroupNavigation key={section.source.key} section={section} groups={sectionGroups[index]}
          revision={groups.revision} sessionsByGroup={sessionsByGroup} active={groups.active}
          pendingSessionId={pendingSessionId} expanded={navigationPins.expandedGroups} searching={searching}
          heading={navigationPins.pins.some((pin) => pin.kind === dimension) ? `其他${label}` : label}
          onToggle={navigationPins.toggleGroup} onSelectSession={onSelectSession} onPrefetchSession={onPrefetchSession}
          onOpenForm={openForm} onNewChat={groups.startChat} onReload={groups.reload} renderSession={groupSessionRow}
          actions={(group) => [...groupFormActions(section.source, group, openForm), ...pinAction({ kind: dimension, id: group.id }, false)]} />;
      }) : null}

      <ConversationNavigation
        destinationHeading={false}
        sessionHeading={!searching ? "最近会话" : recentSessions.length ? "匹配的会话" : ""}
        sessionHeadingAction={searching ? undefined : <SessionSortMenu sort={sessionSort} onSort={setSessionSort} />}
        destinations={[]}
        actions={[]}
        sessions={recentSessions.map(sessionView)}
        sessionGroup={sessionSort === "activity" || sessionSort === "created" ? (session) => {
          const row = sessionById.get(session.id);
          return sessionDayGroup(sessionSort === "created" ? row?.createdAt : row?.updatedAt, groupNow);
        } : undefined}
        sessionDrag={sessionDrag}
        onSessionDelete={deleteHandler}
        onSessionRename={onRenameSession ? (session, title) => onRenameSession(session.id, title) : undefined}
        sessionActions={(session) => {
          const row = allSessions.find((item) => item.id === session.id);
          // 分组已不在的会话也落在这里，同样可以置顶（后端已接受带分组的会话）。
          return row ? pinAction({ kind: "session", id: row.id }, false) : [];
        }}
        onSessionActivate={onSelectSession}
        onSessionPrefetch={onPrefetchSession}
        pendingSessionId={pendingSessionId}
      />
      {activeSessionId && <PluginUiSlot name="drawer.panel" sessionId={activeSessionId} quiet />}
      </div>
      <div className="chat-sidebar__footer">
        {railActions.map((action) => (
          <button key={action.id} type="button" className="chat-sidebar__action"
            onClick={() => activateShellRailAction(action.id)}>
            <span className="chat-sidebar__action-icon" aria-hidden="true"
              dangerouslySetInnerHTML={{ __html: action.iconSvg }} />
            <span>{action.label}</span>
          </button>
        ))}
        <div className="chat-sidebar__footer-row">
          <button type="button" className="chat-sidebar__action"
            onClick={() => onCycleTheme()}
            title={`切换主题，当前为${themeLabel}`}
            aria-label={`切换主题，当前为${themeLabel}`}>
            <SunMoon size={18} aria-hidden="true" />
            <span>主题 · {themeLabel}</span>
          </button>
          <button type="button" className="chat-sidebar__shortcuts" aria-label="键盘快捷键" title={`键盘快捷键（${SHORTCUT_MOD}+/）`}
            onClick={openShortcuts}>
            <Keyboard size={16} aria-hidden="true" />
          </button>
        </div>
      </div>
      <ShortcutsDialog open={shortcutsOpen} onOpenChange={setShortcutsOpen} returnFocus={returnShortcutsFocus} />
      {rail ? <SidebarRailResizer rail={rail} sidebarRef={sidebarRef} /> : null}
      {formRequest && groups ? <GroupFormDialog request={formRequest} groups={groups} onClose={() => setFormRequest(null)}
        onCloseFocus={() => groupFormFocus(sidebarRef.current, formRequest)} /> : null}
    </aside>
  );
});

// 对话框关闭后回到打开它的地方：分组动作回到该行的菜单按钮，新建回到分区的新建按钮。
function groupFormFocus(root: HTMLElement | null, request: GroupFormRequest) {
  const { dimension, label } = request.source.provider;
  const target = request.group
    ? Array.from(root?.querySelectorAll<HTMLElement>("[data-group-key]") ?? [])
      .find((row) => row.dataset.groupKey === groupKey({ dimension, id: request.group!.id }))?.querySelector<HTMLElement>(".navigation-menu-trigger")
    : root?.querySelector<HTMLElement>(`.navigation-section[aria-label="${CSS.escape(label)}"] .navigation-section__header button`);
  target?.focus();
}

/** "最近会话"分组标题行的排序入口；选中项持久化，菜单复用行菜单的浮层样式。 */
function SessionSortMenu({ sort, onSort }: { sort: SessionSortId; onSort: (sort: SessionSortId) => void }) {
  const current = SESSION_SORT_CHOICES.find((choice) => choice.id === sort) ?? SESSION_SORT_CHOICES[0];
  return <DropdownMenu modal={false}>
    <DropdownMenuTrigger asChild>
      <button type="button" className="navigation-section__icon conversation-sort-trigger"
        aria-label={`会话排序：${current.label}`} data-tip="会话排序">
        <ArrowUpDown size={15} aria-hidden="true" />
      </button>
    </DropdownMenuTrigger>
    <DropdownMenuContent className="navigation-row-menu" align="end" sideOffset={4} collisionPadding={12} aria-label="会话排序">
      <DropdownMenuRadioGroup value={sort} aria-label="会话排序">
        {SESSION_SORT_CHOICES.map((choice) => <DropdownMenuRadioItem key={choice.id} value={choice.id}
          className="session-sort-menu__choice" onSelect={() => onSort(choice.id)}>
          <span>{choice.label}</span>
        </DropdownMenuRadioItem>)}
      </DropdownMenuRadioGroup>
    </DropdownMenuContent>
  </DropdownMenu>;
}

/** 侧栏右缘拖拽柄：只写 --chat-rail-width；双击复位，方向键以 0.5rem 步进。 */
function SidebarRailResizer({ rail, sidebarRef }: {
  rail: SidebarRailControl;
  sidebarRef: RefObject<HTMLElement | null>;
}) {
  const [dragging, setDragging] = useState(false);
  const [nowRem, setNowRem] = useState(SIDEBAR_RAIL_MIN_REM);
  useLayoutEffect(() => {
    const aside = sidebarRef.current;
    if (!aside) return;
    const rootPx = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
    setNowRem(aside.getBoundingClientRect().width / rootPx);
  }, [rail.widthRem, sidebarRef]);
  const widthPx = () => sidebarRef.current?.getBoundingClientRect().width ?? 0;
  return <div
    className="chat-sidebar__resizer"
    role="separator"
    aria-orientation="vertical"
    aria-label="侧栏宽度"
    aria-valuemin={SIDEBAR_RAIL_MIN_REM}
    aria-valuemax={SIDEBAR_RAIL_MAX_REM}
    aria-valuenow={Math.round(nowRem * 2) / 2}
    tabIndex={0}
    data-dragging={dragging || undefined}
    onPointerDown={(event) => {
      if (!event.isPrimary || event.button !== 0) return;
      event.preventDefault();
      event.currentTarget.setPointerCapture(event.pointerId);
      rail.dragStart(event.clientX, widthPx());
      setDragging(true);
    }}
    onPointerMove={(event) => {
      if (dragging) rail.dragTo(event.clientX);
    }}
    onPointerUp={(event) => {
      if (!dragging) return;
      event.currentTarget.releasePointerCapture(event.pointerId);
      setDragging(false);
    }}
    onPointerCancel={() => setDragging(false)}
    onDoubleClick={(event) => {
      event.preventDefault();
      rail.reset();
    }}
    onKeyDown={(event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      rail.stepBy(event.key === "ArrowRight" ? SIDEBAR_RAIL_STEP_REM : -SIDEBAR_RAIL_STEP_REM, widthPx());
    }}
  />;
}

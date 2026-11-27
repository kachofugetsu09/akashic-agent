import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { Copy, Ellipsis, PencilLine, Reply } from "lucide-react";
import type { MessageMenuItem, MessageMetaItem } from "@akashic/chat-ui-v1";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { chatPortalRoot } from "./chat-layout";
import { CheckGlyph, IconSwap } from "./icon-swap";
import { PluginUiSlot, type PluginMountExtra } from "./plugin-ui-runtime";

interface MenuEntry {
  id: string;
  label: string;
  icon: ReactNode;
  run: () => void;
}

interface MenuRequest {
  key: number;
  x: number;
  y: number;
  items: readonly MenuEntry[];
  restore: HTMLElement | null;
}

const MAX_ITEMS = 8;
const fullTime = new Intl.DateTimeFormat("zh-CN", { dateStyle: "medium", timeStyle: "medium", hour12: false });
// 每条信息行在自己的根元素上登记当前菜单：右键委托按元素取菜单，不必让每条消息各挂一份 Radix 菜单。
const rowMenus = new WeakMap<Element, () => readonly MenuEntry[]>();
let showMenu: ((request: Omit<MenuRequest, "key">) => void) | null = null;

const metaValid = (item: MessageMetaItem) => typeof item?.id === "string" && typeof item.label === "string"
  && typeof item.value === "string" && (item.text === undefined || typeof item.text === "string");
const menuValid = (item: MessageMenuItem) => typeof item?.id === "string" && typeof item.label === "string"
  && typeof item.run === "function";
const pluginIcon = (svg: string | undefined) => svg
  ? <span className="message-menu__icon" aria-hidden="true" dangerouslySetInnerHTML={{ __html: svg }} /> : <span className="message-menu__icon" />;

// 插件登记的信息项与菜单项：每个插件一份稳定的登记能力，同一插件重复登记时整份替换。
function usePluginEntries() {
  const [meta, setMeta] = useState<ReadonlyMap<string, readonly MessageMetaItem[]>>(new Map());
  const [menu, setMenu] = useState<ReadonlyMap<string, readonly MessageMenuItem[]>>(new Map());
  const owned = useRef(new Map<string, PluginMountExtra>());
  const extra = useCallback((pluginId: string) => {
    let controls = owned.current.get(pluginId);
    if (!controls) {
      controls = {
        meta: { publish: (items: readonly MessageMetaItem[]) => setMeta((current) => new Map(current)
          .set(pluginId, (Array.isArray(items) ? items.filter(metaValid) : []).slice(0, MAX_ITEMS))) },
        menu: { publish: (items: readonly MessageMenuItem[]) => setMenu((current) => new Map(current)
          .set(pluginId, (Array.isArray(items) ? items.filter(menuValid) : []).slice(0, MAX_ITEMS))) },
      };
      owned.current.set(pluginId, controls);
    }
    return controls;
  }, []);
  return { meta: [...meta.values()].flat(), menu: [...menu.entries()], extra };
}

// 回答与提问下方的一行：时间和插件信息是一行淡色小字（点开看明细）；动作不常驻，悬停只露复制与「⋯」，完整菜单在「⋯」和右键里。
export function MessageInfo({ role, lead, timestamp, slots, copied, onCopy, onReply, onEdit }: {
  role: "user" | "assistant";
  /** 时间或发送状态。 */
  lead: ReactNode;
  timestamp?: string;
  /** 完整回答才挂插件的信息与菜单。 */
  slots?: { sessionId: string; messageId: string };
  copied: boolean;
  onCopy?: () => void;
  onReply?: () => void;
  /** 只给用户自己的提问：把原文放回输入框，改一改再问。 */
  onEdit?: () => void;
}) {
  const rowRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const plugins = usePluginEntries();
  const inline = plugins.meta.filter((item) => item.tone === "alert" || item.text);
  const entries = (): MenuEntry[] => [
    ...(onReply ? [{ id: "reply", label: "回复", icon: <Reply size={16} aria-hidden="true" />, run: onReply }] : []),
    ...(onEdit ? [{ id: "edit", label: "放回输入框编辑", icon: <PencilLine size={16} aria-hidden="true" />, run: onEdit }] : []),
    ...(onCopy ? [{ id: "copy", label: "复制", icon: <Copy size={16} aria-hidden="true" />, run: onCopy }] : []),
    ...plugins.menu.flatMap(([pluginId, items]) => items.map((item) => ({
      id: `${pluginId}.${item.id}`, label: item.label, icon: pluginIcon(item.iconSvg), run: () => item.run() }))),
  ];
  const menuRef = useRef(entries);
  menuRef.current = entries;
  useEffect(() => {
    const row = rowRef.current;
    if (!row) return;
    rowMenus.set(row, () => menuRef.current());
    return () => { rowMenus.delete(row); };
  }, []);
  return <>
    <div ref={rowRef} className={`shared-message-meta ${role}`}>
      {plugins.meta.length ? <button type="button" className="message-info" aria-expanded={open} data-perf="message.info"
        aria-label="查看这条消息的信息" onClick={() => setOpen(!open)}>
        {lead}{inline.map((item) => <span key={item.id} data-tone={item.tone}>{item.text ?? item.label}</span>)}
      </button> : lead}
      <div className="shared-message-actions">
        {onCopy ? <button className={copied ? "copied" : ""} type="button" onClick={onCopy} data-perf="message.copy"
          aria-label={copied ? "已复制" : "复制消息"}>
          <IconSwap state={copied} a={<Copy size={16} aria-hidden="true" />} b={<CheckGlyph />} />
        </button> : null}
        <button type="button" aria-label="更多操作" aria-haspopup="menu" data-perf="message.more" onClick={(event) => {
          const box = event.currentTarget.getBoundingClientRect();
          showMenu?.({ x: box.left, y: box.bottom, items: entries(), restore: event.currentTarget });
        }}>
          <Ellipsis size={16} aria-hidden="true" />
        </button>
      </div>
      {slots ? <span className="message-info__sources">
        <PluginUiSlot name="message.meta" sessionId={slots.sessionId} messageId={slots.messageId} quiet extra={plugins.extra} />
        <PluginUiSlot name="message.actions" sessionId={slots.sessionId} messageId={slots.messageId} quiet extra={plugins.extra} />
      </span> : null}
    </div>
    {open && plugins.meta.length ? <dl className={`message-info__detail ${role}`}>
      {timestamp ? <><dt>时间</dt><dd>{fullTime.format(new Date(timestamp))}</dd></> : null}
      {plugins.meta.map((item) => <div key={item.id} data-tone={item.tone}><dt>{item.label}</dt><dd>{item.value}</dd></div>)}
    </dl> : null}
  </>;
}

// 整个对话共用一份菜单：「⋯」和右键都把位置与菜单项交给它，菜单贴着那个点弹出。
export function MessageMenuHost() {
  const [request, setRequest] = useState<MenuRequest | null>(null);
  const [open, setOpen] = useState(false);
  useEffect(() => {
    let key = 0;
    showMenu = (next) => { setRequest({ ...next, key: ++key }); setOpen(true); };
    return () => { showMenu = null; };
  }, []);
  if (!request || !request.items.length) return null;
  return createPortal(<DropdownMenu key={request.key} open={open} onOpenChange={setOpen} modal={false}>
    <DropdownMenuTrigger asChild>
      <span className="message-menu__anchor" style={{ left: request.x, top: request.y }} aria-hidden="true" />
    </DropdownMenuTrigger>
    <DropdownMenuContent className="navigation-row-menu message-menu" align="start" sideOffset={4} collisionPadding={12}
      aria-label="消息操作" onCloseAutoFocus={(event) => { event.preventDefault(); request.restore?.focus({ preventScroll: true }); }}>
      {request.items.map((item) => <DropdownMenuItem key={item.id} className="navigation-row-menu__item"
        data-perf={`message.menu.${item.id}`} onSelect={item.run}>
        {item.icon}<span>{item.label}</span>
      </DropdownMenuItem>)}
    </DropdownMenuContent>
  </DropdownMenu>, chatPortalRoot());
}

// 右键一条消息打开同一份菜单；选着文字、点在链接、代码、图片或插件内容上时保留浏览器自己的菜单。
export function useMessageContextMenu(scrollRef: React.RefObject<HTMLElement | null>) {
  useEffect(() => {
    const scroller = scrollRef.current;
    if (!scroller) return;
    const onContextMenu = (event: MouseEvent) => {
      const target = event.target instanceof Element ? event.target : null;
      if (!target || window.getSelection()?.toString()) return;
      if (target.closest("a, input, textarea, pre, code, img, video, [contenteditable], .plugin-ui-host")) return;
      const row = target.closest("[data-message-id]")?.querySelector(".shared-message-meta");
      const items = row ? rowMenus.get(row)?.() : undefined;
      if (!items?.length || !showMenu) return;
      event.preventDefault();
      showMenu({ x: event.clientX, y: event.clientY, items, restore: null });
    };
    scroller.addEventListener("contextmenu", onContextMenu);
    return () => scroller.removeEventListener("contextmenu", onContextMenu);
  }, [scrollRef]);
}

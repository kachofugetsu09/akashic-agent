import { Ellipsis } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

export interface NavigationRowAction {
  label: string;
  icon: ReactNode;
  disabled?: boolean;
  onSelect: () => void;
}

/** 导航行共享操作菜单；长按不打开会话，滑动仍交给目录滚动。 */
export function NavigationRowMenu({ title, actions = [], className, children }: {
  title: string;
  actions?: NavigationRowAction[];
  className: string;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const rowRef = useRef<HTMLDivElement>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const press = useRef<{ id: number; x: number; y: number } | null>(null);
  const held = useRef(false);
  const cancelPress = () => {
    if (timer.current !== null) clearTimeout(timer.current);
    timer.current = null;
    press.current = null;
  };
  useEffect(() => cancelPress, []);

  return <DropdownMenu open={open} onOpenChange={setOpen} modal={false}>
    <div ref={rowRef} className={`navigation-menu-row ${className}`} data-menu-open={open || undefined}
      onPointerDown={(event) => {
        // 1. 一次新触摸开始计时；第二根手指、鼠标和菜单按钮不参与长按。
        cancelPress();
        held.current = false;
        if (!actions.length || !event.isPrimary || event.pointerType === "mouse"
          || (event.target as Element).closest(".navigation-menu-trigger")) return;
        press.current = { id: event.pointerId, x: event.clientX, y: event.clientY };
        timer.current = setTimeout(() => {
          timer.current = null;
          held.current = true;
          setOpen(true);
        }, 500);
      }}
      onPointerMove={(event) => {
        const start = press.current;
        if (start && (event.pointerId !== start.id
          || Math.hypot(event.clientX - start.x, event.clientY - start.y) > 10)) cancelPress();
      }}
      onPointerUp={cancelPress} onPointerCancel={cancelPress} onPointerLeave={cancelPress}
      onTouchEnd={(event) => {
        // 长按松手不生成兼容鼠标事件，避免焦点回到行上并关闭菜单。
        if (held.current) event.preventDefault();
      }}
      onClickCapture={(event) => {
        // 2. 长按后松手产生的 click 不能选择会话或展开项目。
        if (!held.current) return;
        event.preventDefault();
        event.stopPropagation();
        held.current = false;
      }}
      onContextMenu={(event) => {
        if (!actions.length) return;
        event.preventDefault();
        cancelPress();
        setOpen(true);
      }}
      onKeyDown={(event) => {
        if (actions.length && (event.key === "ContextMenu" || (event.shiftKey && event.key === "F10"))) {
          event.preventDefault();
          setOpen(true);
        }
      }}>
      {children}
      {actions.length ? <>
        <DropdownMenuTrigger asChild>
          <button type="button" className="navigation-menu-trigger" aria-label={`${title} 的操作`}>
            <Ellipsis size={18} aria-hidden="true" />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent className="navigation-row-menu" align="end" sideOffset={4} collisionPadding={12}
          aria-label={`${title} 的操作`} onCloseAutoFocus={(event) => {
            event.preventDefault();
            held.current = false;
            rowRef.current?.querySelector<HTMLButtonElement>("button:not(:disabled)")?.focus({ preventScroll: true });
          }}>
          {actions.map((action) => <DropdownMenuItem key={action.label} disabled={action.disabled}
            onSelect={action.onSelect} className="navigation-row-menu__item">
            {action.icon}<span>{action.label}</span>
          </DropdownMenuItem>)}
        </DropdownMenuContent>
      </> : null}
    </div>
  </DropdownMenu>;
}

import * as DropdownMenuPrimitive from "@radix-ui/react-dropdown-menu";
import { Check, Moon, Sun, SunMoon, type LucideIcon } from "lucide-react";
import type { CSSProperties } from "react";
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuLabel, DropdownMenuRadioGroup, DropdownMenuSeparator, DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  setThemePreference, themeFamilies, useTheme, useThemeSelection, type ThemeMode,
} from "../../../../../packages/akashic-theme/src/theme-runtime";

const MODES: readonly { mode: ThemeMode; label: string; icon: LucideIcon }[] = [
  { mode: "light", label: "亮", icon: Sun },
  { mode: "dark", label: "暗", icon: Moon },
  { mode: "system", label: "跟随系统", icon: SunMoon },
];

// 选中项不关菜单：在手机抽屉里也能连续点几套配色对比，点外面或按 Esc 再收起。
const keepOpen = (event: Event) => event.preventDefault();

/** 侧栏底部的主题入口：上面选明暗，下面选配色，两个维度各点一次就到，不再逐个循环。 */
export function ThemeMenu() {
  const selection = useThemeSelection();
  const theme = useTheme();
  const families = themeFamilies();
  const family = families.find((item) => item.id === selection.family);
  const mode = MODES.find((item) => item.mode === selection.mode) ?? MODES[0];
  const label = family?.label ?? theme.label;
  const ModeIcon = mode.icon;
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button type="button" className="chat-sidebar__icon-action" data-tip={`主题：${label} · ${mode.label}`}
          aria-label={`主题，当前为${label}，${mode.label}`}>
          <ModeIcon size={18} aria-hidden="true" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent side="top" align="end" collisionPadding={8} className="theme-menu">
        <DropdownMenuLabel className="theme-menu__label">明暗</DropdownMenuLabel>
        <DropdownMenuRadioGroup className="theme-menu__modes" value={selection.mode}
          onValueChange={(value) => setThemePreference(selection.family, value as ThemeMode)}>
          {MODES.map(({ mode: value, label: text, icon: Icon }) => (
            <DropdownMenuPrimitive.RadioItem key={value} value={value} className="theme-menu__mode" onSelect={keepOpen}>
              <Icon size={16} aria-hidden="true" />
              <span>{text}</span>
            </DropdownMenuPrimitive.RadioItem>
          ))}
        </DropdownMenuRadioGroup>
        <DropdownMenuSeparator />
        <DropdownMenuLabel className="theme-menu__label">配色</DropdownMenuLabel>
        <DropdownMenuRadioGroup value={selection.family}
          onValueChange={(value) => setThemePreference(value, selection.mode)}>
          {families.map((item) => {
            // 色票跟着当前明暗显示：暗色时看到的就是各配色暗色那一面。
            const shown = item[theme.colorScheme];
            const swatch = {
              "--swatch-paper": shown.material.background,
              "--swatch-ink": shown.material.onSurface,
              "--swatch-accent": shown.material.primary,
            } as CSSProperties;
            return (
              <DropdownMenuPrimitive.RadioItem key={item.id} value={item.id} className="theme-menu__family" onSelect={keepOpen}>
                <span className="theme-menu__swatch" style={swatch} aria-hidden="true" />
                <span className="theme-menu__name">{item.label}</span>
                <DropdownMenuPrimitive.ItemIndicator className="theme-menu__check">
                  <Check size={16} aria-hidden="true" />
                </DropdownMenuPrimitive.ItemIndicator>
              </DropdownMenuPrimitive.RadioItem>
            );
          })}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

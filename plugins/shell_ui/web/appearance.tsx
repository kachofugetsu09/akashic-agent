import { createRoot } from "react-dom/client";
import type { CSSProperties } from "react";
import {
  setThemePreference, themeFamilies, useTheme, useThemeSelection, type ThemeMode, type WebHostContextV1, type WebUiDisposer,
} from "@akashic/web-ui-v1";

const SETTINGS_MOUNT = "shell.settings.v1";
/** 外观排在版本与更新（80）之前：主题设一次就不动，不占侧栏底栏的常驻位置。 */
const APPEARANCE_ORDER = 70;
const APPEARANCE_ICON = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="13.5" cy="6.5" r=".5" fill="currentColor"/><circle cx="17.5" cy="10.5" r=".5" fill="currentColor"/><circle cx="8.5" cy="7.5" r=".5" fill="currentColor"/><circle cx="6.5" cy="12.5" r=".5" fill="currentColor"/><path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10c.926 0 1.648-.746 1.648-1.688 0-.437-.18-.835-.437-1.125-.29-.289-.438-.652-.438-1.125a1.64 1.64 0 0 1 1.668-1.668h1.996c3.051 0 5.555-2.503 5.555-5.554C21.965 6.012 17.461 2 12 2z"/></svg>';

const MODES: readonly { mode: ThemeMode; label: string }[] = [
  { mode: "light", label: "亮" },
  { mode: "dark", label: "暗" },
  { mode: "system", label: "跟随系统" },
];

// 设置·外观：明暗是一排分段单选，配色是带色票的单选列表；选中即生效，没有保存按钮。
function Appearance() {
  const selection = useThemeSelection();
  const theme = useTheme();
  return <div className="shell-appearance">
    <fieldset className="shell-appearance__group">
      <legend>明暗</legend>
      <div className="shell-appearance__modes" role="radiogroup" aria-label="明暗">
        {MODES.map(({ mode, label }) => <label key={mode} className="shell-appearance__mode">
          <input type="radio" name="appearance-mode" value={mode} checked={selection.mode === mode}
            onChange={() => setThemePreference(selection.family, mode)} />
          <span>{label}</span>
        </label>)}
      </div>
    </fieldset>
    <fieldset className="shell-appearance__group">
      <legend>配色</legend>
      <div className="shell-appearance__families" role="radiogroup" aria-label="配色">
        {themeFamilies().map((family) => {
          // 色票跟着当前明暗显示：暗色时看到的就是各配色暗色那一面。
          const shown = family[theme.colorScheme].material;
          const swatch = {
            "--swatch-paper": shown.background, "--swatch-ink": shown.onSurface, "--swatch-accent": shown.primary,
          } as CSSProperties;
          return <label key={family.id} className="shell-appearance__family">
            <input type="radio" name="appearance-family" value={family.id} checked={selection.family === family.id}
              onChange={() => setThemePreference(family.id, selection.mode)} />
            <span className="shell-appearance__swatch" style={swatch} aria-hidden="true" />
            <span>{family.label}</span>
          </label>;
        })}
      </div>
    </fieldset>
  </div>;
}

// 把外观登记成设置工作区的一个普通分节，与其他插件的设置分节同一条路。
export function registerAppearance(ctx: WebHostContextV1): WebUiDisposer {
  return ctx.ui.inject(SETTINGS_MOUNT, (mount) => mount.register({
    id: "appearance", label: "外观", route: "appearance", iconSvg: APPEARANCE_ICON, order: APPEARANCE_ORDER,
    render(host: HTMLElement) {
      const root = createRoot(host);
      root.render(<Appearance />);
      return () => root.unmount();
    },
  }));
}

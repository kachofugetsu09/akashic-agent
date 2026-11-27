import { Component, Suspense, type ErrorInfo, type ReactNode } from "react";
import { Folder } from "lucide-react";
import { useIconProvider } from "./plugin-ui-runtime";

export interface GroupIconColorOption {
  readonly id: string | null;
  readonly label: string;
  readonly cssValue: string;
}

/** 纸面语义色板：直接复用主题校准过的 --ak-chart-1..6，自动适配浅色纸感与深色墨纸。 */
export const GROUP_ICON_COLORS: readonly GroupIconColorOption[] = [
  { id: null, label: "墨色", cssValue: "var(--ak-ink-secondary)" },
  { id: "pine", label: "松绿", cssValue: "var(--ak-chart-3)" },
  { id: "blue", label: "靛蓝", cssValue: "var(--ak-chart-1)" },
  { id: "ochre", label: "赭石", cssValue: "var(--ak-chart-2)" },
  { id: "plum", label: "暮紫", cssValue: "var(--ak-chart-4)" },
  { id: "teal", label: "青瓷", cssValue: "var(--ak-chart-6)" },
  { id: "cinnabar", label: "朱砂", cssValue: "var(--ak-chart-5)" },
];

const COLOR_MAP: Readonly<Record<string, string>> = Object.fromEntries(
  GROUP_ICON_COLORS.filter((item): item is GroupIconColorOption & { id: string } => item.id !== null)
    .map((item) => [item.id, item.cssValue]),
);

export function resolveGroupIconColor(color?: string | null): string | undefined {
  if (!color) return undefined;
  return COLOR_MAP[color.trim()];
}

const DEFAULT_ICON_PROVIDER_ID = "kubai-line";

/** 局部错误边界：把外置图标插件的渲染期异常严格隔离在单枚图标或选择器内部，不拖垮侧栏或表单。 */
class IconProviderBoundary extends Component<{ fallback: ReactNode; children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError(): { failed: boolean } {
    return { failed: true };
  }
  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("[icon.provider] render failed", error, info);
  }
  render(): ReactNode {
    return this.state.failed ? this.props.fallback : this.props.children;
  }
}

/** 侧栏分组图标：历史字符串图标名固定解析到 kubai-line；未装对应图标插件或异常时优雅退回默认 Lucide 文件夹。 */
export function GroupIcon({ name, color, providerId = DEFAULT_ICON_PROVIDER_ID, size = 18, trigger = "parent", className }: {
  name?: string | null;
  color?: string | null;
  providerId?: string;
  size?: number;
  trigger?: "hover" | "parent" | "none";
  className?: string;
}) {
  const entry = useIconProvider(providerId);
  const cls = className ? `group-mark ${className}` : "group-mark";
  const clean = typeof name === "string" && name.trim() ? name.trim() : "folder";
  const cssColor = resolveGroupIconColor(color);
  const fallback = <Folder size={size} strokeWidth={1.75} aria-hidden={true} className={cls} />;

  return <span className="group-mark-host" style={cssColor ? { color: cssColor } : undefined}>
    {entry ? <IconProviderBoundary key={entry.key} fallback={fallback}>
      <Suspense fallback={fallback}>
        <entry.provider.Icon name={clean} size={size} trigger={trigger} className={cls} />
      </Suspense>
    </IconProviderBoundary> : fallback}
  </span>;
}

/**
 * 冻结的 v1 公开图标选择器：
 * - 宿主始终持有预览头、7 色纸面语义色板与恢复默认，与图标插件解耦；
 * - 若已安装匹配的 icon.provider（如 kubai_icons），把选图标区域委托给 provider.Picker（包在局部 ErrorBoundary + Suspense 内）；
 * - 未安装或换代降级时，提供可正常输入、保存、清空且遵守 disabled 的轻量表单，保障旧版 projects 插件 100% 可用。
 */
export function GroupIconPicker({ value, onChange, color = null, onColorChange, disabled = false }: {
  value: string | null;
  onChange: (icon: string | null) => void;
  color?: string | null;
  onColorChange?: (color: string | null) => void;
  disabled?: boolean;
}) {
  const entry = useIconProvider(DEFAULT_ICON_PROVIDER_ID);
  const selected = value?.trim() || "folder";
  const selectedColor = color && COLOR_MAP[color] ? color : null;
  const cssColor = resolveGroupIconColor(selectedColor);
  const canReset = selected !== "folder" || selectedColor !== null;

  const fallbackPicker = <div className="group-icon-picker__fallback">
    <label className="group-icon-picker__input-row">
      <span>图标标识</span>
      <input
        type="text"
        value={value ?? ""}
        disabled={disabled}
        maxLength={64}
        placeholder="留空为默认文件夹，例如 terminal、book-open"
        onChange={(event) => {
          const next = event.target.value.trim();
          onChange(next ? next : null);
        }}
      />
    </label>
    <p className="group-icon-picker__hint">安装 kubai_icons 插件后可按分类浏览全部线性动画图标。</p>
  </div>;

  return <div className="group-icon-picker">
    <div className="group-icon-picker__summary">
      <span className="group-icon-picker__preview" data-kubai-trigger="">
        <GroupIcon name={selected} color={selectedColor} size={20} trigger="parent" />
      </span>
      <span className="group-icon-picker__meta">
        <strong>{selected === "folder" ? "默认文件夹" : selected}</strong>
        <small>{entry ? `${entry.provider.label} · 悬停可预览动效` : "未安装图标扩展包（可输入标识或仅选颜色）"}</small>
      </span>
      {canReset ? <button
        type="button"
        className="group-icon-picker__reset"
        disabled={disabled}
        onClick={() => { onChange(null); onColorChange?.(null); }}
      >
        恢复默认
      </button> : null}
    </div>

    {onColorChange ? <div className="group-icon-picker__swatches" role="radiogroup" aria-label="选择图标颜色">
      {GROUP_ICON_COLORS.map((opt) => {
        const active = selectedColor === opt.id;
        return <button
          key={opt.id ?? "ink"}
          type="button"
          role="radio"
          aria-checked={active}
          aria-label={opt.label}
          title={opt.label}
          disabled={disabled}
          className="group-icon-picker__swatch"
          onClick={() => onColorChange(opt.id)}
        >
          <span className="group-icon-picker__swatch-dot" style={{ background: opt.cssValue }} />
        </button>;
      })}
    </div> : null}

    <div className="group-icon-picker__body" style={cssColor ? { color: cssColor } : undefined}>
      {entry?.provider.Picker ? <IconProviderBoundary key={entry.key} fallback={fallbackPicker}>
        <Suspense fallback={<p className="group-icon-picker__hint">正在加载图标选择器…</p>}>
          <entry.provider.Picker value={value} disabled={disabled} onChange={onChange} />
        </Suspense>
      </IconProviderBoundary> : fallbackPicker}
    </div>
  </div>;
}

import { useSyncExternalStore } from "react";
import catalogJson from "./theme-catalog.json";

export type ThemeStatus = "stable" | "experimental";
export type ThemeColorScheme = "light" | "dark";

export interface ThemeDefinition {
  id: string;
  label: string;
  status: ThemeStatus;
  colorScheme: ThemeColorScheme;
  material: Record<MaterialColorRole, string>;
  domain: Record<DomainColorRole, string>;
}

export interface ThemeSelection {
  requestedThemeId: string;
  effectiveThemeId: string;
  unavailable: boolean;
}

const MATERIAL_COLOR_ROLES = [
  "primary", "onPrimary", "primaryContainer", "onPrimaryContainer",
  "secondary", "onSecondary", "secondaryContainer", "onSecondaryContainer",
  "tertiary", "onTertiary", "tertiaryContainer", "onTertiaryContainer",
  "error", "onError", "errorContainer", "onErrorContainer",
  "background", "onBackground", "surface", "onSurface", "surfaceVariant", "onSurfaceVariant",
  "outline", "outlineVariant", "shadow", "scrim", "inverseSurface", "inverseOnSurface",
  "inversePrimary", "surfaceDim", "surfaceBright", "surfaceContainerLowest",
  "surfaceContainerLow", "surfaceContainer", "surfaceContainerHigh",
  "surfaceContainerHighest", "surfaceTint",
] as const;

const DOMAIN_COLOR_ROLES = [
  "success", "onSuccess", "successContainer", "onSuccessContainer",
  "warning", "onWarning", "warningContainer", "onWarningContainer",
  "trace", "onTrace", "traceContainer", "onTraceContainer",
  "info", "onInfo", "infoContainer", "onInfoContainer",
] as const;

export type MaterialColorRole = (typeof MATERIAL_COLOR_ROLES)[number];
export type DomainColorRole = (typeof DOMAIN_COLOR_ROLES)[number];

type LegacyColorSource = ["material", MaterialColorRole] | ["domain", DomainColorRole];

const LEGACY_COLOR_ALIASES = {
  bgCanvas: ["material", "background"],
  bgSurface: ["material", "surfaceContainerLowest"],
  bgSurfaceLow: ["material", "surfaceContainerLow"],
  bgSurfaceHigh: ["material", "surfaceContainerHigh"],
  textPrimary: ["material", "onSurface"],
  textSecondary: ["material", "onSurfaceVariant"],
  textMuted: ["material", "outline"],
  borderDefault: ["material", "outlineVariant"],
  borderStrong: ["material", "outline"],
  actionPrimary: ["material", "primary"],
  onActionPrimary: ["material", "onPrimary"],
  actionHover: ["material", "primary"],
  actionSoft: ["material", "primaryContainer"],
  actionContainer: ["material", "primaryContainer"],
  onActionContainer: ["material", "onPrimaryContainer"],
  statusError: ["material", "error"],
  statusErrorContainer: ["material", "errorContainer"],
  statusWarning: ["domain", "warning"],
  statusSuccess: ["domain", "success"],
  statusTrace: ["domain", "trace"],
  statusTraceText: ["domain", "trace"],
  statusTraceContainer: ["domain", "traceContainer"],
  shadow: ["material", "shadow"],
  imageOutline: ["material", "outlineVariant"],
} as const satisfies Record<string, LegacyColorSource>;

const THEME_COOKIE = "akashic_theme";
const THEME_ID_PATTERN = /^[a-z0-9][a-z0-9-]{0,63}$/;
const HEX_COLOR_PATTERN = /^#[0-9a-f]{6}(?:[0-9a-f]{2})?$/i;
const THEME_EVENT = "akashic-theme-change";

/** Validate the bundled catalog once and expose immutable theme definitions. */
function validateCatalog(value: unknown): { defaultThemeId: string; themes: ThemeDefinition[] } {
  // 1. 校验目录结构与主题身份
  if (!value || typeof value !== "object") throw new Error("Theme catalog 不是对象");
  const catalog = value as { version?: unknown; defaultThemeId?: unknown; themes?: unknown };
  if (catalog.version !== 2 || typeof catalog.defaultThemeId !== "string" || !Array.isArray(catalog.themes)) {
    throw new Error("Theme catalog 结构无效");
  }
  const themes = catalog.themes.map((raw, index) => {
    if (!raw || typeof raw !== "object") throw new Error(`Theme catalog themes[${index}] 不是对象`);
    const theme = raw as Record<string, unknown>;
    if (typeof theme.id !== "string" || !THEME_ID_PATTERN.test(theme.id)) {
      throw new Error(`Theme catalog themes[${index}].id 无效`);
    }
    if (typeof theme.label !== "string" || !theme.label.trim()) {
      throw new Error(`Theme catalog themes[${index}].label 无效`);
    }
    if (theme.status !== "stable" && theme.status !== "experimental") {
      throw new Error(`Theme catalog themes[${index}].status 无效`);
    }
    if (theme.colorScheme !== "light" && theme.colorScheme !== "dark") {
      throw new Error(`Theme catalog themes[${index}].colorScheme 无效`);
    }

    // 2. 校验所有公开颜色角色完整且可解析
    const material = validateColorGroup(theme, "material", MATERIAL_COLOR_ROLES);
    const domain = validateColorGroup(theme, "domain", DOMAIN_COLOR_ROLES);
    return {
      id: theme.id,
      label: theme.label,
      status: theme.status,
      colorScheme: theme.colorScheme,
      material,
      domain,
    } as ThemeDefinition;
  });

  // 3. 建立目录级不变量
  if (new Set(themes.map((theme) => theme.id)).size !== themes.length) {
    throw new Error("Theme catalog 存在重复 theme id");
  }
  if (!themes.some((theme) => theme.id === catalog.defaultThemeId)) {
    throw new Error("Theme catalog 默认主题不存在");
  }
  return { defaultThemeId: catalog.defaultThemeId, themes };
}

function validateColorGroup<Role extends string>(
  theme: Record<string, unknown>,
  groupName: "material" | "domain",
  roles: readonly Role[],
): Record<Role, string> {
  const rawGroup = theme[groupName];
  if (!rawGroup || typeof rawGroup !== "object") {
    throw new Error(`Theme catalog ${String(theme.id)}.${groupName} 无效`);
  }
  const group = rawGroup as Record<string, unknown>;
  for (const role of roles) {
    if (typeof group[role] !== "string" || !HEX_COLOR_PATTERN.test(group[role])) {
      throw new Error(`Theme catalog ${String(theme.id)}.${groupName}.${role} 无效`);
    }
  }
  return Object.fromEntries(roles.map((role) => [role, group[role]])) as Record<Role, string>;
}

const CATALOG = validateCatalog(catalogJson);
const THEME_BY_ID = new Map(CATALOG.themes.map((theme) => [theme.id, theme]));
let selection: ThemeSelection = {
  requestedThemeId: CATALOG.defaultThemeId,
  effectiveThemeId: CATALOG.defaultThemeId,
  unavailable: false,
};

function cssName(role: string): string {
  return role.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`);
}

function rgbChannels(value: string): string {
  return `${Number.parseInt(value.slice(1, 3), 16)} ${Number.parseInt(value.slice(3, 5), 16)} ${Number.parseInt(value.slice(5, 7), 16)}`;
}

function themeCss(): string {
  return CATALOG.themes.map((theme) => {
    const declarations = MATERIAL_COLOR_ROLES.flatMap((role) => colorDeclarations("md-sys-color", role, theme.material[role]));
    declarations.push(...DOMAIN_COLOR_ROLES.flatMap((role) => colorDeclarations("ak-sys-color", role, theme.domain[role])));
    for (const [legacyRole, [group, role]] of Object.entries(LEGACY_COLOR_ALIASES)) {
      const value = group === "material"
        ? theme.material[role as MaterialColorRole]
        : theme.domain[role as DomainColorRole];
      declarations.push(...colorDeclarations("ak-color", legacyRole, value));
    }
    declarations.push(`--ak-ink-image-filter:${theme.colorScheme === "dark" ? "brightness(0) invert(1)" : "brightness(0)"}`);
    declarations.push(`color-scheme:${theme.colorScheme}`);
    return `:root[data-theme="${theme.id}"]{${declarations.join(";")}}`;
  }).join("\n");
}

function colorDeclarations(namespace: string, role: string, value: string): string[] {
  const name = cssName(role);
  return [`--${namespace}-${name}:${value}`, `--${namespace}-${name}-rgb:${rgbChannels(value)}`];
}

function installThemeCss(): void {
  if (document.getElementById("akashic-theme-catalog")) return;
  const style = document.createElement("style");
  style.id = "akashic-theme-catalog";
  style.textContent = themeCss();
  document.head.prepend(style);
}

function readCookieTheme(): string | null {
  const entry = document.cookie.split(";").map((part) => part.trim()).find((part) => part.startsWith(`${THEME_COOKIE}=`));
  if (!entry) return null;
  const value = decodeURIComponent(entry.slice(THEME_COOKIE.length + 1));
  if (!THEME_ID_PATTERN.test(value)) {
    console.warn("[theme] 丢弃格式无效的主题偏好");
    return null;
  }
  return value;
}

function resolveSelection(requestedThemeId: string): ThemeSelection {
  // warm-paper removed; migrate legacy cookie/prefs onto paper light.
  const normalized = requestedThemeId === "warm-paper" ? "light" : requestedThemeId;
  const effectiveThemeId = THEME_BY_ID.has(normalized) ? normalized : CATALOG.defaultThemeId;
  return {
    requestedThemeId,
    effectiveThemeId,
    unavailable: effectiveThemeId !== requestedThemeId && requestedThemeId !== "warm-paper",
  };
}

function applySelection(next: ThemeSelection): void {
  selection = next;
  const theme = THEME_BY_ID.get(next.effectiveThemeId);
  if (!theme) throw new Error(`Theme catalog 缺少有效主题: ${next.effectiveThemeId}`);
  document.documentElement.dataset.theme = theme.id;
  document.documentElement.style.colorScheme = theme.colorScheme;
  document.querySelector('meta[name="color-scheme"]')?.setAttribute("content", theme.colorScheme);
  window.dispatchEvent(new CustomEvent(THEME_EVENT));
}

export function initializeTheme(): ThemeSelection {
  installThemeCss();
  applySelection(resolveSelection(readCookieTheme() ?? CATALOG.defaultThemeId));
  return selection;
}

export function setTheme(requestedThemeId: string, persist = true): ThemeSelection {
  if (!THEME_ID_PATTERN.test(requestedThemeId)) throw new Error(`Theme id 无效: ${requestedThemeId}`);
  const next = resolveSelection(requestedThemeId);
  applySelection(next);
  if (persist) {
    document.cookie = `${THEME_COOKIE}=${encodeURIComponent(requestedThemeId)}; Path=/; Max-Age=31536000; SameSite=Lax`;
  }
  return next;
}

export function cycleTheme(): ThemeSelection {
  const current = CATALOG.themes.findIndex((theme) => theme.id === selection.effectiveThemeId);
  return setTheme(CATALOG.themes[(current + 1) % CATALOG.themes.length].id);
}

export function currentTheme(): ThemeDefinition {
  const theme = THEME_BY_ID.get(selection.effectiveThemeId);
  if (!theme) throw new Error(`Theme catalog 缺少有效主题: ${selection.effectiveThemeId}`);
  return theme;
}

export function themes(): readonly ThemeDefinition[] {
  return CATALOG.themes;
}

export function themeSelection(): ThemeSelection {
  return selection;
}

export function subscribeTheme(listener: () => void): () => void {
  window.addEventListener(THEME_EVENT, listener);
  return () => window.removeEventListener(THEME_EVENT, listener);
}

export function useTheme(): ThemeDefinition {
  return useSyncExternalStore(subscribeTheme, currentTheme, currentTheme);
}

export function startCrossPortThemeSync(): () => void {
  const refresh = () => applySelection(resolveSelection(readCookieTheme() ?? CATALOG.defaultThemeId));
  window.addEventListener("focus", refresh);
  return () => window.removeEventListener("focus", refresh);
}

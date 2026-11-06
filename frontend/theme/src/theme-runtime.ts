import { useSyncExternalStore } from "react";
import catalogJson from "./theme-catalog.json";

export type ThemeStatus = "stable" | "experimental";
export type ThemeColorScheme = "light" | "dark";

/** 明暗偏好：固定亮、固定暗，或随系统外观切换。 */
export type ThemeMode = ThemeColorScheme | "system";

export interface ThemeDefinition {
  id: string;
  label: string;
  family: string;
  status: ThemeStatus;
  colorScheme: ThemeColorScheme;
  material: Record<MaterialColorRole, string>;
  domain: Record<DomainColorRole, string>;
  chart: Record<ChartColorRole, string>;
}

/** 一套配色：同一组色相关系下的一个亮色主题和一个暗色主题。 */
export interface ThemeFamily {
  id: string;
  label: string;
  light: ThemeDefinition;
  dark: ThemeDefinition;
}

export interface ThemeSelection {
  requestedThemeId: string;
  effectiveThemeId: string;
  family: string;
  mode: ThemeMode;
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

const CHART_COLOR_ROLES = ["1", "2", "3", "4", "5", "6"] as const;
export type ChartColorRole = (typeof CHART_COLOR_ROLES)[number];

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
const THEME_CHANNEL_NAME = "akashic.theme-sync";
// 跟随系统的偏好写成 system-<配色>，与主题 id 共用 cookie 和同步频道。
const SYSTEM_PREFIX = "system-";

/** Validate the bundled catalog once and expose immutable theme definitions. */
function validateCatalog(value: unknown): { defaultThemeId: string; themes: ThemeDefinition[]; families: ThemeFamily[] } {
  // 1. 校验目录结构与主题身份
  if (!value || typeof value !== "object") throw new Error("Theme catalog 不是对象");
  const catalog = value as { version?: unknown; defaultThemeId?: unknown; themes?: unknown; families?: unknown };
  if (catalog.version !== 3 || typeof catalog.defaultThemeId !== "string" || !Array.isArray(catalog.themes)) {
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
    if (typeof theme.family !== "string" || !THEME_ID_PATTERN.test(theme.family)) {
      throw new Error(`Theme catalog themes[${index}].family 无效`);
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
      family: theme.family,
      status: theme.status,
      colorScheme: theme.colorScheme,
      material,
      domain,
      chart: validateColorGroup(theme, "chart", CHART_COLOR_ROLES),
    } as ThemeDefinition;
  });

  // 3. 建立目录级不变量
  if (new Set(themes.map((theme) => theme.id)).size !== themes.length) {
    throw new Error("Theme catalog 存在重复 theme id");
  }
  if (!themes.some((theme) => theme.id === catalog.defaultThemeId)) {
    throw new Error("Theme catalog 默认主题不存在");
  }
  return { defaultThemeId: catalog.defaultThemeId, themes, families: validateFamilies(catalog.families, themes) };
}

/** 每套配色恰好有一个亮色和一个暗色主题，选择器才能只按「配色 × 明暗」两个维度切换。 */
function validateFamilies(value: unknown, themes: ThemeDefinition[]): ThemeFamily[] {
  if (!Array.isArray(value)) throw new Error("Theme catalog families 无效");
  const families = value.map((raw, index) => {
    const family = (raw ?? {}) as { id?: unknown; label?: unknown };
    if (typeof family.id !== "string" || !THEME_ID_PATTERN.test(family.id)
      || typeof family.label !== "string" || !family.label.trim()) {
      throw new Error(`Theme catalog families[${index}] 无效`);
    }
    const members = themes.filter((theme) => theme.family === family.id);
    const light = members.find((theme) => theme.colorScheme === "light");
    const dark = members.find((theme) => theme.colorScheme === "dark");
    if (members.length !== 2 || !light || !dark) {
      throw new Error(`Theme catalog 配色 ${family.id} 需要恰好一个亮色和一个暗色主题`);
    }
    return { id: family.id, label: family.label, light, dark };
  });
  if (new Set(families.map((family) => family.id)).size !== families.length
    || themes.some((theme) => !families.some((family) => family.id === theme.family))) {
    throw new Error("Theme catalog 配色重复或主题引用了不存在的配色");
  }
  return families;
}

function validateColorGroup<Role extends string>(
  theme: Record<string, unknown>,
  groupName: "material" | "domain" | "chart",
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
const FAMILY_BY_ID = new Map(CATALOG.families.map((family) => [family.id, family]));
const DEFAULT_THEME = THEME_BY_ID.get(CATALOG.defaultThemeId)!;
let selection: ThemeSelection = {
  requestedThemeId: DEFAULT_THEME.id,
  effectiveThemeId: DEFAULT_THEME.id,
  family: DEFAULT_THEME.family,
  mode: DEFAULT_THEME.colorScheme,
  unavailable: false,
};

function cssName(role: string): string {
  return role.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`);
}

function rgbChannels(value: string): string {
  return `${Number.parseInt(value.slice(1, 3), 16)} ${Number.parseInt(value.slice(3, 5), 16)} ${Number.parseInt(value.slice(5, 7), 16)}`;
}

/** 浏览器与离线预检使用同一份主题变量。 */
export function themeCss(): string {
  return CATALOG.themes.map((theme) => {
    const declarations = MATERIAL_COLOR_ROLES.flatMap((role) => colorDeclarations("md-sys-color", role, theme.material[role]));
    declarations.push(...CHART_COLOR_ROLES.map((role) => `--ak-chart-${role}:${theme.chart[role]}`));
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

function systemPrefersDark(): boolean {
  return typeof window.matchMedia === "function" && window.matchMedia("(prefers-color-scheme: dark)").matches;
}

function resolveSelection(requestedThemeId: string): ThemeSelection {
  // 1. 跟随系统：按当前系统外观取该配色的亮或暗主题
  const followed = requestedThemeId.startsWith(SYSTEM_PREFIX)
    ? FAMILY_BY_ID.get(requestedThemeId.slice(SYSTEM_PREFIX.length)) : undefined;
  if (followed) {
    const theme = systemPrefersDark() ? followed.dark : followed.light;
    return { requestedThemeId, effectiveThemeId: theme.id, family: followed.id, mode: "system", unavailable: false };
  }
  // 2. 固定主题；warm-paper removed, migrate legacy cookie/prefs onto paper light.
  const normalized = requestedThemeId === "warm-paper" ? "light" : requestedThemeId;
  const theme = THEME_BY_ID.get(normalized) ?? DEFAULT_THEME;
  return {
    requestedThemeId,
    effectiveThemeId: theme.id,
    family: theme.family,
    mode: theme.colorScheme,
    unavailable: theme.id !== requestedThemeId && requestedThemeId !== "warm-paper",
  };
}

function applySelection(next: ThemeSelection): void {
  selection = next;
  const theme = THEME_BY_ID.get(next.effectiveThemeId);
  if (!theme) throw new Error(`Theme catalog 缺少有效主题: ${next.effectiveThemeId}`);
  document.documentElement.dataset.theme = theme.id;
  // markstream/shiki 与 tailwind darkMode:["class"] 都按根元素 .dark 判暗色；
  // 与 data-theme 同步切换，否则暗色下代码块与 dark: 工具类永远落在亮色分支。
  document.documentElement.classList.toggle("dark", theme.colorScheme === "dark");
  document.documentElement.style.colorScheme = theme.colorScheme;
  document.querySelector('meta[name="color-scheme"]')?.setAttribute("content", theme.colorScheme);
  window.dispatchEvent(new CustomEvent(THEME_EVENT));
}

/** 用户主动切换主题时整页淡交；reduced-motion 或不支持 View Transitions 时瞬时切换。 */
function applySelectionWithCrossfade(next: ThemeSelection): void {
  const startViewTransition = (document as Document & {
    startViewTransition?: (update: () => void) => { ready: Promise<void> };
  }).startViewTransition;
  if (!startViewTransition || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    applySelection(next);
    return;
  }
  // View Transition callbacks are deferred. Keep repeated clicks based on the
  // latest choice, and do not let a superseded callback restore an older theme.
  selection = next;
  const transition = startViewTransition.call(document, () => {
    if (selection === next) applySelection(next);
  });
  // Starting another transition or hiding the document can skip the animation.
  void transition.ready.catch(() => {});
}

let themeChannel: BroadcastChannel | null = null;

/** 外层 Shell 与 /chat iframe 是同源不同文档，主题在任一文档切换时经 BroadcastChannel
    即时同步，不再等 focus 重读 cookie。接收方只 apply 不再广播，偏好相同直接忽略，不会成环。
    比较偏好而不是生效主题：「暗」改成「跟随系统」时生效主题可能不变，但选择器要同步。 */
function joinThemeChannel(): BroadcastChannel | null {
  if (typeof BroadcastChannel === "undefined") return null;
  if (themeChannel) return themeChannel;
  themeChannel = new BroadcastChannel(THEME_CHANNEL_NAME);
  themeChannel.addEventListener("message", (event) => {
    if (typeof event.data !== "string" || !THEME_ID_PATTERN.test(event.data)) return;
    if (event.data === selection.requestedThemeId) return;
    applySelection(resolveSelection(event.data));
  });
  return themeChannel;
}

let systemWatched = false;

/** 跟随系统时，系统外观一变就换到同一配色的另一面；每个文档各自监听，不广播。 */
function watchSystemScheme(): void {
  if (systemWatched || typeof window.matchMedia !== "function") return;
  systemWatched = true;
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (selection.mode === "system") applySelectionWithCrossfade(resolveSelection(selection.requestedThemeId));
  });
}

export function initializeTheme(): ThemeSelection {
  installThemeCss();
  joinThemeChannel();
  watchSystemScheme();
  applySelection(resolveSelection(readCookieTheme() ?? CATALOG.defaultThemeId));
  return selection;
}

export function setTheme(requestedThemeId: string, persist = true): ThemeSelection {
  if (!THEME_ID_PATTERN.test(requestedThemeId)) throw new Error(`Theme id 无效: ${requestedThemeId}`);
  const next = resolveSelection(requestedThemeId);
  applySelectionWithCrossfade(next);
  if (persist) {
    document.cookie = `${THEME_COOKIE}=${encodeURIComponent(requestedThemeId)}; Path=/; Max-Age=31536000; SameSite=Lax`;
  }
  joinThemeChannel()?.postMessage(requestedThemeId);
  return next;
}

/** 按「配色 × 明暗」选择；跟随系统存成 system-<配色>。 */
export function setThemePreference(familyId: string, mode: ThemeMode): ThemeSelection {
  const family = FAMILY_BY_ID.get(familyId);
  if (!family) throw new Error(`配色不存在: ${familyId}`);
  return setTheme(mode === "system" ? SYSTEM_PREFIX + family.id : family[mode].id);
}

/** 在当前配色里切换亮与暗。 */
export function cycleTheme(): ThemeSelection {
  return setThemePreference(selection.family, currentTheme().colorScheme === "dark" ? "light" : "dark");
}

export function currentTheme(): ThemeDefinition {
  const theme = THEME_BY_ID.get(selection.effectiveThemeId);
  if (!theme) throw new Error(`Theme catalog 缺少有效主题: ${selection.effectiveThemeId}`);
  return theme;
}

export function themes(): readonly ThemeDefinition[] {
  return CATALOG.themes;
}

export function themeFamilies(): readonly ThemeFamily[] {
  return CATALOG.families;
}

export function themeSelection(): ThemeSelection {
  return selection;
}

export function useThemeSelection(): ThemeSelection {
  return useSyncExternalStore(subscribeTheme, themeSelection, themeSelection);
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

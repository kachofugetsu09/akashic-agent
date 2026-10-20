import type { ComponentType, MouseEvent, ReactNode } from "react";

export type WebUiDisposer = () => void;
export type WebMountCardinality = "single" | "list";

export interface ThemeDefinition {
  readonly id: string;
  readonly label: string;
  readonly status: "stable" | "experimental";
  readonly colorScheme: "light" | "dark";
}

export interface ThemeSelection {
  readonly requestedThemeId: string;
  readonly effectiveThemeId: string;
  readonly unavailable: boolean;
}

export function currentTheme(): ThemeDefinition;
export function subscribeTheme(listener: () => void): WebUiDisposer;
export function themes(): readonly ThemeDefinition[];
export function cycleTheme(): ThemeSelection;
export function useTheme(): ThemeDefinition;

export interface MaterialButtonProps {
  children: ReactNode;
  variant?: "filled" | "tonal" | "outlined" | "text" | "danger";
  disabled?: boolean;
  loading?: boolean;
  className?: string;
  type?: "button" | "submit" | "reset";
  onClick?: (event: MouseEvent<HTMLElement>) => void;
}

export interface MaterialIconButtonProps {
  children: ReactNode;
  variant?: "filled" | "tonal" | "standard" | "danger";
  disabled?: boolean;
  className?: string;
  label: string;
  onClick?: (event: MouseEvent<HTMLElement>) => void;
}

export interface MaterialFilterChipProps {
  children: ReactNode;
  selected?: boolean;
  disabled?: boolean;
  className?: string;
  onClick?: (event: MouseEvent<HTMLElement>) => void;
}

export const MaterialButton: ComponentType<MaterialButtonProps>;
export const MaterialIconButton: ComponentType<MaterialIconButtonProps>;
export const MaterialFilterChip: ComponentType<MaterialFilterChipProps>;

export interface WebMountDefinition {
  id: string;
  cardinality: WebMountCardinality;
}

export interface WebEntry {
  id: string;
  order?: number;
  children?: WebMountDefinition[];
  render?(host: HTMLElement, view: WebEntryView, props?: unknown): void | WebUiDisposer;
  [key: string]: unknown;
}

export interface WebChildRenderOptions {
  onError?(error: Error): void;
}

export interface WebMountView {
  readonly entries: readonly WebEntry[];
  /**
   * Render one child entry into a parent-owned host. When the child's render throws, the host shows
   * the error in `host` and calls `options.onError`, so the parent can revoke what it handed over in `props`.
   */
  render(entryId: string, host: HTMLElement, props?: unknown, options?: WebChildRenderOptions): WebUiDisposer;
  /** Apply the entry owner's isolated stylesheet to a parent-owned child host. */
  style(entryId: string, host: HTMLElement): WebUiDisposer;
}

export interface WebEntryView {
  child(mountId: string): WebMountView;
}

export interface WebMountRegistration {
  register(entry: WebEntry): WebUiDisposer;
}

export interface WebResourceOptions {
  /** The host reads it once while idle after startup, so the first page open already has a value. */
  warm?: boolean;
}

export interface WebRefreshOptions {
  /**
   * Start a new read instead of joining one already in flight. Use it after a write: a read that began
   * before the write could return the old state. Superseded reads never replace `current`.
   */
  fresh?: boolean;
}

/**
 * Data owned by a web module rather than by one rendered page: create it in `activate(ctx)`,
 * read `current` synchronously in `render`, and call `refresh()` to revalidate in the background.
 * Values are kept in memory only; a read whose JSON equals the current value keeps the old object and does not notify.
 */
export interface WebResource<T> {
  /** Last successful value; undefined before the first read. */
  readonly current: T | undefined;
  /** Error of the latest failed read; cleared by the next success. `current` keeps the previous value. */
  readonly error: Error | undefined;
  /** Notified when `current` or `error` changes. */
  subscribe(listener: () => void): WebUiDisposer;
  /** Joins a read already in flight unless `fresh`; `current` keeps its value until a read succeeds. */
  refresh(options?: WebRefreshOptions): Promise<T>;
  /** Accept a value the module already holds (for example a write's response); it supersedes reads in flight. */
  set(value: T): void;
}

export interface WebHostContextV1 {
  readonly data: {
    /** `key` is unique within the module; `load` receives a signal aborted when the module is disposed. */
    resource<T>(key: string, load: (signal: AbortSignal) => Promise<T>, options?: WebResourceOptions): WebResource<T>;
  };
  readonly http: {
    request(path: string, init?: RequestInit): Promise<Response>;
    /** Build one exact-generation WebSocket URL for this module's dashboard route. */
    webSocketUrl(path: string): string;
  };
  readonly ui: {
    inject(
      mountId: string,
      connect: (mount: WebMountRegistration) => WebUiDisposer,
    ): WebUiDisposer;
  };
}

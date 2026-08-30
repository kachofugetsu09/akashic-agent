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

export interface WebMountView {
  readonly entries: readonly WebEntry[];
  render(entryId: string, host: HTMLElement, props?: unknown): WebUiDisposer;
  /** Apply the entry owner's isolated stylesheet to a parent-owned child host. */
  style(entryId: string, host: HTMLElement): WebUiDisposer;
}

export interface WebEntryView {
  child(mountId: string): WebMountView;
}

export interface WebMountRegistration {
  register(entry: WebEntry): WebUiDisposer;
}

export interface WebHostContextV1 {
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

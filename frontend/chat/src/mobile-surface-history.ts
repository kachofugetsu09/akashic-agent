export type MobileSurface =
  | { kind: "chat" }
  | { kind: "plugins" }
  | { kind: "dashboard"; pluginId: string };

interface MobileSurfaceHistoryState {
  akashicMobileSurface: true;
  surface: MobileSurface;
}

interface HistoryWriter {
  pushState(data: unknown, unused: string): void;
  replaceState(data: unknown, unused: string): void;
}

export function mobileSurfaceHistoryState(surface: MobileSurface): MobileSurfaceHistoryState {
  return { akashicMobileSurface: true, surface };
}

export function readMobileSurfaceHistoryState(value: unknown): MobileSurface {
  if (!value || typeof value !== "object") return { kind: "chat" };
  const state = value as Partial<MobileSurfaceHistoryState>;
  if (state.akashicMobileSurface !== true || !state.surface) return { kind: "chat" };
  if (state.surface.kind === "chat" || state.surface.kind === "plugins") return state.surface;
  if (state.surface.kind === "dashboard" && state.surface.pluginId.trim()) return state.surface;
  return { kind: "chat" };
}

export function replaceMobileSurface(history: HistoryWriter, surface: MobileSurface): void {
  history.replaceState(mobileSurfaceHistoryState(surface), "");
}

export function pushMobileSurface(history: HistoryWriter, surface: Exclude<MobileSurface, { kind: "chat" }>): void {
  history.pushState(mobileSurfaceHistoryState(surface), "");
}

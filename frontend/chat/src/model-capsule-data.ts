export interface ChatModelRuntime {
  id: string;
  provider: string;
  model: string;
  sourceId: string;
  sourceName: string;
  reasoningEffort: string;
  supportedReasoningEfforts: string[];
  roles: string[];
}

export const EFFORT_LABELS: Record<string, string> = {
  none: "关闭",
  minimal: "极低",
  low: "低",
  medium: "中",
  high: "高",
  xhigh: "极高",
  max: "最大",
};

export function compatibleEffort(runtime: ChatModelRuntime | undefined, current: string): string {
  if (!runtime) return "";
  const supported = runtime.supportedReasoningEfforts;
  if (current && supported.includes(current)) return current;
  if (runtime.reasoningEffort && supported.includes(runtime.reasoningEffort)) return runtime.reasoningEffort;
  if (supported.includes("medium")) return "medium";
  return supported[0] || "";
}

export function groupModelRuntimes(runtimes: ChatModelRuntime[]) {
  const grouped = new Map<string, Array<{ runtime: ChatModelRuntime; index: number }>>();
  runtimes.forEach((runtime, index) => {
    const source = runtime.sourceName || runtime.provider;
    const items = grouped.get(source);
    if (items) items.push({ runtime, index });
    else grouped.set(source, [{ runtime, index }]);
  });
  return [...grouped.entries()];
}

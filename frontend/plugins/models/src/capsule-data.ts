export interface ChatModelRuntime {
  id: string;
  provider: string;
  model: string;
  sourceId: string;
  sourceName: string;
  /** 驱动声明的品牌图标（data URL）；未声明时为空。 */
  icon: string;
  reasoningEffort: string;
  supportedReasoningEfforts: string[];
  roles: string[];
  /** 服务端能力目录投影；老后端缺省时为 0/空数组。 */
  contextWindow: number;
  inputModalities: string[];
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

export interface ChatModelState {
  generationId: number;
  defaultRuntime: string;
  sessionOverride: string;
  sessionSelection: { modelRef: string; reasoningEffort: string };
  runtimes: ChatModelRuntime[];
  unavailableRuntimes: { id: string; model: string; sourceName: string; availability: "disabled" | "driver_unavailable" }[];
}

const record = (value: unknown): Record<string, unknown> | null =>
  typeof value === "object" && value !== null && !Array.isArray(value) ? value as Record<string, unknown> : null;
const strings = (value: unknown): value is string[] => Array.isArray(value) && value.every((item) => typeof item === "string");

// chat.models 查询结果的数据边界：字段不对就整体拒收，不让半份目录进入选择器。
export function chatModelState(payload: unknown): ChatModelState {
  const body = record(payload);
  const selection = record(body?.sessionSelection);
  if (!body || !Number.isInteger(body.generationId) || typeof body.defaultRuntime !== "string"
    || typeof body.sessionOverride !== "string" || !selection || typeof selection.modelRef !== "string"
    || typeof selection.reasoningEffort !== "string" || !Array.isArray(body.runtimes) || !Array.isArray(body.unavailableRuntimes)) {
    throw new Error("模型列表格式无效");
  }
  const runtimes = body.runtimes.map((value): ChatModelRuntime => {
    const item = record(value);
    if (!item || typeof item.id !== "string" || typeof item.provider !== "string" || typeof item.model !== "string"
      || typeof item.sourceId !== "string" || typeof item.sourceName !== "string" || typeof item.reasoningEffort !== "string"
      || !strings(item.supportedReasoningEfforts) || !strings(item.roles)) throw new Error("模型条目格式无效");
    return {
      id: item.id, provider: item.provider, model: item.model, sourceId: item.sourceId, sourceName: item.sourceName,
      icon: typeof item.icon === "string" && item.icon.startsWith("data:image/svg+xml,") ? item.icon : "",
      reasoningEffort: item.reasoningEffort, supportedReasoningEfforts: item.supportedReasoningEfforts, roles: item.roles,
      contextWindow: typeof item.contextWindow === "number" && Number.isFinite(item.contextWindow) ? item.contextWindow : 0,
      inputModalities: strings(item.inputModalities) ? item.inputModalities : [],
    };
  });
  const unavailableRuntimes = body.unavailableRuntimes.map((value): ChatModelState["unavailableRuntimes"][number] => {
    const item = record(value);
    if (!item || typeof item.id !== "string" || typeof item.model !== "string" || typeof item.sourceName !== "string"
      || (item.availability !== "disabled" && item.availability !== "driver_unavailable")) throw new Error("不可用模型条目格式无效");
    return { id: item.id, model: item.model, sourceName: item.sourceName, availability: item.availability };
  });
  return { generationId: Number(body.generationId), defaultRuntime: body.defaultRuntime, sessionOverride: body.sessionOverride,
    sessionSelection: { modelRef: selection.modelRef, reasoningEffort: selection.reasoningEffort }, runtimes, unavailableRuntimes };
}

// 当前要用的模型解释成用户能处理的一句话；空串表示可以发送。
export function modelProblem(state: ChatModelState | null, modelId: string): string {
  const unavailable = state?.unavailableRuntimes.find((item) => item.id === modelId);
  return !state ? ""
    : state.runtimes.some((item) => item.id === modelId) ? ""
    : unavailable ? unavailable.availability === "disabled"
      ? `连接「${unavailable.sourceName}」已停用，请在模型设置中恢复连接或选择其他模型。`
      : `连接「${unavailable.sourceName}」所需的模型驱动尚未加载或已卸载，请恢复驱动或选择其他模型。`
    : modelId ? "已选的对话模型暂不可用，请在模型设置中恢复或选择其他模型。"
    : state.runtimes.length ? "还没选择默认对话模型。请在下方选择模型，或在模型设置中设为默认。"
    : state.unavailableRuntimes.length ? "已保存的对话连接当前不可用，请在模型设置中恢复连接。"
    : "先连接一个对话模型，之后就可以发送消息。你可以先写下想说的话。";
}

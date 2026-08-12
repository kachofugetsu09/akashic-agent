export type ConnectionKind = "api" | "opencode-go" | "codex";
export type ModelRole = "default" | "fast" | "agent" | "vision";

export interface RuntimeSummary {
  id: string;
  provider: string;
  model: string;
  sourceId: string;
  sourceName: string;
  catalogProvider: string;
  baseUrl: string;
  contextWindow: number;
  maxOutputTokens: number;
  inputModalities: string[];
  reasoningEffort: string;
  supportedReasoningEfforts: string[];
  credential: { id: string; configured: boolean; source: string };
}

export interface RoleBinding {
  modelId: string;
  reasoningEffort: string;
}

export interface SettingsState {
  mode: "needs_setup" | "needs_repair" | "ready";
  workspace: string;
  error?: string;
  activeRuntime: string | null;
  runtimes: RuntimeSummary[];
  roleBindings: Partial<Record<ModelRole, RoleBinding>>;
  modelRevision: number;
  codexConfigured: boolean;
  localOpenCodeConfigured: boolean;
  configRevision: string;
  memory: import("./memory-settings").MemorySettingsState;
}

export interface ModelOption {
  id: string;
  contextWindow?: number;
  maxOutputTokens?: number;
  inputModalities?: string[];
  supportedReasoningEfforts?: string[];
  defaultReasoningEffort?: string;
}

export interface CodexLoginState {
  loginId: string;
  status: "waiting" | "completed" | "failed";
  userCode: string;
  verificationUri: string;
  interval: number;
  error: string;
}

export interface ConnectionGroup {
  sourceId: string;
  sourceName: string;
  provider: string;
  baseUrl: string;
  runtimes: RuntimeSummary[];
}

export interface ConnectionTemplate {
  kind: ConnectionKind;
  provider: string;
  name: string;
  detail: string;
  baseUrl: string;
}

export interface ConnectionDraft {
  sourceId: string;
  sourceName: string;
  kind: ConnectionKind;
  provider: string;
  baseUrl: string;
  apiKey: string;
  credentialId: string;
  model: string;
  reasoningEffort: string;
}

export async function requestSettingsJson<T>(url: string, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, {
      ...init,
      headers: { "Content-Type": "application/json", "X-Akasic-CSRF": "1", ...init.headers },
    });
  } catch (reason) {
    if (reason instanceof TypeError) throw new Error("无法连接 Akashic。请确认服务仍在运行，然后重试。", { cause: reason });
    throw reason;
  }
  const text = await response.text();
  let payload: { detail?: string; message?: string };
  try {
    payload = text ? JSON.parse(text) as { detail?: string; message?: string } : {};
  } catch {
    throw new Error(`设置服务返回了无效响应 (${response.status})`);
  }
  if (!response.ok) throw new Error(payload.detail || payload.message || `请求失败 (${response.status})`);
  return payload as T;
}

export function loadSettingsState(signal?: AbortSignal) {
  return requestSettingsJson<SettingsState>("/api/settings/state", { signal });
}

export function groupConnections(runtimes: RuntimeSummary[], query: string): ConnectionGroup[] {
  const groups = new Map<string, ConnectionGroup>();
  for (const runtime of runtimes) {
    const sourceId = runtime.sourceId || runtime.id;
    const current = groups.get(sourceId);
    if (current) current.runtimes.push(runtime);
    else groups.set(sourceId, {
      sourceId,
      sourceName: runtime.sourceName || runtime.provider,
      provider: runtime.provider,
      baseUrl: runtime.baseUrl,
      runtimes: [runtime],
    });
  }
  const normalized = query.trim().toLowerCase();
  return [...groups.values()].filter((group) =>
    `${group.sourceName} ${group.provider} ${group.runtimes.map((item) => item.model).join(" ")}`
      .toLowerCase().includes(normalized));
}

export function createConnectionDraft(template: ConnectionTemplate, existing?: ConnectionGroup): ConnectionDraft {
  return {
    sourceId: existing?.sourceId || `source-${crypto.randomUUID()}`,
    sourceName: existing?.sourceName || (template.provider ? template.name : ""),
    kind: existing ? connectionKind(existing.provider) : template.kind,
    provider: existing?.provider || template.provider,
    baseUrl: existing?.baseUrl || template.baseUrl,
    apiKey: "",
    credentialId: existing?.runtimes[0]?.credential.id || "",
    model: existing?.runtimes[0]?.model || "",
    reasoningEffort: existing?.runtimes[0]?.reasoningEffort || "",
  };
}

export function discoverConnectionModels(draft: ConnectionDraft, state: SettingsState, signal: AbortSignal) {
  return requestSettingsJson<{ models: ModelOption[] }>("/api/settings/models", {
    method: "POST",
    signal,
    body: JSON.stringify({
      provider: draft.provider,
      model: "",
      api_key: draft.apiKey,
      base_url: draft.baseUrl,
      credential_id: draft.kind === "codex" ? "codex_default" : draft.credentialId,
      use_local_opencode: draft.kind === "opencode-go" && state.localOpenCodeConfigured && !draft.apiKey,
    }),
  });
}

export function applyConnection(draft: ConnectionDraft, state: SettingsState, models: ModelOption[], signal: AbortSignal) {
  const accountCatalog = draft.kind === "codex" || draft.kind === "opencode-go";
  const selected = accountCatalog ? undefined : models.find((item) => item.id === draft.model);
  return requestSettingsJson("/api/settings/apply", {
    method: "POST",
    signal,
    body: JSON.stringify({
      provider: draft.provider,
      model: accountCatalog ? "" : draft.model,
      source_id: draft.sourceId,
      source_name: draft.sourceName,
      api_key: draft.apiKey,
      base_url: draft.baseUrl,
      credential_id: draft.kind === "codex" ? "codex_default" : draft.credentialId,
      use_local_opencode: draft.kind === "opencode-go" && state.localOpenCodeConfigured && !draft.apiKey,
      reasoning_effort: draft.reasoningEffort,
      context_window: selected?.contextWindow || 0,
      max_output_tokens: selected?.maxOutputTokens || 0,
      input_modalities: selected?.inputModalities,
      expected_config_revision: state.configRevision,
      defer_restart: state.runtimes.length === 0,
    }),
  });
}

export function startCodexLogin(signal: AbortSignal) {
  return requestSettingsJson<CodexLoginState>("/api/settings/codex-login", { method: "POST", body: "{}", signal });
}

export function loadCodexLogin(loginId: string, signal: AbortSignal) {
  return requestSettingsJson<CodexLoginState>(`/api/settings/codex-login/${encodeURIComponent(loginId)}`, { signal });
}

export function saveRoleBinding(role: ModelRole, modelId: string, state: SettingsState, signal: AbortSignal) {
  return requestSettingsJson("/api/settings/roles", {
    method: "POST",
    signal,
    body: JSON.stringify({
      role,
      model_id: modelId,
      reasoning_effort: state.roleBindings[role]?.reasoningEffort || "",
      expected_revision: state.modelRevision,
    }),
  });
}

export function settingsErrorMessage(reason: unknown) {
  return reason instanceof Error ? reason.message : String(reason);
}

function connectionKind(provider: string): ConnectionKind {
  if (provider === "codex") return "codex";
  if (provider === "opencode-go") return "opencode-go";
  return "api";
}

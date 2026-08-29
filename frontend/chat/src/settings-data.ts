import { createUuid } from "./browser-uuid.ts";
import type { EmbeddingModelSummary } from "./memory-settings-data.ts";
import { requestSettingsJson } from "./settings-http.ts";

export type ConnectionKind = "api" | "opencode-go" | "codex";
export type ModelRole = "default" | "fast" | "agent" | "vision";

interface CatalogConnection {
  id: string;
  name: string;
  driverId: string;
  authIdentity: string;
  availability: string;
}

interface CatalogModel {
  id: string;
  connectionId: string;
  kind: "chat" | "embedding";
  model: string;
  defaultReasoningEffort: string | null;
  availability: string;
  capabilities: {
    contextWindow: number | null;
    maxOutputTokens: number | null;
    inputModalities: string[];
    supportedReasoningEfforts: string[];
    embeddingDimensions: number | null;
  };
}

export interface ModelCatalogState {
  revision: number;
  connections: CatalogConnection[];
  models: CatalogModel[];
  roleBindings: Partial<Record<ModelRole, string>>;
  defaultEmbeddingModelId: string | null;
}

interface CommandReceipt {
  revision: number;
  status: string;
  attemptId: string | null;
  challenge: Record<string, unknown> | null;
}

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
  mode: "needs_setup" | "ready";
  workspace: string;
  activeRuntime: string | null;
  runtimes: RuntimeSummary[];
  roleBindings: Partial<Record<ModelRole, RoleBinding>>;
  modelRevision: number;
  codexConfigured: boolean;
  localOpenCodeConfigured: boolean;
  configRevision: string;
  embeddingModels: EmbeddingModelSummary[];
  catalog: ModelCatalogState;
}

export interface CodexLoginState {
  loginId: string;
  status: "waiting" | "completed" | "failed";
  userCode: string;
  verificationUri: string;
  interval: number;
  error: string;
  revision: number;
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

export async function loadSettingsState(signal?: AbortSignal): Promise<SettingsState> {
  const catalog = await requestSettingsJson<ModelCatalogState>("/api/settings/model/catalog", { signal });
  const connections = new Map(catalog.connections.map((item) => [item.id, item]));
  const runtimes = catalog.models
    .filter((item) => item.kind === "chat" && item.availability === "available")
    .map((item) => {
      const connection = connections.get(item.connectionId);
      if (!connection) throw new Error(`模型 ${item.id} 缺少连接 ${item.connectionId}`);
      return {
        id: item.id,
        provider: connection.driverId,
        model: item.model,
        sourceId: connection.id,
        sourceName: connection.name,
        catalogProvider: connection.driverId,
        baseUrl: "",
        contextWindow: item.capabilities.contextWindow || 0,
        maxOutputTokens: item.capabilities.maxOutputTokens || 0,
        inputModalities: item.capabilities.inputModalities,
        reasoningEffort: item.defaultReasoningEffort || "",
        supportedReasoningEfforts: item.capabilities.supportedReasoningEfforts,
        credential: { id: connection.authIdentity, configured: true, source: "model-plugin" },
      };
    });
  const roleBindings = Object.fromEntries(
    Object.entries(catalog.roleBindings).map(([role, modelId]) => {
      const model = catalog.models.find((item) => item.id === modelId);
      return [role, { modelId, reasoningEffort: model?.defaultReasoningEffort || "" }];
    }),
  ) as Partial<Record<ModelRole, RoleBinding>>;
  const embeddingModels = catalog.models
    .filter((item) => item.kind === "embedding" && item.availability === "available")
    .map((item) => {
      const connection = connections.get(item.connectionId);
      if (!connection) throw new Error(`模型 ${item.id} 缺少连接 ${item.connectionId}`);
      return {
        id: item.id,
        sourceId: connection.id,
        sourceName: connection.name,
        provider: connection.driverId,
        baseUrl: "",
        model: item.model,
        dimensions: item.capabilities.embeddingDimensions || 0,
        credential: { id: connection.authIdentity, configured: true },
      };
    });
  return {
    mode: runtimes.length ? "ready" : "needs_setup",
    workspace: "",
    activeRuntime: catalog.roleBindings.default || null,
    runtimes,
    roleBindings,
    modelRevision: catalog.revision,
    codexConfigured: catalog.connections.some((item) => item.driverId === "codex"),
    localOpenCodeConfigured: false,
    configRevision: "",
    embeddingModels,
    catalog,
  };
}

export function groupConnections(runtimes: RuntimeSummary[], query: string): ConnectionGroup[] {
  const groups = new Map<string, ConnectionGroup>();
  for (const runtime of runtimes) {
    const current = groups.get(runtime.sourceId);
    if (current) current.runtimes.push(runtime);
    else groups.set(runtime.sourceId, {
      sourceId: runtime.sourceId,
      sourceName: runtime.sourceName,
      provider: runtime.provider,
      baseUrl: runtime.baseUrl,
      runtimes: [runtime],
    });
  }
  const normalized = query.trim().toLowerCase();
  return [...groups.values()].filter((group) =>
    `${group.sourceName} ${group.provider} ${group.runtimes.map((item) => item.model).join(" ")}`.toLowerCase().includes(normalized));
}

export function createConnectionDraft(template: ConnectionTemplate, existing?: ConnectionGroup): ConnectionDraft {
  return {
    sourceId: existing?.sourceId || `source-${createUuid()}`,
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

export async function applyConnection(draft: ConnectionDraft, state: SettingsState, signal: AbortSignal) {
  let revision = state.modelRevision;
  let defaultModelId = "";
  const existing = state.catalog.connections.some((item) => item.id === draft.sourceId);
  if (draft.kind === "opencode-go") {
    const started = await modelCommand({
      type: "start_auth",
      driver_id: "opencode-go",
      connection_id: draft.sourceId,
      input: {
        ...(draft.apiKey ? { api_key: draft.apiKey } : {}),
        ...(draft.baseUrl ? { endpoint: draft.baseUrl } : {}),
        name: draft.sourceName,
        auth_identity: draft.credentialId || draft.sourceId,
      },
    }, signal);
    if (!started.attemptId) throw new Error("OpenCode 登录没有返回 attempt ID");
    const finished = await modelCommand({ type: "finish_auth", expected_revision: revision, attempt_id: started.attemptId }, signal);
    revision = finished.revision;
  } else if (draft.kind === "api") {
    const connection = {
      expected_revision: revision,
      connection_id: draft.sourceId,
      name: draft.sourceName,
      ...(draft.baseUrl ? { endpoint: draft.baseUrl } : {}),
      auth_identity: draft.credentialId || `api:${draft.sourceId}`,
      credential: draft.apiKey ? { driver: "api_key", access_token: draft.apiKey } : null,
      driver_config: { format_version: 1, catalog_provider_id: draft.provider || "openai", allow_unverified_manual: true },
    };
    const known = state.catalog.models.some((item) => item.connectionId === draft.sourceId && item.model === draft.model);
    const model = {
      expected_revision: revision,
      model_id: `${draft.sourceId}__${createUuid()}`,
      connection_id: draft.sourceId,
      kind: "chat",
      model: draft.model,
      capabilities: {
        context_window: null,
        max_output_tokens: null,
        input_modalities: ["text"],
        supports_tool_calls: null,
        supports_parallel_tool_calls: null,
        supported_reasoning_efforts: [],
        embedding_dimensions: null,
        embedding_normalization: null,
      },
      capability_sources: {},
      default_reasoning_effort: draft.reasoningEffort || null,
      driver_config: { format_version: 1 },
    };
    if (!existing) {
      defaultModelId = model.model_id;
      const created = await modelCommand({
        type: "create_connection_with_model",
        connection: { ...connection, driver_id: "openai-compatible", credential: connection.credential || {} },
        model,
      }, signal);
      revision = created.revision;
    } else {
      const receipt = await modelCommand({ ...connection, type: "update_connection" }, signal);
      revision = receipt.revision;
    }
    if (existing && draft.model && !known) {
      defaultModelId = model.model_id;
      const added = await modelCommand({ ...model, type: "add_model", expected_revision: revision }, signal);
      revision = added.revision;
    }
  }
  if (draft.kind !== "api") {
    const synced = await modelCommand({ type: "sync_models", expected_revision: revision, connection_id: draft.sourceId }, signal);
    revision = synced.revision;
  }
  if (!state.catalog.roleBindings.default) {
    if (!defaultModelId) {
      const latest = await requestSettingsJson<ModelCatalogState>("/api/settings/model/catalog", { signal });
      defaultModelId = latest.models.find((item) => item.connectionId === draft.sourceId && item.kind === "chat")?.id || "";
    }
    if (defaultModelId) await modelCommand({ type: "set_default", expected_revision: revision, role: "default", model_id: defaultModelId }, signal);
  }
}

export async function startCodexLogin(draft: ConnectionDraft, signal: AbortSignal) {
  const receipt = await modelCommand({ type: "start_auth", driver_id: "codex", connection_id: draft.sourceId, input: {} }, signal);
  return codexLoginState(receipt);
}

export async function loadCodexLogin(loginId: string, revision: number, signal: AbortSignal) {
  const receipt = await modelCommand({ type: "finish_auth", expected_revision: revision, attempt_id: loginId }, signal);
  return codexLoginState(receipt);
}

export function cancelConnectionAuth(attemptId: string) {
  return requestSettingsJson<CommandReceipt>("/api/settings/model/command", {
    method: "POST",
    keepalive: true,
    body: JSON.stringify({ type: "cancel_auth", attempt_id: attemptId }),
  });
}

export function saveRoleBinding(role: ModelRole, modelId: string, state: SettingsState, signal: AbortSignal) {
  return modelCommand({ type: "set_default", expected_revision: state.modelRevision, role, model_id: modelId }, signal);
}

function modelCommand(payload: Record<string, unknown>, signal: AbortSignal) {
  return requestSettingsJson<CommandReceipt>("/api/settings/model/command", { method: "POST", signal, body: JSON.stringify(payload) });
}

function codexLoginState(receipt: CommandReceipt): CodexLoginState {
  const challenge = receipt.challenge || {};
  return {
    loginId: receipt.attemptId || "",
    status: receipt.status === "committed" ? "completed" : receipt.status === "pending" ? "waiting" : "failed",
    userCode: String(challenge.user_code || ""),
    verificationUri: String(challenge.verification_uri || ""),
    interval: Number(challenge.interval || 5),
    error: "",
    revision: receipt.revision,
  };
}

function connectionKind(provider: string): ConnectionKind {
  if (provider === "codex") return "codex";
  if (provider === "opencode-go") return "opencode-go";
  return "api";
}

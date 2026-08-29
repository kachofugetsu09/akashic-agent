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

export interface ChatModelSummary {
  id: string;
  provider: string;
  model: string;
  sourceId: string;
  sourceName: string;
  baseUrl: string;
  reasoningEffort: string;
  credentialId: string;
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
  models: ChatModelSummary[];
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

export function loadModelCatalog(signal?: AbortSignal): Promise<ModelCatalogState> {
  return requestSettingsJson<ModelCatalogState>("/api/settings/model/catalog", { signal });
}

export function availableChatModels(catalog: ModelCatalogState): ChatModelSummary[] {
  const connections = new Map(catalog.connections.map((item) => [item.id, item]));
  return catalog.models
    .filter((item) => item.kind === "chat" && item.availability === "available")
    .map((item) => {
      const connection = requireConnection(connections, item);
      return {
        id: item.id,
        provider: connection.driverId,
        model: item.model,
        sourceId: connection.id,
        sourceName: connection.name,
        baseUrl: "",
        reasoningEffort: item.defaultReasoningEffort || "",
        credentialId: connection.authIdentity,
      };
    });
}

export function availableEmbeddingModels(catalog: ModelCatalogState): EmbeddingModelSummary[] {
  const connections = new Map(catalog.connections.map((item) => [item.id, item]));
  return catalog.models
    .filter((item) => item.kind === "embedding" && item.availability === "available")
    .map((item) => {
      const connection = requireConnection(connections, item);
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
}

export function hasAvailableChatModel(catalog: ModelCatalogState): boolean {
  return catalog.models.some((item) => item.kind === "chat" && item.availability === "available");
}

export function groupConnections(models: ChatModelSummary[], query: string): ConnectionGroup[] {
  const groups = new Map<string, ConnectionGroup>();
  for (const model of models) {
    const current = groups.get(model.sourceId);
    if (current) current.models.push(model);
    else groups.set(model.sourceId, {
      sourceId: model.sourceId,
      sourceName: model.sourceName,
      provider: model.provider,
      baseUrl: model.baseUrl,
      models: [model],
    });
  }
  const normalized = query.trim().toLowerCase();
  return [...groups.values()].filter((group) =>
    `${group.sourceName} ${group.provider} ${group.models.map((item) => item.model).join(" ")}`.toLowerCase().includes(normalized));
}

export function createConnectionDraft(template: ConnectionTemplate, existing?: ConnectionGroup): ConnectionDraft {
  return {
    sourceId: existing?.sourceId || `source-${createUuid()}`,
    sourceName: existing?.sourceName || (template.provider ? template.name : ""),
    kind: existing ? connectionKind(existing.provider) : template.kind,
    provider: existing?.provider || template.provider,
    baseUrl: existing?.baseUrl || template.baseUrl,
    apiKey: "",
    credentialId: existing?.models[0]?.credentialId || "",
    model: existing?.models[0]?.model || "",
    reasoningEffort: existing?.models[0]?.reasoningEffort || "",
  };
}

export async function applyConnection(draft: ConnectionDraft, catalog: ModelCatalogState, signal: AbortSignal) {
  let revision = catalog.revision;
  let defaultModelId = "";
  const existing = catalog.connections.some((item) => item.id === draft.sourceId);
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
    const known = catalog.models.some((item) => item.connectionId === draft.sourceId && item.model === draft.model);
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
  if (!catalog.roleBindings.default) {
    if (!defaultModelId) {
      const latest = await loadModelCatalog(signal);
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

export function saveRoleBinding(role: ModelRole, modelId: string, catalog: ModelCatalogState, signal: AbortSignal) {
  return modelCommand({ type: "set_default", expected_revision: catalog.revision, role, model_id: modelId }, signal);
}

function requireConnection(connections: ReadonlyMap<string, CatalogConnection>, model: CatalogModel): CatalogConnection {
  const connection = connections.get(model.connectionId);
  if (!connection) throw new Error(`模型 ${model.id} 缺少连接 ${model.connectionId}`);
  return connection;
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

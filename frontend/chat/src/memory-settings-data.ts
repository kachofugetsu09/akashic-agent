import { requestSettingsJson } from "./settings-http.ts";
import { createUuid } from "./browser-uuid.ts";

export interface EmbeddingModelSummary {
  id: string;
  sourceId: string;
  sourceName: string;
  provider: string;
  baseUrl: string;
  model: string;
  dimensions: number;
  credential: { id: string; configured: boolean };
}

export interface EmbeddingDraft {
  sourceName: string;
  baseUrl: string;
  apiKey: string;
  model: string;
  dimensions: number;
}

export async function saveEmbeddingModel(draft: EmbeddingDraft, modelRevision: number, signal: AbortSignal) {
  const sourceId = `embedding-${createUuid()}`;
  const modelId = `${sourceId}__${createUuid()}`;
  const authIdentity = `api:${sourceId}`;
  const created = await modelCommand({
    type: "create_connection_with_model",
    connection: {
      expected_revision: modelRevision,
      connection_id: sourceId,
      name: draft.sourceName,
      driver_id: "openai-compatible",
      endpoint: draft.baseUrl,
      auth_identity: authIdentity,
      credential: { driver: "api_key", access_token: draft.apiKey },
      driver_config: { format_version: 1, catalog_provider_id: "openai", allow_unverified_manual: true },
    },
    model: {
      expected_revision: modelRevision,
      model_id: modelId,
      connection_id: sourceId,
      kind: "embedding",
      model: draft.model,
      capabilities: { embedding_dimensions: draft.dimensions, input_modalities: ["text"] },
      capability_sources: { embedding_dimensions: "user" },
      default_reasoning_effort: null,
      driver_config: { format_version: 1 },
    },
  }, signal);
  await modelCommand({
    type: "set_default",
    expected_revision: created.revision,
    role: null,
    model_id: modelId,
  }, signal);
  return {
    model: {
      id: modelId,
      sourceId,
      sourceName: draft.sourceName,
      provider: "openai-compatible",
      baseUrl: draft.baseUrl,
      model: draft.model,
      dimensions: draft.dimensions,
      credential: { id: authIdentity, configured: true },
    },
  };
}

export function saveDefaultEmbedding(modelId: string, modelRevision: number, signal: AbortSignal) {
  return modelCommand({
    type: "set_default",
    expected_revision: modelRevision,
    role: null,
    model_id: modelId,
  }, signal);
}

function modelCommand(payload: Record<string, unknown>, signal: AbortSignal) {
  return requestSettingsJson<{ revision: number }>("/api/settings/model/command", {
    method: "POST",
    signal,
    body: JSON.stringify(payload),
  });
}

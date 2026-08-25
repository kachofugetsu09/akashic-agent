import { requestSettingsJson } from "./settings-http.ts";

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

export interface MemorySettingsState {
  configured: boolean;
  enabled: boolean;
  embeddingModelId: string;
  embeddingModels: EmbeddingModelSummary[];
  changeLocked: boolean;
  revision: string;
}

export interface EmbeddingDraft {
  sourceName: string;
  baseUrl: string;
  apiKey: string;
  model: string;
}

export function saveMemorySettings(mode: "akasha" | "off", modelId: string, revision: string, signal: AbortSignal) {
  return requestSettingsJson("/api/settings/memory", {
    method: "POST",
    signal,
    body: JSON.stringify({
      enabled: mode !== "off",
      embedding_model_id: mode === "off" ? "" : modelId,
      expected_revision: revision,
    }),
  });
}

export function saveEmbeddingModel(draft: EmbeddingDraft, modelRevision: number, signal: AbortSignal) {
  return requestSettingsJson<{ model: EmbeddingModelSummary }>("/api/settings/embedding-models", {
    method: "POST",
    signal,
    body: JSON.stringify({
      source_name: draft.sourceName,
      provider: "openai",
      base_url: draft.baseUrl,
      api_key: draft.apiKey,
      model: draft.model,
      expected_revision: modelRevision,
    }),
  });
}

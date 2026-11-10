export function htmlNode(markup) {
  const template = document.createElement("template");
  template.innerHTML = markup;
  return template.content.firstElementChild;
}

export function escapeHtml(value) {
  const node = document.createElement("span");
  node.textContent = value;
  return node.innerHTML;
}

export function requireConnectionProps(value, requiredActions = []) {
  if (
    !value
    || typeof value !== "object"
    || typeof value.actions !== "object"
    || requiredActions.some((name) => typeof value.actions[name] !== "function")
    || typeof value.dirty !== "function"
    || typeof value.close !== "function"
    || typeof value.changed !== "function"
    || !value.state
  ) {
    throw new Error("models.connection-types.v1 props 无效");
  }
  if (
    value.state.connection !== null
    && value.state.connection !== undefined
    && (typeof value.state.connection !== "object" || typeof value.state.connection.id !== "string")
  ) {
    throw new Error("models.connection-types.v1 connection 无效");
  }
  return value;
}

export function toDiscoveredModelInput(model, manual = false) {
  const capabilities = model.capabilities ?? {};
  const sources = model.capabilitySources ?? {};
  return {
    kind: "chat",
    discovery_owned: !manual,
    model: model.model,
    capabilities: {
      context_window: capabilities.contextWindow ?? null,
      max_output_tokens: capabilities.maxOutputTokens ?? null,
      input_modalities: capabilities.inputModalities ?? ["text"],
      supports_tool_calls: capabilities.supportsToolCalls ?? null,
      supports_parallel_tool_calls: capabilities.supportsParallelToolCalls ?? null,
      supported_reasoning_efforts: capabilities.supportedReasoningEfforts ?? [],
      embedding_dimensions: capabilities.embeddingDimensions ?? null,
      embedding_normalization: capabilities.embeddingNormalization ?? null,
    },
    capability_sources: {
      context_window: sources.contextWindow ?? "unknown",
      max_output_tokens: sources.maxOutputTokens ?? "unknown",
      input_modalities: sources.inputModalities ?? "unknown",
      tool_calls: sources.toolCalls ?? "unknown",
      parallel_tool_calls: sources.parallelToolCalls ?? "unknown",
      reasoning_efforts: sources.reasoningEfforts ?? "unknown",
      embedding_dimensions: sources.embeddingDimensions ?? "unknown",
      embedding_normalization: sources.embeddingNormalization ?? "unknown",
    },
    default_reasoning_effort: model.defaultReasoningEffort ?? null,
    driver_config: model.driverConfig ?? {format_version: 1},
  };
}

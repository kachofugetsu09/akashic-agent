import { useCallback, useEffect, useRef, useState } from "react";
import { saveMemorySettings, type MemorySettingsState } from "./memory-settings-data";
import { settingsErrorMessage } from "./settings-http.ts";

interface Options {
  memory: MemorySettingsState;
  onComplete: (message: string) => void;
  onError: (message: string) => void;
  onValidationRequired: (hasModels: boolean) => void;
}

/** Own memory selection and serialize its persisted mutation. */
export function useMemorySettings({ memory, onComplete, onError, onValidationRequired }: Options) {
  const [mode, setModeState] = useState<"akasha" | "off">(memory.enabled ? "akasha" : "off");
  const [modelId, setModelId] = useState(memory.embeddingModelId);
  const [saving, setSaving] = useState(false);
  const [validationError, setValidationError] = useState("");
  const requestRef = useRef<AbortController | null>(null);

  useEffect(() => () => requestRef.current?.abort(), []);

  const setMode = useCallback((next: "akasha" | "off") => {
    setModeState(next);
    setValidationError("");
  }, []);

  const selectModel = useCallback((next: string) => {
    setModelId(next);
    setValidationError("");
  }, []);

  const save = useCallback(async () => {
    if (requestRef.current) return;
    if (mode !== "off" && !modelId) {
      setValidationError("启用记忆前，请先添加并选择一个向量模型。");
      onValidationRequired(memory.embeddingModels.length > 0);
      return;
    }
    const controller = new AbortController();
    requestRef.current = controller;
    setSaving(true);
    setValidationError("");
    onError("");
    try {
      await saveMemorySettings(mode, modelId, memory.revision, controller.signal);
      onComplete(mode === "off" ? "已关闭语义记忆" : "Akasha 已启用");
    } catch (reason) {
      if (!controller.signal.aborted) onError(settingsErrorMessage(reason));
    } finally {
      if (requestRef.current === controller) requestRef.current = null;
      if (!controller.signal.aborted) setSaving(false);
    }
  }, [memory.embeddingModels.length, memory.revision, mode, modelId, onComplete, onError, onValidationRequired]);

  return { mode, setMode, modelId, selectModel, saving, validationError, save };
}

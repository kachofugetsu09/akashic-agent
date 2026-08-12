import { useCallback, useEffect, useRef, useState } from "react";
import {
  applyConnection,
  createConnectionDraft,
  discoverConnectionModels,
  loadCodexLogin,
  startCodexLogin,
  type CodexLoginState,
  type ConnectionDraft,
  type ConnectionGroup,
  type ConnectionTemplate,
  type ModelOption,
  type SettingsState,
} from "./settings-data";
import { settingsErrorMessage } from "./settings-http.ts";

interface UseSettingsConnectionOptions {
  template: ConnectionTemplate;
  existing?: ConnectionGroup;
  settings: SettingsState;
  onSaved: (firstConnection: boolean, sourceName: string) => Promise<void>;
  onLoginCompleted: () => Promise<void>;
}

/** Own one connection editor's form, requests, and cancellable login lifecycle. */
export function useSettingsConnection({ template, existing, settings, onSaved, onLoginCompleted }: UseSettingsConnectionOptions) {
  const [draft, setDraft] = useState<ConnectionDraft>(() => createConnectionDraft(template, existing));
  const [models, setModels] = useState<ModelOption[]>([]);
  const [discovering, setDiscovering] = useState(false);
  const [saving, setSaving] = useState(false);
  const [showKey, setShowKey] = useState(false);
  const [error, setError] = useState("");
  const [codexLogin, setCodexLogin] = useState<CodexLoginState | null>(null);
  const discoverRef = useRef<AbortController | null>(null);
  const saveRef = useRef<AbortController | null>(null);
  const loginRef = useRef<AbortController | null>(null);

  useEffect(() => () => {
    discoverRef.current?.abort();
    saveRef.current?.abort();
    loginRef.current?.abort();
  }, []);

  const discover = useCallback(async () => {
    if (discoverRef.current) return;
    const controller = new AbortController();
    discoverRef.current = controller;
    setDiscovering(true);
    setError("");
    try {
      const result = await discoverConnectionModels(draft, settings, controller.signal);
      setModels(result.models);
      if (result.models[0]) {
        setDraft((current) => ({
          ...current,
          model: current.model || result.models[0].id,
          reasoningEffort: current.reasoningEffort || result.models[0].defaultReasoningEffort || "",
        }));
      } else {
        setError("没有发现模型。请确认 Base URL 和认证，或手动填写模型名。");
      }
    } catch (reason) {
      if (!controller.signal.aborted) setError(settingsErrorMessage(reason));
    } finally {
      if (discoverRef.current === controller) discoverRef.current = null;
      if (!controller.signal.aborted) setDiscovering(false);
    }
  }, [draft, settings]);

  const save = useCallback(async () => {
    if (saveRef.current) return;
    const controller = new AbortController();
    saveRef.current = controller;
    setSaving(true);
    setError("");
    try {
      const firstConnection = settings.runtimes.length === 0;
      await applyConnection(draft, settings, models, controller.signal);
      await onSaved(firstConnection, draft.sourceName);
    } catch (reason) {
      if (!controller.signal.aborted) setError(settingsErrorMessage(reason));
    } finally {
      if (saveRef.current === controller) saveRef.current = null;
      if (!controller.signal.aborted) setSaving(false);
    }
  }, [draft, models, onSaved, settings]);

  const beginLogin = useCallback(async () => {
    if (loginRef.current) return;
    const controller = new AbortController();
    loginRef.current = controller;
    setError("");
    try {
      setCodexLogin(await startCodexLogin(controller.signal));
    } catch (reason) {
      if (!controller.signal.aborted) setError(settingsErrorMessage(reason));
    } finally {
      if (loginRef.current === controller) loginRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (codexLogin?.status !== "waiting") return;
    const controller = new AbortController();
    let timeoutId = 0;
    const poll = async () => {
      try {
        const next = await loadCodexLogin(codexLogin.loginId, controller.signal);
        setCodexLogin(next);
        if (next.status === "completed") await onLoginCompleted();
        else if (next.status === "waiting") {
          timeoutId = window.setTimeout(() => void poll(), Math.max(3, next.interval) * 1_000);
        } else if (next.error) setError(next.error);
      } catch (reason) {
        if (!controller.signal.aborted) setError(settingsErrorMessage(reason));
      }
    };
    timeoutId = window.setTimeout(() => void poll(), Math.max(3, codexLogin.interval) * 1_000);
    return () => { controller.abort(); window.clearTimeout(timeoutId); };
  }, [codexLogin, onLoginCompleted]);

  return {
    draft,
    setDraft,
    models,
    discovering,
    saving,
    showKey,
    setShowKey,
    error,
    codexLogin,
    discover,
    save,
    beginLogin,
  };
}

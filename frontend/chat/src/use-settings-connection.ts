import { useCallback, useEffect, useRef, useState } from "react";
import {
  applyConnection,
  cancelConnectionAuth,
  createConnectionDraft,
  loadCodexLogin,
  startCodexLogin,
  type CodexLoginState,
  type ConnectionDraft,
  type ConnectionGroup,
  type ConnectionTemplate,
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
  const [saving, setSaving] = useState(false);
  const [showKey, setShowKey] = useState(false);
  const [error, setError] = useState("");
  const [codexLogin, setCodexLogin] = useState<CodexLoginState | null>(null);
  const saveRef = useRef<AbortController | null>(null);
  const loginRef = useRef<AbortController | null>(null);
  const loginAttemptRef = useRef("");

  useEffect(() => () => {
    saveRef.current?.abort();
    loginRef.current?.abort();
    if (loginAttemptRef.current) {
      void cancelConnectionAuth(loginAttemptRef.current).catch((reason) => {
        console.error("取消模型登录失败", reason);
      });
    }
  }, []);

  const save = useCallback(async () => {
    if (saveRef.current) return;
    const controller = new AbortController();
    saveRef.current = controller;
    setSaving(true);
    setError("");
    try {
      const firstConnection = settings.runtimes.length === 0;
      await applyConnection(draft, settings, controller.signal);
      await onSaved(firstConnection, draft.sourceName);
    } catch (reason) {
      if (!controller.signal.aborted) setError(settingsErrorMessage(reason));
    } finally {
      if (saveRef.current === controller) saveRef.current = null;
      if (!controller.signal.aborted) setSaving(false);
    }
  }, [draft, onSaved, settings]);

  const beginLogin = useCallback(async () => {
    if (loginRef.current || codexLogin?.status === "waiting") return;
    const controller = new AbortController();
    loginRef.current = controller;
    setError("");
    try {
      const started = await startCodexLogin(draft, controller.signal);
      loginAttemptRef.current = started.loginId;
      setCodexLogin(started);
    } catch (reason) {
      if (!controller.signal.aborted) setError(settingsErrorMessage(reason));
    } finally {
      if (loginRef.current === controller) loginRef.current = null;
    }
  }, [codexLogin?.status, draft]);

  useEffect(() => {
    if (codexLogin?.status !== "waiting") return;
    const controller = new AbortController();
    let timeoutId = 0;
    const poll = async () => {
      try {
        const next = await loadCodexLogin(codexLogin.loginId, codexLogin.revision, controller.signal);
        setCodexLogin(next);
        if (next.status === "completed") {
          loginAttemptRef.current = "";
          await onLoginCompleted();
        }
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
    saving,
    showKey,
    setShowKey,
    error,
    codexLogin,
    save,
    beginLogin,
  };
}

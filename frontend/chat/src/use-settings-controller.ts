import { useCallback, useEffect, useRef, useState } from "react";
import {
  loadSettingsState,
  saveRoleBinding,
  type ModelRole,
  type SettingsState,
} from "./settings-data";
import { settingsErrorMessage } from "./settings-http.ts";

/** Own settings page state and serialize refresh and role mutations. */
export function useSettingsController() {
  const [state, setState] = useState<SettingsState | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const refreshRef = useRef<AbortController | null>(null);
  const roleRef = useRef<AbortController | null>(null);

  const refresh = useCallback(async () => {
    refreshRef.current?.abort();
    const controller = new AbortController();
    refreshRef.current = controller;
    try {
      const next = await loadSettingsState(controller.signal);
      if (!controller.signal.aborted) setState(next);
      return next;
    } catch (reason) {
      if (!controller.signal.aborted) setError(settingsErrorMessage(reason));
      return undefined;
    } finally {
      if (refreshRef.current === controller) refreshRef.current = null;
    }
  }, []);

  useEffect(() => {
    void refresh();
    return () => {
      refreshRef.current?.abort();
      roleRef.current?.abort();
    };
  }, [refresh]);

  const updateRole = useCallback(async (role: ModelRole, modelId: string) => {
    if (!state || roleRef.current) return;
    const controller = new AbortController();
    roleRef.current = controller;
    setError("");
    try {
      await saveRoleBinding(role, modelId, state, controller.signal);
      await refresh();
      setNotice(`${roleLabel(role)}已更新；正在运行的任务继续使用旧快照`);
    } catch (reason) {
      if (!controller.signal.aborted) setError(settingsErrorMessage(reason));
    } finally {
      if (roleRef.current === controller) roleRef.current = null;
    }
  }, [refresh, state]);

  return { state, error, setError, notice, setNotice, refresh, updateRole };
}

function roleLabel(role: ModelRole) {
  return { default: "默认模型", agent: "Agent 模型", fast: "轻量模型", vision: "视觉模型" }[role];
}

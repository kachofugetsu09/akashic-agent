import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  loadRuntimeDetail,
  loadRuntimeOverview,
  runtimeErrorMessage,
  runtimeItems,
  type RuntimeDetail,
  type RuntimeOverview,
  type RuntimeView,
} from "./runtime-dashboard-data";

const EMPTY_SELECTIONS: Record<RuntimeView, string> = { documents: "", mcp: "", jobs: "" };

/** Own runtime loading and selection without exposing transport state to the view. */
export function useRuntimeDashboard() {
  const [view, setView] = useState<RuntimeView>("documents");
  const [overview, setOverview] = useState<RuntimeOverview | null>(null);
  const [selectedKeys, setSelectedKeys] = useState(EMPTY_SELECTIONS);
  const [detail, setDetail] = useState<RuntimeDetail | null>(null);
  const [detailOpen, setDetailOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState("");
  const [syncedAt, setSyncedAt] = useState<Date | null>(null);
  const [copyFeedback, setCopyFeedback] = useState("");
  const copyTimerRef = useRef<number | undefined>(undefined);

  const items = useMemo(() => runtimeItems(view, overview), [overview, view]);
  const selectedKey = useMemo(() => {
    const remembered = selectedKeys[view];
    if (items.some((item) => !item.disabled && item.key === remembered)) return remembered;
    return items.find((item) => !item.disabled)?.key ?? "";
  }, [items, selectedKeys, view]);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      setOverview(await loadRuntimeOverview());
      setSyncedAt(new Date());
    } catch (loadError) {
      setError(runtimeErrorMessage(loadError));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (!selectedKey) {
      setDetail(null);
      setDetailLoading(false);
      return;
    }
    const controller = new AbortController();
    setDetail(null);
    setDetailLoading(true);
    setError("");
    void loadRuntimeDetail(view, selectedKey, controller.signal)
      .then(setDetail)
      .catch((loadError: unknown) => {
        if (loadError instanceof DOMException && loadError.name === "AbortError") return;
        setError(runtimeErrorMessage(loadError));
      })
      .finally(() => {
        if (!controller.signal.aborted) setDetailLoading(false);
      });
    return () => controller.abort();
  }, [selectedKey, view]);

  useEffect(() => () => window.clearTimeout(copyTimerRef.current), []);

  const selectView = useCallback((nextView: RuntimeView) => {
    setView(nextView);
    setDetailOpen(false);
  }, []);

  const selectItem = useCallback((key: string) => {
    setSelectedKeys((current) => current[view] === key ? current : { ...current, [view]: key });
    setDetailOpen(true);
  }, [view]);

  const copyDetail = useCallback(async () => {
    if (!detail) return;
    try {
      await navigator.clipboard.writeText(detail.copyText);
      setCopyFeedback("标识已复制");
      window.clearTimeout(copyTimerRef.current);
      copyTimerRef.current = window.setTimeout(() => setCopyFeedback(""), 1500);
    } catch (copyError) {
      setError(runtimeErrorMessage(copyError));
    }
  }, [detail]);

  const closeDetail = useCallback(() => setDetailOpen(false), []);

  return {
    view,
    overview,
    items,
    selectedKey,
    detail,
    detailOpen,
    loading,
    detailLoading,
    error,
    syncedAt,
    copyFeedback,
    refresh,
    selectView,
    selectItem,
    closeDetail,
    copyDetail,
  };
}

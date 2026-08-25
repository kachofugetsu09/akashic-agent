import { Check, ChevronDown, ChevronLeft, ChevronRight, Search, Sparkles } from "lucide-react";
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import codexIcon from "./assets/provider-icons/codex.svg";
import deepseekIcon from "./assets/provider-icons/deepseek.svg";
import opencodeIcon from "./assets/provider-icons/opencode.svg";
import openrouterIcon from "./assets/provider-icons/openrouter.svg";
import { compatibleEffort, EFFORT_LABELS, groupModelRuntimes, type ChatModelRuntime } from "./model-capsule-data";

const COMPACT_PANEL_GAP = 8;
const COMPACT_PANEL_MARGIN = 12;
const COMPACT_PANEL_MIN_WIDTH = 360;
const COMPACT_PANEL_MAX_WIDTH = 420;
const COMPACT_PANEL_MAX_HEIGHT = 416;

export type { ChatModelRuntime } from "./model-capsule-data";

interface ModelCapsulePickerProps {
  defaultRuntime: string;
  runtimes: ChatModelRuntime[];
  selectedRuntimeId: string;
  selectedEffort: string;
  disabled: boolean;
  compact?: boolean;
  onChange: (runtimeId: string, effort: string) => void;
}

const PROVIDER_ICONS: Record<string, string> = {
  codex: codexIcon,
  deepseek: deepseekIcon,
  "opencode-go": opencodeIcon,
  opencode: opencodeIcon,
  openrouter: openrouterIcon,
};

function sourceIcon(runtime: ChatModelRuntime): string {
  const provider = runtime.provider.toLowerCase();
  const source = `${runtime.sourceName} ${runtime.sourceId}`.toLowerCase();
  if (provider.includes("codex") || source.includes("codex")) return codexIcon;
  if (provider.includes("opencode") || source.includes("opencode")) return opencodeIcon;
  if (provider.includes("deepseek") || source.includes("deepseek")) return deepseekIcon;
  if (provider.includes("openrouter") || source.includes("openrouter")) return openrouterIcon;
  return PROVIDER_ICONS[provider] || "";
}

function ModelMark({ runtime }: { runtime: ChatModelRuntime }) {
  const icon = sourceIcon(runtime);
  return (
    <span className="model-capsule__mark" aria-hidden="true">
      {icon ? <img src={icon} alt="" /> : <span>{runtime.sourceName.slice(0, 1).toUpperCase()}</span>}
    </span>
  );
}

export function ModelCapsulePicker({
  defaultRuntime,
  runtimes,
  selectedRuntimeId,
  selectedEffort,
  disabled,
  compact = false,
  onChange,
}: ModelCapsulePickerProps) {
  const [open, setOpen] = useState(false);
  const [view, setView] = useState<"models" | "efforts">("models");
  const [query, setQuery] = useState("");
  const [sourceFilter, setSourceFilter] = useState<string>("all");
  const [compactPanelStyle, setCompactPanelStyle] = useState<CSSProperties | undefined>();
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const defaultOptionRef = useRef<HTMLButtonElement>(null);
  const optionRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const effortRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const effortTriggerRef = useRef<HTMLButtonElement>(null);
  const defaultModel = runtimes.find((runtime) => runtime.id === defaultRuntime) || runtimes[0];
  const explicitModel = runtimes.find((runtime) => runtime.id === selectedRuntimeId);
  const visibleModel = explicitModel || defaultModel;
  const supportedEfforts = visibleModel?.supportedReasoningEfforts;
  const groups = useMemo(() => groupModelRuntimes(runtimes), [runtimes]);
  const visibleEffort = compatibleEffort(visibleModel || defaultModel, selectedEffort);
  const filteredGroups = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return groups
      .filter(([source]) => sourceFilter === "all" || source === sourceFilter)
      .map(([source, models]) => [
        source,
        models.filter(({ runtime }) => {
          if (!needle) return true;
          return `${runtime.model} ${runtime.sourceName} ${runtime.provider}`.toLowerCase().includes(needle);
        }),
      ] as const)
      .filter(([, models]) => models.length > 0);
  }, [groups, query, sourceFilter]);

  useLayoutEffect(() => {
    if (!open || !compact) {
      setCompactPanelStyle(undefined);
      return;
    }
    function placeCompactPanel() {
      const trigger = triggerRef.current;
      if (!trigger) return;
      const rect = trigger.getBoundingClientRect();
      const width = Math.min(
        COMPACT_PANEL_MAX_WIDTH,
        Math.max(COMPACT_PANEL_MIN_WIDTH, Math.min(window.innerWidth * 0.72, COMPACT_PANEL_MAX_WIDTH)),
      );
      const spaceAbove = Math.max(0, rect.top - COMPACT_PANEL_GAP - COMPACT_PANEL_MARGIN);
      const spaceBelow = Math.max(0, window.innerHeight - rect.bottom - COMPACT_PANEL_GAP - COMPACT_PANEL_MARGIN);
      const openUp = spaceAbove >= Math.min(COMPACT_PANEL_MAX_HEIGHT, 280) || spaceAbove >= spaceBelow;
      const height = Math.min(COMPACT_PANEL_MAX_HEIGHT, openUp ? spaceAbove : spaceBelow, Math.max(spaceAbove, spaceBelow));
      const left = Math.max(
        COMPACT_PANEL_MARGIN,
        Math.min(rect.right - width, window.innerWidth - width - COMPACT_PANEL_MARGIN),
      );
      setCompactPanelStyle(
        openUp
          ? { left, width, height, bottom: window.innerHeight - rect.top + COMPACT_PANEL_GAP, top: "auto" }
          : { left, width, height, top: rect.bottom + COMPACT_PANEL_GAP, bottom: "auto" },
      );
    }
    placeCompactPanel();
    window.addEventListener("resize", placeCompactPanel);
    window.addEventListener("scroll", placeCompactPanel, true);
    return () => {
      window.removeEventListener("resize", placeCompactPanel);
      window.removeEventListener("scroll", placeCompactPanel, true);
    };
  }, [compact, open, view]);

  useEffect(() => {
    if (!open) return;
    window.setTimeout(() => {
      if (view === "efforts") {
        const effortIndex = Math.max(0, supportedEfforts?.indexOf(visibleEffort) ?? 0);
        effortRefs.current[effortIndex]?.focus({ preventScroll: true });
      } else {
        searchRef.current?.focus({ preventScroll: true });
      }
    }, 0);
    function closeOnPointer(event: PointerEvent) {
      if (!rootRef.current?.contains(event.target as Node)) closePicker(false);
    }
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key !== "Escape") return;
      closePicker(true);
    }
    document.addEventListener("pointerdown", closeOnPointer);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnPointer);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [open, supportedEfforts, view, visibleEffort]);

  if (!visibleModel || !defaultModel) return null;

  function closePicker(restoreFocus: boolean) {
    setOpen(false);
    setView("models");
    setQuery("");
    setSourceFilter("all");
    if (restoreFocus) triggerRef.current?.focus({ preventScroll: true });
  }

  function choose(runtime: ChatModelRuntime) {
    onChange(runtime.id, compatibleEffort(runtime, selectedEffort));
    if (!runtime.supportedReasoningEfforts.length) closePicker(true);
  }

  function chooseEffort(effort: string) {
    onChange(visibleModel.id, effort);
    closePicker(true);
  }

  function showEfforts() {
    setView("efforts");
  }

  function showModels() {
    setView("models");
    window.setTimeout(() => effortTriggerRef.current?.focus({ preventScroll: true }), 0);
  }

  function movePickerFocus(event: React.KeyboardEvent<HTMLDivElement>) {
    if (!(event.key === "ArrowDown" || event.key === "ArrowUp" || event.key === "Home" || event.key === "End")) return;
    const options = [...event.currentTarget.querySelectorAll<HTMLButtonElement>("button:not(:disabled)")];
    const current = options.indexOf(document.activeElement as HTMLButtonElement);
    const next = event.key === "Home" ? 0 : event.key === "End" ? options.length - 1
      : (Math.max(0, current) + (event.key === "ArrowDown" ? 1 : -1) + options.length) % options.length;
    event.preventDefault();
    options[next]?.focus({ preventScroll: true });
  }

  const panel = open ? (
    <div
      id="model-capsule-panel"
      className={`model-capsule__panel ${compact ? "model-capsule__panel--compact" : ""}`}
      role="dialog"
      aria-label={view === "models" ? "选择模型" : "选择思考强度"}
      style={compact ? compactPanelStyle : undefined}
      onKeyDown={movePickerFocus}
    >
      <header className="model-capsule__header">
        {view === "efforts" ? (
          <button type="button" className="model-capsule__back" onClick={showModels}>
            <ChevronLeft size={17} aria-hidden="true" />
            <span><small>返回</small><strong>思考强度</strong></span>
          </button>
        ) : (
          <strong>选择模型</strong>
        )}
        <small>{view === "models" ? runtimes.length : visibleModel.model}</small>
      </header>
      {view === "models" ? <div className="model-capsule__model-view model-capsule__model-view--split">
        <div className="model-capsule__rails" role="tablist" aria-label="按来源筛选">
          <button type="button" role="tab" aria-selected={sourceFilter === "all"} onClick={() => setSourceFilter("all")}>全部</button>
          {groups.map(([source]) => (
            <button key={source} type="button" role="tab" aria-selected={sourceFilter === source} title={source} onClick={() => setSourceFilter(source)}>
              {source}
            </button>
          ))}
        </div>
        <div className="model-capsule__main">
          <label className="model-capsule__search">
            <Search size={14} aria-hidden="true" />
            <input
              ref={searchRef}
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="搜索模型"
              aria-label="搜索模型"
            />
          </label>
          <div className="model-capsule__list" aria-label="所有供应商的模型">
            <section className="model-capsule__source">
              <div className="model-capsule__source-title"><strong>会话策略</strong></div>
              <div className={`model-capsule__option-wrap ${!selectedRuntimeId ? "is-selected" : ""}`}>
                <button ref={defaultOptionRef} type="button" aria-pressed={!selectedRuntimeId} className="model-capsule__option" onClick={() => { onChange("", ""); closePicker(true); }}>
                  <ModelMark runtime={defaultModel} />
                  <span className="model-capsule__copy"><strong>跟随默认模型</strong><small>{defaultModel.model} · {defaultModel.sourceName}</small></span>
                  {!selectedRuntimeId && <Check size={16} aria-hidden="true" />}
                </button>
              </div>
            </section>
            {filteredGroups.map(([source, models]) => (
              <section className="model-capsule__source" aria-label={source} key={source}>
                <div className="model-capsule__source-title"><strong>{source}</strong><span>{models.length}</span></div>
                {models.map(({ runtime, index }) => {
                  const active = runtime.id === selectedRuntimeId;
                  return (
                    <div className={`model-capsule__option-wrap ${active ? "is-selected" : ""}`} key={runtime.id}>
                      <button
                        ref={(node) => { optionRefs.current[index] = node; }}
                        type="button"
                        aria-pressed={active}
                        className="model-capsule__option"
                        onClick={() => choose(runtime)}
                      >
                        <ModelMark runtime={runtime} />
                        <span className="model-capsule__copy"><strong>{runtime.model}</strong><small>{runtime.sourceName} · {runtime.provider}</small></span>
                        {active && <Check size={16} aria-hidden="true" />}
                      </button>
                    </div>
                  );
                })}
              </section>
            ))}
            {!filteredGroups.length ? <p className="model-capsule__empty">无匹配模型</p> : null}
          </div>
          {visibleModel.supportedReasoningEfforts.length > 0 && (
            <button ref={effortTriggerRef} type="button" className="model-capsule__effort-entry" onClick={showEfforts}>
              <Sparkles size={17} aria-hidden="true" />
              <span><small>{explicitModel ? "思考强度" : "固定当前模型并设置强度"}</small><strong>{EFFORT_LABELS[visibleEffort] || visibleEffort}</strong></span>
              <ChevronRight size={17} aria-hidden="true" />
            </button>
          )}
        </div>
      </div> : (
        <div className="model-capsule__effort-list" aria-label={`${visibleModel.model} 支持的思考强度`}>
          <div className="model-capsule__effort-model">
            <ModelMark runtime={visibleModel} />
            <span className="model-capsule__copy"><strong>{visibleModel.model}：{visibleModel.sourceName}</strong><small>{explicitModel ? "仅影响下一轮及之后的此会话" : "选择强度后，会把此模型固定到当前会话"}</small></span>
          </div>
          {visibleModel.supportedReasoningEfforts.map((effort, index) => (
            <button
              ref={(node) => { effortRefs.current[index] = node; }}
              type="button"
              key={effort}
              aria-pressed={visibleEffort === effort}
              className={`model-capsule__effort-option ${visibleEffort === effort ? "is-selected" : ""}`}
              onClick={() => chooseEffort(effort)}
            >
              <span><strong>{EFFORT_LABELS[effort] || effort}</strong><small>{effort}</small></span>
              {visibleEffort === effort && <Check size={16} aria-hidden="true" />}
            </button>
          ))}
        </div>
      )}
    </div>
  ) : null;

  const trigger = (
        <button
      ref={triggerRef}
      type="button"
      className="model-capsule__trigger"
      aria-controls="model-capsule-panel"
      aria-expanded={open}
      aria-label={compact ? `选择模型，当前 ${visibleModel.model}` : undefined}
      disabled={disabled}
      onClick={() => {
        if (open) closePicker(false);
        else setOpen(true);
      }}
    >
      <ModelMark runtime={visibleModel} />
      {compact ? (
        <span className="model-capsule__name">{visibleModel.model}</span>
      ) : (
        <span className="model-capsule__trigger-copy">
          <strong>{visibleModel.model}：{visibleModel.sourceName}</strong>
          <small>{explicitModel ? (visibleEffort ? `思考 ${EFFORT_LABELS[visibleEffort] || visibleEffort}` : "固定到此会话") : "跟随默认模型"}</small>
        </span>
      )}
      <ChevronDown size={compact ? 12 : 18} aria-hidden="true" />
    </button>
  );

  if (compact) {
    return (
      <div ref={rootRef} className={`model-capsule model-capsule--compact ${open ? "is-open" : ""} ${explicitModel ? "is-pinned" : ""}`}>
        {trigger}
        {panel}
      </div>
    );
  }

  return (
    <div ref={rootRef} className={`model-capsule ${open ? "is-open" : ""} ${explicitModel ? "is-pinned" : ""}`}>
      <div className="model-capsule__shell">
        {panel}
        {trigger}
      </div>
    </div>
  );
}

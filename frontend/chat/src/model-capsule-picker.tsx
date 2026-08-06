import { Check, ChevronDown, Sparkles } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import codexIcon from "./assets/provider-icons/codex.svg";
import deepseekIcon from "./assets/provider-icons/deepseek.svg";
import opencodeIcon from "./assets/provider-icons/opencode.svg";
import openrouterIcon from "./assets/provider-icons/openrouter.svg";

export interface ChatModelRuntime {
  id: string;
  provider: string;
  model: string;
  sourceId: string;
  sourceName: string;
  reasoningEffort: string;
  supportedReasoningEfforts: string[];
  roles: string[];
}

interface ModelCapsulePickerProps {
  defaultRuntime: string;
  runtimes: ChatModelRuntime[];
  selectedRuntimeId: string;
  selectedEffort: string;
  disabled: boolean;
  onChange: (runtimeId: string, effort: string) => void;
}

const PROVIDER_ICONS: Record<string, string> = {
  codex: codexIcon,
  deepseek: deepseekIcon,
  "opencode-go": opencodeIcon,
  opencode: opencodeIcon,
  openrouter: openrouterIcon,
};

const EFFORT_LABELS: Record<string, string> = {
  none: "关闭",
  minimal: "极低",
  low: "低",
  medium: "中",
  high: "高",
  xhigh: "极高",
  max: "最大",
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
  onChange,
}: ModelCapsulePickerProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const optionRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const defaultModel = runtimes.find((runtime) => runtime.id === defaultRuntime) || runtimes[0];
  const explicitModel = runtimes.find((runtime) => runtime.id === selectedRuntimeId);
  const visibleModel = explicitModel || defaultModel;
  const groups = useMemo(() => {
    const grouped = new Map<string, ChatModelRuntime[]>();
    for (const runtime of runtimes) {
      const source = runtime.sourceName || runtime.provider;
      grouped.set(source, [...(grouped.get(source) || []), runtime]);
    }
    return [...grouped.entries()];
  }, [runtimes]);

  useEffect(() => {
    if (!open) return;
    const selectedIndex = Math.max(0, runtimes.findIndex((item) => item.id === selectedRuntimeId));
    window.setTimeout(() => optionRefs.current[selectedIndex]?.focus({ preventScroll: true }), 0);
    function closeOnPointer(event: PointerEvent) {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    }
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key !== "Escape") return;
      setOpen(false);
      triggerRef.current?.focus({ preventScroll: true });
    }
    document.addEventListener("pointerdown", closeOnPointer);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnPointer);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [open, runtimes, selectedRuntimeId]);

  if (!visibleModel || !defaultModel) return null;

  function choose(runtimeId: string, effort: string) {
    onChange(runtimeId, effort);
    setOpen(false);
    triggerRef.current?.focus({ preventScroll: true });
  }

  return (
    <div ref={rootRef} className={`model-capsule ${open ? "is-open" : ""}`}>
      <div className="model-capsule__shell">
        <div className="model-capsule__panel" aria-hidden={!open}>
          <header className="model-capsule__header">
            <div><span>所有供应商</span><strong>选择下一轮使用的模型</strong></div>
            <small>{runtimes.length} 个可用模型</small>
          </header>
          <div className="model-capsule__list" role="listbox" aria-label="所有供应商的模型">
            <section className="model-capsule__source">
              <div className="model-capsule__source-title"><strong>会话策略</strong></div>
              <button type="button" role="option" aria-selected={!selectedRuntimeId} className="model-capsule__option" onClick={() => choose("", "")}>
                <ModelMark runtime={defaultModel} />
                <span className="model-capsule__copy"><strong>跟随默认模型</strong><small>{defaultModel.model}：{defaultModel.sourceName}</small></span>
                {!selectedRuntimeId && <Check size={17} aria-hidden="true" />}
              </button>
            </section>
            {groups.map(([source, models]) => (
              <section className="model-capsule__source" aria-label={source} key={source}>
                <div className="model-capsule__source-title"><strong>{source}</strong><span>{models.length}</span></div>
                {models.map((runtime) => {
                  const index = runtimes.findIndex((item) => item.id === runtime.id);
                  const active = runtime.id === selectedRuntimeId;
                  const efforts = runtime.supportedReasoningEfforts;
                  return (
                    <div className={`model-capsule__option-wrap ${active ? "is-selected" : ""}`} key={runtime.id}>
                      <button
                        ref={(node) => { optionRefs.current[index] = node; }}
                        type="button"
                        role="option"
                        aria-selected={active}
                        className="model-capsule__option"
                        onClick={() => choose(runtime.id, selectedEffort || runtime.reasoningEffort)}
                      >
                        <ModelMark runtime={runtime} />
                        <span className="model-capsule__copy"><strong>{runtime.model}：{runtime.sourceName}</strong><small>{runtime.provider}</small></span>
                        {active && <Check size={17} aria-hidden="true" />}
                      </button>
                      {active && efforts.length > 0 && (
                        <fieldset className="model-capsule__efforts">
                          <legend><Sparkles size={13} aria-hidden="true" />推理强度</legend>
                          {efforts.map((effort) => (
                            <button
                              type="button"
                              key={effort}
                              className={(selectedEffort || runtime.reasoningEffort) === effort ? "is-active" : ""}
                              onClick={() => onChange(runtime.id, effort)}
                            >{EFFORT_LABELS[effort] || effort}</button>
                          ))}
                        </fieldset>
                      )}
                    </div>
                  );
                })}
              </section>
            ))}
          </div>
        </div>
        <button
          ref={triggerRef}
          type="button"
          className="model-capsule__trigger"
          aria-haspopup="listbox"
          aria-expanded={open}
          disabled={disabled}
          onClick={() => setOpen((value) => !value)}
        >
          <ModelMark runtime={visibleModel} />
          <span className="model-capsule__trigger-copy">
            <strong>{visibleModel.model}：{visibleModel.sourceName}</strong>
            <small>{explicitModel ? (selectedEffort ? `推理 ${EFFORT_LABELS[selectedEffort] || selectedEffort}` : "固定到此会话") : "跟随默认模型"}</small>
          </span>
          <ChevronDown size={18} aria-hidden="true" />
        </button>
      </div>
    </div>
  );
}

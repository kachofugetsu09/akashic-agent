import { Check, ChevronDown, ChevronLeft, ChevronRight, Sparkles } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import codexIcon from "./assets/provider-icons/codex.svg";
import deepseekIcon from "./assets/provider-icons/deepseek.svg";
import opencodeIcon from "./assets/provider-icons/opencode.svg";
import openrouterIcon from "./assets/provider-icons/openrouter.svg";
import { compatibleEffort, EFFORT_LABELS, groupModelRuntimes, type ChatModelRuntime } from "./model-capsule-data";

export type { ChatModelRuntime } from "./model-capsule-data";

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
  const [view, setView] = useState<"models" | "efforts">("models");
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
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

  useEffect(() => {
    if (!open) return;
    const selectedIndex = Math.max(0, runtimes.findIndex((item) => item.id === visibleModel?.id));
    window.setTimeout(() => {
      if (view === "efforts") {
        const effortIndex = Math.max(0, supportedEfforts?.indexOf(visibleEffort) ?? 0);
        effortRefs.current[effortIndex]?.focus({ preventScroll: true });
      } else {
        (selectedRuntimeId ? optionRefs.current[selectedIndex] : defaultOptionRef.current)?.focus({ preventScroll: true });
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
  }, [open, runtimes, selectedRuntimeId, supportedEfforts, view, visibleEffort, visibleModel?.id]);

  if (!visibleModel || !defaultModel) return null;

  function closePicker(restoreFocus: boolean) {
    setOpen(false);
    setView("models");
    if (restoreFocus) triggerRef.current?.focus({ preventScroll: true });
  }

  function choose(runtime: ChatModelRuntime) {
    onChange(runtime.id, compatibleEffort(runtime, selectedEffort));
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

  return (
    <div ref={rootRef} className={`model-capsule ${open ? "is-open" : ""} ${explicitModel ? "is-pinned" : ""}`}>
      <div className="model-capsule__shell">
        {open ? <div id="model-capsule-panel" className="model-capsule__panel" role="dialog" aria-label={view === "models" ? "选择模型" : "选择思考强度"} onKeyDown={movePickerFocus}>
          <header className="model-capsule__header">
            {view === "efforts" ? (
              <button type="button" className="model-capsule__back" onClick={showModels}>
                <ChevronLeft size={17} aria-hidden="true" />
                <span><small>返回模型</small><strong>思考强度</strong></span>
              </button>
            ) : (
              <div><span>所有供应商</span><strong>选择下一轮使用的模型</strong></div>
            )}
            <small>{view === "models" ? `${runtimes.length} 个可用模型` : visibleModel.model}</small>
          </header>
          {view === "models" ? <div className="model-capsule__model-view">
            <div className="model-capsule__list" aria-label="所有供应商的模型">
            <section className="model-capsule__source">
              <div className="model-capsule__source-title"><strong>会话策略</strong></div>
              <button ref={defaultOptionRef} type="button" aria-pressed={!selectedRuntimeId} className="model-capsule__option" onClick={() => onChange("", "")}>
                <ModelMark runtime={defaultModel} />
                <span className="model-capsule__copy"><strong>跟随默认模型</strong><small>{defaultModel.model}：{defaultModel.sourceName}</small></span>
                {!selectedRuntimeId && <Check size={17} aria-hidden="true" />}
              </button>
            </section>
            {groups.map(([source, models]) => (
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
                        <span className="model-capsule__copy"><strong>{runtime.model}：{runtime.sourceName}</strong><small>{runtime.provider}</small></span>
                        {active && <Check size={17} aria-hidden="true" />}
                      </button>
                    </div>
                  );
                })}
              </section>
            ))}
            </div>
            {visibleModel.supportedReasoningEfforts.length > 0 && (
              <button ref={effortTriggerRef} type="button" className="model-capsule__effort-entry" onClick={showEfforts}>
                <Sparkles size={17} aria-hidden="true" />
                <span><small>{explicitModel ? "思考强度" : "固定当前模型并设置强度"}</small><strong>{EFFORT_LABELS[visibleEffort] || visibleEffort}</strong></span>
                <ChevronRight size={17} aria-hidden="true" />
              </button>
            )}
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
                  {visibleEffort === effort && <Check size={17} aria-hidden="true" />}
                </button>
              ))}
            </div>
          )}
        </div> : null}
        <button
          ref={triggerRef}
          type="button"
          className="model-capsule__trigger"
          aria-controls="model-capsule-panel"
          aria-expanded={open}
          disabled={disabled}
          onClick={() => {
            if (open) closePicker(false);
            else setOpen(true);
          }}
        >
          <ModelMark runtime={visibleModel} />
          <span className="model-capsule__trigger-copy">
            <strong>{visibleModel.model}：{visibleModel.sourceName}</strong>
            <small>{explicitModel ? (visibleEffort ? `思考 ${EFFORT_LABELS[visibleEffort] || visibleEffort}` : "固定到此会话") : "跟随默认模型"}</small>
          </span>
          <ChevronDown size={18} aria-hidden="true" />
        </button>
      </div>
    </div>
  );
}

import { Check, ChevronDown, Eye, Search, Sparkles, Star } from "lucide-react";
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties, type RefObject } from "react";
import codexIcon from "./assets/provider-icons/codex.svg";
import deepseekIcon from "./assets/provider-icons/deepseek.svg";
import opencodeIcon from "./assets/provider-icons/opencode.svg";
import openrouterIcon from "./assets/provider-icons/openrouter.svg";
import { compatibleEffort, EFFORT_LABELS, groupModelRuntimes, type ChatModelRuntime } from "./model-capsule-data";

const PANEL_GAP = 8;
const PANEL_MARGIN = 12;
const PANEL_MAX_HEIGHT = 380;
const FAVORITES_KEY = "akashic.chat.model-favorites";

export type { ChatModelRuntime } from "./model-capsule-data";

interface ModelCapsulePickerProps {
  defaultRuntime: string;
  runtimes: ChatModelRuntime[];
  selectedRuntimeId: string;
  selectedEffort: string;
  disabled: boolean;
  onChange: (runtimeId: string, effort: string) => void;
}

interface ModelEffortActionProps {
  runtime: ChatModelRuntime;
  effort: string;
  explicit: boolean;
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

// 分组头只放真实品牌 SVG；未知来源不留字母占位块，避免与来源名重复表达。
function GroupIcon({ runtime }: { runtime: ChatModelRuntime }) {
  const icon = sourceIcon(runtime);
  if (!icon) return null;
  return <img className="model-picker__group-icon" src={icon} alt="" aria-hidden="true" />;
}

// Alma 式能力徽标：多模态（图像输入）/ 推理 / 上下文窗口。
function CapabilityMarks({ runtime }: { runtime: ChatModelRuntime }) {
  const vision = runtime.inputModalities.some((modality) => modality === "image" || modality === "video");
  const reasoning = runtime.supportedReasoningEfforts.length > 0;
  const ctx = runtime.contextWindow >= 1e6
    ? `${Math.round(runtime.contextWindow / 1e6)}M`
    : runtime.contextWindow >= 1e3 ? `${Math.round(runtime.contextWindow / 1e3)}K` : "";
  if (!vision && !reasoning && !ctx) return null;
  return (
    <>
      {vision ? <Eye size={11} aria-label="多模态输入" /> : null}
      {reasoning ? <Sparkles size={11} aria-label="支持思考强度" /> : null}
      {ctx ? <i title={`${runtime.contextWindow.toLocaleString()} tokens`}>{ctx}</i> : null}
    </>
  );
}

export function resolveVisibleRuntime(
  runtimes: ChatModelRuntime[],
  selectedRuntimeId: string,
  defaultRuntime: string,
): { visibleModel: ChatModelRuntime | undefined; defaultModel: ChatModelRuntime | undefined; explicitModel: ChatModelRuntime | undefined } {
  const actualDefault = runtimes.find((runtime) => runtime.id === defaultRuntime);
  const defaultModel = actualDefault || runtimes[0];
  const explicitModel = runtimes.find((runtime) => runtime.id === selectedRuntimeId);
  return { visibleModel: explicitModel || defaultModel, defaultModel, explicitModel };
}

function readFavorites(): string[] {
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(FAVORITES_KEY) ?? "[]");
    return Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === "string") : [];
  } catch {
    return [];
  }
}

// 紧凑浮层贴触发器定位：上下择宽处展开，横向钳在视口内边距里。
function useFixedPanelStyle(open: boolean, triggerRef: RefObject<HTMLElement | null>, width: number) {
  const [style, setStyle] = useState<CSSProperties | undefined>();
  useLayoutEffect(() => {
    if (!open) {
      setStyle(undefined);
      return;
    }
    function place() {
      const trigger = triggerRef.current;
      if (!trigger) return;
      const rect = trigger.getBoundingClientRect();
      const spaceAbove = Math.max(0, rect.top - PANEL_GAP - PANEL_MARGIN);
      const spaceBelow = Math.max(0, window.innerHeight - rect.bottom - PANEL_GAP - PANEL_MARGIN);
      const openUp = spaceAbove >= Math.min(PANEL_MAX_HEIGHT, 280) || spaceAbove >= spaceBelow;
      const height = Math.min(PANEL_MAX_HEIGHT, Math.max(spaceAbove, spaceBelow));
      const left = Math.max(PANEL_MARGIN, Math.min(rect.left, window.innerWidth - width - PANEL_MARGIN));
      setStyle(
        openUp
          ? { left, width, height, bottom: window.innerHeight - rect.top + PANEL_GAP, top: "auto" }
          : { left, width, height, top: rect.bottom + PANEL_GAP, bottom: "auto" },
      );
    }
    place();
    window.addEventListener("resize", place);
    // 非捕获：只跟随文档级滚动重定位。面板内部列表滚动不触发，
    // 否则每个滚轮帧都 setState 重渲染，长列表里滚动会被打断。
    window.addEventListener("scroll", place);
    return () => {
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place);
    };
  }, [open, triggerRef, width]);
  return style;
}

function useDismissLayer(open: boolean, rootRef: RefObject<HTMLElement | null>, onClose: (restoreFocus: boolean) => void) {
  useEffect(() => {
    if (!open) return;
    function closeOnPointer(event: PointerEvent) {
      if (!rootRef.current?.contains(event.target as Node)) onClose(false);
    }
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key !== "Escape") return;
      onClose(true);
    }
    document.addEventListener("pointerdown", closeOnPointer);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnPointer);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [open, rootRef, onClose]);
}

// 上下/Home/End 只在 [data-nav] 行之间移动焦点；星星等次级按钮不插进方向链。
function moveRowFocus(event: React.KeyboardEvent<HTMLElement>) {
  if (!(event.key === "ArrowDown" || event.key === "ArrowUp" || event.key === "Home" || event.key === "End")) return;
  const options = [...event.currentTarget.querySelectorAll<HTMLElement>("[data-nav]")];
  if (!options.length) return;
  const current = options.indexOf(document.activeElement as HTMLElement);
  const next = event.key === "Home" ? 0 : event.key === "End" ? options.length - 1
    : (Math.max(0, current) + (event.key === "ArrowDown" ? 1 : -1) + options.length) % options.length;
  event.preventDefault();
  options[next]?.focus({ preventScroll: true });
}

interface ModelRowProps {
  runtime: ChatModelRuntime;
  selected: boolean;
  favorite: boolean;
  onChoose: () => void;
  onToggleFavorite: () => void;
}

function ModelRow({ runtime, selected, favorite, onChoose, onToggleFavorite }: ModelRowProps) {
  return (
    <div
      className={`model-picker__option ${selected ? "is-selected" : ""}`}
      role="option"
      aria-selected={selected}
      tabIndex={-1}
      data-nav
      onClick={onChoose}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onChoose();
        }
      }}
    >
      <Check size={14} className="model-picker__check" aria-hidden="true" />
      <span className="model-picker__copy">
        <strong>{runtime.model}</strong>
        <span className="model-picker__meta">
          <CapabilityMarks runtime={runtime} />
          <em>{runtime.provider}</em>
        </span>
      </span>
      <button
        type="button"
        tabIndex={-1}
        className={`model-picker__star ${favorite ? "is-favorite" : ""}`}
        aria-label={favorite ? "取消收藏" : "收藏"}
        aria-pressed={favorite}
        onClick={(event) => {
          event.stopPropagation();
          onToggleFavorite();
        }}
      >
        <Star size={13} aria-hidden="true" />
      </button>
    </div>
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
  const [query, setQuery] = useState("");
  const [favorites, setFavorites] = useState<string[]>(readFavorites);
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const { visibleModel, defaultModel, explicitModel } = resolveVisibleRuntime(runtimes, selectedRuntimeId, defaultRuntime);
  const panelStyle = useFixedPanelStyle(open, triggerRef, 300);

  const items = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const filtered = needle
      ? runtimes.filter((runtime) =>
          `${runtime.model} ${runtime.sourceName} ${runtime.provider}`.toLowerCase().includes(needle))
      : runtimes;
    const favoriteSet = new Set(favorites);
    const selectedSource = explicitModel ? explicitModel.sourceName : undefined;
    // 1. 收藏组置顶；2. 选中模型所在来源排最前；3. 组内选中项排第一。
    const groups = groupModelRuntimes(filtered).sort(
      ([a], [b]) => (a === selectedSource ? -1 : b === selectedSource ? 1 : 0),
    );
    for (const [, models] of groups) {
      models.sort((a, b) => (a.runtime.id === selectedRuntimeId ? -1 : b.runtime.id === selectedRuntimeId ? 1 : 0));
    }
    return { groups, favorites: filtered.filter((runtime) => favoriteSet.has(runtime.id)), favoriteSet };
  }, [runtimes, query, favorites, explicitModel, selectedRuntimeId]);

  useEffect(() => {
    if (!open) return;
    const focusTimer = window.setTimeout(() => {
      searchRef.current?.focus({ preventScroll: true });
    }, 0);
    return () => window.clearTimeout(focusTimer);
  }, [open]);

  const closePicker = (restoreFocus: boolean) => {
    setOpen(false);
    setQuery("");
    if (restoreFocus) triggerRef.current?.focus({ preventScroll: true });
  };
  useDismissLayer(open, rootRef, closePicker);

  if (!visibleModel || !defaultModel) return null;

  function choose(runtime: ChatModelRuntime) {
    onChange(runtime.id, compatibleEffort(runtime, selectedEffort));
    closePicker(true);
  }

  function toggleFavorite(runtimeId: string) {
    setFavorites((current) => {
      const next = current.includes(runtimeId) ? current.filter((id) => id !== runtimeId) : [...current, runtimeId];
      try {
        localStorage.setItem(FAVORITES_KEY, JSON.stringify(next));
      } catch (error) {
        if (!(error instanceof DOMException)) throw error;
      }
      return next;
    });
  }

  const hasSelection = Boolean(selectedRuntimeId ? explicitModel : runtimes.find((runtime) => runtime.id === defaultRuntime));
  const selectionLabel = hasSelection ? visibleModel.model : selectedRuntimeId || defaultRuntime ? "所选模型不可用" : "选择对话模型";

  return (
    <div ref={rootRef} className={`model-picker ${open ? "is-open" : ""} ${explicitModel ? "is-pinned" : ""}`}>
      <button
        ref={triggerRef}
        type="button"
        className="model-picker__trigger"
        aria-expanded={open}
        aria-label={hasSelection ? `选择模型，当前 ${visibleModel.model}` : selectionLabel}
        disabled={disabled}
        onClick={() => {
          if (open) closePicker(false);
          else setOpen(true);
        }}
      >
        <span className="model-picker__name">{selectionLabel}</span>
        <ChevronDown size={11} aria-hidden="true" />
      </button>
      {open ? (
        <div
          className="model-picker__panel"
          role="dialog"
          aria-label="选择模型"
          style={panelStyle}
          onKeyDown={moveRowFocus}
        >
          <div className="model-picker__search">
            <Search size={14} aria-hidden="true" />
            <input
              ref={searchRef}
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="搜索模型"
              aria-label="搜索模型"
            />
          </div>
          <div className="model-picker__list" role="listbox" aria-label="所有供应商的模型">
            <div
              className={`model-picker__option ${!selectedRuntimeId ? "is-selected" : ""}`}
              role="option"
              aria-selected={!selectedRuntimeId}
              tabIndex={-1}
              data-nav
              onClick={() => {
                onChange("", "");
                closePicker(true);
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  onChange("", "");
                  closePicker(true);
                }
              }}
            >
              <Check size={14} className="model-picker__check" aria-hidden="true" />
              <span className="model-picker__copy">
                <strong>跟随默认模型</strong>
                <span className="model-picker__meta"><em>{defaultModel.model} · {defaultModel.sourceName}</em></span>
              </span>
            </div>
            {items.favorites.length ? (
              <div className="model-picker__group">
                <div className="model-picker__group-title"><Star size={12} aria-hidden="true" />收藏</div>
                {items.favorites.map((runtime) => (
                  <ModelRow
                    key={`fav:${runtime.id}`}
                    runtime={runtime}
                    selected={runtime.id === selectedRuntimeId}
                    favorite
                    onChoose={() => choose(runtime)}
                    onToggleFavorite={() => toggleFavorite(runtime.id)}
                  />
                ))}
              </div>
            ) : null}
            {items.groups.map(([source, models]) => (
              <div className="model-picker__group" aria-label={source} key={source}>
                <div className="model-picker__group-title">
                  <GroupIcon runtime={models[0].runtime} />
                  {source}
                  <span>{models.length}</span>
                </div>
                {models.map(({ runtime }) => (
                  <ModelRow
                    key={runtime.id}
                    runtime={runtime}
                    selected={runtime.id === selectedRuntimeId}
                    favorite={items.favoriteSet.has(runtime.id)}
                    onChoose={() => choose(runtime)}
                    onToggleFavorite={() => toggleFavorite(runtime.id)}
                  />
                ))}
              </div>
            ))}
            {!items.groups.length ? <p className="model-picker__empty">无匹配模型</p> : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}

// 思考强度是与模型选择解耦的独立工具行控件：当前模型不支持推理时整体消失。
export function ModelEffortAction({ runtime, effort, explicit, disabled, onChange }: ModelEffortActionProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const panelStyle = useFixedPanelStyle(open, triggerRef, 176);
  const closePicker = (restoreFocus: boolean) => {
    setOpen(false);
    if (restoreFocus) triggerRef.current?.focus({ preventScroll: true });
  };
  useDismissLayer(open, rootRef, closePicker);

  const supported = runtime.supportedReasoningEfforts;
  if (!supported.length) return null;
  const visibleEffort = compatibleEffort(runtime, effort);

  return (
    <div ref={rootRef} className={`model-effort ${open ? "is-open" : ""} ${visibleEffort ? "is-active" : ""}`}>
      <button
        ref={triggerRef}
        type="button"
        className="model-effort__trigger"
        aria-expanded={open}
        aria-label={`思考强度，当前 ${EFFORT_LABELS[visibleEffort] || visibleEffort}`}
        disabled={disabled}
        onClick={() => {
          if (open) closePicker(false);
          else setOpen(true);
        }}
      >
        <Sparkles size={14} aria-hidden="true" />
        {visibleEffort ? <span className="model-effort__tag">{EFFORT_LABELS[visibleEffort] || visibleEffort}</span> : null}
      </button>
      {open ? (
        <div
          className="model-effort__panel"
          role="dialog"
          aria-label={`${runtime.model} 支持的思考强度`}
          style={panelStyle}
          onKeyDown={moveRowFocus}
        >
          <div className="model-effort__list" role="listbox">
            {supported.map((level) => {
              const active = level === visibleEffort;
              return (
                <div
                  key={level}
                  className={`model-picker__option model-effort__option ${active ? "is-selected" : ""}`}
                  role="option"
                  aria-selected={active}
                  tabIndex={-1}
                  data-nav
                  onClick={() => {
                    onChange(runtime.id, level);
                    closePicker(true);
                  }}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      onChange(runtime.id, level);
                      closePicker(true);
                    }
                  }}
                >
                  <Check size={14} className="model-picker__check" aria-hidden="true" />
                  <span className="model-picker__copy">
                    <strong>{EFFORT_LABELS[level] || level}</strong>
                    <span className="model-picker__meta"><em>{level}</em></span>
                  </span>
                </div>
              );
            })}
          </div>
          {!explicit ? <p className="model-effort__hint">选择强度会把 {runtime.model} 固定到当前会话</p> : null}
        </div>
      ) : null}
    </div>
  );
}

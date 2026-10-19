import { Check, ChevronDown, Eye, Search, Sparkles, Star } from "lucide-react";
import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState, type CSSProperties, type RefObject } from "react";
import { compatibleEffort, EFFORT_LABELS, groupModelRuntimes, type ChatModelRuntime } from "./capsule-data";
import "./capsule.css";

const PANEL_GAP = 8;
const PANEL_MARGIN = 12;
const PANEL_MAX_HEIGHT = 380;
const FAVORITES_KEY = "akashic.chat.model-favorites";

export type { ChatModelRuntime } from "./capsule-data";

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

// 分组头只放驱动声明的品牌 SVG；未声明的来源不留字母占位块，避免与来源名重复表达。
function GroupIcon({ runtime }: { runtime: ChatModelRuntime }) {
  const icon = runtime.icon;
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

/** 面板几何 + 缩放锚点：锚点走自定义属性，和 Radix 浮层用的是同一套表达。 */
type PanelStyle = CSSProperties & Record<"--ak-pop-origin", string>;

// 紧凑浮层贴触发器定位：上下择宽处展开，横向钳在视口内边距里。
function useFixedPanelStyle(open: boolean, triggerRef: RefObject<HTMLElement | null>, width: number) {
  const [style, setStyle] = useState<PanelStyle | undefined>();
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
      const maxHeight = Math.min(PANEL_MAX_HEIGHT, Math.max(spaceAbove, spaceBelow));
      const left = Math.max(PANEL_MARGIN, Math.min(rect.left, window.innerWidth - width - PANEL_MARGIN));
      // 面板就近的那条边与触发钮对齐，缩放锚点取触发钮的水平中心：
      // 面板从这里长出来，人的眼睛才接得上「点的是这个，开的是它」。
      const originX = Math.max(8, Math.min(rect.left + rect.width / 2 - left, width - 8));
      setStyle(
        openUp
          ? { left, width, maxHeight, bottom: window.innerHeight - rect.top + PANEL_GAP, top: "auto", "--ak-pop-origin": `${originX}px bottom` }
          : { left, width, maxHeight, top: rect.bottom + PANEL_GAP, bottom: "auto", "--ak-pop-origin": `${originX}px top` },
      );
    }
    place();
    window.addEventListener("resize", place);
    // 聊天与工作台同页：拖动分隔条改变聊天宽度时窗口不 resize，跟随聊天根节点的尺寸重定位。
    const chat = triggerRef.current?.closest(".akashic-chat");
    const observer = chat ? new ResizeObserver(place) : null;
    if (chat) observer?.observe(chat);
    // 非捕获：只跟随文档级滚动重定位。面板内部列表滚动不触发，
    // 否则每个滚轮帧都 setState 重渲染，长列表里滚动会被打断。
    window.addEventListener("scroll", place);
    return () => {
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place);
      observer?.disconnect();
    };
  }, [open, triggerRef, width]);
  return style;
}

// 窄屏下面板是底部抽屉，遮罩在面板之下、正文之上；宽屏不显示。用 click 而不是 pointerdown 关闭，
// 遮罩撑到点击结束，触摸不会穿透到下面的消息或输入框。遮罩在根节点内，外部关闭逻辑不会先触发。
// 按下时阻止焦点转移：输入区失焦会切回阅读态重新布局，遮罩随之移开，click 就落到别处。
function PickerSheetScrim({ onClose }: { onClose: () => void }) {
  return <div className="picker-sheet-scrim" aria-hidden="true"
    onPointerDown={(event) => event.preventDefault()} onClick={onClose} />;
}

function useDismissLayer(open: boolean, rootRef: RefObject<HTMLElement | null>, onClose: (restoreFocus: boolean) => void) {
  useEffect(() => {
    if (!open) return;
    function closeOnPointer(event: PointerEvent) {
      if (!rootRef.current?.contains(event.target as Node)) onClose(false);
    }
    // 同一页面上还有工作台：焦点已经离开菜单时 Esc 只收起菜单，不把焦点抢回触发按钮。
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key !== "Escape" || event.defaultPrevented) return;
      const active = document.activeElement;
      onClose(active === document.body || !!rootRef.current?.contains(active));
    }
    document.addEventListener("pointerdown", closeOnPointer);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnPointer);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [open, rootRef, onClose]);
}

// 上下/Home/End 只在 [data-nav] 行之间移动焦点；星星等次级按钮不插进方向链，
// 焦点落在次级按钮上时按其所在行定位，方向键回到行间导航。
function moveRowFocus(event: React.KeyboardEvent<HTMLElement>) {
  if (!(event.key === "ArrowDown" || event.key === "ArrowUp" || event.key === "Home" || event.key === "End")) return;
  const options = [...event.currentTarget.querySelectorAll<HTMLElement>("[data-nav]")];
  if (!options.length) return;
  const active = document.activeElement as HTMLElement | null;
  const owningRow = active?.closest?.("[data-nav]") as HTMLElement | null;
  const current = owningRow ? options.indexOf(owningRow) : -1;
  const next = event.key === "Home" ? 0 : event.key === "End" ? options.length - 1
    : (Math.max(0, current) + (event.key === "ArrowDown" ? 1 : -1) + options.length) % options.length;
  event.preventDefault();
  options[next]?.focus({ preventScroll: true });
}

interface ModelRowProps {
  runtime: ChatModelRuntime;
  selected: boolean;
  favorite: boolean;
  /** 所在分组已写明来源时不再逐行重复；收藏组混着不同来源，要写。 */
  showProvider: boolean;
  onChoose: () => void;
  onToggleFavorite: () => void;
}

function ModelRow({ runtime, selected, favorite, showProvider, onChoose, onToggleFavorite }: ModelRowProps) {
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
          return;
        }
        // ArrowRight 进入行内次级动作（收藏星），ArrowLeft 从星星回到行。
        if (event.key === "ArrowRight") {
          event.preventDefault();
          event.currentTarget.querySelector<HTMLElement>(".model-picker__star")?.focus();
        }
      }}
    >
      <Check size={14} className="model-picker__check" aria-hidden="true" />
      <span className="model-picker__copy">
        <strong>{runtime.model}</strong>
        <span className="model-picker__meta">
          <CapabilityMarks runtime={runtime} />
          {showProvider ? <em>{runtime.provider}</em> : null}
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
        onKeyDown={(event) => {
          // Enter/Space 的键盘激活交给按钮原生 click；阻止冒泡，
          // 否则外层行的 onKeyDown 会把同一次按键当成“选择该模型”。
          if (event.key === "Enter" || event.key === " ") {
            event.stopPropagation();
            return;
          }
          if (event.key !== "ArrowLeft") return;
          event.preventDefault();
          event.stopPropagation();
          (event.currentTarget.closest("[data-nav]") as HTMLElement | null)?.focus();
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
  // 行内不再重复写来源，来源靠分组名承接：每组是 listbox 里的 group，组名指向可见标题。
  const groupId = useId();
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
    // 触屏上聚焦搜索框会立刻弹出软键盘盖住半个列表；改为聚焦当前选中行，需要搜索时再点。
    const focusTimer = window.setTimeout(() => {
      if (window.matchMedia("(pointer: coarse)").matches) {
        rootRef.current?.querySelector<HTMLElement>(".model-picker__option[aria-selected='true']")?.focus({ preventScroll: true });
      } else searchRef.current?.focus({ preventScroll: true });
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
  // 输入区触发钮只放模型名本身：OpenRouter 式 "deepseek/deepseek-v4-pro" 去掉厂商前缀，
  // 截断时留下能区分型号的后半段；完整 ID 留在 title 与面板里。先取成字符串副本再切，
  // 直接在 visibleModel.model 上调用会让 React Compiler 认为 explicitModel 可能被改而放弃 memo。
  const fullModelName = `${visibleModel.model}`;
  const selectionLabel = hasSelection ? fullModelName.slice(fullModelName.lastIndexOf("/") + 1) || fullModelName
    : selectedRuntimeId || defaultRuntime ? "所选模型不可用" : "选择对话模型";

  return (
    <div ref={rootRef} className={`model-picker ${open ? "is-open" : ""} ${explicitModel ? "is-pinned" : ""}`}>
      <button
        ref={triggerRef}
        type="button"
        className="model-picker__trigger"
        aria-expanded={open}
        title={hasSelection ? fullModelName : undefined}
        aria-label={hasSelection ? `选择模型，当前 ${fullModelName}` : selectionLabel}
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
              <div className="model-picker__group" role="group" aria-labelledby={`${groupId}-fav`}>
                <div className="model-picker__group-title" role="presentation" id={`${groupId}-fav`}><Star size={12} aria-hidden="true" />收藏</div>
                {items.favorites.map((runtime) => (
                  <ModelRow
                    key={`fav:${runtime.id}`}
                    runtime={runtime}
                    selected={runtime.id === selectedRuntimeId}
                    favorite
                    showProvider
                    onChoose={() => choose(runtime)}
                    onToggleFavorite={() => toggleFavorite(runtime.id)}
                  />
                ))}
              </div>
            ) : null}
            {items.groups.map(([source, models], index) => (
              <div className="model-picker__group" role="group" aria-labelledby={`${groupId}-${index}`} key={source}>
                <div className="model-picker__group-title" role="presentation" id={`${groupId}-${index}`}>
                  <GroupIcon runtime={models[0].runtime} />
                  {source}
                </div>
                {models.map(({ runtime }) => (
                  <ModelRow
                    key={runtime.id}
                    runtime={runtime}
                    selected={runtime.id === selectedRuntimeId}
                    showProvider={runtime.provider.toLowerCase() !== source.toLowerCase()}
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
      {open ? <PickerSheetScrim onClose={() => closePicker(false)} /> : null}
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
                    {/* 只写中文档名：英文值是同一信息，第二行只会让每个选项高一倍。 */}
                    <strong>{EFFORT_LABELS[level] || level}</strong>
                  </span>
                </div>
              );
            })}
          </div>
          {!explicit ? <p className="model-effort__hint">选择强度会把 {runtime.model} 固定到当前会话</p> : null}
        </div>
      ) : null}
      {open ? <PickerSheetScrim onClose={() => closePicker(false)} /> : null}
    </div>
  );
}

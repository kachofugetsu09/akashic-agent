import { Check, ChevronDown, Eye, LayoutGrid, Search, Sparkles, Star } from "lucide-react";
import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState, type CSSProperties, type RefObject } from "react";
import { compatibleEffort, EFFORT_LABELS, type ChatModelRuntime } from "./capsule-data";
import "./picker.css";

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
  // 焦点还在搜索框或来源栏时，下键进第一行、上键进最后一行。
  const next = event.key === "Home" ? 0 : event.key === "End" ? options.length - 1
    : current < 0 ? (event.key === "ArrowDown" ? 0 : options.length - 1)
    : (current + (event.key === "ArrowDown" ? 1 : -1) + options.length) % options.length;
  event.preventDefault();
  options[next]?.focus({ preventScroll: true });
}

interface ModelRowProps {
  runtime: ChatModelRuntime;
  selected: boolean;
  favorite: boolean;
  /** 所在分组已写明来源时不再逐行重复；收藏组混着不同来源，要写。 */
  showSource: boolean;
  onChoose: () => void;
  onToggleFavorite: () => void;
}

function ModelRow({ runtime, selected, favorite, showSource, onChoose, onToggleFavorite }: ModelRowProps) {
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
          {showSource ? <em>{runtime.sourceName}</em> : null}
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

type PickerScope = { kind: "all" } | { kind: "favorites" } | { kind: "source"; id: string };

interface PickerSource {
  id: string;
  name: string;
  icon: string;
  models: ChatModelRuntime[];
}

export interface ModelPickerBodyProps {
  runtimes: ChatModelRuntime[];
  /** 空串表示跟随默认。 */
  selectedId: string;
  /** 首行“跟随默认”；为空时不提供跟随（例如它本身就是全局默认）。 */
  follow: { label: string; detail: string } | null;
  onChoose: (runtimeId: string) => void;
}

// 来源按连接 ID 分组：两个连接同名时也不合并；选中项所在来源排最前、组内选中项排第一。
function pickerSources(runtimes: ChatModelRuntime[], selectedId: string): PickerSource[] {
  const sources = new Map<string, PickerSource>();
  for (const runtime of runtimes) {
    const id = runtime.sourceId || runtime.sourceName;
    const source = sources.get(id) ?? { id, name: runtime.sourceName || runtime.provider, icon: runtime.icon, models: [] };
    source.models.push(runtime);
    sources.set(id, source);
  }
  const pinned = (models: ChatModelRuntime[]) => models.some((runtime) => runtime.id === selectedId);
  return [...sources.values()]
    .map((source) => ({ ...source, models: [...source.models].sort((a, b) => Number(b.id === selectedId) - Number(a.id === selectedId)) }))
    .sort((a, b) => Number(pinned(b.models)) - Number(pinned(a.models)));
}

function useFavorites(): [Set<string>, (runtimeId: string) => void] {
  const [favorites, setFavorites] = useState<string[]>(readFavorites);
  const toggle = (runtimeId: string) => {
    setFavorites((current) => {
      const next = current.includes(runtimeId) ? current.filter((id) => id !== runtimeId) : [...current, runtimeId];
      try {
        localStorage.setItem(FAVORITES_KEY, JSON.stringify(next));
      } catch (error) {
        if (!(error instanceof DOMException)) throw error;
      }
      return next;
    });
  };
  return [useMemo(() => new Set(favorites), [favorites]), toggle];
}

// 左侧来源栏：全部、收藏与每个连接；只有一个来源时不出现，免得只剩一个按钮占位。
function PickerRail({ sources, scope, onScope }: { sources: PickerSource[]; scope: PickerScope; onScope: (scope: PickerScope) => void }) {
  const item = (key: string, label: string, active: boolean, next: PickerScope, mark: React.ReactNode) => (
    <button key={key} type="button" className="model-picker__rail-item" title={label} aria-label={label} aria-pressed={active}
      onClick={() => onScope(next)}>{mark}</button>
  );
  return (
    <nav className="model-picker__rail" aria-label="按来源筛选">
      {item("all", "全部来源", scope.kind === "all", { kind: "all" }, <LayoutGrid size={15} aria-hidden="true" />)}
      {item("fav", "收藏", scope.kind === "favorites", { kind: "favorites" }, <Star size={15} aria-hidden="true" />)}
      <span className="model-picker__rail-rule" aria-hidden="true" />
      {sources.map((source) => item(`s:${source.id}`, source.name, scope.kind === "source" && scope.id === source.id,
        { kind: "source", id: source.id },
        source.icon ? <img src={source.icon} alt="" aria-hidden="true" /> : <span aria-hidden="true">{source.name.slice(0, 1).toUpperCase()}</span>))}
    </nav>
  );
}

// 搜索、来源栏与模型列表；聊天胶囊与设置页的场景选择共用，选择后由外层决定如何关闭。
export function ModelPickerBody({ runtimes, selectedId, follow, onChoose }: ModelPickerBodyProps) {
  const [query, setQuery] = useState("");
  const [scope, setScope] = useState<PickerScope>({ kind: "all" });
  const [favoriteSet, toggleFavorite] = useFavorites();
  const groupId = useId();
  const rootRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const sources = useMemo(() => pickerSources(runtimes, selectedId), [runtimes, selectedId]);
  const needle = query.trim().toLowerCase();
  const matches = (runtime: ChatModelRuntime) => !needle || `${runtime.model} ${runtime.sourceName}`.toLowerCase().includes(needle);
  // 1. 来源栏决定范围；2. 搜索在范围内过滤；3. 收藏组只在“全部”与“收藏”下出现。
  const groups = scope.kind === "favorites" ? [] : sources
    .filter((source) => scope.kind === "all" || source.id === scope.id)
    .map((source) => ({ ...source, models: source.models.filter(matches) }))
    .filter((source) => source.models.length);
  const favorites = scope.kind === "source" ? [] : sources.flatMap((source) => source.models).filter((runtime) => favoriteSet.has(runtime.id) && matches(runtime));
  const showFollow = Boolean(follow) && scope.kind === "all" && !needle;
  const first = favorites[0] ?? groups[0]?.models[0];

  useEffect(() => {
    // 触屏上聚焦搜索框会立刻弹出软键盘盖住半个列表；改为聚焦当前选中行，需要搜索时再点。
    const focusTimer = window.setTimeout(() => {
      if (window.matchMedia("(pointer: coarse)").matches) {
        rootRef.current?.querySelector<HTMLElement>(".model-picker__option[aria-selected='true']")?.focus({ preventScroll: true });
      } else searchRef.current?.focus({ preventScroll: true });
    }, 0);
    return () => window.clearTimeout(focusTimer);
  }, []);

  return (
    <div ref={rootRef} className={`model-picker__body ${sources.length > 1 ? "has-rail" : ""}`} onKeyDown={moveRowFocus}>
      <div className="model-picker__search">
        <Search size={14} aria-hidden="true" />
        <input
          ref={searchRef}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => {
            // Enter 直接选第一个匹配；Esc 先清空搜索，再按一次才由外层关闭。
            if (event.key === "Enter" && first && needle) {
              event.preventDefault();
              onChoose(first.id);
            } else if (event.key === "Escape" && query) {
              event.preventDefault();
              event.stopPropagation();
              setQuery("");
            }
          }}
          placeholder={sources.length > 1 ? "筛选模型或来源" : "筛选模型"}
          aria-label="筛选模型"
        />
      </div>
      {sources.length > 1 ? <PickerRail sources={sources} scope={scope} onScope={setScope} /> : null}
      <div className="model-picker__list" role="listbox" aria-label="可选模型">
        {showFollow && follow ? (
          <div
            className={`model-picker__option model-picker__follow ${!selectedId ? "is-selected" : ""}`}
            role="option"
            aria-selected={!selectedId}
            tabIndex={-1}
            data-nav
            onClick={() => onChoose("")}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                onChoose("");
              }
            }}
          >
            <Check size={14} className="model-picker__check" aria-hidden="true" />
            <span className="model-picker__copy">
              <strong>{follow.label}</strong>
              <span className="model-picker__meta"><em>{follow.detail}</em></span>
            </span>
          </div>
        ) : null}
        {favorites.length ? (
          <div className="model-picker__group" role="group" aria-labelledby={`${groupId}-fav`}>
            <div className="model-picker__group-title" role="presentation" id={`${groupId}-fav`}><Star size={12} aria-hidden="true" />收藏</div>
            {favorites.map((runtime) => (
              <ModelRow key={`fav:${runtime.id}`} runtime={runtime} selected={runtime.id === selectedId} favorite showSource
                onChoose={() => onChoose(runtime.id)} onToggleFavorite={() => toggleFavorite(runtime.id)} />
            ))}
          </div>
        ) : null}
        {groups.map((source, index) => (
          <div className="model-picker__group" role="group" aria-labelledby={`${groupId}-${index}`} key={source.id}>
            <div className="model-picker__group-title" role="presentation" id={`${groupId}-${index}`}>
              {/* 分组头只放驱动声明的品牌 SVG；未声明的来源不留字母占位块，避免与来源名重复表达。 */}
              {source.icon ? <img className="model-picker__group-icon" src={source.icon} alt="" aria-hidden="true" /> : null}
              {source.name}
            </div>
            {source.models.map((runtime) => (
              <ModelRow key={runtime.id} runtime={runtime} selected={runtime.id === selectedId} showSource={false}
                favorite={favoriteSet.has(runtime.id)} onChoose={() => onChoose(runtime.id)} onToggleFavorite={() => toggleFavorite(runtime.id)} />
            ))}
          </div>
        ))}
        {!groups.length && !favorites.length ? (
          <p className="model-picker__empty">{scope.kind === "favorites" && !needle ? "还没有收藏；在模型行右侧点星标收藏" : "没有匹配的模型"}</p>
        ) : null}
      </div>
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
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const { visibleModel, defaultModel, explicitModel } = resolveVisibleRuntime(runtimes, selectedRuntimeId, defaultRuntime);
  const multiSource = new Set(runtimes.map((runtime) => runtime.sourceId)).size > 1;
  const panelStyle = useFixedPanelStyle(open, triggerRef, multiSource ? 380 : 320);

  const closePicker = (restoreFocus: boolean) => {
    setOpen(false);
    if (restoreFocus) triggerRef.current?.focus({ preventScroll: true });
  };
  useDismissLayer(open, rootRef, closePicker);

  if (!visibleModel || !defaultModel) return null;

  function choose(runtimeId: string) {
    const runtime = runtimes.find((item) => item.id === runtimeId);
    onChange(runtimeId, runtime ? compatibleEffort(runtime, selectedEffort) : "");
    closePicker(true);
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
        title={hasSelection ? `${fullModelName} · ${visibleModel.sourceName}` : undefined}
        aria-label={hasSelection ? `选择模型，当前 ${fullModelName}` : selectionLabel}
        disabled={disabled}
        onClick={() => {
          if (open) closePicker(false);
          else setOpen(true);
        }}
      >
        {multiSource && visibleModel.icon ? <img className="model-picker__group-icon" src={visibleModel.icon} alt="" aria-hidden="true" /> : null}
        <span className="model-picker__name">{selectionLabel}</span>
        <ChevronDown size={11} aria-hidden="true" />
      </button>
      {open ? (
        <div className="model-picker__panel" role="dialog" aria-label="选择模型" style={panelStyle}>
          <ModelPickerBody runtimes={runtimes} selectedId={selectedRuntimeId} onChoose={choose}
            follow={{ label: "跟随默认模型", detail: `${defaultModel.model} · ${defaultModel.sourceName}` }} />
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

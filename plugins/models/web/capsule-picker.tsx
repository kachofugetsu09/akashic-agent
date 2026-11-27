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

function resolveVisibleRuntime(
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
type PanelStyle = CSSProperties & Record<"--models-pop-origin", string>;

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
          ? { left, width, maxHeight, bottom: window.innerHeight - rect.top + PANEL_GAP, top: "auto", "--models-pop-origin": `${originX}px bottom` }
          : { left, width, maxHeight, top: rect.bottom + PANEL_GAP, bottom: "auto", "--models-pop-origin": `${originX}px top` },
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

// 搜索命中的那一段加底色；不区分大小写，原文大小写保留。
function Highlight({ text, needle }: { text: string; needle: string }) {
  const at = needle ? text.toLowerCase().indexOf(needle) : -1;
  if (at < 0) return <>{text}</>;
  return <>{text.slice(0, at)}<mark>{text.slice(at, at + needle.length)}</mark>{text.slice(at + needle.length)}</>;
}

interface ModelRowProps {
  id: string;
  runtime: ChatModelRuntime;
  needle: string;
  selected: boolean;
  active: boolean;
  favorite: boolean;
  /** 所在分组已写明来源时不再逐行重复；收藏组混着不同来源，要写。 */
  showSource: boolean;
  onActive: () => void;
  onChoose: () => void;
  onToggleFavorite: () => void;
}

// Cursor 式单行选项：焦点留在搜索框，行只用 aria-activedescendant 标出当前项；勾选放在行尾。
function ModelRow({ id, runtime, needle, selected, active, favorite, showSource, onActive, onChoose, onToggleFavorite }: ModelRowProps) {
  return (
    <div id={id} role="option" aria-selected={selected} data-active={active || undefined}
      aria-description={favorite ? "已收藏" : undefined}
      className={`model-picker__row model-picker__option ${selected ? "is-selected" : ""}`}
      onPointerMove={active ? undefined : onActive} onClick={onChoose}>
      <span className="model-picker__copy">
        <strong><Highlight text={runtime.model} needle={needle} /></strong>
        <span className="model-picker__meta">
          <CapabilityMarks runtime={runtime} />
          {showSource ? <em>{runtime.sourceName}</em> : null}
        </span>
      </span>
      {/* 星标只给指针用；option 里不能嵌可聚焦按钮，键盘与读屏走 Ctrl/⌘+D 收藏当前高亮行。 */}
      <span
        className={`model-picker__star ${favorite ? "is-favorite" : ""}`}
        title={`${favorite ? "取消收藏" : "收藏"}（Ctrl+D）`}
        aria-hidden="true"
        onClick={(event) => {
          event.stopPropagation();
          onToggleFavorite();
        }}
      >
        <Star size={13} />
      </span>
      <Check size={14} className="model-picker__check" aria-hidden="true" />
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

interface PickerOption {
  key: string;
  runtimeId: string;
}

interface PickerView {
  needle: string;
  sources: PickerSource[];
  groups: PickerSource[];
  favorites: ChatModelRuntime[];
  showFollow: boolean;
  options: PickerOption[];
}

// 前缀命中排在包含命中之前（Cursor 的搜索排序）；"vendor/model" 的两段都算前缀。
function searchRank(runtime: ChatModelRuntime, needle: string): number {
  const name = runtime.model.toLowerCase();
  return name.startsWith(needle) || name.includes(`/${needle}`) ? 0 : 1;
}

// 1. 来源栏决定范围；2. 搜索在范围内过滤并按命中位置排序；3. 收藏组只在"全部"与"收藏"下出现；
// 4. 按显示顺序摊平成键盘导航用的选项序列。
function pickerView(runtimes: ChatModelRuntime[], selectedId: string, query: string, scope: PickerScope,
  favoriteSet: Set<string>, hasFollow: boolean): PickerView {
  const needle = query.trim().toLowerCase();
  const sources = pickerSources(runtimes, selectedId);
  const matches = (runtime: ChatModelRuntime) => !needle || `${runtime.model} ${runtime.sourceName}`.toLowerCase().includes(needle);
  const ranked = (models: ChatModelRuntime[]) => needle
    ? models.filter(matches).sort((a, b) => searchRank(a, needle) - searchRank(b, needle)) : models;
  const groups = scope.kind === "favorites" ? [] : sources
    .filter((source) => scope.kind === "all" || source.id === scope.id)
    .map((source) => ({ ...source, models: ranked(source.models) }))
    .filter((source) => source.models.length);
  const favorites = scope.kind === "source" ? [] : ranked(sources.flatMap((source) => source.models).filter((runtime) => favoriteSet.has(runtime.id)));
  const showFollow = hasFollow && scope.kind === "all" && !needle;
  const options: PickerOption[] = [
    ...(showFollow ? [{ key: "follow", runtimeId: "" }] : []),
    ...favorites.map((runtime) => ({ key: `fav:${runtime.id}`, runtimeId: runtime.id })),
    ...groups.flatMap((source) => source.models.map((runtime) => ({ key: `m:${runtime.id}`, runtimeId: runtime.id }))),
  ];
  return { needle, sources, groups, favorites, showFollow, options };
}

// 方向键在选项间循环移动；Home/End 只在搜索框为空时接管（有字时留给光标）。
function nextActiveKey(options: PickerOption[], current: string, key: string, empty: boolean): string | null {
  const index = options.findIndex((option) => option.key === current);
  if (key === "ArrowDown") return options[(index + 1) % options.length]?.key ?? null;
  if (key === "ArrowUp") return options[(index - 1 + options.length) % options.length]?.key ?? null;
  if (empty && key === "Home") return options[0]?.key ?? null;
  if (empty && key === "End") return options.at(-1)?.key ?? null;
  return null;
}

// 搜索、来源栏与模型列表；聊天胶囊与设置页的场景选择共用，选择后由外层决定如何关闭。
export function ModelPickerBody({ runtimes, selectedId, follow, onChoose }: ModelPickerBodyProps) {
  const [query, setQuery] = useState("");
  const [scope, setScope] = useState<PickerScope>({ kind: "all" });
  const [favoriteSet, toggleFavorite] = useFavorites();
  const baseId = useId();
  const searchRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const view = useMemo(() => pickerView(runtimes, selectedId, query, scope, favoriteSet, Boolean(follow)),
    [runtimes, selectedId, query, scope, favoriteSet, follow]);
  const [activeKey, setActiveKey] = useState(() => view.options.find((option) => option.runtimeId === selectedId)?.key ?? "");
  // 搜索或换范围后原高亮项可能不在了：回落到第一项。
  const active = view.options.some((option) => option.key === activeKey) ? activeKey : view.options[0]?.key ?? "";
  const optionId = (key: string) => `${baseId}-${key}`;

  useEffect(() => {
    // 触屏上聚焦搜索框会立刻弹出软键盘盖住半个列表：改为聚焦列表本身，外接键盘照样能用方向键选。
    const coarse = window.matchMedia("(pointer: coarse)").matches;
    (coarse ? listRef.current : searchRef.current)?.focus({ preventScroll: true });
  }, []);
  useLayoutEffect(() => {
    if (active) document.getElementById(`${baseId}-${active}`)?.scrollIntoView({ block: "nearest" });
  }, [active, baseId]);

  // 搜索框与列表共用一套按键：方向键移动高亮行，Enter 选中，Ctrl/⌘+D 收藏当前行，Esc 先清空搜索。
  function onNavKey(event: React.KeyboardEvent<HTMLElement>) {
    const fromSearch = event.target === searchRef.current;
    if (!fromSearch && event.target !== listRef.current) return;
    const current = view.options.find((option) => option.key === active);
    const next = view.options.length ? nextActiveKey(view.options, active, event.key, !fromSearch || !query) : null;
    if (next !== null) {
      event.preventDefault();
      setActiveKey(next);
    } else if (event.key === "Enter" && current) {
      event.preventDefault();
      onChoose(current.runtimeId);
    } else if (event.key.toLowerCase() === "d" && (event.ctrlKey || event.metaKey) && current?.runtimeId) {
      event.preventDefault();
      toggleFavorite(current.runtimeId);
    } else if (event.key === "Escape" && query) {
      event.preventDefault();
      event.stopPropagation();
      setQuery("");
    }
  }

  return (
    <div className={`model-picker__body ${view.sources.length > 1 ? "has-rail" : ""}`} onKeyDown={onNavKey}>
      <div className="model-picker__search">
        <Search size={14} aria-hidden="true" />
        <input ref={searchRef} value={query} role="combobox" aria-expanded="true" aria-autocomplete="list"
          aria-controls={`${baseId}-list`} aria-activedescendant={active ? optionId(active) : undefined}
          onChange={(event) => { setQuery(event.target.value); setActiveKey(""); }}
          placeholder={view.sources.length > 1 ? "搜索模型或来源" : "搜索模型"} aria-label="搜索模型" />
      </div>
      {view.sources.length > 1 ? <PickerRail sources={view.sources} scope={scope} onScope={(next) => { setScope(next); setActiveKey(""); }} /> : null}
      <PickerList listRef={listRef} view={view} baseId={baseId} active={active} selectedId={selectedId} follow={follow} scope={scope}
        favoriteSet={favoriteSet} onActive={setActiveKey} onChoose={onChoose} onToggleFavorite={toggleFavorite} />
    </div>
  );
}

interface PickerListProps {
  listRef: RefObject<HTMLDivElement | null>;
  view: PickerView;
  baseId: string;
  active: string;
  selectedId: string;
  follow: ModelPickerBodyProps["follow"];
  scope: PickerScope;
  favoriteSet: Set<string>;
  onActive: (key: string) => void;
  onChoose: (runtimeId: string) => void;
  onToggleFavorite: (runtimeId: string) => void;
}

// 选项列表：跟随默认、收藏组、按来源分组；每行由 key 对应搜索框的 aria-activedescendant。
function PickerList({ listRef, view, baseId, active, selectedId, follow, scope, favoriteSet, onActive, onChoose, onToggleFavorite }: PickerListProps) {
  const row = (key: string, runtime: ChatModelRuntime, showSource: boolean) => (
    <ModelRow key={key} id={`${baseId}-${key}`} runtime={runtime} needle={view.needle} selected={runtime.id === selectedId}
      active={active === key} favorite={favoriteSet.has(runtime.id)} showSource={showSource} onActive={() => onActive(key)}
      onChoose={() => onChoose(runtime.id)} onToggleFavorite={() => onToggleFavorite(runtime.id)} />
  );
  return (
    <div ref={listRef} className="model-picker__list" id={`${baseId}-list`} role="listbox" aria-label="可选模型" tabIndex={0}
      aria-activedescendant={active ? `${baseId}-${active}` : undefined}>
      {view.showFollow && follow ? (
        <div id={`${baseId}-follow`} role="option" aria-selected={!selectedId} data-active={active === "follow" || undefined}
          className={`model-picker__option model-picker__follow ${!selectedId ? "is-selected" : ""}`}
          onPointerMove={() => onActive("follow")} onClick={() => onChoose("")}>
          <span className="model-picker__copy">
            <strong>{follow.label}</strong>
            <span className="model-picker__meta"><em>{follow.detail}</em></span>
          </span>
          <Check size={14} className="model-picker__check" aria-hidden="true" />
        </div>
      ) : null}
      {view.favorites.length ? (
        <div className="model-picker__group" role="group" aria-labelledby={`${baseId}-fav`}>
          <div className="model-picker__group-title" role="presentation" id={`${baseId}-fav`}><Star size={12} aria-hidden="true" />收藏</div>
          {view.favorites.map((runtime) => row(`fav:${runtime.id}`, runtime, true))}
        </div>
      ) : null}
      {view.groups.map((source, index) => (
        <div className="model-picker__group" role="group" aria-labelledby={`${baseId}-g${index}`} key={source.id}>
          <div className="model-picker__group-title" role="presentation" id={`${baseId}-g${index}`}>
            {/* 分组头只放驱动声明的品牌 SVG；未声明的来源不留字母占位块，避免与来源名重复表达。 */}
            {source.icon ? <img className="model-picker__group-icon" src={source.icon} alt="" aria-hidden="true" /> : null}
            {source.name}
          </div>
          {source.models.map((runtime) => row(`m:${runtime.id}`, runtime, false))}
        </div>
      ))}
      {!view.options.length ? (
        <p className="model-picker__empty">{scope.kind === "favorites" && !view.needle ? "还没有收藏；点模型行右侧的星标，或按 Ctrl+D 收藏高亮行" : "没有匹配的模型"}</p>
      ) : null}
    </div>
  );
}

// 思考强度折成三格刻度：触发钮上只占三个点的宽度，档名留给读屏与面板。
const EFFORT_DOTS: Record<string, number> = { none: 0, minimal: 1, low: 1, medium: 2, high: 3, xhigh: 3, max: 3 };

function EffortDots({ effort }: { effort: string }) {
  const filled = EFFORT_DOTS[effort] ?? 2;
  return (
    <span className="model-picker__dots" aria-hidden="true">
      {[0, 1, 2].map((index) => <i key={index} data-on={index < filled || undefined} />)}
    </span>
  );
}

// 面板底部的思考强度：参数跟着当前模型走（Cursor 把推理档放在模型菜单里，不单独占一个工具按钮）。
function EffortRow({ runtime, effort, explicit, onChange }: { runtime: ChatModelRuntime; effort: string; explicit: boolean;
  onChange: (runtimeId: string, effort: string) => void }) {
  const name = useId();
  return (
    <div className="model-picker__effort">
      {/* 原生 radio：一个 Tab 停靠点，组内方向键换档，与 radio group 的键盘合同一致。 */}
      <div className="model-picker__effort-row" role="radiogroup" aria-label={`${runtime.model} 的思考强度`}>
        <span className="model-picker__effort-label" aria-hidden="true">思考</span>
        {runtime.supportedReasoningEfforts.map((level) => (
          <label key={level} className="model-picker__effort-option">
            <input type="radio" name={name} value={level} checked={level === effort} onChange={() => onChange(runtime.id, level)} />
            {EFFORT_LABELS[level] || level}
          </label>
        ))}
      </div>
      {!explicit ? <p className="model-picker__effort-hint">选择强度会把 {runtime.model} 固定到当前会话</p> : null}
    </div>
  );
}

// 输入区触发钮只放模型名本身：OpenRouter 式 "deepseek/deepseek-v4-pro" 去掉厂商前缀，
// 截断时留下能区分型号的后半段；完整 ID 留在 title 与面板里。
function triggerLabel(model: string, hasSelection: boolean, pinned: boolean): string {
  if (!hasSelection) return pinned ? "所选模型不可用" : "选择对话模型";
  return model.slice(model.lastIndexOf("/") + 1) || model;
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
  // 先取成字符串副本再切：直接在 visibleModel.model 上调用会让 React Compiler 认为 explicitModel 可能被改而放弃 memo。
  const fullModelName = `${visibleModel.model}`;
  const selectionLabel = triggerLabel(fullModelName, hasSelection, Boolean(selectedRuntimeId || defaultRuntime));
  const effort = hasSelection ? compatibleEffort(visibleModel, selectedEffort) : "";
  const effortText = effort ? `，思考强度 ${EFFORT_LABELS[effort] || effort}` : "";

  return (
    <div ref={rootRef} className={`model-picker ${open ? "is-open" : ""} ${explicitModel ? "is-pinned" : ""}`}>
      <button
        ref={triggerRef}
        type="button"
        className="model-picker__trigger"
        aria-expanded={open}
        title={hasSelection ? `${fullModelName} · ${visibleModel.sourceName}${effortText}` : undefined}
        aria-label={hasSelection ? `选择模型，当前 ${fullModelName}${effortText}` : selectionLabel}
        disabled={disabled}
        onClick={() => {
          if (open) closePicker(false);
          else setOpen(true);
        }}
      >
        {multiSource && visibleModel.icon ? <img className="model-picker__group-icon" src={visibleModel.icon} alt="" aria-hidden="true" /> : null}
        {/* 名字最少留约 6 个字符宽：再窄就认不出型号，输入卡据此判断单行放不下、改用两行。 */}
        <span className="model-picker__name" style={{ minWidth: `${Math.min(selectionLabel.length, 6)}ch` }}><bdi>{selectionLabel}</bdi></span>
        {effort ? <EffortDots effort={effort} /> : null}
        <ChevronDown size={9} strokeWidth={2.5} aria-hidden="true" />
      </button>
      {/* 先确定位置再挂列表，避免首次测量触发整份列表布局，随后写入定位又重算一次。 */}
      {open && panelStyle ? (
        <div className="model-picker__panel" role="dialog" aria-label="选择模型" style={panelStyle}>
          <ModelPickerBody runtimes={runtimes} selectedId={selectedRuntimeId} onChoose={choose}
            follow={{ label: "跟随默认模型", detail: `${defaultModel.model} · ${defaultModel.sourceName}` }} />
          {effort ? <EffortRow runtime={visibleModel} effort={effort} explicit={Boolean(explicitModel)} onChange={onChange} /> : null}
        </div>
      ) : null}
      {open ? <PickerSheetScrim onClose={() => closePicker(false)} /> : null}
    </div>
  );
}

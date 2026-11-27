import { useEffect, useMemo, useState } from "react";
import { Folder } from "kubai-icons-react/line";
import { DynamicIcon } from "kubai-icons-react/line/dynamic";
import { Search, X } from "lucide-react";
import type { IconCatalogEntry, IconCategoryGroup } from "./group-icon-catalog";

type CatalogModule = typeof import("./group-icon-catalog");

let catalogPromise: Promise<CatalogModule> | null = null;
function loadCatalog(): Promise<CatalogModule> {
  catalogPromise ??= import("./group-icon-catalog");
  return catalogPromise;
}

const SEARCH_RESULT_LIMIT = 96;

/** 侧栏分组图标：默认 folder 同步渲染，其余图标按官方分类 chunk 懒加载，均支持父行悬停弹簧微动效。 */
export function GroupIcon({ name, size = 18, trigger = "parent", className }: {
  name?: string | null;
  size?: number;
  trigger?: "hover" | "parent" | "none";
  className?: string;
}) {
  const cls = className ? `group-mark ${className}` : "group-mark";
  const clean = typeof name === "string" ? name.trim() : "";
  const fallback = <Folder
    size={size}
    variant="outline"
    strokeWidth={1.75}
    trigger={trigger}
    aria-hidden={true}
    className={cls}
  />;
  if (!clean || clean === "folder") return fallback;
  return <DynamicIcon
    name={clean}
    size={size}
    variant="outline"
    strokeWidth={1.75}
    trigger={trigger}
    aria-hidden={true}
    className={cls}
    fallback={fallback}
  />;
}

/** 供插件表单（例如项目右键「更换图标」）复用的宿主图标选择器：完整接入 kubai-icons/line 官方分类与关键词搜索。 */
export function GroupIconPicker({ value, onChange, disabled = false }: {
  value: string | null;
  onChange: (icon: string | null) => void;
  disabled?: boolean;
}) {
  const [catalog, setCatalog] = useState<CatalogModule | null>(null);
  const [categoryId, setCategoryId] = useState<string>("files");
  const [query, setQuery] = useState("");

  useEffect(() => {
    let live = true;
    void loadCatalog().then((mod) => {
      if (!live) return;
      setCatalog(mod);
      if (value) {
        const hit = mod.ALL_LINE_ICONS.find((item) => item.name === value);
        if (hit) setCategoryId(hit.category);
      }
    });
    return () => { live = false; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const selected = value?.trim() || "folder";
  const keyword = query.trim().toLowerCase();

  const visibleIcons: readonly IconCatalogEntry[] = useMemo(() => {
    if (!catalog) return [];
    if (keyword) {
      const terms = keyword.split(/\s+/).filter(Boolean);
      const matched: IconCatalogEntry[] = [];
      for (const item of catalog.ALL_LINE_ICONS) {
        if (terms.every((term) => item.searchText.includes(term))) {
          matched.push(item);
          if (matched.length >= SEARCH_RESULT_LIMIT) break;
        }
      }
      return matched;
    }
    const activeCat: IconCategoryGroup | undefined = catalog.ICON_CATEGORIES.find((cat) => cat.id === categoryId)
      ?? catalog.ICON_CATEGORIES[0];
    return activeCat?.icons ?? [];
  }, [catalog, categoryId, keyword]);

  return <div className="group-icon-picker">
    <div className="group-icon-picker__summary">
      <span className="group-icon-picker__preview" data-kubai-trigger="">
        <GroupIcon name={selected} size={20} trigger="parent" />
      </span>
      <span className="group-icon-picker__meta">
        <strong>{selected === "folder" ? "默认文件夹" : selected}</strong>
        <small>{catalog ? `共 ${catalog.ALL_LINE_ICONS.length} 枚线性图标 · 悬停可预览动效` : "正在加载图标分类…"}</small>
      </span>
      {selected !== "folder" ? <button
        type="button"
        className="group-icon-picker__reset"
        disabled={disabled}
        onClick={() => onChange(null)}
      >
        恢复默认
      </button> : null}
    </div>

    <label className="group-icon-picker__search">
      <Search size={14} aria-hidden="true" />
      <input
        type="search"
        value={query}
        disabled={disabled}
        placeholder="搜索图标名称或关键词（如 terminal、book、phone）"
        onChange={(event) => setQuery(event.target.value)}
      />
      {query ? <button
        type="button"
        className="group-icon-picker__clear"
        aria-label="清空搜索"
        onClick={() => setQuery("")}
      >
        <X size={13} aria-hidden="true" />
      </button> : null}
    </label>

    {!keyword && catalog ? <div className="group-icon-picker__cats" role="tablist" aria-label="图标分类">
      {catalog.ICON_CATEGORIES.map((cat) => <button
        key={cat.id}
        type="button"
        role="tab"
        aria-selected={categoryId === cat.id}
        className="group-icon-picker__cat"
        onClick={() => setCategoryId(cat.id)}
      >
        <span>{cat.label}</span>
        <small>{cat.icons.length}</small>
      </button>)}
    </div> : null}

    <div className="group-icon-picker__grid" role="radiogroup" aria-label="选择分组图标">
      {!catalog ? <p className="group-icon-picker__empty">正在加载图标目录…</p>
        : visibleIcons.length === 0 ? <p className="group-icon-picker__empty">没有匹配的图标</p>
        : visibleIcons.map((item) => {
          const active = selected === item.name;
          return <button
            key={item.name}
            type="button"
            role="radio"
            aria-checked={active}
            aria-label={item.name}
            title={item.name}
            disabled={disabled}
            data-kubai-trigger=""
            className="group-icon-picker__cell"
            onClick={() => onChange(item.name === "folder" ? null : item.name)}
          >
            <GroupIcon name={item.name} size={20} trigger="parent" />
          </button>;
        })}
    </div>
  </div>;
}

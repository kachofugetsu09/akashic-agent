import categories from "kubai-icons/categories.json";
import lineIcons from "kubai-icons/line/icons.json";

export interface IconCatalogEntry {
  readonly name: string;
  readonly category: string;
  readonly tags: readonly string[];
  readonly searchText: string;
}

export interface IconCategoryGroup {
  readonly id: string;
  readonly label: string;
  readonly en: string;
  readonly icons: readonly IconCatalogEntry[];
}

const CATEGORY_ZH: Readonly<Record<string, string>> = {
  files: "文件文档",
  development: "开发编程",
  devices: "设备硬件",
  science: "科学实验",
  education: "学业阅读",
  tools: "工具效率",
  interface: "界面通用",
  communication: "通讯协作",
  home: "家居生活",
  commerce: "财务消费",
  media: "影音创作",
  charts: "数据图表",
  time: "时间日程",
  travel: "旅行地图",
  nature: "自然植物",
  weather: "天气气候",
  food: "饮食咖啡",
  transport: "交通出行",
  people: "人物社交",
  security: "安全隐私",
  health: "健康医疗",
  sports: "运动游戏",
  text: "文字编辑",
  layout: "布局视图",
  arrows: "方向箭头",
  shapes: "基础图形",
};

// 把项目高频分类排在前面，其余按官方顺序排列。
const PRIORITY: readonly string[] = [
  "files", "development", "devices", "education", "science", "tools",
  "commerce", "communication", "media", "home", "interface", "charts",
];

const entries: IconCatalogEntry[] = (lineIcons as { name: string; category: string; tags?: string[] }[]).map((item) => {
  const tags = Array.isArray(item.tags) ? item.tags : [];
  const zh = CATEGORY_ZH[item.category] ?? item.category;
  return {
    name: item.name,
    category: item.category,
    tags,
    searchText: `${item.name} ${tags.join(" ")} ${item.category} ${zh}`.toLowerCase(),
  };
});

const grouped = new Map<string, IconCatalogEntry[]>();
for (const entry of entries) {
  const bucket = grouped.get(entry.category);
  if (bucket) bucket.push(entry);
  else grouped.set(entry.category, [entry]);
}

const categoryMap = new Map((categories as { id: string; en: string }[]).map((item) => [item.id, item]));
const orderedIds = [
  ...PRIORITY.filter((id) => grouped.has(id)),
  ...[...grouped.keys()].filter((id) => !PRIORITY.includes(id)),
];

export const ICON_CATEGORIES: readonly IconCategoryGroup[] = orderedIds.map((id) => {
  const meta = categoryMap.get(id);
  const en = meta?.en ?? id;
  return {
    id,
    label: CATEGORY_ZH[id] ?? en,
    en,
    icons: grouped.get(id) ?? [],
  };
});

export const ALL_LINE_ICONS: readonly IconCatalogEntry[] = entries;
export const ALL_LINE_ICON_NAMES: ReadonlySet<string> = new Set(entries.map((entry) => entry.name));

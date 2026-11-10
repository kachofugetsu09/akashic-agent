// 宿主图表工具包 window.ak：生成文档只给数据、选型与注释；坐标、刻度、配色、标签避让、窄屏重排和读数由这里负责。
// 构建时打包成 IIFE 内联进 frame.html，预检与客户端执行同一份代码；颜色全部是 CSS 变量，主题切换无需重画。

type ColorKey = number | "gray" | "hot";
type Format = "raw" | "%" | "wan" | "si" | ((value: number) => string);
type Target = string | HTMLElement;
type Value = number | null | undefined;

export interface SeriesInput { name?: string; values?: readonly Value[]; color?: ColorKey }
// 注释主句写 text；label 是 text 的别名，便于里程碑沿用常见写法。
export interface Note { at: string | number; text?: string; label?: string; sub?: string; series?: string }
export interface RefLine { value: number; label?: string }
interface Common { label?: string; fmt?: Format; unit?: string; height?: number }
export interface BarsOptions extends Common {
  x: readonly (string | number)[]; series: readonly SeriesInput[]; stack?: boolean; horizontal?: boolean;
  annotate?: readonly Note[]; refLine?: RefLine; legend?: boolean;
  sub?: readonly string[]; range?: readonly ({ lo: number; hi: number } | null)[]; rangeLabels?: readonly [string, string];
  colors?: readonly ColorKey[];
}
export interface LineOptions extends Common {
  x: readonly (string | number)[]; series: readonly SeriesInput[]; area?: boolean; yMin?: number;
  milestones?: readonly Note[]; annotate?: readonly Note[]; spans?: readonly { from: string | number; to: string | number; label: string }[];
  projectFrom?: string | number; refLine?: RefLine;
}
export interface BarlistOptions extends Common {
  items: readonly { label: string; value: number; note?: string }[]; sort?: boolean; limit?: number; max?: number; color?: ColorKey;
}
export interface ScatterOptions extends Common {
  points: readonly { x: number; y: number; label?: string; size?: number; group?: string }[]; xTitle?: string; yTitle?: string;
  xFmt?: Format; yFmt?: Format; highlight?: readonly string[]; refX?: RefLine; refY?: RefLine; trend?: boolean;
}
export interface DumbbellOptions extends Common {
  x: readonly (string | number)[]; from: SeriesInput; to: SeriesInput; sort?: boolean;
}
export interface HeatmapOptions extends Common {
  rows: readonly (string | number)[]; cols: readonly (string | number)[]; values: readonly (readonly number[])[]; color?: ColorKey; showValues?: boolean;
}
export interface ShareOptions extends Common { x: readonly (string | number)[]; series: readonly SeriesInput[]; annotate?: readonly Note[] }
export interface TreemapOptions extends Common { nodes: readonly { name: string; value: number; group?: string }[] }
export interface Handle { redraw(): void }

interface Series { name: string; color: string; values: (number | null)[] }
interface Hit { x0: number; x1: number; y0: number; y1: number; cx?: number; html: string }
interface Box { x: number; y: number; w: number; h: number }
interface Drawing { html: string; hits: Hit[] }

const FONT = 'system-ui, -apple-system, "Segoe UI", "PingFang SC", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif';
const ISO = /^(\d{4})-(\d{2})-(\d{2})/;
const measureContext = document.createElement("canvas").getContext("2d");
let sequence = 0;

const round = (value: number): number => Math.round(value * 10) / 10;
const clamp = (value: number, low: number, high: number): number => Math.max(low, Math.min(high, value));
const esc = (value: unknown): string => String(value ?? "").replace(/[&<>"']/g, char => `&#${char.charCodeAt(0)};`);
const textWidth = (text: string, size = 12, weight = 400): number => {
  if (!measureContext) return String(text).length * size * 0.6;
  measureContext.font = `${weight} ${size}px ${FONT}`;
  return measureContext.measureText(String(text)).width;
};
const shortLabel = (value: string): string => {
  const match = ISO.exec(value);
  return match ? `${Number(match[2])}/${Number(match[3])}` : value;
};

// 系列色按固定顺序取宿主变量；文字永远用墨色。
function color(key: ColorKey | undefined, index: number): string {
  if (key === "gray") return "var(--k-gray)";
  if (key === "hot") return "var(--k-hot)";
  const slot = typeof key === "number" && Number.isInteger(key) && key > 0 ? key - 1 : index;
  return `var(--k${(slot % 6) + 1})`;
}

// 数值格式：raw 原样千分位，% 百分比，wan 中文万/亿，si 用 K/M/B；也接受函数。先按显示精度舍入再决定单位，避免 "10,000"。
function formatter(format: Format | undefined, unit = ""): (value: number) => string {
  if (typeof format === "function") return format;
  const units: readonly [number, string, number][] = format === "wan" ? [[1e8, "亿", 1], [1e4, "万", 1]]
    : format === "si" ? [[1e9, "B", 1], [1e6, "M", 1], [1e3, "K", 10]] : [];
  return value => {
    const size = Math.abs(value);
    if (format === "%") return `${size >= 10 || Number.isInteger(value) ? Math.round(value) : round(value)}%`;
    const scaled = units.find(([base, , least], at) => Math.abs(round(value / base)) >= least
      && (at < units.length - 1 || Math.abs(Number(value.toFixed(1))) >= base * least));
    if (scaled) return `${round(value / scaled[0])}${scaled[1]}${unit}`;
    const plain = Number.isInteger(value) ? value : Number(value.toPrecision(size < 1 ? 2 : Math.max(2, Math.floor(Math.log10(size)) + 2)));
    return (plain === 0 ? 0 : plain).toLocaleString("en-US", { maximumFractionDigits: 6 }) + unit;
  };
}

// 刻度取 1/2/2.5/5 的整倍数，范围包住全部数据；超出浮点可绘制精度的数域直接报错，不进入无界循环。
function niceScale(low: number, high: number, count: number): { low: number; high: number; ticks: number[] } {
  if (!(high > low)) high = low + (Math.abs(low) * 0.1 || 1);
  const raw = (high - low) / count, power = 10 ** Math.floor(Math.log10(raw)), ratio = raw / power;
  const step = (ratio <= 1 ? 1 : ratio <= 2 ? 2 : ratio <= 2.5 ? 2.5 : ratio <= 5 ? 5 : 10) * power;
  const start = Math.floor(low / step + 1e-9) * step, end = Math.ceil(high / step - 1e-9) * step, steps = Math.round((end - start) / step);
  if (![step, start, end, steps].every(Number.isFinite) || !(step > 0) || steps < 1 || steps > 20 || start + step === start)
    throw new Error("ak: 数值范围超出可绘制精度，请先换算单位（如万、亿）再画");
  const ticks = Array.from({ length: steps + 1 }, (_, index) => Number((start + index * step).toPrecision(12)));
  return { low: start, high: end, ticks };
}

// 横轴标签：日期多时按月，否则按宽度抽稀，保证不重叠。
function xTicks(labels: readonly string[], width: number): { index: number; text: string }[] {
  if (labels.length > 35 && labels.every(label => ISO.test(label))) {
    // 月份边界再按像素抽稀；跨年时每段新年份的第一个保留刻度写年份。
    const years = new Set(labels.map(label => label.slice(0, 4))).size > 1, months: { index: number; month: string; year: string }[] = [];
    let previous = "";
    labels.forEach((label, index) => {
      const match = ISO.exec(label)!;
      if (match[1] + match[2] !== previous && (index > 0 || Number(match[3]) <= 3)) months.push({ index, month: `${Number(match[2])}月`, year: match[1] });
      previous = match[1] + match[2];
    });
    const gap = Math.max(...months.map(tick => textWidth(years ? tick.year : tick.month))) + 12, pixel = width / labels.length;
    const kept: { index: number; text: string }[] = [];
    let year = "", last = -Infinity;
    for (const tick of months) {
      if ((tick.index - last) * pixel < gap) continue;
      kept.push({ index: tick.index, text: years && tick.year !== year ? tick.year : tick.month });
      year = tick.year;
      last = tick.index;
    }
    return kept;
  }
  const short = labels.map(shortLabel), widest = Math.max(0, ...short.map(text => textWidth(text))) + 12;
  const step = Math.max(1, Math.ceil(widest / (width / Math.max(1, labels.length))));
  return short.map((text, index) => ({ index, text })).filter(tick => tick.index % step === 0);
}

// 输入在这里校验：错误抛给预检，模型据此修正，而不是静默画出错图。
function readSeries(chart: string, inputs: readonly SeriesInput[] | undefined, length: number, nullable: boolean): Series[] {
  if (!Array.isArray(inputs) || !inputs.length) throw new Error(`ak.${chart}: series 至少需要一个系列`);
  return inputs.map((input: SeriesInput, index: number): Series => {
    const values: readonly Value[] = input?.values ?? [];
    if (values.length !== length) throw new Error(`ak.${chart}: series[${index}].values 有 ${values.length} 个值，x 有 ${length} 个`);
    return {
      name: String(input.name ?? `系列 ${index + 1}`), color: color(input.color, index),
      values: values.map((value, at) => {
        if (value == null && nullable) return null;
        if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`ak.${chart}: series[${index}].values[${at}] 不是有限数字`);
        return value;
      }),
    };
  });
}

function indexOf(labels: readonly string[], at: string | number): number {
  const found = labels.indexOf(String(at));
  if (found >= 0) return found;
  return typeof at === "number" && Number.isInteger(at) && at >= 0 && at < labels.length ? at : -1;
}

// 标签避让：按优先级放置，取第一个落在图内且不与已放标签重叠的候选位置。
class Placement {
  private readonly boxes: Box[] = [];
  constructor(readonly width: number, private readonly height: number) {}
  block(box: Box): void { this.boxes.push(box); }
  // 候选先水平收进图内；都冲突时 force 取第一个候选，宁可贴近也不静默丢掉注释。
  place(candidates: readonly Box[], force = false): Box | null {
    const inside = candidates.map(box => ({ ...box, x: clamp(box.x, 0, Math.max(0, this.width - box.w)) }))
      .filter(box => box.y >= 0 && box.y + box.h <= this.height);
    const free = inside.find(box => !this.boxes.some(other => box.x < other.x + other.w + 2 && other.x < box.x + box.w + 2
      && box.y < other.y + other.h + 1 && other.y < box.y + box.h + 1));
    const first = candidates[0], fallback = first && { ...first, x: clamp(first.x, 0, Math.max(0, this.width - first.w)), y: clamp(first.y, 0, Math.max(0, this.height - first.h)) };
    const chosen = free ?? (force ? fallback ?? null : null);
    if (chosen) this.boxes.push(chosen);
    return chosen;
  }
}

// 按可用宽度逐字折行：中文按字，含空格的英文优先在空格处断开。
function wrap(text: string, limit: number, weight = 400): string[] {
  const lines: string[] = [];
  let line = "";
  for (const char of Array.from(text)) {
    if (line && textWidth(line + char, 12, weight) > limit) {
      const space = line.lastIndexOf(" ");
      lines.push(space > 0 && char !== " " ? line.slice(0, space) : line);
      line = space > 0 && char !== " " ? line.slice(space + 1) + char : char.trimStart();
    } else line += char;
  }
  return line ? [...lines, line] : lines;
}

// 注释文字：粗体主句 + 可选次级说明，超出图宽时折行；依次尝试上方（居中、左右错开）、右、左、下方和更远的上方。
function annotation(placement: Placement, note: Note, x: number, y: number, gap = 8): string {
  const text = String(note.text ?? note.label ?? "");
  if (!text) return "";
  const limit = Math.max(48, Math.min(240, placement.width - 4)), head = wrap(text, limit, 600), tail = note.sub ? wrap(note.sub, limit) : [];
  const w = Math.max(...head.map(line => textWidth(line, 12, 600)), ...tail.map(line => textWidth(line))), h = 16 * (head.length + tail.length);
  const box = (left: number, top: number): Box => ({ x: left, y: top, w, h });
  const chosen = placement.place([
    box(x - w / 2, y - gap - h), box(x - w + 8, y - gap - h), box(x - 8, y - gap - h),
    box(x + gap, y - h / 2), box(x - gap - w, y - h / 2), box(x - w / 2, y + gap),
    box(x - w / 2, y - gap - h - 20), box(x - w - gap, y - gap - h), box(x + gap, y - gap - h)], true);
  if (!chosen) return "";
  const tx = round(chosen.x + w / 2);
  return [...head.map(line => [line, ' class="ak-b"']), ...tail.map(line => [line, ""])].map(([line, kind], index) =>
    `<text${kind} x="${tx}" y="${round(chosen.y + 12 + index * 16)}" text-anchor="middle">${esc(line)}</text>`).join("");
}

// 线尾标签：按数据位置排序后上下推开，整体留在绘图区内。
function spread(items: { y: number; html: (y: number) => string }[], low: number, high: number, gap = 15): string {
  const sorted = [...items].sort((a, b) => a.y - b.y);
  sorted.forEach((item, index) => { if (index) item.y = Math.max(item.y, sorted[index - 1].y + gap); });
  for (let index = sorted.length - 1, limit = high; index >= 0; index -= 1, limit -= gap) sorted[index].y = Math.min(sorted[index].y, limit);
  return sorted.map(item => item.html(Math.max(low, item.y))).join("");
}

const svgOpen = (width: number, height: number, label?: string): string =>
  `<svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img" aria-label="${esc(label || "图表")}">`;
const legend = (items: readonly [string, string | null, string][]): string =>
  `<div class="ak-legend">${items.map(([fill, value, name]) =>
    `<span><i style="background:${fill}"></i>${value == null ? "" : `<b>${esc(value)}</b> `}${esc(name)}</span>`).join("")}</div>`;
const tip = (title: string, rows: readonly [string, string, string][], total?: string): string =>
  `<b>${esc(title)}</b>${total == null ? "" : ` · 合计 <b>${esc(total)}</b>`}`
  + rows.map(([name, value, fill]) => `<br><i style="background:${fill}"></i>${esc(name)} <b>${esc(value)}</b>`).join("");
const horizontalGrid = (ticks: readonly number[], y: (value: number) => number, left: number, right: number, format: (value: number) => string): string =>
  ticks.map(value => `<line class="ak-grid" x1="${left}" x2="${right}" y1="${round(y(value))}" y2="${round(y(value))}"/>`
    + `<text x="${left - 8}" y="${round(y(value)) + 4}" text-anchor="end">${esc(format(value))}</text>`).join("");
const xLabel = (x: number, y: number, text: string, width: number): string => {
  const half = textWidth(text) / 2;
  return `<text x="${round(clamp(x, half, width - half))}" y="${y}" text-anchor="middle">${esc(text)}</text>`;
};

const observers = new WeakMap<HTMLElement, ResizeObserver>();

// 挂载：按容器实际宽度绘制，宽度变化时重画；同一容器再次挂载时先断开旧的监听。
function mount(target: Target, draw: (width: number, id: string) => Drawing): Handle {
  const element = typeof target === "string" ? document.querySelector<HTMLElement>(target) : target;
  if (!(element instanceof HTMLElement)) throw new Error(`ak: 找不到容器 ${String(target)}`);
  element.classList.add("ak-chart");
  let width = 0;
  const render = (): void => {
    const next = Math.max(160, Math.floor(element.clientWidth || 640));
    if (next === width) return;
    width = next;
    const drawing = draw(next, `ak${(sequence += 1)}`);
    element.innerHTML = drawing.html;
    const svg = element.querySelector("svg");
    if (svg && drawing.hits.length) interact(element, svg, drawing.hits);
  };
  render();
  observers.get(element)?.disconnect();
  const observer = new ResizeObserver(render);
  observer.observe(element);
  observers.set(element, observer);
  return { redraw: () => { width = 0; render(); } };
}

// 读数：指针悬停显示浮层，方向键在数据点间移动，同步给屏幕阅读器。
function interact(element: HTMLElement, svg: SVGSVGElement, hits: readonly Hit[]): void {
  const panel = document.createElement("div"), live = document.createElement("span"), cross = svg.querySelector<SVGLineElement>(".ak-cross");
  panel.className = "ak-tip";
  panel.hidden = true;
  live.className = "ak-sr";
  live.setAttribute("role", "status");
  element.append(panel, live);
  let current = -1;
  const scale = (): number => svg.getBoundingClientRect().width / (svg.viewBox.baseVal.width || 1);
  const hide = (): void => { panel.hidden = true; current = -1; cross?.style.setProperty("visibility", "hidden"); };
  const show = (index: number, px?: number, py?: number): void => {
    const hit = hits[index];
    current = index;
    panel.innerHTML = hit.html;
    panel.hidden = false;
    live.textContent = panel.textContent;
    const box = element.getBoundingClientRect(), frame = svg.getBoundingClientRect(), ratio = scale();
    const x = frame.left - box.left + (px ?? (hit.x0 + hit.x1) / 2) * ratio, y = frame.top - box.top + (py ?? hit.y0) * ratio;
    const left = x + 14 + panel.offsetWidth > element.clientWidth ? Math.max(0, x - panel.offsetWidth - 14) : x + 14;
    panel.style.left = `${left}px`;
    panel.style.top = `${Math.max(0, y - panel.offsetHeight - 10)}px`;
    if (cross && hit.cx != null) {
      cross.setAttribute("x1", String(hit.cx));
      cross.setAttribute("x2", String(hit.cx));
      cross.style.visibility = "visible";
    }
  };
  svg.tabIndex = 0;
  svg.addEventListener("pointermove", event => {
    const frame = svg.getBoundingClientRect(), ratio = scale(), x = (event.clientX - frame.left) / ratio, y = (event.clientY - frame.top) / ratio;
    const index = hits.findIndex(hit => x >= hit.x0 && x <= hit.x1 && y >= hit.y0 && y <= hit.y1);
    if (index < 0) hide(); else show(index, x, y);
  });
  svg.addEventListener("pointerleave", hide);
  svg.addEventListener("blur", hide);
  svg.addEventListener("keydown", event => {
    if (event.key === "Escape") return hide();
    const step = ({ ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 } as Record<string, number>)[event.key];
    if (!step) return;
    event.preventDefault();
    show(current < 0 ? (step > 0 ? 0 : hits.length - 1) : (current + step + hits.length) % hits.length);
  });
}

// 柱状图：竖向堆叠或分组；horizontal 时为横条，可带估计区间与参考线。
function bars(target: Target, options: BarsOptions): Handle {
  const labels = (options?.x ?? []).map(String);
  if (!labels.length) throw new Error("ak.bars: x 不能为空");
  const series = readSeries("bars", options.series, labels.length, false), format = formatter(options.fmt, options.unit);
  if (options.horizontal) checkHorizontal(options, labels.length, series);
  return mount(target, (width, id) => options.horizontal
    ? horizontalBars(width, id, options, labels, series, format)
    : verticalBars(width, options, labels, series, format));
}

// 横条只表达非负量；区间必须是有限数字且 lo ≤ hi，不在绘制时悄悄改写。
function checkHorizontal(options: BarsOptions, count: number, series: readonly Series[]): void {
  if (series.some(item => item.values.some(value => (value ?? 0) < 0))) throw new Error("ak.bars: 横向条不支持负数，请改用竖向柱");
  (options.range ?? []).slice(0, count).forEach((range, index) => {
    if (range == null) return;
    if (![range.lo, range.hi].every(value => typeof value === "number" && Number.isFinite(value))) throw new Error(`ak.bars: range[${index}] 的 lo、hi 必须是有限数字`);
    if (range.lo < 0 || range.lo > range.hi) throw new Error(`ak.bars: range[${index}] 需要 0 ≤ lo ≤ hi`);
  });
}

function verticalBars(width: number, options: BarsOptions, labels: string[], series: Series[], format: (value: number) => string): Drawing {
  const stack = options.stack !== false, count = labels.length, values = (index: number) => series.map(item => item.values[index] ?? 0);
  const positive = labels.map((_, index) => values(index).reduce((sum, value) => sum + Math.max(0, value), 0));
  const negative = labels.map((_, index) => values(index).reduce((sum, value) => sum + Math.min(0, value), 0));
  const tops = labels.map((_, index) => stack ? positive[index] : Math.max(0, ...values(index)));
  const high = Math.max(0, ...tops, options.refLine?.value ?? 0), low = Math.min(0, ...(stack ? negative : series.flatMap(item => item.values as number[])));
  const height = options.height ?? (width < 480 ? 220 : 260), top = options.annotate?.length ? (options.annotate.some(note => note.sub) ? 44 : 28) : 12, bottom = 24;
  const scale = niceScale(low, high, height < 240 ? 3 : 4), left = Math.max(...scale.ticks.map(value => textWidth(format(value)))) + 10;
  const plot = width - left - 4, band = plot / count, bar = Math.max(1, Math.min(band * (count > 40 ? 0.8 : 0.66), 44));
  const y = (value: number): number => top + (height - top - bottom) - (value - scale.low) / (scale.high - scale.low) * (height - top - bottom);
  const placement = new Placement(width, height);
  placement.block({ x: 0, y: 0, w: left - 4, h: height });
  placement.block({ x: 0, y: height - bottom + 4, w: width, h: bottom });
  let marks = "";
  const hits = labels.map((label, index) => {
    const cx = left + band * (index + 0.5);
    let up = 0, down = 0;
    series.forEach((item, at) => {
      const value = item.values[index] ?? 0, slot = stack ? bar : bar / series.length;
      const x = stack ? cx - bar / 2 : cx - bar / 2 + at * slot, base = stack ? (value > 0 ? up : down) : 0, end = base + value;
      if (stack) { if (value > 0) up = end; else down = end; }
      const y0 = y(Math.max(base, end)), y1 = y(Math.min(base, end));
      marks += `<rect x="${round(x)}" y="${round(y0)}" width="${round(Math.max(1, slot - (stack ? 0 : 1)))}" height="${round(Math.max(0.5, y1 - y0))}" style="fill:${item.color}"${stack && series.length > 1 && bar > 6 ? ' class="ak-sep"' : ""}/>`;
    });
    placement.block({ x: cx - bar / 2, y: y(tops[index]), w: bar, h: y(0) - y(tops[index]) });
    return { x0: cx - band / 2, x1: cx + band / 2, y0: top, y1: height - bottom, html: tip(shortLabel(label),
      [...series].reverse().map(item => [item.name, format(item.values[index] ?? 0), item.color]), stack && series.length > 1 ? format(positive[index] + negative[index]) : undefined) };
  });
  let grid = horizontalGrid(scale.ticks, y, left, width - 4, format) + `<line class="ak-base" x1="${left}" x2="${width - 4}" y1="${round(y(0))}" y2="${round(y(0))}"/>`;
  xTicks(labels, plot).forEach(tick => { grid += xLabel(left + band * (tick.index + 0.5), height - 6, tick.text, width); });
  marks += referenceLine(placement, options.refLine, format, { x1: left, x2: width - 4, y: y(options.refLine?.value ?? 0) });
  for (const note of options.annotate ?? []) {
    const index = indexOf(labels, note.at);
    if (index >= 0) marks += annotation(placement, note, left + band * (index + 0.5), y(tops[index]), 6);
  }
  const head = series.length > 1 && options.legend !== false
    ? legend(series.map(item => [item.color, format(item.values.reduce<number>((sum, value) => sum + (value ?? 0), 0)), item.name])) : "";
  return { html: head + svgOpen(width, height, options.label) + grid + marks + "</svg>", hits };
}

// 参考线标签写"数值 · 名称"；名称里已含数值时不重复，优先放在右端上下，其次左端。
function referenceText(line: RefLine, format: (value: number) => string): string {
  const value = format(line.value);
  return !line.label ? value : line.label.includes(value) ? line.label : `${value} · ${line.label}`;
}

function referenceLine(placement: Placement, line: RefLine | undefined, format: (value: number) => string, at: { x1: number; x2: number; y: number }): string {
  if (!line) return "";
  const text = referenceText(line, format), width = textWidth(text, 12, 600);
  const box = placement.place([{ x: at.x2 - width, y: at.y - 18, w: width, h: 14 }, { x: at.x2 - width, y: at.y + 4, w: width, h: 14 },
    { x: at.x1 + 4, y: at.y - 18, w: width, h: 14 }, { x: at.x1 + 4, y: at.y + 4, w: width, h: 14 }], true);
  return `<line class="ak-ref" x1="${at.x1}" x2="${at.x2}" y1="${round(at.y)}" y2="${round(at.y)}"/>`
    + (box ? `<text class="ak-b" x="${round(box.x)}" y="${round(box.y + 11)}">${esc(text)}</text>` : "");
}

function horizontalBars(width: number, id: string, options: BarsOptions, labels: string[], series: Series[], format: (value: number) => string): Drawing {
  const ranges = options.range ?? [], subs = options.sub ?? [], narrow = width < 420, single = series.length === 1;
  const totals = labels.map((_, index) => series.reduce((sum, item) => sum + Math.max(0, item.values[index] ?? 0), 0));
  const texts = labels.map((_, index) => { const range = ranges[index]; return range ? `${format(range.lo)}–${format(range.hi)}` : format(totals[index]); });
  const high = Math.max(0, options.refLine?.value ?? 0, ...labels.map((_, index) => Math.max(totals[index], ranges[index]?.hi ?? 0)));
  const scale = niceScale(0, high, 3), labelWidth = narrow ? 0 : Math.min(width * 0.34, Math.max(...labels.map(text => textWidth(text, 13, 600)), ...subs.map(text => textWidth(text))) + 14);
  const valueWidth = Math.max(...texts.map(text => textWidth(text, 13, 600))) + 12, thick = 18, row = thick + (narrow ? 30 : subs.some(Boolean) ? 22 : 14);
  const plot = Math.max(40, width - labelWidth - valueWidth - 4), height = 4 + labels.length * row + 24;
  const x = (value: number): number => labelWidth + clamp(value, 0, scale.high) / scale.high * plot;
  const rowColor = (index: number): string => options.colors ? color(options.colors[index], index) : series[0].color;
  let defs = "", marks = "";
  const hits = labels.map((label, index) => {
    const y0 = 4 + index * row + (narrow ? 22 : (row - thick) / 2), middle = y0 + thick / 2, range = ranges[index];
    marks += narrow
      ? `<text class="ak-b" x="0" y="${round(y0 - 7)}">${esc(label)}${subs[index] ? `<tspan class="ak-m" font-weight="400"> ${esc(subs[index])}</tspan>` : ""}</text>`
      : `<text class="ak-b" x="0" y="${round(subs[index] ? middle - 2 : middle + 4)}">${esc(label)}</text>`
        + (subs[index] ? `<text class="ak-m" x="0" y="${round(middle + 13)}">${esc(subs[index])}</text>` : "");
    if (range) {
      const fill = rowColor(index), low = Math.min(range.lo, range.hi), up = Math.max(range.lo, range.hi);
      defs += `<pattern id="${id}h${index}" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)"><rect width="6" height="6" style="fill:${fill};opacity:.14"/><rect width="2.2" height="6" style="fill:${fill}"/></pattern>`;
      marks += `<rect x="${labelWidth}" y="${round(y0)}" width="${round(x(low) - labelWidth)}" height="${thick}" style="fill:${fill}"/>`
        + `<rect x="${round(x(low))}" y="${round(y0)}" width="${round(x(up) - x(low))}" height="${thick}" style="fill:url(#${id}h${index})"/>`;
    } else {
      let start = 0;
      series.forEach(item => {
        const value = Math.max(0, item.values[index] ?? 0);
        marks += `<rect x="${round(x(start))}" y="${round(y0)}" width="${round(x(start + value) - x(start))}" height="${thick}" style="fill:${single ? rowColor(index) : item.color}"${single ? "" : ' class="ak-sep"'}/>`;
        start += value;
      });
    }
    marks += `<text class="ak-b" x="${width}" y="${round(middle + 5)}" text-anchor="end">${esc(texts[index])}</text>`;
    return { x0: 0, x1: width, y0: 4 + index * row, y1: 4 + (index + 1) * row, html: tip(label, range
      ? [[options.rangeLabels?.[1] ?? "估计区间", texts[index], rowColor(index)]] : series.map(item => [item.name, format(item.values[index] ?? 0), single ? rowColor(index) : item.color])) };
  });
  const placement = new Placement(width, height), axis = 4 + labels.length * row + 16;
  if (options.refLine) {
    const rx = x(options.refLine.value), text = referenceText(options.refLine, format);
    marks += `<line class="ak-ref" x1="${round(rx)}" x2="${round(rx)}" y1="2" y2="${axis - 12}"/>` + xLabel(rx, axis, text, width).replace("<text", '<text class="ak-b"');
    placement.block({ x: rx - textWidth(text, 12, 600) / 2 - 6, y: axis - 12, w: textWidth(text, 12, 600) + 12, h: 16 });
  }
  scale.ticks.forEach(value => {
    const tx = x(value), half = textWidth(format(value)) / 2;
    if (placement.place([{ x: clamp(tx - half, 0, width - 2 * half), y: axis - 12, w: 2 * half, h: 16 }])) marks += xLabel(tx, axis, format(value), width - valueWidth);
  });
  const head = ranges.some(Boolean)
    ? legend([["var(--ak-color-text-muted)", null, options.rangeLabels?.[0] ?? "下限估计"], ["repeating-linear-gradient(45deg, var(--ak-color-text-muted) 0 2px, transparent 2px 5px)", null, options.rangeLabels?.[1] ?? "估计区间"]])
    : series.length > 1 && options.legend !== false ? legend(series.map(item => [item.color, null, item.name])) : "";
  return { html: head + svgOpen(width, height, options.label) + `<defs>${defs}</defs>` + marks + "</svg>", hits };
}

// 折线 / 面积：里程碑、区间、外推虚线；多系列在线尾直接标名。
function line(target: Target, options: LineOptions): Handle {
  const labels = (options?.x ?? []).map(String);
  if (!labels.length) throw new Error("ak.line: x 不能为空");
  const series = readSeries("line", options.series, labels.length, true), format = formatter(options.fmt, options.unit);
  if (series.every(item => item.values.every(value => value == null))) throw new Error("ak.line: 没有可画的数值");
  if (options.projectFrom != null && indexOf(labels, options.projectFrom) < 0) throw new Error("ak.line: projectFrom 不是 x 中的值或有效下标");
  const most = Math.max(...series.flatMap(item => item.values.filter((value): value is number => value != null)), options.refLine?.value ?? -Infinity);
  if (options.yMin != null && !(Number.isFinite(options.yMin) && options.yMin < most)) throw new Error("ak.line: yMin 必须是有限数字且小于数据最大值");
  return mount(target, (width, id) => drawLine(width, id, options, labels, series, format));
}

function drawLine(width: number, id: string, options: LineOptions, labels: string[], series: Series[], format: (value: number) => string): Drawing {
  const count = labels.length, numbers = series.flatMap(item => item.values.filter((value): value is number => value != null));
  const most = Math.max(...numbers, options.refLine?.value ?? -Infinity), lowest = Math.min(...numbers, options.refLine?.value ?? Infinity);
  const least = options.yMin ?? (lowest >= 0 && lowest < most * 0.5 ? 0 : lowest);
  const notes = [...(options.milestones ?? []), ...(options.annotate ?? [])], multi = series.length > 1;
  const height = options.height ?? (width < 480 ? 220 : 280), top = notes.length ? (notes.some(note => note.sub) ? 44 : 28) : 12, bottom = 24;
  const scale = niceScale(least, most, height < 240 ? 3 : 4), left = Math.max(...scale.ticks.map(value => textWidth(format(value)))) + 10;
  const right = multi ? Math.min(width * 0.3, Math.max(...series.map(item => textWidth(item.name, 12, 600))) + 30) : 12;
  const plot = width - left - right, base = height - bottom;
  const x = (index: number): number => left + (count === 1 ? plot / 2 : index / (count - 1) * plot);
  const y = (value: number): number => base - (clamp(value, scale.low, scale.high) - scale.low) / (scale.high - scale.low) * (base - top);
  const project = options.projectFrom == null ? count - 1 : indexOf(labels, options.projectFrom);
  const placement = new Placement(width, height);
  placement.block({ x: 0, y: 0, w: left - 4, h: height });
  placement.block({ x: 0, y: height - bottom + 4, w: width, h: bottom });
  let grid = horizontalGrid(scale.ticks, y, left, left + plot, format) + `<line class="ak-base" x1="${left}" x2="${left + plot}" y1="${base}" y2="${base}"/>`;
  xTicks(labels, plot).forEach(tick => { grid += xLabel(x(tick.index), height - 6, tick.text, width); });
  let defs = "", marks = "";
  const ends: { y: number; html: (y: number) => string }[] = [];
  series.forEach((item, index) => {
    const drawn = drawSeries(item, `${id}g${index}`, { x, y, base }, options.area ?? !multi, project, placement);
    defs += drawn.defs;
    marks += drawn.marks;
    if (multi && drawn.last) ends.push({ y: drawn.last[1] + 4, html: at => `<rect x="${round(left + plot + 12)}" y="${round(at - 8)}" width="8" height="8" rx="2" style="fill:${item.color}"/>`
      + `<text class="ak-b" x="${round(left + plot + 24)}" y="${round(at)}">${esc(item.name)}</text>` });
  });
  marks += spread(ends, top + 8, base + 4);
  marks += lineNotes(placement, options, labels, series, notes, { x, y, base, top });
  marks += referenceLine(placement, options.refLine, format, { x1: left, x2: left + plot, y: y(options.refLine?.value ?? 0) });
  marks += `<line class="ak-cross" x1="0" x2="0" y1="${top}" y2="${base}" style="visibility:hidden"/>`;
  const step = count > 1 ? plot / (count - 1) : plot;
  const hits = labels.map((label, index) => ({ x0: x(index) - step / 2, x1: x(index) + step / 2, y0: 0, y1: height, cx: round(x(index)),
    html: tip(shortLabel(label), series.filter(item => item.values[index] != null).map(item => [item.name, format(item.values[index]!), item.color])) }));
  return { html: svgOpen(width, height, options.label) + `<defs>${defs}</defs>` + grid + marks + "</svg>", hits };
}

type Point = readonly [number, number];

// 把系列切成连续非空段：描边、面积、孤立点和避让都按段处理，不跨缺失值连线。
function runs(values: readonly (number | null)[]): number[][] {
  const out: number[][] = [];
  values.forEach((value, index) => {
    if (value == null) return;
    const last = out[out.length - 1];
    if (last && last[last.length - 1] === index - 1) last.push(index); else out.push([index]);
  });
  return out;
}

// 单个系列：每段画面积、实线和外推虚线，孤立点画圆点，并沿线每隔约 8px 登记避让块。
function drawSeries(item: Series, gradient: string, axes: { x: (index: number) => number; y: (value: number) => number; base: number },
  area: boolean, project: number, placement: Placement): { defs: string; marks: string; last: Point | null } {
  const point = (at: number): Point => [round(axes.x(at)), round(axes.y(item.values[at]!))];
  const stroke = (indices: readonly number[], dash: string): string => indices.length < 2 ? ""
    : `<path d="${indices.map((at, order) => `${order ? "L" : "M"}${point(at).join(" ")}`).join("")}" fill="none" style="stroke:${item.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"${dash}/>`;
  let marks = "", last: Point | null = null;
  for (const run of runs(item.values)) {
    const points = run.map(point);
    points.forEach((from, order) => {
      const to = points[order + 1] ?? from, steps = Math.max(1, Math.ceil(Math.hypot(to[0] - from[0], to[1] - from[1]) / 8));
      for (let step = 0; step < steps; step += 1) placement.block({ x: from[0] + (to[0] - from[0]) * step / steps - 2, y: from[1] + (to[1] - from[1]) * step / steps - 2, w: 4, h: 4 });
    });
    last = points[points.length - 1];
    if (points.length === 1) { marks += `<circle cx="${last[0]}" cy="${last[1]}" r="3" style="fill:${item.color}"/>`; continue; }
    if (area) marks += `<path d="M${points[0][0]} ${axes.base}L${points.map(at => at.join(" ")).join("L")}L${last[0]} ${axes.base}Z" style="fill:url(#${gradient})"/>`;
    marks += stroke(run.filter(at => at <= project), "") + stroke(run.filter(at => at >= project), ' stroke-dasharray="4 4"');
  }
  const defs = area ? `<linearGradient id="${gradient}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" style="stop-color:${item.color};stop-opacity:.24"/><stop offset="1" style="stop-color:${item.color};stop-opacity:.02"/></linearGradient>` : "";
  return { defs, marks, last };
}

// 里程碑画点和落地虚线，区间标在底部；注释可用 series 指定挂在哪条线上。
function lineNotes(placement: Placement, options: LineOptions, labels: string[], series: Series[], notes: Note[],
  axes: { x: (index: number) => number; y: (value: number) => number; base: number; top: number }): string {
  let marks = "";
  for (const span of options.spans ?? []) {
    const from = indexOf(labels, span.from), to = indexOf(labels, span.to);
    if (from < 0 || to < 0) continue;
    const middle = (axes.x(from) + axes.x(to)) / 2, width = textWidth(span.label);
    if (width + 8 < Math.abs(axes.x(to) - axes.x(from)) && placement.place([{ x: middle - width / 2, y: axes.base - 20, w: width, h: 14 }]))
      marks += `<text class="ak-m" x="${round(middle)}" y="${axes.base - 8}" text-anchor="middle">${esc(span.label)}</text>`;
  }
  const milestones = new Set(options.milestones ?? []);
  notes.forEach((note, order) => {
    const index = indexOf(labels, note.at);
    const owner = series.find(item => item.name === note.series) ?? series.find(item => index >= 0 && item.values[index] != null);
    const value = owner && index >= 0 ? owner.values[index] : null;
    if (owner == null || value == null) return;
    const px = axes.x(index), py = axes.y(value), last = milestones.has(note) && order === (options.milestones?.length ?? 0) - 1 && milestones.size > 1;
    if (milestones.has(note)) {
      marks += `<line class="ak-lead" x1="${round(px)}" x2="${round(px)}" y1="${round(py)}" y2="${axes.base}"/>` + (last
        ? `<circle cx="${round(px)}" cy="${round(py)}" r="10" style="fill:none;stroke:var(--k-hot);opacity:.4"/><circle cx="${round(px)}" cy="${round(py)}" r="5.5" style="fill:var(--k-hot)"/>`
        : `<circle cx="${round(px)}" cy="${round(py)}" r="4" style="fill:var(--ak-color-bg-surface);stroke:${owner.color}" stroke-width="2"/>`);
    }
    placement.block({ x: px - 6, y: py - 6, w: 12, h: 12 });
    marks += annotation(placement, note, px, py, 10);
  });
  return marks;
}

// 排行横条列表：标签、内联条、数值三列；窄屏条换到下一行。
function barlist(target: Target, options: BarlistOptions): Handle {
  const format = formatter(options?.fmt, options?.unit), fill = color(options?.color ?? 1, 0);
  let items = (options?.items ?? []).map((item, index) => {
    if (typeof item?.value !== "number" || !Number.isFinite(item.value) || item.value < 0) throw new Error(`ak.barlist: items[${index}].value 必须是非负有限数字`);
    return { label: String(item.label ?? ""), value: item.value, note: item.note };
  });
  if (!items.length) throw new Error("ak.barlist: items 不能为空");
  if (options.sort !== false) items = [...items].sort((a, b) => b.value - a.value);
  if (options.limit) items = items.slice(0, options.limit);
  const most = options.max ?? Math.max(...items.map(item => item.value)) ?? 1;
  return mount(target, width => ({ hits: [], html: `<div class="ak-barlist${width < 420 ? " narrow" : ""}" role="list" aria-label="${esc(options.label || "排行")}">`
    + items.map(item => {
      const name = `<span class="ak-l${/[\\/]/.test(item.label) ? " path" : ""}">${esc(item.label)}${item.note ? `<small>${esc(item.note)}</small>` : ""}</span>`;
      const bar = `<span class="ak-bar" aria-hidden="true"><i style="width:${round(clamp(item.value / (most || 1) * 100, 0.5, 100))}%;background:${fill}"></i></span>`;
      const value = `<b class="ak-v">${esc(format(item.value))}</b>`;
      return `<div class="ak-row" role="listitem">${width < 420 ? name + value + bar : name + bar + value}</div>`;
    }).join("") + "</div>" }));
}

// 迷你趋势线：放在 hero 数字或表格单元格里，只表达走势，末点标出。
function spark(target: Target, values: readonly number[], options: { color?: ColorKey; label?: string } = {}): Handle {
  if (!Array.isArray(values) || values.length < 2 || values.some(value => typeof value !== "number" || !Number.isFinite(value)))
    throw new Error("ak.spark: 需要至少两个有限数字");
  const low = Math.min(...values), high = Math.max(...values), stroke = color(options.color ?? 1, 0);
  return mount(target, () => {
    const width = 96, height = 28, x = (index: number) => 2 + index / (values.length - 1) * (width - 6);
    const y = (value: number) => height - 3 - (high > low ? (value - low) / (high - low) : 0.5) * (height - 6);
    const path = values.map((value, index) => `${index ? "L" : "M"}${round(x(index))} ${round(y(value))}`).join("");
    return { hits: [], html: `<svg class="ak-spark" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img" aria-label="${esc(options.label || "趋势")}">`
      + `<path d="${path}" fill="none" style="stroke:${stroke}" stroke-width="1.5" stroke-linejoin="round"/>`
      + `<circle cx="${round(x(values.length - 1))}" cy="${round(y(values[values.length - 1]))}" r="2.5" style="fill:${stroke}"/></svg>` };
  });
}

// 散点 / 气泡：两个量的关系；少量点直接标名，多点只标 highlight，可加参考线分象限和最小二乘趋势线。
function scatter(target: Target, options: ScatterOptions): Handle {
  const points = (options?.points ?? []).map((point, index) => {
    if (![point?.x, point?.y].every(value => typeof value === "number" && Number.isFinite(value))) throw new Error(`ak.scatter: points[${index}] 的 x、y 必须是有限数字`);
    if (point.size != null && !(typeof point.size === "number" && Number.isFinite(point.size) && point.size >= 0)) throw new Error(`ak.scatter: points[${index}].size 必须是非负有限数字`);
    return { x: point.x, y: point.y, label: String(point.label ?? ""), size: point.size ?? 0, group: String(point.group ?? "") };
  });
  if (!points.length) throw new Error("ak.scatter: points 不能为空");
  const groups = [...new Set(points.map(point => point.group))], fx = formatter(options.xFmt ?? options.fmt, ""), fy = formatter(options.yFmt ?? options.fmt, options.unit);
  return mount(target, width => {
    const height = options.height ?? (width < 480 ? 240 : 300), top = 12, bottom = options.xTitle ? 40 : 24;
    const sx = niceScale(Math.min(...points.map(point => point.x), options.refX?.value ?? Infinity), Math.max(...points.map(point => point.x), options.refX?.value ?? -Infinity), width < 480 ? 3 : 5);
    const sy = niceScale(Math.min(...points.map(point => point.y), options.refY?.value ?? Infinity), Math.max(...points.map(point => point.y), options.refY?.value ?? -Infinity), height < 260 ? 3 : 4);
    const left = Math.max(...sy.ticks.map(value => textWidth(fy(value)))) + 10, right = width - 8, base = height - bottom;
    const x = (value: number): number => left + (value - sx.low) / (sx.high - sx.low) * (right - left);
    const y = (value: number): number => base - (value - sy.low) / (sy.high - sy.low) * (base - top);
    const biggest = Math.max(...points.map(point => point.size)) || 1, radius = (size: number): number => size ? 3 + Math.sqrt(size / biggest) * 15 : 4.5;
    const placement = new Placement(width, height), wanted = new Set(options.highlight ?? (points.length <= 12 ? points.map(point => point.label) : []));
    let grid = horizontalGrid(sy.ticks, y, left, right, fy) + `<line class="ak-base" x1="${left}" x2="${right}" y1="${base}" y2="${base}"/>`;
    sx.ticks.forEach(value => { grid += xLabel(x(value), base + 16, fx(value), width); });
    if (options.xTitle) grid += `<text class="ak-m" x="${right}" y="${height - 4}" text-anchor="end">${esc(options.xTitle)} →</text>`;
    if (options.yTitle) grid += `<text class="ak-m" x="${left + 4}" y="${top + 4}">↑ ${esc(options.yTitle)}</text>`;
    let marks = "";
    if (options.refX) marks += `<line class="ak-lead" x1="${round(x(options.refX.value))}" x2="${round(x(options.refX.value))}" y1="${top}" y2="${base}"/>`;
    marks += referenceLine(placement, options.refY, fy, { x1: left, x2: right, y: y(options.refY?.value ?? 0) }).replace('class="ak-ref"', 'class="ak-lead"');
    if (options.trend && points.length > 2) marks += trendLine(points, x, y, sx);
    const order = [...points].sort((a, b) => b.size - a.size);
    order.forEach(point => {
      const fill = color(undefined, groups.indexOf(point.group)), r = radius(point.size), hot = wanted.has(point.label) && point.label;
      marks += `<circle cx="${round(x(point.x))}" cy="${round(y(point.y))}" r="${round(r)}" style="fill:${fill};fill-opacity:${point.size ? 0.55 : 0.85};stroke:${hot ? "var(--ak-color-text-primary)" : fill}" stroke-width="${hot ? 1.5 : 1}"/>`;
      placement.block({ x: x(point.x) - r, y: y(point.y) - r, w: 2 * r, h: 2 * r });
    });
    points.forEach(point => { if (point.label && wanted.has(point.label)) marks += annotation(placement, { at: 0, text: point.label }, x(point.x), y(point.y), radius(point.size) + 3); });
    const hits = points.map(point => { const r = Math.max(8, radius(point.size)); return { x0: x(point.x) - r, x1: x(point.x) + r, y0: y(point.y) - r, y1: y(point.y) + r,
      html: tip(point.label || point.group || "点", [[options.xTitle ?? "x", fx(point.x), color(undefined, groups.indexOf(point.group))], [options.yTitle ?? "y", fy(point.y), color(undefined, groups.indexOf(point.group))]]) }; });
    const head = groups.length > 1 ? legend(groups.map((group, index) => [color(undefined, index), null, group || "其他"])) : "";
    return { html: head + svgOpen(width, height, options.label) + grid + marks + "</svg>", hits };
  });
}

function trendLine(points: readonly { x: number; y: number }[], x: (value: number) => number, y: (value: number) => number, scale: { low: number; high: number }): string {
  const n = points.length, mx = points.reduce((sum, point) => sum + point.x, 0) / n, my = points.reduce((sum, point) => sum + point.y, 0) / n;
  const sxx = points.reduce((sum, point) => sum + (point.x - mx) ** 2, 0);
  if (!sxx) return "";
  const slope = points.reduce((sum, point) => sum + (point.x - mx) * (point.y - my), 0) / sxx, at = (value: number): number => my + slope * (value - mx);
  return `<line class="ak-lead" x1="${round(x(scale.low))}" x2="${round(x(scale.high))}" y1="${round(y(at(scale.low)))}" y2="${round(y(at(scale.high)))}" style="stroke-dasharray:6 4"/>`;
}

// 哑铃：每行两个时点或两种口径的对比，线段长度就是差值；可按变化量排序。
function dumbbell(target: Target, options: DumbbellOptions): Handle {
  const labels = (options?.x ?? []).map(String);
  if (!labels.length) throw new Error("ak.dumbbell: x 不能为空");
  const [from, to] = readSeries("dumbbell", [options.from ?? {}, { color: 2, ...options.to }], labels.length, false), format = formatter(options.fmt, options.unit);
  const rows = labels.map((label, index) => ({ label, a: from.values[index]!, b: to.values[index]! }));
  if (options.sort !== false) rows.sort((m, n) => (n.b - n.a) - (m.b - m.a));
  return mount(target, width => {
    const narrow = width < 420, labelWidth = narrow ? 0 : Math.min(width * 0.3, Math.max(...labels.map(text => textWidth(text, 12, 600))) + 12);
    const row = narrow ? 44 : 28, height = rows.length * row + 24, scale = niceScale(Math.min(0, ...rows.flatMap(item => [item.a, item.b])), Math.max(...rows.flatMap(item => [item.a, item.b])), 4);
    const valueWidth = Math.max(...rows.flatMap(item => [item.a, item.b]).map(value => textWidth(format(value)))) + 8;
    const x = (value: number): number => labelWidth + valueWidth + (value - scale.low) / (scale.high - scale.low) * (width - labelWidth - 2 * valueWidth);
    let marks = scale.ticks.map(value => `<line class="ak-grid" x1="${round(x(value))}" x2="${round(x(value))}" y1="0" y2="${height - 20}"/>` + xLabel(x(value), height - 4, format(value), width)).join("");
    const hits = rows.map((item, index) => {
      const cy = index * row + (narrow ? 30 : row / 2), up = item.b >= item.a;
      marks += narrow ? `<text class="ak-b" x="0" y="${cy - 12}">${esc(item.label)}</text>` : `<text class="ak-b" x="0" y="${cy + 4}">${esc(item.label)}</text>`;
      marks += `<line x1="${round(x(item.a))}" x2="${round(x(item.b))}" y1="${cy}" y2="${cy}" style="stroke:var(--k-gray)" stroke-width="3"/>`
        + `<circle cx="${round(x(item.a))}" cy="${cy}" r="5" style="fill:${from.color}"/><circle cx="${round(x(item.b))}" cy="${cy}" r="5" style="fill:${to.color}"/>`
        + `<text x="${round(x(item.a) + (up ? -9 : 9))}" y="${cy + 4}" text-anchor="${up ? "end" : "start"}">${esc(format(item.a))}</text>`
        + `<text class="ak-b" x="${round(x(item.b) + (up ? 9 : -9))}" y="${cy + 4}" text-anchor="${up ? "start" : "end"}">${esc(format(item.b))}</text>`;
      return { x0: 0, x1: width, y0: index * row, y1: (index + 1) * row, html: tip(item.label, [[from.name, format(item.a), from.color], [to.name, format(item.b), to.color]]) };
    });
    return { html: legend([[from.color, null, from.name], [to.color, null, to.name]]) + svgOpen(width, height, options.label) + marks + "</svg>", hits };
  });
}

// 热力图：两个周期维度的规律，单色相由浅到深，最大格描边，下方附色阶。
function heatmap(target: Target, options: HeatmapOptions): Handle {
  const rows = (options?.rows ?? []).map(String), cols = (options?.cols ?? []).map(String);
  if (!rows.length || !cols.length) throw new Error("ak.heatmap: rows 和 cols 不能为空");
  if (!Array.isArray(options.values) || options.values.length !== rows.length || options.values.some(row => !Array.isArray(row) || row.length !== cols.length))
    throw new Error(`ak.heatmap: values 必须是 ${rows.length} 行 × ${cols.length} 列`);
  const grid = rows.map((_, r) => cols.map((_, c) => {
    const value = options.values?.[r]?.[c];
    if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`ak.heatmap: values[${r}][${c}] 不是有限数字`);
    return value;
  }));
  const format = formatter(options.fmt, options.unit), flat = grid.flat(), low = Math.min(...flat), high = Math.max(...flat), base = color(options.color ?? 1, 0);
  const tone = (value: number): string => `color-mix(in oklab, ${base} ${Math.round(8 + 90 * (high > low ? (value - low) / (high - low) : 1))}%, var(--ak-color-bg-surface))`;
  return mount(target, width => {
    const left = Math.max(...rows.map(text => textWidth(text))) + 10, cell = (width - left) / cols.length, tall = options.height ? options.height / rows.length : clamp(cell * 0.9, 14, 28), gap = cell > 10 ? 2 : 1;
    const bottom = rows.length * tall, height = bottom + 46;
    let marks = "", best = [0, 0];
    const hits: Hit[] = [];
    grid.forEach((line, r) => {
      marks += `<text x="${left - 8}" y="${round(r * tall + tall / 2 + 4)}" text-anchor="end">${esc(rows[r])}</text>`;
      line.forEach((value, c) => {
        if (value > grid[best[0]][best[1]]) best = [r, c];
        const cx = left + c * cell, cy = r * tall;
        marks += `<rect x="${round(cx)}" y="${round(cy)}" width="${round(cell - gap)}" height="${round(tall - gap)}" rx="${cell > 12 ? 2 : 1}" style="fill:${tone(value)}"/>`;
        if (options.showValues && cell >= 28) marks += `<text x="${round(cx + (cell - gap) / 2)}" y="${round(cy + tall / 2 + 4)}" text-anchor="middle" style="fill:${value > low + (high - low) * 0.55 ? "var(--ak-color-bg-surface)" : "var(--ak-color-text-primary)"}">${esc(format(value))}</text>`;
        hits.push({ x0: cx, x1: cx + cell, y0: cy, y1: cy + tall, html: tip(`${rows[r]} · ${cols[c]}`, [[options.label || "数值", format(value), tone(value)]]) });
      });
    });
    marks += `<rect x="${round(left + best[1] * cell - 1)}" y="${round(best[0] * tall - 1)}" width="${round(cell - gap + 2)}" height="${round(tall - gap + 2)}" rx="2" style="fill:none;stroke:var(--ak-color-text-primary)" stroke-width="1.5"/>`;
    xTicks(cols, width - left).forEach(tick => { marks += xLabel(left + cell * (tick.index + 0.5), bottom + 16, tick.text, width); });
    const scaleY = bottom + 28;
    marks += `<text class="ak-m" x="${left}" y="${scaleY + 11}">${esc(format(low))}</text>`;
    const start = left + textWidth(format(low)) + 6;
    [0, 0.25, 0.5, 0.75, 1].forEach((step, index) => { marks += `<rect x="${round(start + index * 16)}" y="${scaleY + 1}" width="14" height="12" rx="2" style="fill:${tone(low + step * (high - low))}"/>`; });
    marks += `<text class="ak-m" x="${round(start + 84)}" y="${scaleY + 11}">${esc(format(high))}${options.label ? ` · ${esc(options.label)}` : ""}</text>`;
    return { html: svgOpen(width, height, options.label) + marks + "</svg>", hits };
  });
}

// 100% 份额面积：构成随时间变化，右侧直接写名称和最新份额。
function share(target: Target, options: ShareOptions): Handle {
  const labels = (options?.x ?? []).map(String);
  if (labels.length < 2) throw new Error("ak.share: x 至少需要两个时点");
  const series = readSeries("share", options.series, labels.length, false);
  if (series.some(item => item.values.some(value => value! < 0))) throw new Error("ak.share: 份额不支持负数");
  const totals = labels.map((_, index) => series.reduce((sum, item) => sum + Math.max(0, item.values[index]!), 0));
  if (totals.some(total => !Number.isFinite(total))) throw new Error("ak.share: 每个时点的合计必须是有限数字");
  const shares = series.map(item => item.values.map((value, index) => totals[index] ? Math.max(0, value!) / totals[index] * 100 : 0));
  return mount(target, width => {
    const count = labels.length, last = count - 1, names = series.map((item, index) => `${item.name} ${Math.round(shares[index][last])}%`);
    const height = options.height ?? (width < 480 ? 220 : 260), top = options.annotate?.length ? 26 : 8, base = height - 24, left = textWidth("100%") + 10;
    const plot = width - left - Math.min(width * 0.34, Math.max(...names.map(text => textWidth(text, 12, 600))) + 12);
    const x = (index: number): number => left + index / last * plot, y = (value: number): number => base - value / 100 * (base - top);
    let marks = "";
    const sums = labels.map(() => 0), ends: { y: number; html: (y: number) => string }[] = [];
    series.forEach((item, index) => {
      const low = [...sums], high = sums.map((sum, at) => sum + shares[index][at]);
      high.forEach((value, at) => { sums[at] = value; });
      marks += `<path d="M${high.map((value, at) => `${round(x(at))} ${round(y(value))}`).join("L")}L${low.map((value, at) => `${round(x(at))} ${round(y(value))}`).reverse().join("L")}Z" class="ak-sep" style="fill:${item.color}"/>`;
      if (shares[index][last] > 0) ends.push({ y: y((low[last] + high[last]) / 2) + 4, html: at => `<text class="ak-b" x="${round(left + plot + 6)}" y="${round(at)}">${esc(names[index])}</text>` });
    });
    marks += spread(ends, top + 8, base + 4);
    [0, 50, 100].forEach(value => { marks += `<text x="${left - 8}" y="${round(y(value)) + 4}" text-anchor="end">${value}%</text>`; });
    xTicks(labels, plot).forEach(tick => { marks += xLabel(x(tick.index), height - 6, tick.text, width); });
    for (const note of options.annotate ?? []) {
      const index = indexOf(labels, note.at), text = String(note.text ?? note.label ?? "");
      if (index < 0 || !text) continue;
      marks += `<line x1="${round(x(index))}" x2="${round(x(index))}" y1="${top - 4}" y2="${base}" style="stroke:var(--ak-color-bg-canvas)" stroke-width="2"/>`
        + `<text class="ak-b" x="${round(clamp(x(index), textWidth(text, 12, 600) / 2, width - textWidth(text, 12, 600) / 2))}" y="${top - 9}" text-anchor="middle">${esc(text)}</text>`;
    }
    marks += `<line class="ak-cross" x1="0" x2="0" y1="${top}" y2="${base}" style="visibility:hidden"/>`;
    const step = plot / last;
    const hits = labels.map((label, at) => ({ x0: x(at) - step / 2, x1: x(at) + step / 2, y0: 0, y1: height, cx: round(x(at)),
      html: tip(shortLabel(label), [...series].reverse().map(item => [item.name, `${round(shares[series.indexOf(item)][at])}%`, item.color])) }));
    return { html: svgOpen(width, height, options.label) + marks + "</svg>", hits };
  });
}

interface Area<T> { item: T; value: number; x: number; y: number; w: number; h: number }

// squarify：按面积切块并让每块尽量接近正方形。
function squarify<T>(items: readonly { item: T; value: number }[], x: number, y: number, w: number, h: number): Area<T>[] {
  const out: Area<T>[] = [], total = items.reduce((sum, entry) => sum + entry.value, 0);
  if (!total || w <= 0 || h <= 0) return out;
  // 先换算成占比再乘面积，极小或极大的数值都不会溢出。
  const unit = w * h, weight = (entry: { value: number }): number => entry.value / total;
  let rest = [...items];
  while (rest.length) {
    const side = Math.min(w, h);
    let row: typeof rest = [], best = Infinity;
    for (const entry of rest) {
      const next = [...row, entry], area = next.reduce((sum, item) => sum + weight(item) * unit, 0), sizes = next.map(item => weight(item) * unit);
      const worst = Math.max(side * side * Math.max(...sizes) / (area * area), area * area / (side * side * Math.min(...sizes)));
      if (worst > best) break;
      best = worst;
      row = next;
    }
    const area = row.reduce((sum, entry) => sum + weight(entry) * unit, 0);
    if (w >= h) { const cw = area / h; let cy = y; row.forEach(entry => { const eh = weight(entry) * unit / cw; out.push({ ...entry, x, y: cy, w: cw, h: eh }); cy += eh; }); x += cw; w -= cw; }
    else { const ch = area / w; let cx = x; row.forEach(entry => { const ew = weight(entry) * unit / ch; out.push({ ...entry, x: cx, y, w: ew, h: ch }); cx += ew; }); y += ch; h -= ch; }
    rest = rest.slice(row.length);
  }
  return out;
}

// 矩形树图：层级加大小；先按组切块再切节点，组色区分，组内按大小深浅。
function treemap(target: Target, options: TreemapOptions): Handle {
  const nodes = (options?.nodes ?? []).map((node, index) => {
    if (typeof node?.value !== "number" || !Number.isFinite(node.value) || node.value < 0) throw new Error(`ak.treemap: nodes[${index}].value 必须是非负有限数字`);
    return { name: String(node.name ?? ""), value: node.value, group: String(node.group ?? "") };
  }).filter(node => node.value > 0);
  if (!nodes.length) throw new Error("ak.treemap: nodes 需要至少一个正数 value");
  const format = formatter(options.fmt, options.unit), names = [...new Set(nodes.map(node => node.group))], total = nodes.reduce((sum, node) => sum + node.value, 0);
  if (!Number.isFinite(total)) throw new Error("ak.treemap: value 合计必须是有限数字");
  const groups = names.map((name, index) => ({ item: { name, fill: color(undefined, index) }, value: nodes.filter(node => node.group === name).reduce((sum, node) => sum + node.value, 0) })).sort((a, b) => b.value - a.value);
  return mount(target, width => {
    const height = options.height ?? (width < 480 ? 280 : 340), hits: Hit[] = [];
    let marks = "";
    squarify(groups, 0, 0, width, height).forEach(group => {
      const head = names.length > 1 && group.w > 70 && group.h > 56 ? 18 : 0;
      const inside = nodes.filter(node => node.group === group.item.name).sort((a, b) => b.value - a.value), top = inside[0].value;
      if (head) marks += `<text class="ak-b" x="${round(group.x + 4)}" y="${round(group.y + 13)}">${esc(group.item.name)} <tspan class="ak-m" font-weight="400">${esc(format(group.value))}</tspan></text>`;
      squarify(inside.map(node => ({ item: node, value: node.value })), group.x + 1, group.y + head + 1, group.w - 2, group.h - head - 2).forEach(cell => {
        marks += `<rect x="${round(cell.x + 1)}" y="${round(cell.y + 1)}" width="${round(Math.max(0, cell.w - 2))}" height="${round(Math.max(0, cell.h - 2))}" rx="2" style="fill:color-mix(in oklab, ${group.item.fill} ${Math.round(16 + cell.value / top * 30)}%, var(--ak-color-bg-surface))"/>`;
        if (cell.w > textWidth(cell.item.name, 12, 600) + 10 && cell.h > 20) marks += `<text class="ak-b" x="${round(cell.x + 6)}" y="${round(cell.y + 16)}">${esc(cell.item.name)}</text>`;
        if (cell.w > textWidth(format(cell.value)) + 10 && cell.h > 36) marks += `<text class="ak-b" style="font-weight:400" x="${round(cell.x + 6)}" y="${round(cell.y + 31)}">${esc(format(cell.value))}</text>`;
        hits.push({ x0: cell.x, x1: cell.x + cell.w, y0: cell.y, y1: cell.y + cell.h, html: tip(cell.item.name, [[cell.item.group || "大小", `${format(cell.value)} · ${round(cell.value / total * 100)}%`, group.item.fill]]) });
      });
    });
    return { html: svgOpen(width, height, options.label) + marks + "</svg>", hits };
  });
}

export const ak = { bars, line, scatter, dumbbell, heatmap, share, treemap, barlist, spark, fmt: (value: number, format?: Format, unit?: string): string => formatter(format, unit)(value) };

declare global { interface Window { ak: typeof ak } }
window.ak = ak;

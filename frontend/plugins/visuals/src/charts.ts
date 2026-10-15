// 宿主图表工具包 window.ak：生成文档只给数据、选型与注释；坐标、刻度、配色、标签避让、窄屏重排和读数由这里负责。
// 构建时打包成 IIFE 内联进 frame.html，预检与客户端执行同一份代码；颜色全部是 CSS 变量，主题切换无需重画。

type ColorKey = number | "gray" | "hot";
type Format = "raw" | "%" | "wan" | "si" | ((value: number) => string);
type Target = string | HTMLElement;
type Value = number | null | undefined;

export interface SeriesInput { name?: string; values?: readonly Value[]; color?: ColorKey }
// 注释主句写 text；label 是 text 的别名，模型常用两种写法。
export interface Note { at: string | number; text?: string; label?: string; sub?: string; series?: string }
export interface RefLine { value: number; label?: string }
interface Common { label?: string; fmt?: Format; unit?: string; height?: number }
export interface BarsOptions extends Common {
  x: readonly (string | number)[]; series: readonly SeriesInput[]; stack?: boolean; horizontal?: boolean;
  annotate?: readonly Note[]; refLine?: RefLine; legend?: boolean;
  sub?: readonly string[]; range?: readonly ({ lo: number; hi: number } | null)[]; rangeLabels?: readonly [string, string];
  colors?: readonly ColorKey[];
}
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
    // 月份边界再按像素抽稀；跨年时一月写成年份。
    const years = new Set(labels.map(label => label.slice(0, 4))).size > 1, months: { index: number; text: string }[] = [];
    let previous = "";
    labels.forEach((label, index) => {
      const match = ISO.exec(label)!, month = match[1] + match[2];
      if (month !== previous && (index > 0 || Number(match[3]) <= 3))
        months.push({ index, text: years && match[2] === "01" ? match[1] : `${Number(match[2])}月` });
      previous = month;
    });
    const gap = Math.max(...months.map(tick => textWidth(tick.text))) + 12, pixel = width / labels.length;
    const kept: { index: number; text: string }[] = [];
    for (const tick of months) if (!kept.length || (tick.index - kept[kept.length - 1].index) * pixel >= gap) kept.push(tick);
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

export const ak = { bars, fmt: (value: number, format?: Format, unit?: string): string => formatter(format, unit)(value) };

declare global { interface Window { ak: typeof ak } }
window.ak = ak;

import * as Plot from "@observablehq/plot";

/** 为任意 Plot marks 提供纸面默认值与容器宽度；不限制图表种类。 */
function plot(host: HTMLElement, options: (width: number) => Plot.PlotOptions): () => void {
  let width = 0;
  const draw = () => {
    width = host.clientWidth;
    const value = options(width);
    const chart = Plot.plot({ height: 280, marginLeft: 42, marginRight: 14, ...value, width,
      x: { tickSize: 0, label: null, ...value.x },
      y: { tickSize: 0, grid: true, label: null, ...value.y },
      color: { range: [1, 2, 3, 4, 5, 6].map(n => `var(--ak-chart-${n})`), legend: true, ...value.color },
      style: typeof value.style === "string" ? value.style : { background: "transparent", color: "var(--ak-color-text-secondary)", fontFamily: "inherit", fontSize: "12px", ...value.style },
    });
    // Plot 把图例放在图前；阅读插图把它放到坐标轴后，保留原生图例行为。
    if (chart instanceof HTMLElement) {
      const svg = chart.querySelector(":scope > svg");
      if (svg) chart.prepend(svg);
    }
    host.replaceChildren(chart);
  };
  // 改 DOM 放在下一帧，避免在同一轮尺寸通知里再次改变被观察节点的高度。
  const observer = new ResizeObserver(() => { if (host.clientWidth !== width) requestAnimationFrame(draw); });
  observer.observe(host);
  draw();
  return draw;
}

/** 参数仍是原生 range；统一标签、读数和操作色，回调只在用户输入时触发。 */
function param(host: HTMLElement, options: {
  label: string; min: number; max: number; step: number; value: number;
  format?: (value: number) => string; onInput?: (value: number) => void;
}): HTMLInputElement {
  const label = document.createElement("label");
  label.className = "ak-param";
  const heading = document.createElement("span");
  heading.textContent = options.label;
  const output = document.createElement("output");
  const input = document.createElement("input");
  input.type = "range";
  input.min = String(options.min); input.max = String(options.max); input.step = String(options.step); input.value = String(options.value);
  input.setAttribute("aria-label", options.label);
  const update = () => {
    const value = input.valueAsNumber;
    output.textContent = options.format ? options.format(value) : String(value);
    input.setAttribute("aria-valuetext", output.textContent);
    input.style.setProperty("--range-fill", `${100 * (value - options.min) / (options.max - options.min)}%`);
  };
  input.addEventListener("input", () => { update(); options.onInput?.(input.valueAsNumber); });
  update();
  label.append(heading, output, input);
  host.append(label);
  return input;
}

Object.assign(globalThis, { Plot, ak: { plot, param } });

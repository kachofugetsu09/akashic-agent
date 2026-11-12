import "./plugin-ui.css";

// 值班面板：只认识“来源”和“事件”，不认识任何一种具体的状态；多一个来源就多一行。
const ICON = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><circle cx="8" cy="8" r="5.5"/><path d="M8 5v3l2 1.5"/></svg>';
const TO = { wake: "叫醒", inbox: "待看", log: "记录" };
const FILES = [["WATCH.md", "嘱托"], ["notes.md", "便签"], ["inbox.md", "待看"]];

function h(tag, className, text, children) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  for (const child of children ?? []) if (child) node.append(child);
  return node;
}
const minutes = (at) => Number(at.slice(0, 2)) * 60 + Number(at.slice(3, 5));
const left = (at) => `${(minutes(at) / 1440 * 100).toFixed(2)}%`;
const tickKind = (tick) => tick.rounds.length ? "wake" : tick.verdicts.some((v) => v.to === "inbox") ? "inbox" : "quiet";

function summary(data) {
  const events = data.ticks.reduce((sum, tick) => sum + tick.verdicts.length, 0);
  const rounds = data.ticks.reduce((sum, tick) => sum + tick.rounds.length, 0);
  const said = data.talk.filter((row) => row.kind === "push").length;
  return `看了 ${events} 件事，叫醒主 Agent ${rounds} 次，对你说了 ${said} 句`;
}

function button(className, text, onClick, label) {
  const node = h("button", className, text);
  node.type = "button";
  if (label) node.setAttribute("aria-label", label);
  node.addEventListener("click", onClick);
  return node;
}

function compactView(data, panel) {
  const rounds = data.ticks.flatMap((tick) => tick.rounds.map((round) => ({ at: tick.at, text: `${round.label}：${round.status}` })));
  const said = data.talk.filter((row) => row.kind === "push").map((row) => ({ at: row.at, text: `说了一句：${row.text}` }));
  const recent = [...rounds, ...said].sort((a, b) => b.at.localeCompare(a.at)).slice(0, 4);
  return h("div", "watch-compact", null, [
    h("p", "watch-line", `今天${summary(data)}`),
    h("p", "watch-sub", `每 ${data.interval} 分钟看一次 · 上次 ${data.ticks.at(-1)?.at ?? "还没有"}`),
    recent.length ? h("ul", "watch-mini", null, recent.map((row) => h("li", null, null, [h("time", null, row.at), h("span", null, row.text)]))) : null,
    button("watch-more", "看全部", () => panel.setSize("expanded")),
  ]);
}

function timeline(data, selected, onPick) {
  const end = data.date === data.today ? data.now : "23:59";
  const lane = (name, body) => h("div", "watch-lane", null, [h("span", "watch-lane-name", name), body]);
  const rows = data.lanes.map((item) => lane(item.source, h("div", "watch-band", null, item.segments.map((segment, index) => {
    const until = item.segments[index + 1]?.at ?? end;
    const node = h("span", "watch-seg", segment.text);
    node.style.left = left(segment.at);
    node.style.width = `${Math.max(0, (minutes(until) - minutes(segment.at)) / 1440 * 100).toFixed(2)}%`;
    node.title = `${segment.at} 起：${segment.text}`;
    return node;
  }))));
  const talk = h("div", "watch-band", null, data.talk.map((row) => {
    const node = h("i", `watch-mark watch-mark--${row.kind}`);
    node.style.left = left(row.at);
    node.title = `${row.at} ${row.kind === "user" ? "他说" : "主动消息"}：${row.text}`;
    return node;
  }));
  const ticks = h("div", "watch-band watch-ticks", null, data.ticks.map((tick, index) => {
    const node = button(`watch-tick watch-tick--${tickKind(tick)}`, null, () => onPick(index), `${tick.at} 这一轮`);
    node.style.left = left(tick.at);
    node.title = tick.at;
    node.setAttribute("aria-pressed", String(index === selected));
    return node;
  }));
  return h("div", "watch-day", null, [...rows, lane("对话", talk), lane("值班", ticks),
    h("div", "watch-hours", null, [0, 3, 6, 9, 12, 15, 18, 21, 24].map((hour) => h("span", null, String(hour)))),
    h("p", "watch-legend", "圆点是值班的每一轮：浅色只记录，深色进了待看，大的那个叫醒了主 Agent。对话一行里深色竖线是他说话，蓝色是主动消息。")]);
}

function tickView(data, tick, query) {
  const env = data.lanes.map((item) => {
    const segment = [...item.segments].reverse().find((row) => row.at <= tick.at);
    return segment ? h("li", null, null, [h("span", "watch-key", item.source), h("span", null, segment.text, [h("small", null, ` ${segment.since.slice(5, 16).replace("T", " ")} 起`)])]) : null;
  });
  const verdicts = tick.verdicts.map((item) => h("div", "watch-ev", null, [
    h("div", "watch-key", `${item.source} #${item.id}`),
    h("p", null, null, [h("b", `watch-to watch-to--${item.to}`, TO[item.to]), document.createTextNode(item.title)]),
    item.why ? h("p", "watch-why", item.why) : null]));
  const failures = tick.failures.map((item) => h("p", "watch-why", `模型失败：${item.text}`));
  const rounds = tick.rounds.map((round) => {
    const steps = h("ol", "watch-steps", null, [h("li", "watch-why", "正在读取…")]);
    query("round", { session: round.session }, { cache: "memory" }).then((result) => {
      steps.replaceChildren(...result.steps.map((step) => step.kind === "text"
        ? h("li", "watch-said", step.text)
        : h("li", null, null, [h("code", null, step.tool), document.createTextNode(step.text), step.result ? h("small", null, step.result) : null])));
      if (!result.steps.length) steps.replaceChildren(h("li", "watch-why", "这一轮没有留下记录"));
    }).catch((error) => steps.replaceChildren(h("li", "watch-why", `读取失败：${error instanceof Error ? error.message : error}`)));
    return h("div", null, null, [h("p", "watch-round", `${round.label}：${round.status}`), steps]);
  });
  const memory = tick.memory?.length ? h("div", null, null, [h("p", "watch-round", "它改了记忆"),
    h("ul", "watch-env", null, tick.memory.map((item) => h("li", null, null, [h("span", "watch-key", item.mark === "+" ? "新增" : "删去"), h("span", null, item.text)])))]) : null;
  const column = (title, children, empty) => h("section", "watch-col", null, [h("h4", null, title), ...(children.filter(Boolean).length ? children : [h("p", "watch-why", empty)])]);
  return h("div", "watch-flow", null, [
    h("h3", null, `${tick.at} 这一轮`),
    h("div", "watch-cols", null, [
      column("① 值班员手里有什么（各来源最近一条状态）", [h("ul", "watch-env", null, env)], "还没有来源报告过状态"),
      column("② 新进来的事件和去向", [...verdicts, ...failures], "这一轮没有新事件"),
      column("③ 主 Agent 做了什么", [...rounds, memory], "这一轮没有叫醒它"),
    ])]);
}

function mount(host, context) {
  const state = { data: null, date: null, tab: "today", selected: -1, error: "" };
  const shift = (days) => { const day = new Date(`${state.data.date}T00:00:00Z`); day.setUTCDate(day.getUTCDate() + days); load(day.toISOString().slice(0, 10)); };
  function draw() {
    if (state.error) return host.replaceChildren(h("p", "watch-why", state.error));
    if (!state.data) return host.replaceChildren(h("p", "watch-why", "正在读取…"));
    const data = state.data;
    if (context.panel.size() === "compact") return host.replaceChildren(compactView(data, context.panel));
    const next = button("watch-nav", "›", () => shift(1), "后一天");
    next.disabled = data.date >= data.today;
    const tabs = [["today", "动线"], ...FILES].map(([key, label]) => {
      const node = button("watch-tab", label, () => { state.tab = key; draw(); });
      node.setAttribute("aria-selected", String(state.tab === key));
      return node;
    });
    const tick = data.ticks[state.selected];
    const body = state.tab === "today"
      ? [timeline(data, state.selected, (index) => { state.selected = index; draw(); }), tick ? tickView(data, tick, context.query) : h("p", "watch-why", "这一天还没有值班记录")]
      : [h("pre", "watch-file", data.files[state.tab] || "（空）")];
    host.replaceChildren(h("div", "watch-page", null, [
      h("header", "watch-head", null, [button("watch-nav", "‹", () => shift(-1), "前一天"), h("b", null, data.date), next, h("span", "watch-line", summary(data))]),
      h("div", "watch-tabs", null, tabs), ...body]));
  }
  async function load(date) {
    try {
      state.data = await context.query("day", date ? { date } : {}, { cache: "none" });
      state.error = "";
      const picked = state.data.ticks.findLastIndex((tick) => tick.rounds.length);
      state.selected = picked >= 0 ? picked : state.data.ticks.length - 1;
    } catch (error) {
      state.error = `读取失败：${error instanceof Error ? error.message : error}`;
    }
    draw();
  }
  const stops = [context.onInvalidate(draw), context.onShown(() => load(state.data?.date === state.data?.today ? undefined : state.data?.date))];
  draw();
  void load();
  return () => { for (const stop of stops) stop(); host.replaceChildren(); };
}

const panel = {
  label: "值班",
  iconSvg: ICON,
  async status({ query }) {
    const data = await query("day", {}, { cache: "none" });
    const said = data.talk.filter((row) => row.kind === "push").length;
    return data.ticks.length ? { text: `今天说了 ${said} 句` } : null;
  },
  mount,
};

export default { slots: { "sidebar.panels": panel } };

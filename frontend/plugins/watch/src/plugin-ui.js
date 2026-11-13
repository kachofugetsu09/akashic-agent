import "./plugin-ui.css";

// 值班面板：只认识“来源”“事件”“回合”，不认识任何一种具体的来源；多一个来源就多一行，画法都一样。
const ICON = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><circle cx="8" cy="8" r="5.5"/><path d="M8 5v3l2 1.5"/></svg>';
const TO = { wake: "叫醒", inbox: "待看", log: "记录" };
const WHO = { user: "你说", push: "它对你说" };
const STATUS = { complete: "做完了", timeout: "超时", failed: "失败", error: "出错", cancelled: "被取消" };
const status = (round) => STATUS[round.status] ?? round.status;
const FILES = [["WATCH.md", "嘱托"], ["notes.md", "便签"], ["inbox.md", "待看"]];

function h(tag, className, text, children) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  for (const child of children ?? []) if (child) node.append(child);
  return node;
}
const left = (at) => `${((Number(at.slice(0, 2)) * 60 + Number(at.slice(3, 5))) / 1440 * 100).toFixed(2)}%`;
const fmt = (value) => value >= 1e9 ? `${(value / 1e9).toFixed(2)}B` : value >= 1e6 ? `${(value / 1e6).toFixed(1)}M`
  : value >= 1e4 ? `${Math.round(value / 1e3)}k` : value >= 1e3 ? `${(value / 1e3).toFixed(1)}k` : String(value);
const hitRate = (item) => item.cache_reported_calls && item.hit_input_tokens
  ? `${Math.round(item.hit_tokens / item.hit_input_tokens * 100)}%` : "—";
const usageText = (item) => item.reported_calls
  ? `${item.calls} 次 · 输入 ${fmt(item.input_tokens)} · 命中 ${hitRate(item)} · 输出 ${fmt(item.output_tokens)}`
  : `${item.calls} 次 · 用量未知`;
const briefUsage = (label, total) => !total?.calls ? `${label} 无`
  : total.reported_calls ? `${label} ${fmt(total.input_tokens)}（命中 ${hitRate(total)}）`
  : `${label} ${total.calls} 次`;
const allRounds = (data) => data.ticks.flatMap((tick) => tick.rounds.map((round) => ({ ...round, at: tick.at, tick })));
const allVerdicts = (data) => data.ticks.flatMap((tick) => tick.verdicts.map((item) => ({ ...item, at: tick.at })));

function usageList(models) {
  if (!models?.length) return null;
  return h("ul", "watch-usage-list", null, models.map((item) =>
    h("li", null, null, [h("span", "watch-model", item.model), h("span", "watch-stat", usageText(item))])));
}

function summary(data) {
  const said = data.talk.filter((row) => row.kind === "push").length;
  return `看了 ${allVerdicts(data).length} 件事，叫醒主 Agent ${allRounds(data).length} 次，对你说了 ${said} 句`;
}

function button(className, text, onClick, label) {
  const node = h("button", className, text);
  node.type = "button";
  if (label) node.setAttribute("aria-label", label);
  node.addEventListener("click", onClick);
  return node;
}

function section(title, hint, children) {
  return h("section", "watch-sec", null, [h("h3", null, title, [hint ? h("small", null, hint) : null]), ...children]);
}

function compactView(data, panel) {
  const rounds = allRounds(data).map((round) => ({ at: round.at, text: `${round.label}：${round.said || status(round)}` }));
  const said = data.talk.filter((row) => row.kind === "push").map((row) => ({ at: row.at, text: `说了一句：${row.text}` }));
  const recent = [...rounds, ...said].sort((a, b) => b.at.localeCompare(a.at)).slice(0, 4);
  const hasUsage = data.usage?.watcher?.total?.calls || data.usage?.agent?.total?.calls;
  return h("div", "watch-compact", null, [
    h("p", "watch-line", `今天${summary(data)}`),
    h("p", "watch-sub", `每 ${data.interval} 分钟看一次 · 上次 ${data.ticks.at(-1)?.at ?? "还没有"}`),
    hasUsage ? h("p", "watch-sub watch-stat", `${briefUsage("值班", data.usage.watcher.total)} · ${briefUsage("唤醒", data.usage.agent.total)}`) : null,
    recent.length ? h("ul", "watch-mini", null, recent.map((row) => h("li", null, null, [h("time", null, row.at), h("span", null, row.text)]))) : null,
    button("watch-more", "看全部", () => panel.setSize("expanded")),
  ]);
}

// 一个回合展开后的内容：它收尾时的交代、改了记忆的哪几行、当时各来源报的状态、一步一步做了什么。
function roundBody(data, round, query) {
  const steps = h("ol", "watch-steps", null, [h("li", "watch-why", "正在读取…")]);
  query("round", { session: round.session }, { cache: "memory" }).then((result) => {
    steps.replaceChildren(...result.steps.map((step) => step.kind === "text"
      ? h("li", "watch-said", step.text)
      : h("li", null, null, [h("code", null, step.tool), document.createTextNode(step.text), step.result ? h("small", null, step.result) : null])));
    if (!result.steps.length) steps.replaceChildren(h("li", "watch-why", "这一轮没有留下记录"));
  }).catch((error) => steps.replaceChildren(h("li", "watch-why", `读取失败：${error instanceof Error ? error.message : error}`)));
  const memory = round.tick.memory.map((item) => h("li", null, null, [h("span", "watch-key", item.mark === "+" ? "新增" : "删去"), h("span", null, item.text)]));
  const env = data.lanes.map((item) => {
    const segment = [...item.segments].reverse().find((row) => row.at <= round.at);
    return segment ? h("li", null, null, [h("span", "watch-key", item.source), h("span", null, segment.text)]) : null;
  }).filter(Boolean);
  const fold = (title, body) => h("details", "watch-fold", null, [h("summary", null, title), body]);
  return h("div", "watch-round-body", null, [
    round.said ? h("p", "watch-said", round.said) : h("p", "watch-why", "它这一轮没有留下交代"),
    memory.length ? h("div", null, null, [h("h4", null, "它改了记忆"), h("ul", "watch-env", null, memory)]) : null,
    env.length ? fold(`当时各来源报告的状态（${env.length}）`, h("ul", "watch-env", null, env)) : null,
    fold(round.steps ? `一步一步做了什么（${round.steps} 步）` : "一步一步做了什么", steps),
    usageList(round.usage?.models),
  ]);
}

function roundsView(data, query) {
  const rounds = allRounds(data).reverse();
  if (!rounds.length) return h("p", "watch-why", "这一天没有叫醒过主 Agent");
  return h("div", "watch-rounds", null, rounds.map((round) => {
    const brief = (round.said || status(round)).split("\n")[0];
    const meta = [round.steps ? `${round.steps} 步` : "", round.tick.memory.length ? "改了记忆" : "", round.status === "complete" ? "" : status(round)].filter(Boolean).join(" · ");
    const card = h("details", "watch-round", null, [h("summary", null, null, [
      h("time", null, round.at), h("b", null, round.label), h("span", "watch-brief", brief), h("span", "watch-stat", meta)])]);
    card.addEventListener("toggle", () => { if (card.open && card.childElementCount === 1) card.append(roundBody(data, round, query)); });
    return card;
  }));
}

// 每个来源一行，画法相同：竖线落在事件被判断的时刻，越要紧越粗越高。点一行，下面的事件只看这个来源。
function sourcesView(data, state, redraw) {
  const groups = new Map();
  for (const item of allVerdicts(data)) groups.set(item.source, [...(groups.get(item.source) ?? []), item]);
  const weight = (items) => items.filter((item) => item.to === "wake").length * 1e6 + items.filter((item) => item.to === "inbox").length * 1e3 + items.length;
  const band = (marks) => h("span", "watch-band", null, marks.map(([at, kind, title]) => {
    const node = h("i", `watch-mark watch-mark--${kind}`);
    node.style.left = left(at);
    node.title = `${at} ${title}`;
    return node;
  }));
  const fixed = (name, marks, count) => h("div", "watch-lane", null, [h("span", "watch-lane-name", name), band(marks), h("span", "watch-stat", count)]);
  const rows = [...groups].sort((a, b) => weight(b[1]) - weight(a[1])).map(([source, items]) => {
    const kept = items.filter((item) => item.to !== "log").length;
    const row = button("watch-lane", null, () => { state.source = state.source === source ? "" : source; redraw(); });
    row.append(h("span", "watch-lane-name", source), band(items.map((item) => [item.at, item.to, `${TO[item.to]}：${item.title}`])),
      h("span", "watch-stat", kept ? `${items.length} 条 · ${kept} 条要紧` : `${items.length} 条`));
    row.setAttribute("aria-pressed", String(state.source === source));
    row.title = `只看 ${source} 的事件`;
    return row;
  });
  const rounds = allRounds(data);
  return h("div", "watch-day", null, [
    h("div", "watch-hours", null, [0, 3, 6, 9, 12, 15, 18, 21, 24].map((hour) => h("span", null, String(hour)))),
    fixed("你和它的对话", data.talk.map((row) => [row.at, row.kind, `${WHO[row.kind]}：${row.text}`]), `${data.talk.length} 句`),
    fixed("主 Agent 醒来", rounds.map((round) => [round.at, "wake", round.label]), `${rounds.length} 次`),
    ...rows,
    h("p", "watch-legend", null, [["log", "只记录"], ["inbox", "进了待看"], ["wake", "叫醒了主 Agent"], ["push", "它对你说"]]
      .map(([kind, text]) => h("span", null, null, [h("i", `watch-mark watch-mark--${kind}`), document.createTextNode(text)]))),
  ]);
}

// 事件流：新的在上。默认只留要紧的（进了待看、叫醒了主 Agent）和你们之间的对话，只记录的收起来。
function feedView(data, state, redraw) {
  const verdicts = allVerdicts(data).filter((item) => !state.source || item.source === state.source);
  const kept = verdicts.filter((item) => item.to !== "log");
  const shown = state.all ? verdicts : kept;
  const talk = state.source ? [] : data.talk;
  const rows = [...shown.map((item) => ({ at: item.at, item })), ...talk.map((row) => ({ at: row.at, row }))]
    .sort((a, b) => b.at.localeCompare(a.at)).map(({ at, item, row }) => row
      ? h("div", `watch-row watch-row--${row.kind}`, null, [h("time", null, at), h("span", "watch-key", WHO[row.kind]), h("span", "watch-title", row.text)])
      : h("details", "watch-row", null, [
        h("summary", null, null, [h("time", null, at), h("span", "watch-key", item.source), h("span", "watch-title", item.title), h("b", `watch-to watch-to--${item.to}`, TO[item.to])]),
        h("p", "watch-why", item.why || "值班员没有写理由")]));
  const pick = (all, text) => {
    const node = button("watch-pick", text, () => { state.all = all; redraw(); });
    node.setAttribute("aria-pressed", String(state.all === all));
    return node;
  };
  return h("div", "watch-feed", null, [
    h("div", "watch-picks", null, [pick(false, `要紧的 ${kept.length}`), pick(true, `全部 ${verdicts.length}`),
      state.source ? button("watch-pick watch-pick--source", `只看 ${state.source} ×`, () => { state.source = ""; redraw(); }, "取消只看这个来源") : null]),
    rows.length ? h("div", null, null, rows) : h("p", "watch-why", state.all ? "这一天还没有事件" : "没有要紧的事，其余都只记录了"),
  ]);
}

function usageView(data) {
  const card = (label, group, empty) => h("div", "watch-usage-card", null, [h("b", null, label),
    group?.total?.calls ? usageList(group.models) : h("p", "watch-why", empty)]);
  return h("div", "watch-usage-bar", null, [card("值班员", data.usage?.watcher, "还没有调用记录"), card("主 Agent（被叫醒时）", data.usage?.agent, "这一天没有叫醒")]);
}

function mount(host, context) {
  const state = { data: null, tab: "today", source: "", all: false, error: "" };
  const shift = (days) => { const day = new Date(`${state.data.date}T00:00:00Z`); day.setUTCDate(day.getUTCDate() + days); load(day.toISOString().slice(0, 10)); };
  function draw() {
    if (state.error) return host.replaceChildren(h("p", "watch-why", state.error));
    if (!state.data) return host.replaceChildren(h("p", "watch-why", "正在读取…"));
    const data = state.data;
    if (context.panel.size() === "compact") return host.replaceChildren(compactView(data, context.panel));
    const next = button("watch-nav", "›", () => shift(1), "后一天");
    next.disabled = data.date >= data.today;
    const tabs = [["today", "这一天"], ...FILES].map(([key, label]) => {
      const node = button("watch-tab", label, () => { state.tab = key; draw(); });
      node.setAttribute("aria-selected", String(state.tab === key));
      return node;
    });
    // 选来源、切“要紧的/全部”只重画下面两块，上面已经展开的回合不动。
    const sources = h("div");
    const feed = h("div");
    const redraw = () => { sources.replaceChildren(sourcesView(data, state, redraw)); feed.replaceChildren(feedView(data, state, redraw)); };
    redraw();
    const body = state.tab === "today" ? [
      section("主 Agent 做了什么", "点开看它的交代和每一步", [roundsView(data, context.query)]),
      section("各来源什么时候来了事", "点一行只看这个来源", [sources]),
      section("事件和值班员的判断", "点一条看理由", [feed]),
      section("用量", "", [usageView(data)]),
    ] : [h("pre", "watch-file", data.files[state.tab] || "（空）")];
    host.replaceChildren(h("div", "watch-page", null, [
      h("header", "watch-head", null, [button("watch-nav", "‹", () => shift(-1), "前一天"), h("b", null, data.date), next, h("span", "watch-line", summary(data))]),
      h("div", "watch-tabs", null, tabs), ...body]));
  }
  async function load(date) {
    try {
      state.data = await context.query("day", date ? { date } : {}, { cache: "none" });
      state.error = "";
      state.source = "";
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

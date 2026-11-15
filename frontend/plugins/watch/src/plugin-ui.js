import "./plugin-ui.css";

// 值班面板：只认识“来源”“事件”“回合”，不认识任何一种具体的来源；多一个来源就多一行，画法都一样。
const ICON = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><circle cx="8" cy="8" r="5.5"/><path d="M8 5v3l2 1.5"/></svg>';
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

// 面板顶上的一句话：到你面前的有几句，它动过几处，外面的事有多少只记了一笔。
function lede(data) {
  const said = data.talk.filter((row) => row.kind === "push").length;
  const did = allRounds(data).reduce((total, round) => total + round.left.filter((item) => item.kind !== "said").length, 0);
  const verdicts = allVerdicts(data);
  const quiet = verdicts.filter((item) => item.to === "log").length;
  return `它对你说了 ${said} 句话，自己动了 ${did} 处。外面来了 ${verdicts.length} 件事，${quiet} 件只记了一笔。`;
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

// 行内只认两种记号：`代码` 和 **强调**；其余按原文显示。
function rich(text) {
  return String(text).split(/(`[^`]+`)/).filter(Boolean).map((part) => part.startsWith("`") && part.length > 2
    ? h("code", null, part.slice(1, -1)) : document.createTextNode(part.replaceAll("**", "")));
}
const LEFT = { rule: "改了嘱托", note: "记了便签", memory: "改了记忆", file: "改了文件" };

// 一个后台回合一行：它对你说的话直接摆出来，改过的东西逐条列出；点开是当时的状态和每一步。
function roundRow(data, round, query) {
  const said = round.left.filter((item) => item.kind === "said");
  const did = round.left.filter((item) => item.kind !== "said");
  const to = said.length ? ["对你说", "said"] : did.length ? ["悄悄做了", "did"] : ["没留下什么", ""];
  const meta = [round.steps ? `${round.steps} 步` : "", round.status === "complete" ? "" : status(round)].filter(Boolean).join(" · ");
  const row = h("details", "watch-r", null, [h("summary", null, null, [h("time", null, round.at), h("div", null, null, [
    ...said.map((item) => h("p", "watch-push", null, rich(item.text))),
    h("p", null, null, [h("span", "watch-who", round.label), document.createTextNode(did.length || said.length ? "" : (round.said || status(round)).split("\n")[0])]),
    did.length ? h("ul", "watch-left", null, did.map((item) => h("li", null, null, [h("span", null, LEFT[item.kind]),
      h("span", null, null, item.text ? rich(item.text) : [document.createTextNode(item.path)])]))) : null,
    meta ? h("p", "watch-sub", meta) : null,
  ]), h("span", `watch-to watch-to--${to[1]}`, to[0])])]);
  row.addEventListener("toggle", () => { if (row.open && row.childElementCount === 1) row.append(roundBody(data, round, query)); });
  return row;
}

// 只记了一笔的事不逐条占行：两件要紧事之间的折成一行，点开才列出来。
function quietRow(items) {
  const count = new Map();
  for (const item of items) count.set(item.source, (count.get(item.source) ?? 0) + 1);
  const names = [...count].sort((a, b) => b[1] - a[1]).map(([source, total]) => `${source} ${total}`).join("、");
  return h("details", "watch-r watch-r--quiet", null, [
    h("summary", null, null, [h("time"), h("p", "watch-fold", `${items.length} 件只记了一笔：${names}`)]),
    h("div", "watch-more", null, items.map((item) => {
      const line = h("div", null, null, [h("time", null, item.at), h("span", null, item.source), h("span", null, item.title)]);
      line.title = item.why || "值班员没有写理由";
      return line;
    }))]);
}

// 这一天的流水，新的在上：接下来要回来的时刻、你们的对话、后台回合、要紧的事件、各来源报告的状态变化。
function ledgerView(data, query) {
  const rows = [];
  const pushed = allRounds(data).flatMap((round) => round.left.filter((item) => item.kind === "said").map((item) => item.text.slice(0, 40)));
  for (const round of allRounds(data)) rows.push({ at: round.at, rank: 2, node: roundRow(data, round, query) });
  for (const row of data.talk) {
    if (row.kind === "push" && pushed.some((text) => row.text.startsWith(text))) continue;
    rows.push({ at: row.at, rank: 3, node: h("div", "watch-r", null, [h("time", null, row.at),
      row.kind === "push" ? h("p", "watch-push", null, rich(row.text)) : h("p", "watch-you", null, [h("span", "watch-who", "你"), document.createTextNode(row.text)]),
      row.kind === "push" ? h("span", "watch-to watch-to--said", "对你说") : null]) });
  }
  for (const lane of data.lanes) for (const segment of lane.segments) {
    if (segment.at !== "00:00") rows.push({ at: segment.at, rank: 0, node: h("div", "watch-mark", null, [h("time", null, segment.at),
      h("span", null, segment.text, [h("small", null, lane.source)])]) });
  }
  let quiet = [];
  const flush = (at) => { if (quiet.length) rows.push({ at, rank: 1, node: quietRow(quiet.reverse()) }); quiet = []; };
  for (const item of allVerdicts(data)) {
    if (item.to === "log") { quiet.push(item); continue; }
    flush(quiet.at(-1)?.at);
    const row = h("details", "watch-r", null, [h("summary", null, null, [h("time", null, item.at),
      h("p", null, null, [h("span", "watch-who", item.source), document.createTextNode(item.title)]),
      h("span", `watch-to watch-to--${item.to}`, item.to === "wake" ? "叫醒了它" : "进了待看")]), h("p", "watch-sub watch-why", item.why || "值班员没有写理由")]);
    rows.push({ at: item.at, rank: 1, node: row });
  }
  flush(quiet.at(-1)?.at);
  rows.sort((a, b) => b.at.localeCompare(a.at) || b.rank - a.rank);
  const today = data.date === data.today;
  const next = today ? orders(data.files["WATCH.md"]).wake.filter((item) => item.at?.startsWith(data.date) && item.at.slice(11) > data.now)
    .sort((a, b) => b.at.localeCompare(a.at)).map((item) => h("div", "watch-r watch-r--next", null, [h("time", null, item.at.slice(11)),
      h("p", null, null, [h("span", "watch-who", "它会自己回来"), ...rich(item.title)]), h("span", "watch-to", "今天")])) : [];
  if (!rows.length && !next.length) return h("p", "watch-why", "这一天还没有记录");
  return h("div", "watch-ledger", null, [...next, today ? h("div", "watch-now", `现在 ${data.now}`) : null, ...rows.map((row) => row.node)]);
}

// 嘱托文件按它自己的格式读成条目：行首的方括号是到期日、长期和回来的时刻，其余是原话。
function orders(text) {
  const groups = { wake: [], care: [] };
  let current = null;
  for (const line of String(text ?? "").split("\n")) {
    if (line.startsWith("## ")) { current = line.includes("叫醒") ? groups.wake : groups.care; continue; }
    if (!current || !line.startsWith("- ")) continue;
    let rest = line.slice(2);
    const tags = [];
    for (let match; (match = rest.match(/^\[([^\]]+)\]\s*/));) { tags.push(match[1]); rest = rest.slice(match[0].length); }
    rest = rest.replace(/^到点(叫我|查看|提醒)[：:]\s*/, "");
    const cut = rest.search(/[：:]/);
    const short = cut > 0 && cut <= 18;
    current.push({ title: short ? rest.slice(0, cut) : rest.split(/[，。；（]/)[0], body: short ? rest.slice(cut + 1).trim() : rest,
      until: tags.find((tag) => tag.startsWith("到 "))?.slice(2), at: tags.find((tag) => /^\d{4}-\d\d-\d\d \d\d:\d\d$/.test(tag)),
      when: tags.find((tag) => /^(每天|工作日|周末|周.) \d\d:\d\d$/.test(tag)), long: tags.includes("长期") });
  }
  return groups;
}

function ordersView(data, state) {
  const { wake, care } = orders(data.files["WATCH.md"]);
  const days = (value) => Math.round((new Date(value.slice(0, 10)) - new Date(data.today)) / 864e5);
  const ahead = (value) => { const left = days(value); return left < 0 ? "已过" : left === 0 ? "今天" : left === 1 ? "明天" : `${left} 天后`; };
  const timed = wake.filter((item) => item.at || item.when).sort((a, b) => (a.at ?? "9").localeCompare(b.at ?? "9"));
  const item = (row) => h("details", "watch-item", null, [h("summary", null, null, [h("span", null, null, rich(row.title)),
    h("span", `watch-stat${!row.long && row.until && days(row.until) <= 7 ? " watch-soon" : ""}`, row.long ? "一直有效" : row.until ? `${ahead(row.until)}到期` : "")]),
    h("p", null, null, rich(row.body))]);
  const list = (rows, empty) => rows.length ? h("div", "watch-list", null, rows.map(item)) : h("p", "watch-why", empty);
  const raw = button("watch-more", state.raw ? "收起原文" : "查看原文", () => { state.raw = !state.raw; state.redraw(); });
  raw.setAttribute("aria-expanded", String(Boolean(state.raw)));
  return [
    section("到点它会自己回来", `${timed.length} 条`, [timed.length ? h("div", "watch-ledger watch-agenda", null, timed.map((row) =>
      h("div", "watch-r", null, [h("time", null, row.at ? `${row.at.slice(5, 10)} ${row.at.slice(11)}` : row.when), h("p", null, null, rich(row.body)),
        h("span", "watch-to", row.at ? ahead(row.at) : "")]))) : h("p", "watch-why", "没有定下回来的时刻")]),
    section("出了这些事，它会被叫醒", `${wake.length - timed.length} 条`, [list(wake.filter((row) => !row.at && !row.when), "没有叫醒的条件")]),
    section("这些事不打扰你", `${care.length} 条`, [list(care, "没有留意的条目")]),
    raw, state.raw ? h("pre", "watch-file", data.files["WATCH.md"] || "（空）") : null,
  ];
}

// 便签是它自己写的 Markdown：顶层加粗的一行当小标题，下面的条目各占一行，日期单独一列。
function notesView(text) {
  const out = [];
  let rows = null;
  const open = (title) => { rows = h("div", "watch-ledger watch-notes"); out.push(section(title, "", [rows])); };
  for (const line of String(text ?? "").split("\n")) {
    const item = line.match(/^(\s*)- (.+)/);
    if (!item) continue;
    const head = !item[1] && item[2].match(/^\*\*(.+?)\*\*[：:]?\s*(.*)$/);
    if (head) { open(head[1]); if (!head[2]) continue; }
    if (!rows) open("便签");
    const body = head ? head[2] : item[2];
    const dated = body.match(/^\[?(?:\d{4}-)?(\d\d-\d\d)[^\s\]]*(?: \d\d:\d\d)?\]?\s+(.*)$/);
    rows.append(h("div", "watch-r", null, [h("time", null, dated ? dated[1] : ""), h("p", null, null, rich(dated ? dated[2] : body))]));
  }
  return out.length ? out : [h("p", "watch-why", "便签是空的")];
}

function usageView(data) {
  const card = (label, group, empty) => h("div", "watch-usage-card", null, [h("b", null, label),
    group?.total?.calls ? usageList(group.models) : h("p", "watch-why", empty)]);
  return h("div", "watch-usage-bar", null, [card("值班员", data.usage?.watcher, "还没有调用记录"), card("主 Agent（被叫醒时）", data.usage?.agent, "这一天没有叫醒")]);
}

function mount(host, context) {
  const state = { data: null, tab: "today", raw: false, redraw: () => {}, error: "" };
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
    state.redraw = draw;
    const body = state.tab === "today" ? [h("p", "watch-lede", lede(data)), ledgerView(data, context.query), section("用量", "", [usageView(data)])]
      : state.tab === "WATCH.md" ? ordersView(data, state)
      : state.tab === "notes.md" ? notesView(data.files["notes.md"])
      : [h("pre", "watch-file", data.files[state.tab] || "（空）")];
    host.replaceChildren(h("div", "watch-page", null, [
      h("header", "watch-head", null, [button("watch-nav", "‹", () => shift(-1), "前一天"), h("b", null, data.date), next]),
      h("div", "watch-tabs", null, tabs), ...body]));
  }
  async function load(date) {
    try {
      state.data = await context.query("day", date ? { date } : {}, { cache: "none" });
      state.error = "";
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

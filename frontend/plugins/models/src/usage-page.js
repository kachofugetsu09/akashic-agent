const ICON = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 3v16a2 2 0 0 0 2 2h16"/><path d="M7 16l4-4 4 4 5-6"/></svg>';
const RANGES = [[1, "今天"], [7, "7 天"], [30, "30 天"]];
const SOURCE_LABEL = {conversation: "对话", programmatic: "程序调用", wake: "主动消息", drift: "漫游", subagent: "子智能体"};
const MODE_KEY = "akashic.models.quota-mode";
const NUMBER = new Intl.NumberFormat("zh-CN");
const DAY = new Intl.DateTimeFormat("zh-CN", {month: "numeric", day: "numeric"});
const RESET = new Intl.DateTimeFormat("zh-CN", {month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false});

const short = (value) => value >= 1e9 ? `${(value / 1e9).toFixed(2)}B` : value >= 1e6 ? `${(value / 1e6).toFixed(1)}M`
  : value >= 1e4 ? `${Math.round(value / 1e3)}K` : value >= 1e3 ? `${(value / 1e3).toFixed(1)}K` : NUMBER.format(value);
const ratio = (item) => item.prompt_tokens ? `${Math.round(item.hit_tokens / item.prompt_tokens * 100)}%` : "—";
const hourDate = (key) => new Date(`${key}:00:00Z`);

// 本地零点往前推 days-1 天，小时桶是 UTC，归桶时换成本地时间。
function rangeStart(days) {
  const start = new Date();
  start.setHours(0, 0, 0, 0);
  start.setDate(start.getDate() - days + 1);
  return start;
}

function readMode() {
  try { return window.localStorage.getItem(MODE_KEY) === "left" ? "left" : "used"; } catch { return "used"; }
}

function saveMode(mode) {
  try { window.localStorage.setItem(MODE_KEY, mode); } catch { /* 私密窗口等不可写时只在本页生效 */ }
}

// 桶边界按本地日历生成：多天逐日 setDate，今天逐小时直到明天零点，夏令时那天自然是 23 或 25 桶。
function bucketStarts(days) {
  const start = rangeStart(days);
  const end = new Date(start);
  end.setDate(end.getDate() + days);
  const starts = [];
  if (days === 1) for (let at = start.getTime(); at < end.getTime(); at += 3600e3) starts.push(at);
  else for (const day = new Date(start); day < end; day.setDate(day.getDate() + 1)) starts.push(day.getTime());
  return {starts, end: end.getTime()};
}

// 每桶记命中与未命中输入；UTC 小时记录归到它所在的本地桶。
function buckets(hours, days) {
  const {starts, end} = bucketStarts(days);
  const items = starts.map((at) => ({hit: 0, miss: 0, output: 0, at}));
  for (const [key, , prompt, hit, output] of hours) {
    const time = hourDate(key).getTime();
    if (time < starts[0] || time >= end) continue;
    const item = items[starts.findLastIndex((at) => at <= time)];
    item.hit += hit;
    item.miss += prompt - hit;
    item.output += output;
  }
  return items;
}

function totalsSince(hours, days) {
  const start = rangeStart(days);
  const total = {turns: 0, prompt_tokens: 0, hit_tokens: 0, output_tokens: 0, calls: 0};
  for (const [key, turns, prompt, hit, output, calls] of hours) {
    if (hourDate(key) < start) continue;
    total.turns += turns; total.prompt_tokens += prompt; total.hit_tokens += hit; total.output_tokens += output; total.calls += calls;
  }
  return total;
}

// 一张订阅卡：连接名、套餐与账号，每个窗口一条；已用/剩余由页面统一切换。
function quotaCard(quota, name, mode) {
  const card = document.createElement("article");
  card.className = "usage-quota-card";
  card.innerHTML = `<header><strong></strong><small></small></header><div class="usage-quota-windows"></div><p class="usage-quota-note"></p>`;
  card.querySelector("strong").textContent = name;
  card.querySelector("small").textContent = [quota.plan, quota.account].filter(Boolean).join(" · ");
  const rows = card.querySelector(".usage-quota-windows");
  for (const span of quota.windows) {
    const used = Math.max(0, Math.min(100, Math.round(span.usedPercent)));
    const shown = mode === "left" ? 100 - used : used;
    const row = document.createElement("div");
    row.className = "usage-quota-window";
    row.innerHTML = `<span></span><b></b><span class="settings-quota-bar"><i></i></span><small></small>`;
    row.children[0].textContent = span.label;
    row.children[1].textContent = `${mode === "left" ? "剩余" : "已用"} ${shown}%`;
    row.children[2].classList.toggle("is-high", used >= 80);
    row.querySelector("i").style.inlineSize = `${shown}%`;
    row.children[3].textContent = span.resetsAt ? `${RESET.format(new Date(span.resetsAt))} 重置` : "";
    rows.append(row);
  }
  if (quota.balance) rows.insertAdjacentHTML("afterbegin", `<p class="usage-quota-balance"></p>`);
  if (quota.balance) rows.firstElementChild.textContent = `余额 ${quota.balance}`;
  card.querySelector(".usage-quota-note").textContent = [quota.note, quota.error && `最近一次读取失败：${quota.error}`].filter(Boolean).join("；");
  return card;
}

// 按模型或来源的一张小表：名称、输入（带占比条）、命中、输出、调用。
function breakdown(title, groups, label) {
  const section = document.createElement("section");
  section.className = "usage-breakdown";
  section.innerHTML = `<div class="settings-section-bar"><h2 class="settings-section-label"></h2></div><div class="usage-table" role="table"></div>`;
  section.querySelector("h2").textContent = title;
  const table = section.querySelector(".usage-table");
  table.innerHTML = `<div role="row" class="usage-table-head"><span role="columnheader">名称</span><span role="columnheader">输入</span><span role="columnheader">缓存命中</span><span role="columnheader">输出</span><span role="columnheader">调用</span></div>`;
  const max = Math.max(1, ...groups.map((item) => item.prompt_tokens));
  for (const item of groups) {
    const row = document.createElement("div");
    row.setAttribute("role", "row");
    row.innerHTML = `<span role="cell"></span><span role="cell"><b></b><i class="usage-share"><i></i></i></span><span role="cell"></span><span role="cell"></span><span role="cell"></span>`;
    const cells = row.children;
    cells[0].textContent = label(item);
    cells[1].querySelector("b").textContent = short(item.prompt_tokens);
    cells[1].querySelector("i i").style.inlineSize = `${item.prompt_tokens / max * 100}%`;
    cells[2].textContent = ratio(item);
    cells[3].textContent = short(item.output_tokens);
    cells[4].textContent = NUMBER.format(item.calls);
    table.append(row);
  }
  if (!groups.length) table.insertAdjacentHTML("beforeend", `<p class="usage-empty">这段时间没有调用记录。</p>`);
  return section;
}

// 柱状图：每桶一根，深色是未命中输入，浅色是缓存命中；悬停给出数值。
function chart(items, days) {
  const figure = document.createElement("div");
  figure.className = "usage-chart";
  figure.setAttribute("role", "img");
  const max = Math.max(1, ...items.map((item) => item.hit + item.miss));
  const now = Date.now();
  for (const item of items) {
    const bar = document.createElement("span");
    const when = days === 1 ? `${new Date(item.at).getHours()} 时` : DAY.format(new Date(item.at));
    bar.title = `${when} · 输入 ${short(item.hit + item.miss)}（命中 ${short(item.hit)}）· 输出 ${short(item.output)}`;
    bar.dataset.future = String(item.at > now);
    bar.innerHTML = `<i style="block-size:${item.miss / max * 100}%"></i><i class="is-hit" style="block-size:${item.hit / max * 100}%"></i>`;
    figure.append(bar);
  }
  const total = items.reduce((sum, item) => sum + item.hit + item.miss, 0);
  figure.setAttribute("aria-label", `${days === 1 ? "今天每小时" : `近 ${days} 天每天`}的输入 Token，合计 ${short(total)}`);
  const axis = document.createElement("div");
  axis.className = "usage-axis";
  const labels = days === 1 ? ["0 时", "12 时", "23 时"] : [DAY.format(new Date(items[0].at)), DAY.format(new Date(items.at(-1).at))];
  axis.replaceChildren(...labels.map((text) => Object.assign(document.createElement("span"), {textContent: text})));
  const wrap = document.createElement("div");
  wrap.append(figure, axis);
  return wrap;
}

// 订阅额度区：读取失败也留在页面上说明原因，刷新按钮始终可以再点。
function quotaPanel(ctx, page, catalogResource, isClosed) {
  const section = page.querySelector("[data-quota-section]");
  const grid = page.querySelector("[data-quotas]");
  const note = page.querySelector("[data-quota-error]");
  const refresh = page.querySelector("[data-quota-refresh]");
  let mode = readMode();
  let quotas = [];
  const draw = () => {
    const names = new Map((catalogResource.current?.connections ?? []).map((item) => [item.id, item.name]));
    section.hidden = !quotas.length && !note.textContent;
    grid.replaceChildren(...quotas.map((quota) => quotaCard(quota, names.get(quota.connectionId) ?? quota.connectionId, mode)));
    for (const button of page.querySelectorAll("[data-mode]")) button.setAttribute("aria-pressed", String(button.dataset.mode === mode));
  };
  const load = (fresh) => {
    refresh.disabled = true;
    return ctx.http.request(`/api/dashboard/models/quotas${fresh ? "?fresh=true" : ""}`)
      .then((response) => response.ok ? response.json() : Promise.reject(new Error(`额度读取失败（${response.status}）`)))
      .then((body) => { quotas = body.quotas; note.textContent = ""; })
      .catch((error) => { note.textContent = `${quotas.length ? "未能刷新，显示的是上次结果：" : ""}${error.message}`; })
      .finally(() => { if (isClosed()) return; refresh.disabled = false; draw(); });
  };
  const setMode = (next) => { mode = next; saveMode(mode); draw(); };
  return {load, setMode};
}

// Token 与缓存区：每次切换范围换一个代号，迟到的旧回包与它安排的重试都作废。
function usagePanel(ctx, page, isClosed) {
  const status = page.querySelector(".usage-status");
  const content = page.querySelector("[data-usage]");
  let days = 1;
  let usage = null;
  let timer = 0;
  let generation = 0;
  const draw = () => {
    for (const button of page.querySelectorAll("[data-days]")) button.setAttribute("aria-pressed", String(Number(button.dataset.days) === days));
    if (!usage) return;
    const total = totalsSince(usage.hours, usage.days);
    const metrics = document.createElement("div");
    metrics.className = "usage-metrics";
    for (const [value, label] of [[short(total.prompt_tokens), "输入 Token"], [ratio(total), "缓存命中"],
      [short(total.output_tokens), "输出 Token"], [NUMBER.format(total.calls), "模型调用"]]) {
      metrics.insertAdjacentHTML("beforeend", `<div><b>${value}</b><span>${label}</span></div>`);
    }
    content.replaceChildren(metrics, chart(buckets(usage.hours, usage.days), usage.days),
      breakdown("按模型", usage.models, (item) => item.model),
      breakdown("按来源", usage.sources, (item) => SOURCE_LABEL[item.source] ?? item.source));
  };
  // 冷启动时扫描逐次推进：未完成先显示进度，挂载期间退避着接着读。
  const load = (ticket, attempt = 0) => {
    const since = encodeURIComponent(rangeStart(days).toISOString());
    const asked = days;
    ctx.http.request(`/api/dashboard/models/usage?since=${since}`)
      .then((response) => response.ok ? response.json() : Promise.reject(new Error(`用量读取失败（${response.status}）`)))
      .then((body) => {
        if (isClosed() || ticket !== generation) return;
        if (body.complete === false) {
          status.textContent = `正在统计 ${NUMBER.format(body.scanned_sessions)} / ${NUMBER.format(body.total_sessions)} 个会话…`;
          timer = window.setTimeout(() => load(ticket, attempt + 1), Math.min(2000, 400 * (attempt + 1)));
          return;
        }
        status.textContent = "";
        usage = {...body, days: asked};
        draw();
      })
      .catch((error) => {
        if (!isClosed() && ticket === generation) status.textContent = `${usage ? "未能更新，显示的是上次结果：" : ""}${error.message}`;
      });
  };
  const select = (next) => {
    days = next;
    window.clearTimeout(timer);
    status.textContent = "正在读取用量…";
    draw();
    load(++generation);
  };
  return {select, stop: () => { generation += 1; window.clearTimeout(timer); }};
}

// 设置「用量」：订阅额度在上，Token、缓存命中与调用按所选范围汇总在下。
export function usageSection(ctx, catalogResource) {
  return {
    id: "models.usage", label: "用量", iconSvg: ICON, route: "usage", order: 31,
    render(host, _view, props = {}) {
      const page = document.createElement("main");
      page.className = `settings-page models-usage ${props.embedded ? "settings-page--embedded" : ""}`;
      page.innerHTML = `<div class="settings-shell">
        <header class="settings-header"><div><h1>用量</h1><p>订阅额度由各服务商提供；Token 与缓存命中来自模型实报的调用记录。</p></div></header>
        <section class="settings-section" data-quota-section hidden>
          <div class="settings-section-bar"><h2 class="settings-section-label">订阅额度</h2>
            <div class="usage-toggle" role="group" aria-label="额度显示"><button type="button" data-mode="used">已用</button><button type="button" data-mode="left">剩余</button></div>
            <button type="button" class="settings-text-button" data-quota-refresh>刷新</button></div>
          <p class="usage-status" data-quota-error role="status"></p>
          <div class="usage-quota-grid" data-quotas></div>
        </section>
        <section class="settings-section">
          <div class="settings-section-bar"><h2 class="settings-section-label">Token 与缓存</h2>
            <div class="usage-toggle" role="group" aria-label="时间范围">${RANGES.map(([days, label]) => `<button type="button" data-days="${days}">${label}</button>`).join("")}</div></div>
          <p class="usage-status" role="status"></p>
          <div data-usage></div>
        </section>
      </div>`;
      host.replaceChildren(page);
      let closed = false;
      const isClosed = () => closed;
      const quota = quotaPanel(ctx, page, catalogResource, isClosed);
      const usage = usagePanel(ctx, page, isClosed);
      page.addEventListener("click", (event) => {
        const button = event.target instanceof Element ? event.target.closest("button") : null;
        if (!button) return;
        if (button.dataset.mode) quota.setMode(button.dataset.mode);
        else if (button.dataset.days) usage.select(Number(button.dataset.days));
        else if (button.hasAttribute("data-quota-refresh")) void quota.load(true);
      });
      void quota.load(false);
      usage.select(1);
      return () => {
        closed = true;
        usage.stop();
        host.replaceChildren();
      };
    },
  };
}

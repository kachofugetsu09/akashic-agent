const ICON = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 3v16a2 2 0 0 0 2 2h16"/><path d="M7 16l4-4 4 4 5-6"/></svg>';

const number = (value) => new Intl.NumberFormat("zh-CN").format(Number(value || 0));
const rateOf = (totals) => Number(totals?.prompt_tokens) > 0 ? Number(totals.hit_tokens) / Number(totals.prompt_tokens) : null;
const percent = (value) => typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "—";

function shortTime(value) {
  const date = new Date(String(value || ""));
  if (Number.isNaN(date.getTime())) return String(value || "—");
  return new Intl.DateTimeFormat("zh-CN", {month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false}).format(date);
}

function metric(element, totals, title) {
  const turns = Number(totals?.turns || 0);
  element.classList.toggle("empty", turns === 0);
  element.querySelector("strong").textContent = turns === 0 ? "暂无记录" : percent(rateOf(totals));
  element.querySelector("span").textContent = turns === 0 ? title : `${title} · ${number(turns)} 轮`;
}

// 每轮一行；展开后显示会话与 Prompt / Hit / Miss 原始数。
function turnRow(turn) {
  const item = document.createElement("article");
  item.className = "models-cache-turn";
  const rate = rateOf(turn);
  item.innerHTML = `<button type="button" aria-expanded="false">
      <span class="models-cache-turn__copy"><strong></strong><small></small></span>
      <span class="models-cache-turn__values"><strong></strong><small></small></span>
    </button><dl hidden></dl>`;
  const [title, meta, value, tokens] = item.querySelectorAll("button strong, button small");
  title.textContent = turn.preview || "（无内容）";
  meta.textContent = `${turn.proactive ? "主动" : "被动"} · ${shortTime(turn.recorded_at)}`;
  value.textContent = percent(rate);
  value.classList.toggle("low", typeof rate === "number" && rate < 0.5);
  tokens.textContent = `${number(turn.hit_tokens)} / ${number(turn.prompt_tokens)}`;
  const detail = item.querySelector("dl");
  const miss = Math.max(0, Number(turn.prompt_tokens) - Number(turn.hit_tokens));
  for (const [label, text] of [["Session", turn.session_id], ["Prompt", number(turn.prompt_tokens)],
    ["Hit", number(turn.hit_tokens)], ["Miss", number(miss)], ["时间", shortTime(turn.recorded_at)]]) {
    const term = document.createElement("dt");
    const description = document.createElement("dd");
    term.textContent = label;
    description.textContent = text;
    detail.append(term, description);
  }
  const trigger = item.querySelector("button");
  trigger.addEventListener("click", () => {
    detail.hidden = !detail.hidden;
    trigger.setAttribute("aria-expanded", String(!detail.hidden));
  });
  return item;
}

function render(page, usage) {
  const recent = rateOf(usage.recent_passive);
  const ring = page.querySelector(".models-cache-ring");
  ring.style.setProperty("--models-cache-rate", `${Math.max(0, Math.min(1, recent || 0)) * 100}%`);
  ring.querySelector("strong").textContent = percent(recent);
  metric(page.querySelector("[data-passive]"), usage.passive, "被动总览");
  metric(page.querySelector("[data-proactive]"), usage.proactive, "主动链路");
  page.querySelector("[data-total]").textContent = `${number(usage.total)} 轮`;
  const turns = Array.isArray(usage.turns) ? usage.turns : [];
  const list = page.querySelector("[data-turns]");
  if (turns.length) list.replaceChildren(...turns.map(turnRow));
  else list.innerHTML = '<p class="models-cache-empty">暂无 KV Cache 记录。</p>';
}

// 设置中的“缓存效率”分节：打开时按需计算，不在后台持续同步。
export function cacheUsageSection(ctx) {
  return {
    id: "models.cache-usage", label: "缓存效率", iconSvg: ICON, route: "cache-usage", order: 31,
    render(host) {
      let active = true;
      const page = document.createElement("main");
      page.className = "settings-page models-cache";
      page.innerHTML = `<div class="settings-shell">
        <header class="settings-header"><div><h1>缓存效率</h1><p>按 Turn 汇总模型调用的 KV Cache 命中，数据来自模型调用记录。</p></div></header>
        <p class="models-cache-status" role="status">正在读取 KV Cache…</p>
        <div data-content hidden>
          <section class="models-cache-overview" aria-label="KV Cache 概览">
            <div class="models-cache-current"><div class="models-cache-ring"><strong>—</strong></div><span>近期被动复用</span></div>
            <div class="models-cache-sources"><div data-passive><strong>—</strong><span>被动总览</span></div><div data-proactive><strong>—</strong><span>主动链路</span></div></div>
          </section>
          <section class="settings-section"><header><h2>最近 Turn</h2><span data-total></span></header><div data-turns></div></section>
        </div>
      </div>`;
      host.replaceChildren(page);
      async function load() {
        // 每次读取只推进一段历史；离开页面后不再发起下一次请求。
        while (active) {
          const response = await ctx.http.request("/api/dashboard/models/cache-usage");
          if (!response.ok) throw new Error(`读取失败（${response.status}）`);
          const usage = await response.json();
          if (!active) return;
          if (!usage.complete) {
            page.querySelector(".models-cache-status").textContent =
              `正在读取历史记录（${number(usage.scanned_sessions)} / ${number(usage.total_sessions)} 个会话）…`;
            continue;
          }
          render(page, usage);
          page.querySelector(".models-cache-status").remove();
          page.querySelector("[data-content]").hidden = false;
          return;
        }
      }
      load().catch((error) => {
        if (!active) return;
        const status = page.querySelector(".models-cache-status");
        status.classList.add("error");
        status.textContent = `KV Cache 读取失败：${error instanceof Error ? error.message : String(error)}`;
      });
      return () => { active = false; host.replaceChildren(); };
    },
  };
}

let d = null;
async function h(a, s) {
  if (!d) throw new Error("Akasha 工作台面板未激活");
  const e = await d(a, s), n = await e.json();
  if (!e.ok) throw new Error(String(n.detail ?? n.message ?? `HTTP ${e.status}`));
  return n;
}
function t(a) {
  return String(a).replace(/[&<>"']/g, (s) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  })[s] ?? s);
}
function $(a) {
  return a.split("/").map(encodeURIComponent).join("/");
}
function p(a) {
  if (!a) return "—";
  const s = new Date(String(a));
  return Number.isNaN(s.getTime()) ? String(a) : new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: !1
  }).format(s);
}
function u(a, s = 3) {
  if (a == null || a === "") return "—";
  const e = Number(a);
  return Number.isFinite(e) ? e.toFixed(s) : "—";
}
function g(a) {
  const s = a.sources ?? [];
  return s.length ? s.join(" · ") : a.first_relation ? a.first_relation : a.graph_only ? "仅由关系补全" : "外部线索";
}
function m(a, s) {
  return a.length ? `
    <ol class="akasha-evidence-list">
      ${a.map((e, n) => {
    var o;
    const r = e.score ?? e.value ?? e.completion_mass ?? e.seed_score, l = (o = e.relation_path) != null && o.length ? `<span class="akasha-path" title="${t(e.relation_path.join(" → "))}">${t(e.relation_path.join(" → "))}</span>` : "";
    return `
          <li class="akasha-evidence">
            <span class="akasha-evidence-rank" aria-hidden="true">${n + 1}</span>
            <div class="akasha-evidence-main">
              <p>${t(e.user_text || "（空消息）")}</p>
              ${e.assistant_preview ? `<p class="akasha-assistant">${t(e.assistant_preview)}</p>` : ""}
            </div>
            <div class="akasha-evidence-meta">
              <time class="akasha-chip akasha-chip--time">${t(p(e.ts))}</time>
              <span class="akasha-chip">${t(g(e))}</span>
              ${r == null ? "" : `<b class="akasha-chip akasha-chip--score">${u(r)}</b>`}
            </div>
            ${l}
          </li>
        `;
  }).join("")}
    </ol>
  ` : `<p class="akasha-empty">${t(s)}</p>`;
}
function c(a, s, e, n, r, l, o = !1) {
  return `
    <details class="akasha-section akasha-lane akasha-lane--${t(e)}" ${o ? "open" : ""}>
      <summary>
        <span class="akasha-lane-copy">
          <strong>${t(a)}</strong>
          <small>${t(s)}</small>
        </span>
        <span class="akasha-lane-count">${t(String(r))}</span>
      </summary>
      ${m(n, l)}
    </details>
  `;
}
function i(a, s, e) {
  return `
    <div class="akasha-metric">
      <dt>${t(a)}</dt>
      <dd>${t(String(s))}</dd>
      <p>${t(e)}</p>
    </div>
  `;
}
function y(a, s) {
  const e = s.filters.q ?? "", n = a.querySelector("[data-akasha-search]");
  if (n) {
    document.activeElement !== n && n.value !== e && (n.value = e);
    return;
  }
  a.innerHTML = `
    <div class="akasha-filter">
      <label>
        <span>搜索检索记录</span>
        <input
          type="search"
          value="${t(e)}"
          placeholder="Query、回复或 Session"
          data-akasha-search
        />
      </label>
      <md-text-button data-akasha-clear ${e ? "" : "disabled"}>清空</md-text-button>
    </div>
  `;
  const r = a.querySelector("[data-akasha-search]"), l = a.querySelector("[data-akasha-clear]");
  let o = 0;
  const v = () => {
    window.clearTimeout(o), o = window.setTimeout(() => {
      const _ = r.value.trim();
      _ ? s.setFilter("q", _) : s.clearFilter("q");
    }, 200);
  }, k = () => {
    r.value = "", s.clearFilter("q");
  };
  return r.addEventListener("input", v), l.addEventListener("click", k), () => {
    window.clearTimeout(o), r.removeEventListener("input", v), l.removeEventListener("click", k);
  };
}
function f(a, s) {
  const e = a.recall_capture_available ? a.left_count + a.right_count : a.left_count;
  return `
    <article class="akasha-inspector">
      <header class="akasha-query">
        <div>
          <h2>${t(a.query_text)}</h2>
          <p class="akasha-query-meta">${t(p(a.ts))} · seq ${a.seq}<span>${t(a.session_key)}</span></p>
        </div>
        ${s ? '<md-icon-button class="akasha-close" data-akasha-close aria-label="关闭详情"><span aria-hidden="true">×</span></md-icon-button>' : ""}
      </header>

      <section class="akasha-overview" aria-labelledby="akasha-overview-title">
        <div class="akasha-overview-heading">
          <div>
            <h3 id="akasha-overview-title">${e} 条记忆参与回答</h3>
          </div>
          <p>${a.inject_chars > 0 ? `已写入 ${a.inject_chars} 字上下文` : "没有写入 Prompt"}</p>
        </div>
        <dl class="akasha-metrics">
          ${i("直接线索", a.seed_count, "Dense、BM25 与时序")}
          ${i("精确回忆", a.left_count, "语义最接近的历史")}
          ${i("模式联想", a.recall_capture_available ? a.right_count : "—", a.recall_capture_available ? `${a.basin_count} 个情景簇` : "本轮未记录")}
        </dl>
      </section>

      <details class="akasha-answer">
        <summary>
          <span><strong>助手回复</strong><small>查看这一轮的完整回答</small></span>
          <span class="akasha-answer-action">展开</span>
        </summary>
        <div class="akasha-answer-body">${t(a.assistant_text || "（助手没有文本回复）")}</div>
      </details>

      <section class="akasha-evidence-group" aria-labelledby="akasha-evidence-title">
        <div class="akasha-section-heading">
          <h3 id="akasha-evidence-title">记忆证据</h3>
          <small>选择一组展开查看</small>
        </div>
        <div class="akasha-lanes">
        ${c("直接线索", "最初命中的消息", "seed", a.seeds, a.seeds.length, "这一轮没有形成可持久化线索。")}
        ${a.activation_capture_available ? c("图扩散候选", "由关系网络补入的候选", "activation", a.activation_items, a.activation_items.length, "图扩散没有增加候选。") : ""}
        ${c("精确回忆", "语义最接近的历史消息", "precise", a.left, a.left_count, "没有精确命中。")}
        ${c("模式联想", "跨关系补全且已与精确结果去重", "completion", a.right, a.recall_capture_available ? a.right_count : "未记录", "没有产生模式联想。")}
        ${a.tool_left_count ? c("工具精确回忆", "recall_memory 的语义命中", "precise", a.tool_left, a.tool_left_count, "工具没有产生精确命中。") : ""}
        ${a.tool_right_count ? c("工具模式联想", "recall_memory 的图关系结果", "completion", a.tool_right, a.tool_right_count, "工具没有产生模式联想。") : ""}
        </div>
      </section>

      <details class="akasha-learning">
        <summary><span><strong>学习变化与技术指标</strong><small>${a.activation_count} 条扩散候选 · ${a.pushes} 次扩散</small></span></summary>
        <dl>
          ${i("惊喜度", u(a.surprise), "当前 cue 与已有模式的差异")}
          ${i("观察质量", u(a.observed_mass), "由外部证据支持的学习质量")}
          ${i("再激活", u(a.reactivated_mass), "已有关系重新获得的活性")}
          ${i("增强 / 抑制", `${u(a.potentiated_mass)} / ${u(a.inhibited_mass)}`, "连接预算内的竞争结果")}
        </dl>
      </details>

      <details class="akasha-prompt">
        <summary><span><strong>写入 Prompt 的记忆</strong><small>${a.inject_chars} 字 · 原始上下文预览</small></span></summary>
        <pre>${t(a.text_block_preview || "这一轮没有注入记忆。")}</pre>
      </details>
    </article>
  `;
}
const b = {
  id: "akasha-inspector",
  label: "Akasha 检索",
  viewLabel: "Akasha 检索",
  pageSize: 25,
  rowKey: "query_id",
  countTitle(a) {
    return `${a} 轮检索`;
  },
  columns: [
    { key: "session_key", label: "会话", width: 120, fmt: "mono-session", cellClass: "mono cell-session", rawTitle: !0 },
    {
      key: "ts",
      label: "时间",
      width: 110,
      cellClass: "mono cell-time",
      rawTitle: !0,
      renderCell(a) {
        return t(p(a));
      }
    },
    { key: "query_text", label: "用户问题", flex: !0, fmt: "text-preview", cellClass: "content-preview" },
    { key: "seed_count", label: "线索", width: 64, fmt: "metric", cellClass: "mono cell-metric", align: "right" },
    { key: "completion_count", label: "召回", width: 64, fmt: "metric", cellClass: "mono cell-metric", align: "right" }
  ],
  renderFilters: y,
  async getCount({ signal: a }) {
    try {
      const s = await h("/api/dashboard/akasha-inspector/overview", { signal: a });
      return s.available ? s.total : null;
    } catch (s) {
      if (a.aborted) throw s;
      return null;
    }
  },
  async fetchPage({ page: a, pageSize: s, filters: e, signal: n }) {
    const r = new URLSearchParams({
      page: String(a),
      page_size: String(s)
    });
    e != null && e.session_key && r.set("session_key", e.session_key), e != null && e.q && r.set("q", e.q);
    const l = await h(
      `/api/dashboard/akasha-inspector/turns?${r.toString()}`,
      { signal: n }
    );
    return { items: l.items, total: l.total };
  },
  async fetchDetail(a, { signal: s }) {
    return h(
      `/api/dashboard/akasha-inspector/turns/${$(String(a.query_id ?? ""))}`,
      { signal: s }
    );
  },
  renderDetail(a, s, e) {
    var r;
    if (!a) {
      s.innerHTML = `
        <div class="akasha-detail-empty">
          <div class="akasha-detail-empty__title">Akasha Inspector</div>
          <div class="akasha-detail-empty__text">选择一轮检索，查看它从哪些线索开始、扩散到哪里，以及最终进入 Prompt 的内容。</div>
        </div>
      `;
      return;
    }
    s.innerHTML = f(
      a,
      e == null ? void 0 : e.closePane
    ), (r = s.querySelector("[data-akasha-close]")) == null || r.addEventListener(
      "click",
      () => {
        var l;
        return (l = e == null ? void 0 : e.closePane) == null ? void 0 : l.call(e);
      }
    );
    const n = Array.from(s.querySelectorAll(".akasha-lane"));
    for (const l of n)
      l.addEventListener("toggle", () => {
        if (l.open)
          for (const o of n)
            o !== l && (o.open = !1);
      });
  }
};
function w(a) {
  d = a.http.request;
  const s = a.ui.inject("workbench.panels.v2", (e) => e.register(b));
  return () => {
    s(), d = null;
  };
}
export {
  w as activate
};

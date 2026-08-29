const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;");

function shortTime(value) {
  const parsed = new Date(String(value || ""));
  if (Number.isNaN(parsed.getTime())) return String(value || "—");
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(parsed);
}

function score(item) {
  const value = Number(item.score);
  return Number.isFinite(value) ? value.toFixed(3) : "";
}

function memoryList(items, empty) {
  if (!items.length) {
    return `<p class="akasha-mobile-empty">${escapeHtml(empty)}</p>`;
  }
  return `
    <ol class="akasha-mobile-memories">
      ${items.map((item) => `
        <li>
          <div>
            <p>${escapeHtml(item.user_preview || item.user_text || "（空消息）")}</p>
            ${item.assistant_preview
              ? `<p class="akasha-mobile-assistant">${escapeHtml(item.assistant_preview)}</p>`
              : ""}
          </div>
          <span>${escapeHtml(shortTime(item.ts))}${score(item) ? ` · ${score(item)}` : ""}</span>
        </li>
      `).join("")}
    </ol>
  `;
}

function recallSection(title, items, lane, captureAvailable = true) {
  const count = captureAvailable ? items.length : "未记录";
  const empty = captureAvailable
    ? "本轮没有命中"
    : "这一轮没有保存模式补全读出";
  return `
    <details class="akasha-mobile-recall akasha-mobile-recall--${escapeHtml(lane)}" ${items.length ? "open" : ""}>
      <summary>
        <span>${escapeHtml(title)}</span>
        <b>${escapeHtml(count)}</b>
      </summary>
      ${memoryList(items, empty)}
    </details>
  `;
}

function renderRecent(items, total) {
  if (!items.length) {
    return `
      <section class="akasha-mobile-inspector">
        <header>
          <h2>Akasha Inspector</h2>
          <p>还没有可检查的检索记录。</p>
        </header>
      </section>
    `;
  }
  return `
    <section class="akasha-mobile-inspector">
      <header>
        <h2>Akasha Inspector</h2>
        <p>最近 30 轮，共 ${Number(total || 0)} 轮。点开一轮查看线索和模式补全。</p>
      </header>
      <ol class="akasha-mobile-turns">
        ${items.map((item) => `
          <li>
            <button type="button" data-akasha-query="${escapeHtml(item.query_id)}">
              <span>${escapeHtml(item.query_preview || item.query_text || "（空消息）")}</span>
              <small>${escapeHtml(shortTime(item.ts))} · ${Number(item.seed_count || 0)} seeds · ${item.recall_capture_available === false ? "未记录 recall" : `${Number(item.completion_count || 0)} recall`}</small>
            </button>
          </li>
        `).join("")}
      </ol>
    </section>
  `;
}

function renderDetail(item) {
  const left = Array.isArray(item.left) ? item.left : [];
  const right = Array.isArray(item.right) ? item.right : [];
  const toolLeft = Array.isArray(item.tool_left) ? item.tool_left : [];
  const toolRight = Array.isArray(item.tool_right) ? item.tool_right : [];
  const recallCaptured = item.recall_capture_available !== false;
  return `
    <section class="akasha-mobile-inspector">
      <button class="akasha-mobile-back" type="button" data-akasha-back>返回检索列表</button>
      <header>
        <p class="akasha-mobile-time">${escapeHtml(shortTime(item.ts))}</p>
        <h2>${escapeHtml(item.query_text || "（空消息）")}</h2>
        <dl class="akasha-mobile-metrics">
          <div><dt>线索</dt><dd>${Number(item.seed_count || 0)}</dd></div>
          <div><dt>补全</dt><dd>${Number(item.activation_count || 0)}</dd></div>
          <div><dt>左脑</dt><dd>${left.length}</dd></div>
          <div><dt>右脑</dt><dd>${recallCaptured ? right.length : "—"}</dd></div>
        </dl>
      </header>
      <div class="akasha-mobile-detail-lanes">
        ${recallSection("左脑 · 精确回忆", left, "precise")}
        ${recallSection("右脑 · 模式补全", right, "completion", recallCaptured)}
        ${toolLeft.length ? recallSection("工具回忆 · 精确回忆", toolLeft, "precise") : ""}
        ${toolRight.length ? recallSection("工具回忆 · 模式补全", toolRight, "completion") : ""}
      </div>
      <p class="akasha-mobile-convergence">${Number(item.pushes || 0)} 次扩散，残余 ${Number(item.residual_l1 || 0).toExponential(2)}</p>
    </section>
  `;
}

function mountRecall(host, context) {
  if (!context.capabilities?.queryTransports?.includes("https")) {
    host.innerHTML = '<p class="akasha-mobile-error">当前客户端版本不支持 Akasha 轻量数据通道，请更新后查看本轮记忆。</p>';
    return undefined;
  }
  host.innerHTML = '<p class="akasha-mobile-loading">正在读取本轮记忆…</p>';
  let active = true;
  let retryTimer;
  const activeMessage = context.messageId.startsWith("assistant:");
  const waitToRetry = () => new Promise((resolve) => {
    retryTimer = setTimeout(resolve, 250);
  });
  const load = async () => {
    while (active) {
      const result = await context.query(
        "recall.current",
        { message_id: context.messageId },
        { cache: activeMessage ? "none" : "immutable", transport: "https" },
      );
      if (!active) return;
      const left = Array.isArray(result.left) ? result.left : [];
      const right = Array.isArray(result.right) ? result.right : [];
      const toolLeft = Array.isArray(result.tool_left) ? result.tool_left : [];
      const toolRight = Array.isArray(result.tool_right) ? result.tool_right : [];
      const recallCaptured = result.recall_capture_available !== false;
      if (result.pending !== true || result.recall_capture_available === true) {
        host.innerHTML = `
          <div class="akasha-mobile-recall-group">
            ${recallSection("左脑 · 精确回忆", left, "precise")}
            ${recallSection("右脑 · 模式补全", right, "completion", recallCaptured)}
            ${toolLeft.length ? recallSection("工具回忆 · 精确回忆", toolLeft, "precise") : ""}
            ${toolRight.length ? recallSection("工具回忆 · 模式补全", toolRight, "completion") : ""}
          </div>
        `;
      }
      if (result.pending === true) {
        await waitToRetry();
        continue;
      }
      return;
    }
  };
  load().catch((error) => {
    if (active) {
      host.innerHTML = `<p class="akasha-mobile-error">${escapeHtml(error.message || "记忆读取失败")}</p>`;
    }
  });
  return () => {
    active = false;
    if (retryTimer !== undefined) clearTimeout(retryTimer);
  };
}

function mountInspector(host, context) {
  let active = true;
  let recent = null;
  host.innerHTML = '<p class="akasha-mobile-loading">正在读取检索记录…</p>';

  const showRecent = () => {
    if (!recent) return;
    host.innerHTML = renderRecent(recent.items, recent.total);
    host.querySelectorAll("[data-akasha-query]").forEach((button) => {
      button.addEventListener("click", () => {
        const queryId = button.getAttribute("data-akasha-query");
        if (!queryId) return;
        host.innerHTML = '<p class="akasha-mobile-loading">正在读取这一轮…</p>';
        context.query("inspector.detail", { query_id: queryId }).then((item) => {
          if (!active) return;
          host.innerHTML = renderDetail(item);
          host.querySelector("[data-akasha-back]")?.addEventListener("click", showRecent);
        }).catch((error) => {
          if (active) {
            host.innerHTML = `<p class="akasha-mobile-error">${escapeHtml(error.message || "检索记录读取失败")}</p>`;
          }
        });
      });
    });
  };

  context.query("inspector.recent").then((result) => {
    if (!active) return;
    recent = {
      items: Array.isArray(result.items) ? result.items : [],
      total: Number(result.total || 0),
    };
    showRecent();
  }).catch((error) => {
    if (active) {
      host.innerHTML = `<p class="akasha-mobile-error">${escapeHtml(error.message || "检索记录读取失败")}</p>`;
    }
  });
  return () => { active = false; };
}

export default {
  slots: {
    "turn.before_reasoning": {
      mount: mountRecall,
    },
  },
  dashboard: {
    mount: mountInspector,
  },
};

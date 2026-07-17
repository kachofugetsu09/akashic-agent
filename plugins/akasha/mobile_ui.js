const number = (value) => new Intl.NumberFormat("zh-CN").format(Number(value || 0));

const escapeHtml = (value) => String(value)
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;");

function shortTime(value) {
  const date = new Date(String(value || ""));
  if (Number.isNaN(date.getTime())) return String(value || "—");
  return new Intl.DateTimeFormat("zh-CN", {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);
}

function recallGroup(side, title, items) {
  const rows = items.length
    ? items.map((item) => `
      <li>
        <span>${escapeHtml(item.summary || "未命名记忆")}</span>
        ${item.preview ? `<small>${escapeHtml(item.preview)}</small>` : ""}
        <b>${Number(item.score || 0).toFixed(3)}</b>
      </li>`).join("")
    : '<li class="akasha-recall-empty">本轮没有命中</li>';
  return `
    <details class="akasha-recall ${side}">
      <summary><span class="akasha-recall-mark"></span><strong>${title}</strong><span>${items.length} 条</span></summary>
      <ol>${rows}</ol>
    </details>`;
}

function createElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function createLane(side, label, count) {
  const lane = createElement("span", `akasha-inspector-lane ${side}`);
  lane.append(
    createElement("i", "akasha-inspector-lane__mark"),
    createElement("span", "", label),
    createElement("strong", "", number(count)),
  );
  return lane;
}

function createRecallSection(side, title, items) {
  const section = createElement("section", `akasha-inspector-recall ${side}`);
  const heading = createElement("div", "akasha-inspector-recall__heading");
  heading.append(
    createElement("i", "akasha-inspector-recall__mark"),
    createElement("strong", "", title),
    createElement("span", "", `${number(items.length)} 条`),
  );
  const list = createElement("ol", "akasha-inspector-recall__list");
  if (items.length === 0) {
    list.append(createElement("li", "akasha-inspector-recall__empty", "本轮没有命中"));
  } else {
    for (const item of items) {
      const row = createElement("li", "akasha-inspector-memory");
      const copy = createElement("span", "akasha-inspector-memory__copy");
      copy.append(createElement("strong", "", item.summary || "未命名记忆"));
      if (item.preview) copy.append(createElement("small", "", item.preview));
      row.append(copy, createElement("b", "", Number(item.score || 0).toFixed(3)));
      list.append(row);
    }
  }
  section.append(heading, list);
  return section;
}

function renderInspectionDetail(container, result) {
  const left = Array.isArray(result.left) ? result.left : [];
  const right = Array.isArray(result.right) ? result.right : [];
  const query = createElement("section", "akasha-inspector-query");
  query.append(
    createElement("strong", "", "本轮问题"),
    createElement("p", "", result.query_text || "（没有问题原文）"),
  );
  container.replaceChildren(
    query,
    createRecallSection("left", "左脑 · 精确回忆", left),
    createRecallSection("right", "右脑 · 联想记忆", right),
  );
}

function inspectionRow(item, context, options) {
  const article = createElement("article", `akasha-inspection ${options.featured ? "featured" : ""}`);
  const trigger = createElement("button", "akasha-inspection__trigger");
  trigger.type = "button";
  trigger.setAttribute("aria-expanded", "false");

  const copy = createElement("span", "akasha-inspection__copy");
  if (options.featured) copy.append(createElement("small", "akasha-inspection__eyebrow", "最近一次检索"));
  copy.append(createElement("strong", "", item.query_preview || item.query_text || "（没有问题摘要）"));
  copy.append(createElement(
    "small",
    "akasha-inspection__meta",
    `${shortTime(item.ts)} · 注入 ${number(item.inject_chars)} 字`,
  ));
  const lanes = createElement("span", "akasha-inspection__lanes");
  lanes.append(
    createLane("left", "精确", item.left_count),
    createLane("right", "联想", item.right_count),
  );
  const chevron = createElement("i", "akasha-inspection__chevron");
  trigger.append(copy, lanes, chevron);

  const detail = createElement("div", "akasha-inspection__detail");
  detail.setAttribute("aria-hidden", "true");
  detail.inert = true;
  const detailBody = createElement("div", "akasha-inspection__detail-body");
  const detailContent = createElement("div", "akasha-inspection__detail-content");
  detailBody.append(detailContent);
  detail.append(detailBody);
  article.append(trigger, detail);

  let loaded = false;
  let loading = false;
  const close = () => {
    article.classList.remove("is-expanded");
    trigger.setAttribute("aria-expanded", "false");
    detail.setAttribute("aria-hidden", "true");
    detail.inert = true;
  };
  const open = () => {
    options.onOpen(close);
    article.classList.add("is-expanded");
    trigger.setAttribute("aria-expanded", "true");
    detail.setAttribute("aria-hidden", "false");
    detail.inert = false;
  };
  const load = async () => {
    if (loading) return;
    loading = true;
    detailContent.replaceChildren(createElement("p", "akasha-inspector-inline-state", "正在读取这轮回忆…"));
    try {
      const result = await context.request("inspector.detail", { query_id: item.query_id });
      if (!options.isActive()) return;
      renderInspectionDetail(detailContent, result);
      loaded = true;
    } catch (error) {
      if (!options.isActive()) return;
      const state = createElement("div", "akasha-inspector-inline-state error");
      state.append(createElement(
        "span",
        "",
        error instanceof Error ? `这轮回忆读取失败：${error.message}` : "这轮回忆读取失败",
      ));
      const retry = createElement("button", "", "重试");
      retry.type = "button";
      retry.addEventListener("click", () => load());
      state.append(retry);
      detailContent.replaceChildren(state);
    } finally {
      loading = false;
    }
  };
  trigger.addEventListener("click", () => {
    if (trigger.getAttribute("aria-expanded") === "true") {
      close();
      options.onClose(close);
      return;
    }
    open();
    if (!loaded) void load();
  });
  return { article, close };
}

const dashboard = {
  mount(host, context) {
    let active = true;
    let expanded = null;
    host.className += " akasha-inspector";
    host.innerHTML = `
      <div class="akasha-inspector-state" role="status">
        <span>正在读取最近回忆…</span>
        <button type="button" hidden>重试</button>
      </div>
      <div class="akasha-inspector-content" hidden>
        <div class="akasha-inspector-latest"></div>
        <section class="akasha-inspector-history" aria-labelledby="akasha-inspector-history-title">
          <header>
            <h2 id="akasha-inspector-history-title">更早的检索</h2>
            <span></span>
          </header>
          <div class="akasha-inspector-list"></div>
        </section>
      </div>`;
    const state = host.querySelector(".akasha-inspector-state");
    const stateText = state.querySelector("span");
    const retry = state.querySelector("button");
    const content = host.querySelector(".akasha-inspector-content");
    const latest = host.querySelector(".akasha-inspector-latest");
    const list = host.querySelector(".akasha-inspector-list");
    const history = host.querySelector(".akasha-inspector-history");
    const historyCount = history.querySelector("header span");

    const onOpen = (close) => {
      if (expanded && expanded !== close) expanded();
      expanded = close;
    };
    const onClose = (close) => {
      if (expanded === close) expanded = null;
    };
    const rowOptions = (featured) => ({
      featured,
      isActive: () => active,
      onOpen,
      onClose,
    });

    const load = async () => {
      state.className = "akasha-inspector-state";
      stateText.textContent = "正在读取最近回忆…";
      retry.hidden = true;
      state.hidden = false;
      content.hidden = true;
      expanded = null;
      latest.replaceChildren();
      list.replaceChildren();
      try {
        const result = await context.request("inspector.recent");
        if (!active) return;
        const items = Array.isArray(result.items) ? result.items : [];
        if (items.length === 0) {
          stateText.textContent = "还没有可检查的 Akasha 检索记录。";
          return;
        }
        latest.append(inspectionRow(items[0], context, rowOptions(true)).article);
        const older = items.slice(1);
        const olderTotal = Math.max(Number(result.total || items.length) - 1, 0);
        historyCount.textContent = older.length < olderTotal
          ? `最近 ${number(older.length)} / ${number(olderTotal)} 轮`
          : `${number(olderTotal)} 轮`;
        if (older.length === 0) {
          list.append(createElement("p", "akasha-inspector-empty", "没有更早的检索。"));
        } else {
          list.append(...older.map((item) => inspectionRow(item, context, rowOptions(false)).article));
        }
        state.hidden = true;
        content.hidden = false;
      } catch (error) {
        if (!active) return;
        state.className = "akasha-inspector-state error";
        stateText.textContent = error instanceof Error
          ? `回忆检查读取失败：${error.message}`
          : "回忆检查读取失败";
        retry.hidden = false;
      }
    };
    retry.addEventListener("click", () => void load());
    void load();
    return () => {
      active = false;
      expanded = null;
    };
  },
};

export default {
  slots: {
    "turn.before_reasoning": {
      mount(host, context) {
        host.innerHTML = '<div class="akasha-recall-loading">正在读取本轮记忆…</div>';
        let active = true;
        context.query("recall.current", { message_id: context.messageId }).then((result) => {
          if (!active) return;
          const left = Array.isArray(result.left) ? result.left : [];
          const right = Array.isArray(result.right) ? result.right : [];
          host.innerHTML = `<div class="akasha-recall-group">${recallGroup("left", "左脑 · 精确回忆", left)}${recallGroup("right", "右脑 · 联想记忆", right)}</div>`;
        }).catch((error) => {
          if (active) host.innerHTML = `<div class="akasha-recall-error">${escapeHtml(error.message || "记忆读取失败")}</div>`;
        });
        return () => { active = false; };
      },
    },
  },
  navigation: {
    label: "Akasha",
    description: "检查最近回答实际召回的左右脑记忆",
  },
  dashboard,
};

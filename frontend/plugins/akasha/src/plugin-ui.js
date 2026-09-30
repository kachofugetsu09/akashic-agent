import "./plugin-ui.css";

const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (character) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
})[character]);

function shortTime(value) {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false,
  }).format(new Date(value));
}

function checkResult(result) {
  if (result.schema !== "akasha.queries.v1") throw new Error("查询格式不受支持，请更新页面");
  return result;
}

function originText(source) {
  if (source.kind === "context") return `会话 ${source.session_id} · ${source.source} · 截至 #${source.through_seq}`;
  if (source.kind === "tool") return `会话 ${source.session_id} · 调用 ${source.call_ref.message_id}:${source.call_ref.part_index}`;
  return `独立查询 ${source.key}`;
}

export function renderDetail(item) {
  checkResult(item);
  const lanes = [["dense", "精确回忆"], ["completion", "模式补全"]];
  return `<section class="akasha-plugin-ui-inspector">
    <button type="button" data-akasha-back>返回检索列表</button>
    <header><h2>${escapeHtml(item.query_text + (item.query_text_truncated ? "…" : ""))}</h2><p>${escapeHtml(shortTime(item.ts))}</p>
      <p>命中 ${item.hit_count} 项记忆，向上下文提供 ${item.presented_count} 条消息。</p>
      <p>${escapeHtml(originText(item.source))}</p>
      <p>这是查询和材料记录，不证明模型已经使用这些内容。</p></header>
    ${lanes.map(([lane, title]) => `<details open class="akasha-plugin-ui-recall akasha-plugin-ui-recall--${lane === "completion" ? "completion" : "precise"}">
      <summary>${title}</summary>
      <ol class="akasha-plugin-ui-memories">${item.hits.filter((hit) => hit.lane === lane).map((hit) => `
        <li><div><p>得分 ${Number(hit.score).toFixed(3)} · ${escapeHtml(hit.sources.join(" · "))}</p>
        ${hit.messages.map((message) => `<article>
          <p>${escapeHtml(message.preview || "（非文本消息）")}${message.truncated ? "…（正文预览）" : ""}</p>
          <small>${escapeHtml(message.message_id)} · ${message.presented ? "已提供" : "未提供"}</small>
        </article>`).join("")}</div></li>`).join("") || "<li>本次没有命中</li>"}</ol>
    </details>`).join("")}
    <p>图版本 ${item.graph_version} · ${item.pushes} 次扩散 · 残余 ${Number(item.residual_l1).toExponential(2)}</p>
  </section>`;
}

function renderRecent(result) {
  checkResult(result);
  return `<section class="akasha-plugin-ui-inspector"><header><h2>Akasha Inspector</h2>
    <p>实际查询共 ${result.total} 次。选择一条查看命中与呈现记录。</p></header>
    ${result.items.length ? `<ol class="akasha-plugin-ui-turns">${result.items.map((item) => `
      <li><button type="button" data-akasha-query="${escapeHtml(item.query_id)}">
        <span>${escapeHtml(item.query_text + (item.query_text_truncated ? "…" : ""))}</span><small>${escapeHtml(shortTime(item.ts))} · 命中 ${item.hit_count} 项 · 提供 ${item.presented_count} 条</small>
      </button></li>`).join("")}</ol>` : "<p>还没有查询记录。使用记忆召回后，可在这里查看。</p>"}
    <nav aria-label="检索分页"><button type="button" data-akasha-prev ${result.page === 1 ? "disabled" : ""}>上一页</button>
      <span>第 ${result.page} 页</span>
      <button type="button" data-akasha-next ${result.page * result.page_size >= result.total ? "disabled" : ""}>下一页</button></nav></section>`;
}

export function mount(host, context) {
  let active = true;
  let recent;
  let requestId = 0;
  const failed = (error, request) => {
    if (active && request === requestId) host.innerHTML = `<p role="alert">${escapeHtml(error.message)}。请重新打开 Inspector。</p>`;
  };
  const showRecent = () => {
    if (!active) return;
    host.innerHTML = renderRecent(recent);
    host.querySelector("[data-akasha-prev]").addEventListener("click", () => loadPage(recent.page - 1));
    host.querySelector("[data-akasha-next]").addEventListener("click", () => loadPage(recent.page + 1));
    host.querySelectorAll("[data-akasha-query]").forEach((button) => {
      button.addEventListener("click", () => {
        const request = ++requestId;
        host.innerHTML = '<p role="status">正在读取检索记录…</p>';
        context.query("inspector.detail", { query_id: button.getAttribute("data-akasha-query") }).then((item) => {
          if (!active || request !== requestId) return;
          host.innerHTML = renderDetail(item);
          host.querySelector("[data-akasha-back]").addEventListener("click", showRecent);
        }).catch((error) => failed(error, request));
      });
    });
  };
  const loadPage = (page) => {
    const request = ++requestId;
    host.innerHTML = '<p role="status">正在读取查询列表…</p>';
    context.query("inspector.recent", { page }).then((result) => {
      if (!active || request !== requestId) return;
      recent = result;
      showRecent();
    }).catch((error) => failed(error, request));
  };
  loadPage(1);
  return () => { active = false; };
}

/** 聚合事实在服务端确认闭合前仍可变化，不进入页面结果缓存。 */
async function readRecall(context, isActive) {
  const query = async (messageId, offset) => {
    while (true) {
      if (!isActive()) throw new Error("召回面板已卸载");
      const result = await context.query("recall.turn", {
        message_id: messageId, source: context.block?.source ?? "", ...(offset ? { offset } : {}),
      }, { cache: "none" });
      // 兼容查找按服务端有界页推进，不是等待新事实的轮询。
      if (!result.legacy_pending) return result;
    }
  };
  const result = await query(context.messageId, 0);
  const items = [...result.items];
  let pending = result.pending;
  let offset = result.next_offset;
  while (offset != null) {
    const page = await query(result.input_message_id, offset);
    items.push(...page.items);
    pending = page.pending;
    offset = page.next_offset;
  }
  return { ...result, items, pending };
}

const OPEN_VIEW_PREFIX = "akasha.recall.open:";

/** 展开标志是可选的本页状态；浏览器拒绝存储时保留查询和清理行为。 */
function readOpenedView(key) {
  try {
    const value = sessionStorage.getItem(key) ?? "";
    sessionStorage.removeItem(key);
    return new Set(value.split(",").filter(lane => lane === "dense" || lane === "completion"));
  } catch (error) {
    if (!(error instanceof DOMException) || !["SecurityError", "QuotaExceededError"].includes(error.name)) throw error;
    console.warn("[akasha-ui] 浏览器未允许恢复展开状态", error.name);
    return new Set();
  }
}

/** 仅保留至多64个非空展开标志，不存放召回正文或结果。 */
function saveOpenedView(key, lanes) {
  try {
    if (!lanes.length) { sessionStorage.removeItem(key); return; }
    const keys = Array.from({length: sessionStorage.length}, (_, index) => sessionStorage.key(index))
      .filter(item => item?.startsWith(OPEN_VIEW_PREFIX) && item !== key);
    while (keys.length >= 64) sessionStorage.removeItem(keys.shift());
    sessionStorage.setItem(key, lanes.join(","));
  } catch (error) {
    if (!(error instanceof DOMException) || !["SecurityError", "QuotaExceededError"].includes(error.name)) throw error;
    console.warn("[akasha-ui] 浏览器未允许保存展开状态", error.name);
  }
}

/** 先计算展示内容；相同快照不改写真实卡片、选择与焦点。 */
function renderRecall(result) {
  return result.items.length ? `<div class="akasha-plugin-ui-recall-group">${[
    ["dense", "左脑 · 精确回忆", "precise"], ["completion", "右脑 · 模式补全", "completion"],
  ].map(([lane, title, style]) => {
    // 多次真实查询可以命中同一回忆；卡片按消息成员展示一次，原查询留在 Inspector。
    const hits = [...new Map(result.items.flatMap((item) => item.hits.filter((hit) => hit.lane === lane))
      .map((hit) => [JSON.stringify(hit.messages.map((message) => message.message_id)), hit])).values()];
    return `<details data-lane="${lane}" class="akasha-plugin-ui-recall akasha-plugin-ui-recall--${style}">
      <summary><span>${title}</span><b>${hits.length}</b></summary>
      <ol class="akasha-plugin-ui-memories">${hits.map((hit) => `<li><div>${hit.messages.map((message) =>
        `<p>${escapeHtml(message.preview || "（非文本消息）")}${message.truncated ? "…" : ""}</p>`).join("")}</div></li>`).join("")
        || '<li class="akasha-plugin-ui-empty">本次没有命中</li>'}</ol></details>`;
  }).join("")}</div>` : "";
}

/** 已接纳 Input 持有面板；事实失效只刷新内容，不重新挂载。 */
export function mountRecall(host, context) {
  if (typeof context.onInvalidate !== "function") throw new Error("聊天页面不支持召回刷新，请刷新页面后重试");
  let active = true;
  let loading = false;
  let dirty = false;
  let loaded = false;
  let loadingTimer;
  let rendered = null;
  let unsubscribe;
  // 只交接展开状态，不缓存消息、召回结果或插件版本。
  const keyFor = (messageId) => `${OPEN_VIEW_PREFIX}${JSON.stringify([context.sessionId, messageId, context.turnId, context.block?.source])}`;
  let viewKey = keyFor(context.messageId);
  const remembered = readOpenedView(viewKey);
  const content = document.createElement("div");
  const status = document.createElement("p");
  status.className = "akasha-plugin-ui-query-status";
  status.setAttribute("role", "status");
  status.hidden = true;
  host.replaceChildren(content, status);

  const load = async () => {
    if (!active) return;
    if (loading) { dirty = true; return; }
    loading = true;
    dirty = false;
    let settled = false;
    // 成功空快照也是已读；后续刷新不再显示首次加载占位。
    if (!loaded && status.hidden) loadingTimer = setTimeout(() => {
      if (!active) return;
      status.textContent = "正在读取召回记录…";
      status.hidden = false;
    }, 150);
    try {
      const result = await readRecall(context, () => active);
      if (!active) return;
      // 历史 fallback 使用服务端解析的 Input 交接展开状态，不猜测归属。
      if (result.input_message_id) {
        const resolvedKey = keyFor(result.input_message_id);
        if (viewKey !== resolvedKey) {
          viewKey = resolvedKey;
          if (!loaded) for (const lane of readOpenedView(viewKey)) remembered.add(lane);
        }
      }
      const markup = renderRecall(result);
      if (markup !== rendered) {
        const opened = new Set([
          ...(!loaded ? remembered : []),
          ...Array.from(content.querySelectorAll("details[open]"), (item) => item.dataset.lane),
        ]);
        content.innerHTML = markup;
        content.querySelectorAll("details").forEach((item) => { item.open = opened.has(item.dataset.lane); });
        rendered = markup;
      }
      loaded = true;
      status.hidden = true;
      settled = result.pending === false;
    } catch (error) {
      // 失败只更新局部状态，重试沿同一查询入口走，不销毁召回卡片。
      if (active) {
        const retry = document.createElement("button");
        retry.type = "button";
        const stale = error.code === "plugin_ui_stale_revision" || error.code === "plugin_ui_unavailable";
        retry.textContent = stale ? "刷新页面" : "重试";
        retry.addEventListener("click", () => { if (stale) window.location.reload(); else void load(); });
        status.replaceChildren(document.createTextNode(`情景记忆展示暂不可用：${error.message} `), retry);
        status.hidden = false;
      }
    } finally {
      loading = false;
      clearTimeout(loadingTimer);
      // 读取期间收到的提交不能丢失，即使这份旧快照已经声称闭合。
      if (active && dirty) void load();
      else if (active && settled) { unsubscribe?.(); unsubscribe = undefined; }
    }
  };
  unsubscribe = context.onInvalidate(() => { void load(); });
  void load();
  return () => {
    active = false;
    unsubscribe?.();
    clearTimeout(loadingTimer);
    if (loaded) saveOpenedView(viewKey,
      Array.from(content.querySelectorAll("details[open]"), item => item.dataset.lane));
  };
}

export default {
  slots: { "turn.before_reasoning": { mount: mountRecall } },
  dashboard: { mount },
};

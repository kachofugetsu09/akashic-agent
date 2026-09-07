import "./mobile.css";

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
  return `<section class="akasha-mobile-inspector">
    <button type="button" data-akasha-back>返回检索列表</button>
    <header><h2>${escapeHtml(item.query_text + (item.query_text_truncated ? "…" : ""))}</h2><p>${escapeHtml(shortTime(item.ts))}</p>
      <p>命中 ${item.hit_count} 项记忆，向上下文提供 ${item.presented_count} 条消息。</p>
      <p>${escapeHtml(originText(item.source))}</p>
      <p>这是查询和材料记录，不证明模型已经使用这些内容。</p></header>
    ${lanes.map(([lane, title]) => `<details open class="akasha-mobile-recall akasha-mobile-recall--${lane === "completion" ? "completion" : "precise"}">
      <summary>${title}</summary>
      <ol class="akasha-mobile-memories">${item.hits.filter((hit) => hit.lane === lane).map((hit) => `
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
  return `<section class="akasha-mobile-inspector"><header><h2>Akasha Inspector</h2>
    <p>实际查询共 ${result.total} 次。选择一条查看命中与呈现记录。</p></header>
    ${result.items.length ? `<ol class="akasha-mobile-turns">${result.items.map((item) => `
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

export default { slots: {}, dashboard: { mount } };

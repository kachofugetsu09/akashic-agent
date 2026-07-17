const escapeHtml = (value) => String(value)
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;");

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

export default {
  slots: {
    "turn.before_reasoning": {
      mount(host, context) {
        host.innerHTML = '<div class="akasha-recall-loading">正在读取本轮记忆…</div>';
        let active = true;
        context.query(
          "recall.current",
          { message_id: context.messageId },
          { cache: "immutable" },
        ).then((result) => {
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
};

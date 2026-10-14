import "./message-ui.css";

const number = (value) => new Intl.NumberFormat("zh-CN").format(Number(value || 0));

// 回答尾注：显示本轮全部模型调用的真实输出 token；没有完整用量时不占位。
function answerUsage(host, context) {
  let active = true;
  if (!context.messageId || !context.sessionId) throw new Error("Turn token 统计缺少消息上下文");
  host.classList.add("models-usage-tail-host");
  context.query("usage.message", { message_id: context.messageId }, { cache: "immutable" }).then((result) => {
    if (!active || typeof result.output_tokens !== "number") return;
    const row = document.createElement("div");
    row.className = "models-usage-tail";
    row.setAttribute("aria-label", "本轮输出 Token");
    row.innerHTML = `<span>输出 <strong>${number(result.output_tokens)} tokens</strong></span>`;
    host.replaceChildren(row);
  }).catch((error) => {
    if (!active) return;
    host.classList.add("models-usage-tail-host--error");
    host.textContent = error instanceof Error ? `Token 统计不可用：${error.message}` : "Token 统计不可用";
  });
  return () => { active = false; };
}

export default { slots: { "turn.after_answer": { mount: answerUsage } } };

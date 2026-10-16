import "./message-ui.css";

const OPEN_INTENT = "workspace.changes.open.v1";
const RETRY_MS = [500, 1000, 2000, 4000, 8000, 15000];
// 这些情况与文件无关（闲聊、没设目录、不是仓库、没有记录），回复下面不占位。
const SILENT = new Set(["no_directory", "missing", "not_repository", "not_recorded"]);

// 回复下方的改动摘要：只在完成的最终回答下出现；统计中先显示占位，结束后换成“改动了 N 个文件”。
function answerChanges(host, context) {
  let active = true;
  let timer = 0;
  let summary = null;
  if (!context.messageId || !context.sessionId) throw new Error("改动摘要缺少消息上下文");
  host.classList.add("changes-tail-host");

  function render() {
    if (!summary) return;
    if (summary.phase === "pending") return host.replaceChildren(line("统计改动中…", "is-pending"));
    if (summary.error) return host.replaceChildren(...(SILENT.has(summary.error.reason) ? [] : [line(`改动未统计：${summary.error.message}`, "is-muted")]));
    if (summary.phase !== "ended" || !summary.files) return host.replaceChildren();
    const text = `改动了 ${summary.files} 个文件 +${summary.added} −${summary.removed}${summary.uncertain ? "（可能不全）" : ""}`;
    if (!summary.path || !context.workbench.accepts(OPEN_INTENT)) return host.replaceChildren(line(text, ""));
    const button = document.createElement("button");
    button.type = "button";
    button.className = "changes-tail is-link";
    button.textContent = text;
    button.title = "在工作台的“改动”里查看";
    button.addEventListener("click", () => void context.workbench.open(OPEN_INTENT, { path: summary.path }));
    host.replaceChildren(button);
  }

  function load(attempt) {
    context.query("turn.message", { message_id: context.messageId }, { cache: "none" }).then((result) => {
      if (!active) return;
      summary = result;
      render();
      if (result.phase !== "pending") return;
      // 重试用完仍在统计（例如收尾卡住）：撤掉占位，不让“统计改动中…”一直挂着。
      if (attempt < RETRY_MS.length) timer = window.setTimeout(() => load(attempt + 1), RETRY_MS[attempt]);
      else host.replaceChildren();
    }).catch((error) => {
      if (active) context.reportError(error instanceof Error ? error.message : String(error));
    });
  }

  load(0);
  const stop = context.onInvalidate(render);
  return () => {
    active = false;
    window.clearTimeout(timer);
    stop();
  };
}

function line(text, state) {
  const node = document.createElement("div");
  node.className = `changes-tail ${state}`;
  node.textContent = text;
  return node;
}

export default { slots: { "turn.after_answer": { mount: answerChanges } } };

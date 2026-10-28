import "./plugin-ui.css";
import { mountMemoryBadge, mountMemoryField } from "./scope-ui.tsx";

const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (character) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
})[character]);

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

const BRAIN = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 18V5"/><path d="M15 13a4.17 4.17 0 0 1-3-4 4.17 4.17 0 0 1-3 4"/><path d="M17.598 6.5A3 3 0 1 0 12 5a3 3 0 1 0-5.598 1.5"/><path d="M17.997 5.125a4 4 0 0 1 2.526 5.77"/><path d="M18 18a4 4 0 0 0 2-7.464"/><path d="M19.967 17.483A4 4 0 1 1 12 18a4 4 0 1 1-7.967-.517"/><path d="M6 18a4 4 0 0 1-2-7.464"/><path d="M6.003 5.125a4 4 0 0 0-2.526 5.77"/></svg>';

function shortDate(value) {
  const date = new Date(value);
  const day = `${date.getMonth() + 1}月${date.getDate()}日`;
  return date.getFullYear() === new Date().getFullYear() ? day : `${date.getFullYear()}年${day}`;
}

const leadMessage = (hit) => hit.messages.find((message) => message.author === "user") ?? hit.messages[0];

/** 多次查询、相邻窗口会命中同一段对话；按开头那条消息合并成一段，给模型看过的排前面。 */
function recallView(result) {
  const merged = new Map();
  for (const hit of result.items.flatMap((item) => item.hits)) {
    const key = leadMessage(hit)?.message_id ?? JSON.stringify(hit.messages.map((message) => message.message_id));
    const known = merged.get(key);
    if (!known) { merged.set(key, { ...hit, messages: [...hit.messages] }); continue; }
    const ids = new Set(known.messages.map((message) => message.message_id));
    known.messages.push(...hit.messages.filter((message) => !ids.has(message.message_id)));
  }
  const hits = [...merged.values()];
  const shown = hits.filter((hit) => hit.messages.some((message) => message.presented));
  return { shown, rest: hits.filter((hit) => !shown.includes(hit)), count: hits.length };
}

function hitRow(hit) {
  const first = leadMessage(hit);
  const title = (first?.preview || "（非文本消息）").split("\n").find((line) => line.trim()) ?? "";
  // 工具调用等没有正文的步骤不列出。
  const lines = hit.messages.filter((message) => message.preview).map((message) => `<p><i>${message.author === "user" ? "你" : "我"}</i><span>${
    escapeHtml(message.preview)}${message.truncated ? "…" : ""}</span></p>`).join("");
  return `<li><button type="button" class="akasha-recall-row" aria-expanded="false">
    <time>${first?.recorded_at ? escapeHtml(shortDate(first.recorded_at)) : ""}</time><span>${escapeHtml(title)}</span>
    ${hit.lane === "completion" ? "<small>联想</small>" : ""}</button><div class="akasha-recall-said" hidden>${lines}</div></li>`;
}

/** 回想面板：给模型看过的直接列出，其余折起；点一行就地展开原话，不再查询。 */
function renderPanel(view) {
  const shown = view.shown.length ? `<section><h3>给模型看了 <b>${view.shown.length}</b></h3>
    <ol>${view.shown.map(hitRow).join("")}</ol></section>` : "";
  const rest = view.rest.length ? `<section><button type="button" class="akasha-recall-more" aria-expanded="false">也想起了 ${
    view.rest.length} 段</button><ol hidden>${view.rest.map(hitRow).join("")}</ol></section>` : "";
  return shown + rest;
}

let openPanel = null;
let closeTimer;
const hoverable = () => window.matchMedia("(hover: hover)").matches;

function closePanel() {
  clearTimeout(closeTimer);
  openPanel?.close();
}

/** 面板挂到 document.body：不继承宿主样式，也不被消息裁切；滚动、Esc 或点外部即收起。 */
function showPanel(anchor, view, side) {
  clearTimeout(closeTimer);
  if (openPanel?.anchor === anchor) return;
  closePanel();
  const panel = document.createElement("div");
  panel.className = "akasha-recall-panel";
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-label", "回想");
  panel.innerHTML = renderPanel(view);
  document.body.append(panel);
  const place = () => {
    const box = anchor.getBoundingClientRect();
    const width = panel.offsetWidth, height = panel.offsetHeight;
    let left = box.left, top = box.bottom + 6;
    if (side === "right") {
      left = box.right + 12 + width <= innerWidth - 12 ? box.right + 12 : box.left - 12 - width;
      top = box.top + box.height / 2 - height / 2;
    } else if (top + height > innerHeight - 12) top = box.top - 6 - height;
    panel.style.left = `${Math.max(12, Math.min(left, innerWidth - width - 12))}px`;
    panel.style.top = `${Math.max(12, Math.min(top, innerHeight - height - 12))}px`;
  };
  panel.addEventListener("click", (event) => {
    const toggle = event.target.closest("button[aria-expanded]");
    if (!toggle) return;
    const open = toggle.getAttribute("aria-expanded") !== "true";
    toggle.setAttribute("aria-expanded", String(open));
    toggle.nextElementSibling.hidden = !open;
    place();
  });
  panel.addEventListener("pointerenter", () => clearTimeout(closeTimer));
  panel.addEventListener("pointerleave", () => { if (hoverable()) closeTimer = setTimeout(closePanel, 180); });
  const outside = (event) => { if (!panel.contains(event.target) && !anchor.contains(event.target)) closePanel(); };
  const key = (event) => { if (event.key === "Escape") { closePanel(); anchor.focus(); } };
  const scroll = (event) => { if (!panel.contains(event.target)) closePanel(); };
  document.addEventListener("pointerdown", outside, true);
  document.addEventListener("keydown", key);
  document.addEventListener("scroll", scroll, true);
  window.addEventListener("resize", closePanel);
  anchor.setAttribute("aria-expanded", "true");
  place();
  openPanel = { anchor, close() {
    panel.remove();
    anchor.setAttribute("aria-expanded", "false");
    document.removeEventListener("pointerdown", outside, true);
    document.removeEventListener("keydown", key);
    document.removeEventListener("scroll", scroll, true);
    window.removeEventListener("resize", closePanel);
    openPanel = null;
  } };
}

/** 悬停打开（有悬停的指针），点按切换（触屏与键盘）。 */
function attachPanel(anchor, getView, side) {
  anchor.setAttribute("aria-haspopup", "dialog");
  anchor.setAttribute("aria-expanded", "false");
  anchor.addEventListener("pointerenter", () => { if (hoverable()) showPanel(anchor, getView(), side); });
  anchor.addEventListener("pointerleave", () => { if (hoverable()) closeTimer = setTimeout(closePanel, 180); });
  anchor.addEventListener("click", () => {
    if (openPanel?.anchor === anchor && !hoverable()) closePanel();
    else showPanel(anchor, getView(), side);
  });
}

// 同一轮的召回按会话与消息登记，回答工具栏凭自己的消息认领，不重复查询。
const turnViews = new Map();
const viewWatchers = new Set();
const viewKey = (sessionId, messageId) => `${sessionId}\n${messageId}`;

function publishView(sessionId, ids, view) {
  for (const id of ids) turnViews.set(viewKey(sessionId, id), view);
  viewWatchers.forEach((watch) => watch());
}

function retractView(sessionId, ids, view) {
  for (const id of ids) if (turnViews.get(viewKey(sessionId, id)) === view) turnViews.delete(viewKey(sessionId, id));
  viewWatchers.forEach((watch) => watch());
}

/** 回答工具栏里的脑图标：本轮有召回才出现，面板在按钮右侧。 */
export function mountActions(host, context) {
  const key = viewKey(context.sessionId, context.messageId);
  let button = null;
  const sync = () => {
    const view = turnViews.get(key);
    if (!view?.count) { if (openPanel?.anchor === button) closePanel(); button?.remove(); button = null; return; }
    if (button) return;
    button = document.createElement("button");
    button.type = "button";
    button.innerHTML = BRAIN;
    button.setAttribute("aria-label", "回想");
    button.dataset.tip = "回想";
    attachPanel(button, () => turnViews.get(key), "right");
    host.replaceChildren(button);
  };
  viewWatchers.add(sync);
  sync();
  return () => {
    viewWatchers.delete(sync);
    if (openPanel?.anchor === button) closePanel();
  };
}

/** 已接纳 Input 持有一行「N 段回想」；事实失效只刷新内容，不重新挂载。 */
export function mountRecall(host, context) {
  if (typeof context.onInvalidate !== "function") throw new Error("聊天页面不支持召回刷新，请刷新页面后重试");
  let active = true;
  let loading = false;
  let dirty = false;
  let loaded = false;
  let loadingTimer;
  let rendered = null;
  let unsubscribe;
  let view = null;
  let claimed = [];
  const content = document.createElement("div");
  const status = document.createElement("p");
  status.className = "akasha-plugin-ui-query-status";
  status.setAttribute("role", "status");
  status.hidden = true;
  host.replaceChildren(content, status);

  // 1. 先算出这一行的文字；相同快照不改写按钮，保留焦点与已打开的面板。
  const show = (next) => {
    const markup = next.count ? `${BRAIN}<span>${next.count} 段回想</span>` : "";
    if (markup === rendered) return;
    rendered = markup;
    if (!markup) { if (openPanel?.anchor === content.querySelector(".akasha-recall-badge")) closePanel(); content.replaceChildren(); return; }
    let badge = content.querySelector(".akasha-recall-badge");
    if (!badge) {
      const group = document.createElement("div");
      group.className = "akasha-plugin-ui-recall-group";
      badge = document.createElement("button");
      badge.type = "button";
      badge.className = "akasha-recall-badge";
      attachPanel(badge, () => view, "below");
      group.append(badge);
      content.replaceChildren(group);
    }
    badge.innerHTML = markup;
  };

  const load = async () => {
    if (!active) return;
    if (loading) { dirty = true; return; }
    loading = true;
    dirty = false;
    let settled = false;
    // 2. 成功空快照也是已读；后续刷新不再显示首次加载占位。
    if (!loaded && status.hidden) loadingTimer = setTimeout(() => {
      if (!active) return;
      status.innerHTML = `${BRAIN}<span class="akasha-recall-shimmer">回想中…</span>`;
      status.hidden = false;
    }, 150);
    try {
      const result = await readRecall(context, () => active);
      if (!active) return;
      const next = recallView(result);
      // 3. 换下旧登记再登记新快照，回答工具栏随之出现或更新。
      retractView(context.sessionId, claimed, view);
      view = next;
      claimed = Array.isArray(result.message_ids) ? result.message_ids : [];
      publishView(context.sessionId, claimed, view);
      show(view);
      loaded = true;
      status.hidden = true;
      settled = result.pending === false;
    } catch (error) {
      // 4. 失败只更新局部状态，重试沿同一查询入口走，不销毁已有的回想。
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
    if (openPanel?.anchor === content.querySelector(".akasha-recall-badge")) closePanel();
    retractView(context.sessionId, claimed, view);
  };
}

export default {
  slots: {
    "turn.before_reasoning": { mount: mountRecall },
    "message.actions": { mount: mountActions },
    "scope.fields": { mount: mountMemoryField },
    "scope.badge": { mount: mountMemoryBadge },
  },
};

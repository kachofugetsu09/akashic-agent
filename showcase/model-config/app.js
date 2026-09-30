/*
 * Akashic 模型连接 showcase —— 无依赖单页原型。
 * 形态与 plugins/<name>/web_module.js 一致：宿主渲染外壳，连接工作区由"插件"各自渲染。
 * 数据全部为演示桩，不接真实 API。
 *
 * 交互链（对齐 dsh / magpie 的真实流程）：
 *   连接 = 凭据 + 目录 + 开放度。探测把 endpoint 当前值（含未保存的 key）发出去，
 *   候选模型直接落进管理表：新模型默认勾选、已有行保留用户调过的字段。
 *   开放是连接级勾选，可见是角色级勾选，当前模型固定到会话是会话级。
 */

"use strict";

/* ---------- 基础 ---------- */

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};
const svg = (paths, size = 16, width = 1.7) => {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("viewBox", "0 0 24 24");
  s.setAttribute("width", String(size));
  s.setAttribute("height", String(size));
  s.setAttribute("fill", "none");
  s.setAttribute("stroke", "currentColor");
  s.setAttribute("stroke-width", String(width));
  s.setAttribute("stroke-linecap", "round");
  s.setAttribute("stroke-linejoin", "round");
  s.setAttribute("aria-hidden", "true");
  for (const d of paths) {
    const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
    p.setAttribute("d", d);
    s.append(p);
  }
  return s;
};
const ICONS = {
  search: ["M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16z", "m21 21-4.3-4.3"],
  chevronR: ["m9 18 6-6-6-6"],
  chevronD: ["m6 9 6 6 6-6"],
  plus: ["M12 5v14", "M5 12h14"],
  close: ["M18 6 6 18", "m6 6 12 12"],
  check: ["M20 6 9 17l-5-5"],
  grid: ["M3 3h7v7H3z", "M14 3h7v7H7z", "M3 14h7v7H3z", "M14 14h7v7h-7z"],
  spark: ["M12 3v3", "M12 18v3", "M5.6 5.6l2.1 2.1", "M16.3 16.3l2.1 2.1", "M3 12h3", "M18 12h3", "M5.6 18.4l2.1-2.1", "M16.3 7.7l2.1-2.1"],
  play: ["M6 4.5v15l13-7.5z"],
  refresh: ["M21 12a9 9 0 1 1-2.64-6.36", "M21 3v6h-6"],
  trash: ["M3 6h18", "M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2", "M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"],
  link: ["M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71", "M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"],
  eye: ["M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z", "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z"],
};
const ico = (name, size) => svg(ICONS[name], size);
const iconCls = (name, size, cls) => {
  const s = svg(ICONS[name], size);
  s.setAttribute("class", cls);
  return s;
};

function iconImg(name) {
  const img = document.createElement("img");
  img.src = `assets/icons/${name}.svg`;
  img.alt = "";
  return img;
}
function mark(name, letter) {
  const m = el("span", "mark");
  if (name) m.append(iconImg(name));
  else m.append(el("span", "fallback", (letter || "?").slice(0, 1).toUpperCase()));
  return m;
}
function slug(text) {
  return (text || "").toLowerCase().trim().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "conn";
}

function toast(text) {
  const region = $("#toastRegion");
  const t = el("div", "toast", text);
  t.setAttribute("role", "status");
  region.replaceChildren(t);
  setTimeout(() => { if (t.isConnected) t.remove(); }, 3200);
}

/* ---------- 演示数据 ---------- */

const EFFORT_LABELS = { none: "关闭", minimal: "极低", low: "低", medium: "中", high: "高", xhigh: "极高", max: "最大" };
const fmtCtx = n => n == null ? "" : n >= 1e6 ? `${n / 1e6}M` : `${Math.round(n / 1e3)}K`;

/*
 * 驱动即插件贡献的连接类型（models.connection-types.v1 插槽）。
 * catalog 是"探测"返回的目录：每个候选自带能力（context / efforts / images / free），
 * 能力映射由各驱动插件负责 —— 这是组合卖点：宿主不理解 vendor，插件理解。
 */
const KINDS = {
  codex: {
    label: "Codex 订阅", plugin: "codex", auth: "oauth",
    need: "浏览器登录",
    catalog: () => [
      { id: "gpt-5.4-codex", name: "GPT-5.4 Codex", context: 400e3, efforts: ["low", "medium", "high", "xhigh"], images: true },
      { id: "gpt-5.4", name: "GPT-5.4", context: 400e3, efforts: ["low", "medium", "high"], images: true },
      { id: "codex-mini-latest", name: "Codex Mini", context: 200e3, efforts: ["low", "medium", "high"] },
    ],
  },
  "opencode-go": {
    label: "OpenCode Go", plugin: "opencode-go", auth: "key-or-local",
    need: "Key 或本机登录",
    catalog: () => [
      { id: "opencode/glm-5", name: "GLM 5", context: 200e3, efforts: ["low", "medium", "high"], images: true },
      { id: "opencode/kimi-k3", name: "Kimi K3", context: 256e3, efforts: ["medium", "high"], free: true },
    ],
  },
  "openai-compatible": {
    label: "OpenAI 兼容服务", plugin: "openai-compatible", auth: "key",
    need: "API Key",
    // 不同 preset 共享此驱动，目录按 baseURL 区分 —— 插件探测后返回各自候选。
    catalog: (conn) => {
      const host = conn.baseURL || "";
      if (host.includes("deepseek")) return [
        { id: "deepseek-chat", name: "DeepSeek Chat", context: 128e3, efforts: [] },
        { id: "deepseek-reasoner", name: "DeepSeek Reasoner", context: 128e3, efforts: ["low", "medium", "high"] },
      ];
      if (host.includes("openrouter")) return [
        { id: "anthropic/claude-sonnet-5", name: "Claude Sonnet 5", context: 200e3, efforts: ["low", "medium", "high"], images: true },
        { id: "anthropic/claude-opus-5", name: "Claude Opus 5", context: 200e3, efforts: ["low", "medium", "high", "xhigh"], images: true },
        { id: "openai/gpt-5.5-mini", name: "GPT-5.5 Mini", context: 1e6, efforts: ["minimal", "low", "medium", "high"], images: true },
        { id: "z-ai/glm-5.2:free", name: "GLM 5.2", context: 128e3, efforts: [], free: true },
        { id: "google/gemini-3-flash", name: "Gemini 3 Flash", context: 1e6, efforts: ["low", "medium"], images: true },
        { id: "moonshot/kimi-k3", name: "Kimi K3", context: 256e3, efforts: ["medium", "high"] },
        { id: "qwen/qwen3-235b", name: "Qwen3 235B", context: 128e3, efforts: ["low", "medium", "high"] },
      ];
      if (host.includes("11434") || host.includes("1234")) return [
        { id: "qwen3:32b", context: 32e3, efforts: [], images: true },
        { id: "qwen3:8b", context: 32e3, efforts: ["low"] },
        { id: "llama4:scout", context: 128e3, efforts: [] },
      ];
      return [
        { id: "claude-sonnet-5", name: "Claude Sonnet 5", context: 200e3, efforts: ["low", "medium", "high"], images: true },
        { id: "gpt-5.5", name: "GPT-5.5", context: 400e3, efforts: ["minimal", "low", "medium", "high"], images: true },
      ];
    },
  },
};

const TEMPLATE_GROUPS = [
  {
    id: "plugins", title: "来自插件", note: "由已安装插件贡献的连接方式",
    items: [
      { id: "codex", name: "Codex 订阅", icon: "codex", kind: "codex", need: "OAuth 登录", plugin: "codex 插件" },
      { id: "opencode-go", name: "OpenCode Go", icon: "opencode", kind: "opencode-go", need: "Key 或本机登录", plugin: "opencode-go 插件" },
    ],
  },
  {
    id: "vendors", title: "厂商", note: "模型自己的官方服务",
    items: [
      { id: "deepseek", name: "DeepSeek", icon: "deepseek", kind: "openai-compatible", need: "API Key", baseURL: "https://api.deepseek.com/v1" },
      { id: "openai", name: "OpenAI", kind: "openai-compatible", need: "API Key", baseURL: "https://api.openai.com/v1" },
      { id: "anthropic", name: "Anthropic", kind: "openai-compatible", need: "API Key", baseURL: "https://api.anthropic.com/v1" },
    ],
  },
  {
    id: "relays", title: "聚合与转发", note: "一个 Key 访问多家模型",
    items: [
      { id: "openrouter", name: "OpenRouter", icon: "openrouter", kind: "openai-compatible", need: "API Key", baseURL: "https://openrouter.ai/api/v1" },
      { id: "custom", name: "自定义兼容端点", kind: "openai-compatible", need: "名称 + Base URL + Key", custom: true },
    ],
  },
  {
    id: "local", title: "本机", note: "本机或局域网内运行的服务",
    items: [
      { id: "ollama", name: "Ollama", kind: "openai-compatible", need: "无需密钥", baseURL: "http://127.0.0.1:11434/v1", keyless: true },
      { id: "lmstudio", name: "LM Studio", kind: "openai-compatible", need: "无需密钥", baseURL: "http://127.0.0.1:1234/v1", keyless: true },
    ],
  },
];

// 一条连接的完整演示态。models[].exposed = 是否开放给角色/会话挑选。
const state = {
  view: "connections",
  firstRun: false,
  query: "",
  openWs: null,       // 展开工作区的连接 id，或 "draft"
  draft: null,        // 未保存的新连接草稿
  connections: [
    {
      id: "codex-main", name: "Codex 订阅", kind: "codex", icon: "codex",
      auth: "oauth", credential: "configured", account: "hua@example.com",
      models: KINDS.codex.catalog().map((m, i) => ({ ...m, exposed: i < 2 })),
    },
    {
      id: "deepseek", name: "DeepSeek 官方", kind: "openai-compatible", icon: "deepseek",
      auth: "key", credential: "configured", baseURL: "https://api.deepseek.com/v1",
      models: KINDS["openai-compatible"].catalog({ baseURL: "https://api.deepseek.com/v1" }).map(m => ({ ...m, exposed: true })),
    },
    {
      id: "openrouter", name: "OpenRouter", kind: "openai-compatible", icon: "openrouter",
      auth: "key", credential: "configured", baseURL: "https://openrouter.ai/api/v1",
      models: KINDS["openai-compatible"].catalog({ baseURL: "https://openrouter.ai/api/v1" })
        .map((m, i) => ({ ...m, exposed: [0, 2, 3].includes(i) })),
    },
    {
      id: "oc-go", name: "OpenCode Go", kind: "opencode-go", icon: "opencode",
      auth: "key-or-local", credential: "missing",
      models: [],
    },
  ],
  roles: {
    default: { connection: "deepseek", model: "deepseek-chat" },
    agent: { connection: "codex-main", model: "gpt-5.4-codex" },
    fast: { connection: "deepseek", model: "deepseek-chat" },
    vision: null,
  },
  // 角色级可见性：role -> Set(隐藏的模型 id)。目录全局共享，可见性私有。
  roleHidden: { fast: new Set(["anthropic/claude-opus-5", "gpt-5.4-codex"]), vision: new Set(), agent: new Set(), default: new Set() },
  selection: null,
  effort: "medium",
};
const ROLE_ROWS = [
  ["default", "默认模型", "普通对话与系统默认"],
  ["agent", "Agent 模型", "被动回复与计划任务"],
  ["fast", "轻量模型", "压缩、标签与后台提取"],
  ["vision", "视觉模型", "看图与含图对话"],
];

/* ---------- 派生 ---------- */

function exposedModels(c) {
  return c.models.filter(m => m.exposed !== false);
}
function allModels() {
  return state.connections.flatMap(c => exposedModels(c).map(m => ({ ...m, connection: c })));
}
function modelsForRole(role) {
  const hidden = state.roleHidden[role] ?? new Set();
  return allModels().filter(m => !hidden.has(m.id));
}
function findModel(ref) {
  if (!ref) return null;
  const c = state.connections.find(x => x.id === ref.connection);
  const m = c?.models.find(x => x.id === ref.model);
  return c && m ? { ...m, connection: c } : null;
}
function defaultRef() { return state.roles.default; }
function effectiveRef() { return state.selection ?? defaultRef(); }
function connHost(c) {
  if (c.account) return `已登录 ${c.account}`;
  try { return c.baseURL ? new URL(c.baseURL).host : KINDS[c.kind]?.label ?? c.kind; }
  catch { return c.baseURL || c.kind; }
}
// 谁在用：绑定到这条连接的角色的当前模型
function usersOf(c) {
  const out = [];
  for (const [role, label] of ROLE_ROWS) {
    const ref = state.roles[role];
    if (ref?.connection === c.id) out.push({ role, label, model: ref.model });
  }
  return out;
}
function credState(c) {
  if (c.enabled === false) return "off";
  if (c.credential === "missing") return "missing";
  return "ok";
}

/* ---------- 视图路由 ---------- */

const app = $("#app");
function setView(view) {
  state.view = view;
  document.querySelectorAll(".demo-views button").forEach(b =>
    b.setAttribute("aria-pressed", String(b.dataset.view === view)));
  render();
}
document.querySelectorAll(".demo-views button").forEach(b =>
  b.addEventListener("click", () => setView(b.dataset.view)));

$("#firstRunToggle").addEventListener("click", e => {
  state.firstRun = !state.firstRun;
  e.currentTarget.setAttribute("aria-pressed", String(state.firstRun));
  if (state.view !== "connections") setView("connections"); else render();
});
$("#themeToggle").addEventListener("click", e => {
  const dark = document.documentElement.dataset.theme !== "dark";
  document.documentElement.dataset.theme = dark ? "dark" : "";
  e.currentTarget.setAttribute("aria-pressed", String(dark));
});

function render() {
  app.replaceChildren();
  if (state.view === "connections") renderSettings();
  else if (state.view === "composer") renderComposer();
  else renderNotes();
}

/* ---------- 连接页 ---------- */

function renderSettings() {
  app.replaceChildren(); // 行内交互直接调本函数重绘，不能只 append
  const page = el("main", "settings-page");
  const shell = el("div", "settings-shell");
  page.append(shell);

  const firstRun = state.firstRun || state.connections.length === 0;

  // 首跑：整页即引导 —— 不套壳设置页；结构对齐 onboarding 的 kicker + 内容区。
  if (firstRun) {
    const onb = el("main", "onb");
    onb.append(el("span", "onb-kicker", "初始配置"));
    onb.append(el("h1", "", "先连一个模型"));
    onb.append(el("p", "onb-lead", "登录订阅或粘贴 API Key；模型目录自动同步，之后可改。"));
    if (state.draft) {
      const ws = el("div", "conn-item is-open is-draft");
      ws.append(draftRowHead(), workspace(state.draft));
      onb.append(ws);
    } else {
      onb.append(templateRows(t => openDraft(t)));
    }
    onb.append(el("p", "onb-foot", "连接后可在「设置 → 模型」管理。"));
    shell.append(onb);
    app.append(page);
    return;
  }

  const head = el("header", "settings-header");
  head.append(el("h1", "", "模型"));
  shell.append(head);

  const connSec = el("section", "settings-section");
  const connHead = el("header", "");
  const connTitle = el("div", "");
  connTitle.append(el("h2", "", "连接"));
  connTitle.append(el("p", "", "表单与目录探测由来源插件渲染。"));
  connHead.append(connTitle, el("span", "count", `${state.connections.length} 条`));
  connSec.append(connHead);

  {
    const list = el("div", "connection-list");
    const q = state.query.trim().toLowerCase();
    if (state.draft) {
      const ws = el("div", "conn-item is-open is-draft");
      ws.append(draftRowHead(), workspace(state.draft));
      list.append(ws);
    }
    for (const c of state.connections) {
      if (q && !`${c.name} ${c.kind} ${c.models.map(m => m.id).join(" ")}`.toLowerCase().includes(q)) continue;
      const item = el("div", "conn-item" + (state.openWs === c.id ? " is-open" : ""));
      item.append(connectionRow(c));
      if (state.openWs === c.id) item.append(workspace(c));
      list.append(item);
    }
    connSec.append(list);
    const add = el("button", "add-connection");
    add.type = "button";
    add.append(ico("plus", 13), document.createTextNode("添加连接"));
    add.addEventListener("click", () => openAddSheet());
    connSec.append(add);
  }
  shell.append(connSec);

  const roleSec = el("section", "settings-section");
  const roleHead = el("header", "");
  const roleTitle = el("div", "");
  roleTitle.append(el("h2", "", "系统模型"));
  roleTitle.append(el("p", "", "每类工作用的模型；可再收窄可见集。"));
  roleHead.append(roleTitle);
  roleSec.append(roleHead);
  const roleList = el("div", "role-list");
  for (const [role, label, detail] of ROLE_ROWS) roleList.append(roleRow(role, label, detail));
  roleSec.append(roleList);
  shell.append(roleSec);

  app.append(page);
}

/* ---------- 连接行（折叠态） ---------- */

function connectionRow(c) {
  const row = el("button", "connection-row");
  row.type = "button";
  const open = state.openWs === c.id;
  row.setAttribute("aria-expanded", String(open));
  row.setAttribute("aria-label", `${open ? "收起" : "管理"}连接 ${c.name}`);

  row.append(mark(c.icon, c.name));
  const copy = el("span", "conn-copy");
  const name = el("span", "conn-name");
  name.append(el("span", "", c.name));
  name.append(el("span", "plugin-tag", KINDS[c.kind].plugin));
  if (c.enabled === false) name.append(el("span", "badge off", "停用"));
  copy.append(name);
  const sub = el("span", "conn-sub");
  sub.append(el("span", "mono", connHost(c)));
  const exposed = exposedModels(c).length;
  sub.append(el("span", "models", `${exposed} 模型`));
  const users = usersOf(c);
  if (users.length) sub.append(el("span", "uses-count", `${users.length} 角色`));
  copy.append(sub);
  row.append(copy);

  const st = credState(c);
  const dot = el("span", "state-dot" + (st === "missing" ? " missing" : st === "off" ? " off" : ""));
  row.append(dot);
  if (st === "missing") row.append(el("span", "conn-state", "缺少密钥"));
  row.append(iconCls(open ? "chevronD" : "chevronR", 15, "conn-chevron"));
  row.addEventListener("click", () => {
    state.openWs = open ? null : c.id;
    renderSettings();
  });
  return row;
}

function draftRowHead() {
  const row = el("div", "connection-row draft-head");
  row.append(mark(state.draft.icon, state.draft.name));
  const copy = el("span", "conn-copy");
  const name = el("span", "conn-name");
  name.append(el("span", "", `新建 · ${state.draft.name}`));
  name.append(el("span", "plugin-tag", KINDS[state.draft.kind].plugin));
  copy.append(name);
  copy.append(el("span", "conn-sub", "未保存"));
  row.append(copy);
  const cancel = el("button", "icon-button");
  cancel.type = "button";
  cancel.setAttribute("aria-label", "放弃新建");
  cancel.append(ico("close", 15));
  cancel.addEventListener("click", () => { state.draft = null; state.openWs = null; renderSettings(); });
  row.append(cancel);
  return row;
}

/* ---------- 连接工作区（行内展开） ---------- */

/*
 * 三段：凭据（保存后折叠成摘要）、模型管理表、底部动作条。
 * 探测用表单当前值 —— key 不必先保存（dsh 的 ProbeTarget 语义）。
 */
function workspace(c) {
  const isDraft = c === state.draft;
  const ws = el("div", "workspace");
  ws.addEventListener("click", e => e.stopPropagation());

  // --- 谁在用（magpie 的 agent chips）：点开即改该角色的模型 ---
  const users = isDraft ? [] : usersOf(c);
  if (users.length) {
    const chips = el("div", "use-chips");
    chips.append(el("span", "use-cap", "在用"));
    for (const u of users) {
      const ch = el("button", "use-chip");
      ch.type = "button";
      ch.title = `改${u.label}`;
      ch.append(el("span", "", u.label), el("span", "mono", u.model));
      ch.addEventListener("click", () => openModelPicker(ch, {
        current: state.roles[u.role], forRole: u.role,
        allowFollowDefault: false, showEffort: false, title: `选择${u.label}`,
        onChoose: ref => { state.roles[u.role] = ref; closeModelPicker(); renderSettings(); },
      }));
      chips.append(ch);
    }
    ws.append(chips);
  }

  // --- 凭据区 ---
  ws.append(credSection(c, isDraft));
  ws.append(el("div", "ws-rule"));

  // --- 模型管理表 ---
  const ms = el("section", "model-sec");
  const msHead = el("div", "ms-head");
  const msTitle = el("div", "ms-title");
  const open = c.models.filter(m => m.exposed !== false).length;
  msTitle.append(el("strong", "", "模型"));
  msTitle.append(el("span", "mono", c.models.length ? `${open}/${c.models.length} 已开放` : "目录为空"));
  msHead.append(msTitle, el("span", "grow"));

  const filter = textInput("", c.models.length > 8 ? `过滤 ${c.models.length} 个模型…` : "过滤模型…", "search");
  filter.classList.add("ms-filter");
  filter.hidden = c.models.length < 5;
  msHead.append(filter);

  const probeBtn = el("button", "btn small", "探测");
  probeBtn.type = "button";
  probeBtn.appendChild(svg(ICONS.refresh, 13));
  const blockReason = probeBlocker(c);
  probeBtn.disabled = !!blockReason;
  if (blockReason) probeBtn.title = blockReason;
  probeBtn.addEventListener("click", () => runProbe(c, ms, probeBtn));
  msHead.append(probeBtn);
  // 探测门槛跟随表单当前值：凭据输入冒泡到这里，实时解锁
  ws.addEventListener("input", () => {
    const reason = probeBlocker(c);
    probeBtn.disabled = !!reason;
    probeBtn.title = reason ?? "";
  });
  ms.append(msHead);

  const table = el("div", "model-table");
  ms.append(table);
  const status = el("p", "form-status");
  if (c.probeMsg) { status.textContent = c.probeMsg.text; if (c.probeMsg.cls) status.classList.add(c.probeMsg.cls); }
  else status.hidden = true;
  ms.append(status);
  const why = el("p", "ms-why");
  ms.append(why);

  const TABLE_CAP = 6;
  const drawTable = () => {
    table.replaceChildren();
    const f = filter.value.trim().toLowerCase();
    const items = c.models.filter(m => !f || `${m.id} ${m.name ?? ""}`.toLowerCase().includes(f));
    if (!c.models.length) {
      const empty = el("div", "ms-empty");
      empty.append(el("p", "", "目录为空。"));
      const manual = el("button", "btn small", "手动添加模型");
      manual.type = "button";
      manual.addEventListener("click", () => {
        c.models.push({ id: "", exposed: true, efforts: [], manual: true, _edit: true });
        drawTable();
      });
      empty.append(manual);
      table.append(empty);
    }
    // 列表有上限：超出就收进过滤，不无限摊开（极端目录交给覆盖层/过滤）
    for (const m of items.slice(0, TABLE_CAP)) table.append(modelRow(c, m, drawTable));
    if (!f && items.length > TABLE_CAP) {
      const more = el("button", "ms-more", `其余 ${items.length - TABLE_CAP} 个 · 输入过滤`);
      more.type = "button";
      more.addEventListener("click", () => { filter.hidden = false; filter.focus(); });
      table.append(more);
    }
    const n = c.models.filter(m => m.exposed !== false).length;
    msTitle.lastElementChild.textContent = c.models.length ? `${n}/${c.models.length} 开放` : "";
    why.textContent = c.models.length && n === 0 ? "全不开放 = 角色见完整目录。" : "";
  };
  filter.addEventListener("input", drawTable);
  c._drawTable = drawTable;
  c._status = status;
  drawTable();
  ws.append(ms);

  // --- 动作条 ---
  const bar = el("div", "ws-bar");
  if (!isDraft) {
    const del = el("button", "btn danger-quiet", "移除连接");
    del.type = "button";
    del.addEventListener("click", () => {
      if (del.dataset.arm) {
        state.connections = state.connections.filter(x => x !== c);
        for (const r of Object.keys(state.roles)) if (state.roles[r]?.connection === c.id) state.roles[r] = null;
        state.openWs = null;
        toast(`${c.name} 已移除`);
        renderSettings();
      } else {
        del.dataset.arm = "1";
        del.textContent = "再点一次确认移除";
        del.classList.add("armed");
      }
    });
    bar.append(del);
  }
  bar.append(el("span", "grow"));
  const cancel = el("button", "btn", isDraft ? "放弃" : "收起");
  cancel.type = "button";
  cancel.addEventListener("click", () => {
    if (isDraft) state.draft = null;
    state.openWs = null;
    renderSettings();
  });
  const save = el("button", "btn primary", isDraft ? "保存连接" : "保存");
  save.type = "button";
  save.addEventListener("click", () => saveConnection(c, isDraft, status));
  bar.append(cancel, save);
  ws.append(bar);
  return ws;
}

// 凭据就绪判定 —— 探测的门槛，逐 kind 由插件语义决定
function credReady(c) {
  if (c.auth === "oauth") return !!c.account;
  if (c.auth === "key-or-local") return true; // key 留空时走本机登录
  if (c.custom && !c.baseURL?.trim()) return false;
  if (!c.keyless && c.credential !== "configured" && !c.key) return false;
  return true;
}
function probeBlocker(c) {
  if (c.auth === "oauth") return c.account ? null : "先登录再探测目录";
  if (c.auth === "key-or-local") return null;
  if (c.custom && !c.baseURL?.trim()) return "先填 Base URL";
  if (!c.keyless && c.credential !== "configured" && !c.key) return "先填 API Key";
  return null;
}

// 凭据区：已配置 = 折叠摘要行；缺密钥/草稿/编辑中 = 完整表单（插件渲染，小字标边界）
function credSection(c, isDraft) {
  const sec = el("section", "cred-sec");
  if (!isDraft && c.credential === "configured" && !c._editingCred) {
    const line = el("button", "cred-summary");
    line.type = "button";
    line.append(ico("link", 14));
    line.append(el("span", "", c.auth === "oauth" ? `订阅登录 · ${c.account}` : c.keyless ? `本机端点 · ${c.baseURL}` : `${connHost(c)} · 密钥已存`));
    line.append(el("span", "grow"));
    line.append(el("span", "linklike", "编辑凭据"));
    line.addEventListener("click", () => { c._editingCred = true; renderSettings(); });
    sec.append(line);
    return sec;
  }
  const note = el("p", "form-note", `${KINDS[c.kind].plugin} 插件渲染`);
  sec.append(note);
  sec.append(credForm(c, isDraft));
  return sec;
}

function credForm(c, isDraft) {
  const form = el("div", "conn-form");
  const kind = KINDS[c.kind];

  if (c.auth === "oauth") {
    const card = el("div", "auth-card");
    const who = el("div", "who");
    who.append(el("strong", "", c.account ? `已登录 ${c.account}` : "使用 Codex 订阅"),
      el("small", "", c.account ? "令牌由插件保管" : "浏览器登录，目录随之同步"));
    const btn = el("button", "btn primary", c.account ? "重新登录" : "打开浏览器登录");
    btn.type = "button";
    card.append(who, btn);
    const status = el("p", "form-status");
    status.hidden = true;
    btn.addEventListener("click", () => {
      btn.disabled = true;
      status.hidden = false;
      status.textContent = "等待浏览器回执…";
      setTimeout(() => {
        c.account = "hua@example.com";
        c.credential = "configured";
        status.textContent = "已登录，正在读取目录…";
        status.className = "form-status ok";
        // 订阅类连接：登录即探测，目录自动落表（magpie 语义）
        setTimeout(() => { runProbe(c, null, null); renderSettings(); }, 500);
      }, 1100);
    });
    form.append(card, status);
    return form;
  }

  // key / keyless / key-or-local：预填模板的最少字段
  if (isDraft && c.custom) {
    const name = textInput(c.name === "自定义兼容端点" ? "" : c.name, "例如：公司网关");
    name.addEventListener("input", () => { c.name = name.value || "自定义兼容端点"; c._slug = slug(name.value); });
    form.append(field("名称", name));
  } else if (!isDraft) {
    const name = textInput(c.name, "连接名称");
    name.addEventListener("input", () => { c.name = name.value || c.name; });
    form.append(field("名称", name));
  }
  // URL 字段只在草稿自定义端点或编辑既有连接时出现；preset 草稿只剩 key（magpie 同语义）
  if ((isDraft && c.custom) || (!isDraft && (c._editingCred || c.baseURL))) {
    const url = textInput(c.baseURL ?? "", "https://…/v1");
    url.addEventListener("input", () => { c.baseURL = url.value; });
    form.append(field(c.keyless ? "端点" : "Base URL", url));
  } else if (isDraft && c.baseURL) {
    form.append(el("p", "form-note", `端点 ${(() => { try { return new URL(c.baseURL).host } catch { return c.baseURL } })()}`));
  }
  if (!c.keyless) {
    const key = textInput("", isDraft ? "sk-…" : "留空保留现有密钥", "password");
    key.addEventListener("input", () => { c.key = key.value; });
    form.append(field(c.auth === "key-or-local" ? "API Key（可留空）" : "API Key", key));
  }
  const hide = !isDraft && c._editingCred
    ? el("button", "btn small", "收起凭据")
    : null;
  if (hide) {
    hide.type = "button";
    hide.addEventListener("click", () => { c._editingCred = false; renderSettings(); });
    form.append(hide);
  }
  return form;
}

// 探测：把表单当前值（含未保存 key）发出去，结果进候选覆盖层 ——
// 勾选哪些开放，采纳才落表（dsh 的 fetch modal 语义：快开快关，批量决定）。
function runProbe(c, ms, btn) {
  if (btn) btn.disabled = true;
  c.probeMsg = { text: "正在读取…", cls: "" };
  if (c._status) { c._status.hidden = false; c._status.className = "form-status"; c._status.textContent = c.probeMsg.text; }
  setTimeout(() => {
    if (btn) btn.disabled = false;
    const found = KINDS[c.kind].catalog(c);
    c.probeMsg = { text: `读到 ${found.length} 个`, cls: "ok" };
    state.openWs = state.draft === c ? "draft" : c.id;
    renderSettings();
    openCandidates(c, found);
  }, 850);
}

// 候选覆盖层：探测到的目录 = 勾选「开放谁」。已有行回显其开放态，新模型预选。
function openCandidates(c, found) {
  const known = new Map(c.models.filter(m => m.id).map(m => [m.id, m]));
  const checks = new Map();

  const scrim = document.createElement("dialog");
  scrim.className = "scrim";
  const sheet = el("section", "mini-sheet");
  scrim.append(sheet);
  document.body.append(scrim);
  scrim.showModal();

  const head = el("div", "sheet-head");
  head.append(el("h2", "", `目录 · ${c.name}`));
  const close = el("button", "icon-button");
  close.type = "button";
  close.setAttribute("aria-label", "关闭");
  close.append(ico("close", 16));
  close.addEventListener("click", () => scrim.close());
  head.append(close);

  const body = el("div", "sheet-body");
  const tools = el("div", "cand-tools");
  const q = textInput("", "过滤…", "search");
  q.classList.add("ms-filter");
  q.style.inlineSize = "auto";
  const all = el("button", "btn small", "全选");
  const none = el("button", "btn small", "全不选");
  tools.append(q, all, none);
  const list = el("div", "cand-list");
  body.append(tools, list);

  const draw = () => {
    list.replaceChildren();
    const f = q.value.trim().toLowerCase();
    for (const m of found) {
      if (f && !`${m.id} ${m.name ?? ""}`.toLowerCase().includes(f)) continue;
      const lab = el("label", "vis-row");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      // 新候选预选；已有行回显当前开放态
      cb.checked = checks.has(m.id) ? checks.get(m.id) : (known.has(m.id) ? known.get(m.id).exposed !== false : true);
      cb.addEventListener("change", () => checks.set(m.id, cb.checked));
      lab.append(cb, el("span", "", m.name && m.name !== m.id ? m.name : m.id));
      if (m.name && m.name !== m.id) lab.append(el("code", "", m.id));
      if (m.images) lab.append(el("span", "badge mm", "多模态"));
      if (m.free) lab.append(el("span", "badge free", "free"));
      if (m.context) lab.append(el("span", "badge ctx", fmtCtx(m.context)));
      if (known.has(m.id)) lab.append(el("span", "badge", "已有"));
      list.append(lab);
    }
  };
  q.addEventListener("input", draw);
  const setAll = v => {
    const f = q.value.trim().toLowerCase();
    for (const m of found) {
      if (f && !`${m.id} ${m.name ?? ""}`.toLowerCase().includes(f)) continue;
      checks.set(m.id, v);
    }
    draw();
  };
  all.addEventListener("click", () => setAll(true));
  none.addEventListener("click", () => setAll(false));
  draw();

  const foot = el("div", "form-actions");
  const cancel = el("button", "btn", "取消");
  cancel.type = "button";
  cancel.addEventListener("click", () => scrim.close());
  const adopt = el("button", "btn primary", "开放所选");
  adopt.type = "button";
  adopt.addEventListener("click", () => {
    for (const m of found) {
      const want = checks.has(m.id) ? checks.get(m.id) : (known.has(m.id) ? known.get(m.id).exposed !== false : true);
      const row = known.get(m.id);
      if (row) row.exposed = want;
      else if (want) c.models.push({ ...m, exposed: true });
    }
    c.probeMsg = { text: `${c.models.filter(m => m.exposed !== false).length} 个开放`, cls: "ok" };
    scrim.close();
    renderSettings();
  });
  foot.append(el("span", "grow"), cancel, adopt);
  body.append(foot);
  sheet.append(head, body);
  scrim.addEventListener("click", e => { if (e.target === scrim) scrim.close(); });
  scrim.addEventListener("close", () => scrim.remove());
  q.focus();
}

/* ---------- 模型管理表 ---------- */

function testDot(m) {
  const d = el("span", "tdot" + (m.test?.state ? " " + m.test.state : ""));
  d.title = !m.test ? "未实测" : m.test.state === "wait" ? "测试中…"
    : m.test.state === "ok" ? `应答 ${m.test.ms} ms` : m.test.err;
  return d;
}

function modelRow(c, m, redraw) {
  const wrap = el("div", "mrow" + (m._edit ? " is-edit" : ""));
  const line = el("div", "mrow-line");

  // 开放勾选 = dsh 候选 picker 的去模态化：勾选即草稿，保存时提交
  const tick = document.createElement("input");
  tick.type = "checkbox";
  tick.checked = m.exposed !== false;
  tick.setAttribute("aria-label", `开放 ${m.id || "新模型"}`);
  tick.addEventListener("change", () => { m.exposed = tick.checked; redraw(); });
  line.append(tick);

  const name = el("span", "mname");
  name.append(el("span", "", m.name && m.name !== m.id ? m.name : (m.id || "未命名")));
  if (m.name && m.name !== m.id) name.append(el("code", "", m.id));
  if (m.manual) name.append(el("span", "badge", "手填"));
  line.append(name);
  if (m.images) line.append(el("span", "badge mm", "多模态"));
  if (m.free) line.append(el("span", "badge free", "free"));
  if (m.context) line.append(el("span", "badge ctx", fmtCtx(m.context)));
  const lv = m.efforts?.length
    ? el("span", "mlevels", (m.kept ?? m.efforts).map(e => EFFORT_LABELS[e] ?? e).join("·"))
    : null;
  if (lv) { lv.title = `可用强度 ${m.efforts.map(e => EFFORT_LABELS[e]).join(" / ")}`; line.append(lv); }
  line.append(el("span", "grow"));
  line.append(testDot(m));

  // 实测：dsh 没有，magpie 是右键菜单 —— 这里做成行尾按钮（键盘可达）
  const test = el("button", "icon-button test-btn", "");
  test.type = "button";
  test.title = "实测此模型";
  test.setAttribute("aria-label", `实测 ${m.id || "此模型"}`);
  test.append(svg(ICONS.play, 12));
  test.addEventListener("click", () => {
    m.test = { state: "wait" };
    redraw();
    setTimeout(() => {
      m.test = Math.random() > 0.15 ? { state: "ok", ms: 40 + Math.floor(Math.random() * 300) } : { state: "bad", err: "401 · 密钥被拒" };
      redraw();
    }, 350 + Math.random() * 500);
  });
  line.append(test);

  const toggle = el("button", "icon-button");
  toggle.type = "button";
  toggle.setAttribute("aria-expanded", String(!!m._edit));
  toggle.setAttribute("aria-label", `${m.id || "模型"} 高级设置`);
  toggle.append(svg(m._edit ? ICONS.chevronD : ICONS.chevronR, 13));
  toggle.addEventListener("click", () => { m._edit = !m._edit; redraw(); });
  line.append(toggle);
  wrap.append(line);

  if (m._edit) wrap.append(modelEditor(c, m, redraw));
  return wrap;
}

// 行内展开：显示名 / 上下文窗口 / 强度子集 / 图像输入 —— magpie 的 names 子表 +
// dsh 的容量字段；手填行的 id 也在此编辑。
function modelEditor(c, m, redraw) {
  const box = el("div", "medit");
  if (m.manual) {
    const id = textInput(m.id, "provider/model-id");
    id.addEventListener("change", () => { m.id = id.value.trim(); });
    box.append(field("模型 ID", id));
  }
  const nm = textInput(m.name && m.name !== m.id ? m.name : "", m.id || "显示名");
  nm.addEventListener("change", () => { m.name = nm.value.trim() || undefined; redraw(); });
  box.append(field("显示名", nm));

  const cx = textInput(m.context ? fmtCtx(m.context) : "", "256K · 留空用厂商值");
  cx.addEventListener("change", () => {
    const v = cx.value.trim().toLowerCase();
    const n = v.endsWith("m") ? parseFloat(v) * 1e6 : v.endsWith("k") ? parseFloat(v) * 1e3 : parseFloat(v);
    if (!v || Number.isFinite(n) && n > 0) { m.context = v ? Math.round(n) : undefined; redraw(); }
    else toast("上下文窗口需是 128K / 1M 这样的数值");
  });
  box.append(field("上下文窗口", cx));

  if (m.efforts?.length) {
    const lv = el("div", "effort-ticks");
    for (const e of m.efforts) {
      const lab = el("label", "tick");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = (m.kept ?? m.efforts).includes(e);
      cb.addEventListener("change", () => {
        const kept = m.efforts.filter(x => x === e ? cb.checked : (m.kept ?? m.efforts).includes(x));
        if (!kept.length) { cb.checked = true; toast("至少保留一级强度"); return; }
        m.kept = kept.length === m.efforts.length ? undefined : kept;
        redraw();
      });
      lab.append(cb, document.createTextNode(EFFORT_LABELS[e] ?? e));
      lv.append(lab);
    }
    box.append(field("强度", lv));
  }
  const img = el("label", "tick");
  const icb = document.createElement("input");
  icb.type = "checkbox";
  icb.checked = !!m.images;
  icb.addEventListener("change", () => { m.images = icb.checked; });
  img.append(icb, document.createTextNode("图像输入"));
  box.append(field("能力", img));

  const row = el("div", "medit-actions");
  const rst = el("button", "btn small", "恢复默认");
  rst.type = "button";
  rst.addEventListener("click", () => {
    delete m.name; delete m.kept; delete m.images; delete m.context;
    const base = KINDS[c.kind].catalog(c).find(x => x.id === m.id);
    if (base) Object.assign(m, { context: base.context, efforts: base.efforts, images: base.images, free: base.free });
    redraw();
  });
  const rm = el("button", "btn small danger-quiet", "移除");
  rm.type = "button";
  rm.addEventListener("click", () => { c.models = c.models.filter(x => x !== m); redraw(); });
  row.append(rst, rm);
  box.append(row);
  return box;
}

function saveConnection(c, isDraft, status) {
  if (isDraft) {
    const id = c._slug || slug(c.name);
    if (state.connections.some(x => x.id === id)) { status.hidden = false; status.className = "form-status err"; status.textContent = `标识 ${id} 已被占用，换个名称。`; return; }
    if (!credReady(c)) { status.hidden = false; status.className = "form-status err"; status.textContent = "凭据不完整：先填密钥或登录。"; return; }
    state.connections.push({ ...c, id, credential: "configured" });
    state.draft = null;
    toast(`已连接 ${c.name}`);
  } else {
    if (c.key) { c.credential = "configured"; delete c.key; }
    c._editingCred = false;
    toast(`${c.name} 已保存`);
  }
  state.openWs = null;
  renderSettings();
}

/* ---------- 添加连接 sheet ---------- */

function templateRows(onPick) {
  const box = el("div", "");
  for (const group of TEMPLATE_GROUPS) {
    const g = el("section", "tpl-group");
    const head = el("header", "");
    head.append(el("h4", "", group.title), el("small", "", group.note));
    g.append(head);
    const rows = el("div", "tpl-rows");
    for (const t of group.items) {
      const row = el("button", "tpl-row");
      row.type = "button";
      const ic = el("span", "tpl-icon");
      if (t.icon) ic.append(iconImg(t.icon));
      else ic.append(el("span", "fallback", t.name.slice(0, 1)));
      row.append(ic, el("span", "tpl-name", t.name));
      row.append(el("span", "plugin-tag", t.plugin ?? `${KINDS[t.kind].plugin} 插件`));
      row.append(el("span", "tpl-need", t.need));
      row.addEventListener("click", () => onPick(t));
      rows.append(row);
    }
    g.append(rows);
    box.append(g);
  }
  return box;
}

function openAddSheet() {
  const scrim = document.createElement("dialog");
  scrim.className = "scrim";
  const sheet = el("section", "sheet");
  scrim.append(sheet);
  document.body.append(scrim);
  scrim.showModal();

  const head = el("div", "sheet-head");
  head.append(el("h2", "", "添加连接"));
  const close = el("button", "icon-button");
  close.type = "button";
  close.setAttribute("aria-label", "关闭");
  close.append(ico("close", 16));
  close.addEventListener("click", () => scrim.close());
  head.append(close);
  const body = el("div", "sheet-body");
  const search = el("label", "settings-search");
  search.append(ico("search", 15));
  const input = document.createElement("input");
  input.placeholder = "找厂商或服务…";
  input.setAttribute("aria-label", "搜索连接方式");
  search.append(input);
  body.append(search);
  const host = el("div", "");
  body.append(host);
  const draw = () => {
    host.replaceChildren();
    const q = input.value.trim().toLowerCase();
    const filtered = TEMPLATE_GROUPS.map(g => ({
      ...g, items: g.items.filter(t => !q || t.name.toLowerCase().includes(q)),
    })).filter(g => g.items.length);
    if (!filtered.length) host.append(el("p", "picker-empty", "没有这个名字的连接方式。"));
    for (const group of filtered) {
      const g = el("section", "tpl-group");
      const h = el("header", "");
      h.append(el("h4", "", group.title), el("small", "", group.note));
      g.append(h);
      const rows = el("div", "tpl-rows");
      for (const t of group.items) {
        const row = el("button", "tpl-row");
        row.type = "button";
        const icn = el("span", "tpl-icon");
        if (t.icon) icn.append(iconImg(t.icon)); else icn.append(el("span", "fallback", t.name.slice(0, 1)));
        row.append(icn, el("span", "tpl-name", t.name), el("span", "plugin-tag", t.plugin ?? `${KINDS[t.kind].plugin} 插件`), el("span", "tpl-need", t.need));
        row.addEventListener("click", () => { scrim.close(); openDraft(t); });
        rows.append(row);
      }
      g.append(rows);
      host.append(g);
    }
  };
  input.addEventListener("input", draw);
  draw();
  sheet.append(head, body);
  scrim.addEventListener("click", e => { if (e.target === scrim) scrim.close(); });
  scrim.addEventListener("close", () => scrim.remove());
  input.focus();
}

// 新连接不再是独立 dialog：草稿工作区直接插在列表顶部 —— 加完即见模型表。
function openDraft(t) {
  state.draft = {
    name: t.name, kind: t.kind, icon: t.icon ?? null,
    auth: KINDS[t.kind].auth, keyless: !!t.keyless, custom: !!t.custom,
    credential: "missing", baseURL: t.baseURL ?? "", models: [],
  };
  state.openWs = "draft";
  if (state.view !== "connections") setView("connections");
  else renderSettings();
}

/* ---------- 角色行 + 可见性 ---------- */

function roleRow(role, label, detail) {
  const row = el("button", "role-row");
  row.type = "button";
  const copy = el("span", "role-copy");
  copy.append(el("strong", "", label), el("small", "", detail));
  const cur = el("span", "role-current");
  const bound = state.roles[role];
  const m = findModel(bound);
  if (m) cur.append(el("span", "mono", `${m.id}`), el("span", "", `· ${m.connection.name}`));
  else cur.append(el("span", "unset", "尚未配置"));
  cur.append(svg(ICONS.chevronR, 14));
  row.append(copy, cur);

  // 第二行：此角色可见的模型数（magpie 的 per-agent hidden 语义）
  const vis = el("button", "role-vis");
  vis.type = "button";
  const shown = modelsForRole(role).length, total = allModels().length;
  vis.append(ico("eye", 12), document.createTextNode(`可见 ${shown}/${total}`));
  vis.title = "管理此角色能看到的模型";
  vis.addEventListener("click", e => { e.stopPropagation(); openVisibility(role, label); });

  const wrap = el("div", "role-wrap");
  row.addEventListener("click", () => {
    openModelPicker(cur, {
      current: bound,
      title: `选择${label}`,
      allowFollowDefault: false,
      showEffort: false,
      forRole: role,
      onChoose: ref => {
        state.roles[role] = ref;
        closeModelPicker();
        toast(`${label} 已改为 ${ref.model}`);
        renderSettings();
      },
    });
  });
  wrap.append(row, vis);
  return wrap;
}

// 角色级可见性：按连接分组的勾选清单；在用的模型不可隐藏（magpie 同语义）。
function openVisibility(role, label) {
  const scrim = document.createElement("dialog");
  scrim.className = "scrim";
  const sheet = el("section", "sheet");
  scrim.append(sheet);
  document.body.append(scrim);
  scrim.showModal();
  const head = el("div", "sheet-head");
  head.append(el("h2", "", `${label} · 可见模型`));
  const close = el("button", "icon-button");
  close.type = "button";
  close.setAttribute("aria-label", "关闭");
  close.append(ico("close", 16));
  close.addEventListener("click", () => scrim.close());
  head.append(close);
  const body = el("div", "sheet-body");
  body.append(el("p", "form-note", "勾掉不给它看的；在用的不可隐藏。"));

  const hidden = state.roleHidden[role] ?? new Set();
  state.roleHidden[role] = hidden;
  const current = state.roles[role]?.model;

  for (const c of state.connections) {
    const exposed = exposedModels(c);
    if (!exposed.length) continue;
    const g = el("section", "tpl-group");
    const h = el("header", "");
    h.append(el("h4", "", c.name), el("small", "", `${exposed.length} 个已开放`));
    g.append(h);
    const rows = el("div", "vis-rows");
    for (const m of exposed) {
      const lab = el("label", "vis-row");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      const inUse = m.id === current;
      cb.checked = !hidden.has(m.id) || inUse;
      cb.disabled = inUse;
      if (inUse) lab.title = "角色当前正在使用";
      cb.addEventListener("change", () => { cb.checked ? hidden.delete(m.id) : hidden.add(m.id); });
      lab.append(cb, el("span", "", m.name && m.name !== m.id ? m.name : m.id));
      if (m.name && m.name !== m.id) lab.append(el("code", "", m.id));
      if (inUse) lab.append(el("span", "badge", "在用"));
      rows.append(lab);
    }
    g.append(rows);
    body.append(g);
  }
  const done = el("button", "btn primary", "完成");
  done.type = "button";
  done.addEventListener("click", () => { scrim.close(); renderSettings(); });
  const foot = el("div", "form-actions");
  foot.append(el("span", "grow"), done);
  body.append(foot);
  sheet.append(head, body);
  scrim.addEventListener("click", e => { if (e.target === scrim) scrim.close(); });
  scrim.addEventListener("close", () => { scrim.remove(); renderSettings(); });
}

/* ---------- 共享模型选择器 ---------- */

let openPicker = null;
function closeModelPicker() {
  if (openPicker?._esc) document.removeEventListener("keydown", openPicker._esc, true);
  openPicker?.remove();
  openPicker = null;
  document.removeEventListener("pointerdown", outsidePicker, true);
}
function outsidePicker(e) {
  if (openPicker && !openPicker.contains(e.target) && !openPicker._anchor?.contains(e.target)) closeModelPicker();
}

/*
 * 一个选择器，三个使用点：composer 胶囊、角色行、「谁在用」chip。
 * 布局 = magpie 的「图标导轨 + 分组列表」+ akashic 的「跟随默认」行；
 * effort 是底部常驻分段条。只列已开放、且对当前角色未隐藏的模型。
 */
function openModelPicker(anchor, { current, onChoose, allowFollowDefault = true, showEffort = true, forRole = null, title = "选择模型" } = {}) {
  closeModelPicker();
  const panel = el("div", "picker");
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-label", title);
  panel._anchor = anchor;

  const body = el("div", "picker-body");
  const rail = el("nav", "picker-rail");
  rail.setAttribute("aria-label", "按连接筛选");
  const main = el("div", "picker-main");
  body.append(rail, main);

  const search = el("label", "settings-search");
  search.append(ico("search", 14));
  const input = document.createElement("input");
  input.placeholder = "搜索模型";
  input.setAttribute("aria-label", "搜索模型");
  search.append(input);
  const list = el("div", "picker-list");
  main.append(search, list);

  const effStrip = el("div", "effort-strip");
  const foot = el("div", "picker-foot");
  const manage = document.createElement("a");
  manage.href = "#";
  manage.append(document.createTextNode("管理连接"), svg(ICONS.chevronR, 13));
  manage.addEventListener("click", e => { e.preventDefault(); closeModelPicker(); setView("connections"); });
  foot.append(manage);
  panel.append(body, effStrip, foot);

  let railSel = "all";
  let chosen = current === undefined ? effectiveRef() : current;
  let effort = state.effort;

  const hidden = forRole ? state.roleHidden[forRole] ?? new Set() : new Set();
  const conns = state.connections.filter(c => c.enabled !== false && exposedModels(c).length);
  const allItem = el("button", "rail-item");
  allItem.type = "button";
  allItem.setAttribute("aria-label", "全部连接");
  allItem.setAttribute("aria-pressed", "true");
  allItem.append(ico("grid", 15));
  rail.append(allItem, el("span", "rail-sep"));
  const railButtons = new Map([["all", allItem]]);
  for (const c of conns) {
    const b = el("button", "rail-item");
    b.type = "button";
    b.setAttribute("aria-label", c.name);
    b.title = c.name;
    if (c.icon) b.append(iconImg(c.icon)); else b.append(el("span", "fallback", c.name.slice(0, 1)));
    b.addEventListener("click", () => { railSel = c.id; syncRail(); drawList(); });
    rail.append(b);
    railButtons.set(c.id, b);
  }
  allItem.addEventListener("click", () => { railSel = "all"; syncRail(); drawList(); });
  function syncRail() {
    for (const [id, b] of railButtons) b.setAttribute("aria-pressed", String(id === railSel));
  }

  function drawList() {
    list.replaceChildren();
    const q = input.value.trim().toLowerCase();
    const pick = ref => {
      chosen = ref;
      effort = compatibleEffort(findModel(ref));
      onChoose?.(ref);
      drawList(); drawEffort();
    };

    if (allowFollowDefault && railSel === "all" && !q) {
      const g = el("section", "");
      g.append(el("div", "picker-group-title")).append(el("strong", "", "会话策略"));
      const dm = findModel(defaultRef());
      const opt = el("button", "picker-option" + (chosen === null ? " is-selected" : ""));
      opt.type = "button";
      opt.append(mark(dm?.connection.icon, "默"));
      const copy = el("span", "copy");
      copy.append(el("strong", "", "跟随默认模型"), el("small", "", dm ? `${dm.id} · ${dm.connection.name}` : "默认模型尚未配置"));
      opt.append(copy);
      if (chosen === null) opt.append(iconCls("check", 15, "check"));
      opt.addEventListener("click", () => pick(null));
      g.append(opt);
      list.append(g);
    }

    let any = false;
    for (const c of conns) {
      if (railSel !== "all" && c.id !== railSel) continue;
      const items = exposedModels(c).filter(m => !hidden.has(m.id))
        .filter(m => !q || `${m.id} ${m.name ?? ""} ${c.name}`.toLowerCase().includes(q));
      if (!items.length) continue;
      any = true;
      const g = el("section", "");
      const gt = el("div", "picker-group-title");
      gt.append(el("strong", "", c.name), el("span", "mono", `${items.length}`));
      g.append(gt);
      for (const m of items) {
        const selected = chosen && chosen.connection === c.id && chosen.model === m.id;
        const opt = el("button", "picker-option" + (selected ? " is-selected" : ""));
        opt.type = "button";
        opt.append(mark(c.icon, m.name ?? m.id));
        const copy = el("span", "copy");
        copy.append(el("strong", "", m.name && m.name !== m.id ? m.name : m.id));
        copy.append(el("small", "mono", m.id));
        opt.append(copy);
        if (m.free) opt.append(el("span", "badge free", "free"));
        if (m.context) opt.append(el("span", "badge", fmtCtx(m.context)));
        if (m.test?.state === "ok") opt.append(el("span", "badge ms", `${m.test.ms}ms`));
        if (selected) opt.append(iconCls("check", 15, "check"));
        opt.addEventListener("click", () => pick({ connection: c.id, model: m.id }));
        g.append(opt);
      }
      list.append(g);
    }
    if (!any) list.append(el("p", "picker-empty", "无匹配模型"));
  }

  function compatibleEffort(m) {
    const avail = m?.kept ?? m?.efforts;
    if (!avail?.length) return "";
    if (avail.includes(state.effort)) return state.effort;
    return avail.includes("medium") ? "medium" : avail[0];
  }

  function drawEffort() {
    effStrip.replaceChildren();
    const m = findModel(chosen ?? defaultRef());
    const avail = m?.kept ?? m?.efforts;
    if (!showEffort || !avail?.length) { effStrip.hidden = true; return; }
    effStrip.hidden = false;
    const head = el("span", "effort-head");
    head.append(ico("spark", 14), el("strong", "", "思考强度"));
    const segs = el("div", "effort-segs");
    for (const e of avail) {
      const b = el("button", "", EFFORT_LABELS[e] ?? e);
      b.type = "button";
      b.setAttribute("aria-pressed", String(e === effort));
      b.addEventListener("click", () => {
        if (chosen === null) { chosen = defaultRef(); onChoose?.(chosen); }
        effort = e;
        state.effort = e;
        drawList(); drawEffort();
      });
      segs.append(b);
    }
    const cur = el("span", "mono", EFFORT_LABELS[effort] ?? effort);
    cur.style.cssText = "font-size:12px;color:var(--ak-ink-muted)";
    effStrip.append(head, segs, cur);
  }

  input.addEventListener("input", drawList);
  drawList(); drawEffort();

  document.body.append(panel);
  const r = anchor.getBoundingClientRect();
  panel.style.position = "fixed";
  panel.style.inlineSize = Math.min(560, innerWidth - 24) + "px";
  panel.style.maxBlockSize = Math.min(430, innerHeight - 24) + "px";
  const w = panel.offsetWidth, h = panel.offsetHeight;
  let x = Math.min(Math.max(12, r.right - w), innerWidth - w - 12);
  let y = r.top - 8 - h;
  if (y < 12) y = Math.min(r.bottom + 8, innerHeight - h - 12);
  panel.style.left = x + "px";
  panel.style.top = y + "px";

  panel.addEventListener("keydown", e => {
    if (e.key === "Escape") { e.stopPropagation(); closeModelPicker(); anchor.focus(); }
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      const opts = [...list.querySelectorAll("button")];
      const at = opts.indexOf(document.activeElement);
      const next = (at + (e.key === "ArrowDown" ? 1 : -1) + opts.length) % opts.length;
      e.preventDefault();
      opts[next]?.focus();
    }
  });
  const esc = e => {
    if (e.key !== "Escape") return;
    e.stopPropagation();
    closeModelPicker();
    anchor.focus();
  };
  panel._esc = esc;
  document.addEventListener("keydown", esc, true);
  document.addEventListener("pointerdown", outsidePicker, true);
  openPicker = panel;
  input.focus();
}

/* ---------- composer 演示 ---------- */

function renderComposer() {
  app.replaceChildren();
  const stage = el("main", "composer-stage");
  const col = el("div", "composer-demo");
  const note = el("p", "", "对话输入区的模型胶囊。选择器只列已开放的模型；实测延迟随模型一起带进来。");
  note.style.cssText = "color:var(--ak-ink-muted);font-size:13px;margin:0 0 14px 4px";
  col.append(note);

  const card = el("div", "composer-card");
  const inputArea = el("div", "composer-input", "给 Akashic 留言…");
  const footBar = el("div", "composer-foot");

  const eff = findModel(effectiveRef());
  const capsule = el("button", "capsule");
  capsule.type = "button";
  capsule.append(mark(eff?.connection.icon, "模"));
  const dm = findModel(defaultRef());
  capsule.append(el("strong", "", state.selection ? (eff?.name ?? eff?.id ?? "?") : `跟随默认${dm ? `：${dm.name ?? dm.id}` : ""}`));
  const avail = eff?.kept ?? eff?.efforts;
  if (avail?.length) capsule.append(el("span", "eff", EFFORT_LABELS[state.effort] ?? state.effort));
  capsule.append(svg(ICONS.chevronD, 13));
  capsule.addEventListener("click", () => {
    if (openPicker) { closeModelPicker(); return; }
    openModelPicker(capsule, {
      onChoose: ref => { state.selection = ref; closeModelPicker(); renderComposer(); },
    });
  });

  footBar.append(capsule, el("span", "grow"));
  const send = el("span", "send-dot");
  send.append(svg(["M12 19V5", "m5 12 7-7 7 7"], 15));
  footBar.append(send);
  card.append(inputArea, footBar);
  col.append(card);

  const hint = el("p", "", "导轨按连接过滤 · 底部一行完成思考强度 · 强度子集受连接里「开放强度」约束");
  hint.style.cssText = "color:var(--ak-ink-muted);font-size:12.5px;margin:12px 0 0 4px";
  col.append(hint);
  stage.append(col);
  app.append(stage);
}

/* ---------- 设计说明 ---------- */

function renderNotes() {
  app.replaceChildren();
  const w = el("main", "notes-wrap");
  w.innerHTML = `
    <h1>模型连接 · 交互重构说明</h1>
    <p>交互链：连接持凭据 → 探测读目录 → 覆盖层勾选开放 → 行内逐模型管理 → 角色收窄可见集 → 会话固定模型与强度。</p>

    <h2>对照真实实现</h2>
    <table class="notes-table">
      <tr><th>环节</th><th>dsh</th><th>magpie</th><th>此原型</th></tr>
      <tr><td>建连接</td><td>Add 卡内两模式，凭据+目录一趟完成</td><td>sheet 分组，preset 只要 key</td><td>sheet 选模板 → 草稿工作区插在列表顶部</td></tr>
      <tr><td>探测</td><td>表单当前值直发 endpoint（含未保存 key）</td><td>保存后自动问 vendor</td><td>同 dsh：探测按钮用草稿值；订阅登录即探测</td></tr>
      <tr><td>候选挑选</td><td>modal：搜索+全选+checkbox 采纳</td><td>芯片云点击切换</td><td><b>覆盖层勾选开放</b>：新模型预选、已有行回显开放态；快开快关，适合长目录</td></tr>
      <tr><td>逐模型管理</td><td>行内展开改 id/容量/模态</td><td>names 子表：改名/图像/强度子集</td><td>行内展开：显示名、上下文、开放强度、图像</td></tr>
      <tr><td>多模态</td><td>input modalities checkbox</td><td>「Accepts images」勾</td><td>目录/候选/选择器统一「多模态」徽标</td></tr>
      <tr><td>实测</td><td>无</td><td>右键单测 / 全测</td><td>行尾按钮（键盘可达）</td></tr>
      <tr><td>谁在用</td><td>无</td><td>provider 行 agent chips</td><td>连接行计数 + 工作区顶部 chips 直接改</td></tr>
      <tr><td>可见性</td><td>目录级 models 数组</td><td>每 agent hidden</td><td>开放（连接级）× 可见（角色级）两层</td></tr>
    </table>

    <h2>布局取舍</h2>
    <ul>
      <li><b>覆盖层 vs 行内</b>：探测结果与角色可见性都是「可能极长的勾选集」→ 做成可快速开合的覆盖层；管理表行内但有 6 行上限，溢出交给过滤。</li>
      <li><b>凭据折叠</b>：就绪后收成一行摘要；凭据一次性的，管理是日常的。</li>
      <li><b>文案节食</b>：行内只剩判断所需的位（名称/id/能力徽标/开放计数），解释性文字移到 title。</li>
    </ul>

    <h2>插件边界</h2>
    <ul>
      <li>凭据表单与目录应答由来源插件渲染/提供（oauth / key / 本机各不同）；宿主持有列表、表格、选择器与保存。</li>
      <li>探测返回的能力映射（context/efforts/images/free）由插件给 —— 宿主不理解 vendor 差异。</li>
    </ul>`;
  app.append(w);
}

/* ---------- 工具 ---------- */

function field(labelText, input) {
  const f = el("label", "field");
  f.append(el("span", "", labelText), input);
  return f;
}
function textInput(value = "", placeholder = "", type = "text") {
  const i = document.createElement("input");
  i.type = type;
  i.value = value;
  i.placeholder = placeholder;
  return i;
}

/* ---------- 启动 ---------- */
render();

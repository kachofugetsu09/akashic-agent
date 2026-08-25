import {
  AlertCircle,
  ArrowUp,
  Check,
  ChevronDown,
  ChevronRight,
  Copy,
  LoaderCircle,
  MessageSquarePlus,
  PanelLeft,
  Paperclip,
  Plus,
  Search,
  Sparkles,
  Star,
  Wrench,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState, type Dispatch, type SetStateAction } from "react";
import { MaterialIconButton } from "../../theme/src/material-react";
import "./styles.css";
import "./chat-product-variants.css";

export type ChatProductVariantId = "liked" | "waku" | "grok" | "m3";

type VariantSpec = {
  id: ChatProductVariantId;
  label: string;
  tagline: string;
  stack: string;
  summary: string;
  material: string;
};

type DemoSession = {
  id: string;
  title: string;
  preview: string;
  time: string;
  active: boolean;
  status: "idle" | "running" | "waiting";
  group: "today" | "week" | "older";
};

const VARIANTS: VariantSpec[] = [
  {
    id: "liked",
    label: "精选",
    tagline: "交替过程 · 浅灰无边框 · 克制输入",
    stack: "Akashic 交替 thinking/tool · 无卡片过程轨 · Waku 模型弹层 · streamdown",
    summary:
      "保留交替 thinking/tool，但过程区去掉卡片：无边框、比正文小一号、浅灰字；工具展开也是轻量明细。模型弹层仍抄 Waku；空态输入为长条椭圆。",
    material: "发送 filled · 选中 secondary-container · surface 阶梯",
  },
  {
    id: "waku",
    label: "Waku 对照",
    tagline: "原教旨布局",
    stack: "react-markdown + markdown-veil · Virtuoso · 提交节流 ≤8.3Hz · 非 Markstream",
    summary: "侧栏 #f3f3f3、760/540/720 几何、activity 卡。流式妙招：chunk 合并提交 + 虚拟列表 + veil 淡入，不用逐 token 全量重排。",
    material: "对照",
  },
  {
    id: "grok",
    label: "Grok 对照",
    tagline: "Sand 暖纸感",
    stack: "自研 assistantContentBlocks · sand-message-prose · 非 Markstream",
    summary: "分区会话、690 共轴、暖灰纸面与工具卡。模型通常在设置/顶栏，本对照保留舞台顶栏。",
    material: "对照",
  },
  {
    id: "m3",
    label: "M3 初稿",
    tagline: "上一版关键点",
    stack: "@material/web filter-chip · 偏方输入框",
    summary: "上一轮方案，方便对比：输入框偏方、过程折叠成一张卡、模型 chip 在顶栏。",
    material: "发送 · tonal 新会话 · filter-chip",
  },
];

const SESSIONS: DemoSession[] = [
  {
    id: "s1",
    title: "整理本周阅读笔记",
    preview: "正在核对冲突结论…",
    time: "刚刚",
    active: true,
    status: "running",
    group: "today",
  },
  {
    id: "s2",
    title: "检查定时任务失败原因",
    preview: "等待确认是否重跑 wake",
    time: "14:20",
    active: false,
    status: "waiting",
    group: "today",
  },
  {
    id: "s3",
    title: "对比两个模型的工具轨迹",
    preview: "9 条消息 · 已完成",
    time: "昨天",
    active: false,
    status: "idle",
    group: "week",
  },
  {
    id: "s4",
    title: "写 Gate 验收清单",
    preview: "草稿已落盘",
    time: "周一",
    active: false,
    status: "idle",
    group: "week",
  },
  {
    id: "s5",
    title: "迁移插件安装链说明",
    preview: "归档",
    time: "8/12",
    active: false,
    status: "idle",
    group: "older",
  },
];

const GROUP_LABEL = { today: "今天", week: "本周", older: "更早" } as const;

const DEMO_MODELS = [
  { id: "flash", provider: "opencode", name: "deepseek-v4-flash", favorite: true },
  { id: "pro", provider: "opencode", name: "deepseek-v4", favorite: false },
  { id: "codex", provider: "codex", name: "gpt-5.4", favorite: true },
  { id: "or", provider: "openrouter", name: "claude-sonnet", favorite: false },
] as const;

/** Bold product study: clone Waku/Grok languages; Material only on critical actions. */
export function ChatProductVariants() {
  const [variant, setVariant] = useState<ChatProductVariantId>("liked");
  const [text, setText] = useState("");
  const [openTool, setOpenTool] = useState(true);
  const [collapsed, setCollapsed] = useState<Set<DemoSession["group"]>>(new Set(["older"]));
  const [model, setModel] = useState<(typeof DEMO_MODELS)[number]["id"]>("flash");
  const spec = VARIANTS.find((item) => item.id === variant) ?? VARIANTS[0];

  return (
    <div className="cpv-page" data-variant={variant}>
      <header className="cpv-chooser" aria-label="对话产品方案">
        <div className="cpv-chooser__intro">
          <span>AKASHIC · BOLD STUDY</span>
          <h1>精选：交替过程 + 浅灰过程轨 + 克制输入</h1>
          <p>
            流式：两边都不用 Markstream。我们产品侧是 <b>streamdown</b>；Waku 是 react-markdown + veil + ≤8.3Hz 提交；Grok 是自研 block 解析。
            默认看「精选」——保留你喜欢的交替 thinking/tool，并抄模型弹层与长条输入。
          </p>
        </div>
        <div className="cpv-chooser__cards" role="radiogroup" aria-label="方案">
          {VARIANTS.map((item) => (
            <button
              key={item.id}
              type="button"
              role="radio"
              aria-checked={variant === item.id}
              className={variant === item.id ? "active" : undefined}
              onClick={() => setVariant(item.id)}
            >
              <strong>{item.label}</strong>
              <span>{item.tagline}</span>
            </button>
          ))}
        </div>
        <aside className="cpv-chooser__brief" aria-live="polite">
          <h2>{spec.label}</h2>
          <p className="cpv-stack">{spec.stack}</p>
          <p>{spec.summary}</p>
          <p className="cpv-m3-note"><b>Material：</b>{spec.material}</p>
        </aside>
      </header>

      <div className="cpv-stage" aria-label={`${spec.label}预览`}>
        {variant === "liked" ? (
          <LikedFrame
            sessions={SESSIONS}
            collapsed={collapsed}
            onToggle={(g) => toggleGroup(g, setCollapsed)}
            openTool={openTool}
            onToggleTool={() => setOpenTool((v) => !v)}
            text={text}
            onText={setText}
            model={model}
            onModel={setModel}
          />
        ) : null}
        {variant === "waku" ? (
          <WakuFrame
            sessions={SESSIONS}
            collapsed={collapsed}
            onToggle={(g) => toggleGroup(g, setCollapsed)}
            openTool={openTool}
            onToggleTool={() => setOpenTool((v) => !v)}
            text={text}
            onText={setText}
          />
        ) : null}
        {variant === "grok" ? (
          <GrokFrame
            sessions={SESSIONS}
            collapsed={collapsed}
            onToggle={(g) => toggleGroup(g, setCollapsed)}
            openTool={openTool}
            onToggleTool={() => setOpenTool((v) => !v)}
            text={text}
            onText={setText}
          />
        ) : null}
        {variant === "m3" ? (
          <M3Frame
            sessions={SESSIONS}
            collapsed={collapsed}
            onToggle={(g) => toggleGroup(g, setCollapsed)}
            openTool={openTool}
            onToggleTool={() => setOpenTool((v) => !v)}
            text={text}
            onText={setText}
            model={model}
            onModel={setModel}
          />
        ) : null}
      </div>
    </div>
  );
}

function toggleGroup(
  group: DemoSession["group"],
  setCollapsed: Dispatch<SetStateAction<Set<DemoSession["group"]>>>,
) {
  setCollapsed((current) => {
    const next = new Set(current);
    if (next.has(group)) next.delete(group);
    else next.add(group);
    return next;
  });
}

/* ——— Liked / curated ——— */

function LikedFrame(props: FrameProps & {
  model: (typeof DEMO_MODELS)[number]["id"];
  onModel: (id: (typeof DEMO_MODELS)[number]["id"]) => void;
}) {
  const [toolOpen, setToolOpen] = useState(true);

  return (
    <div className="liked-root">
      <aside className="liked-sidebar">
        <div className="liked-sidebar__new">
          <MaterialIconButton variant="tonal" label="新会话" onClick={() => undefined}>
            <MessageSquarePlus size={18} aria-hidden="true" />
          </MaterialIconButton>
          <span>新会话</span>
        </div>
        <div className="liked-sidebar__search">
          <Search size={14} />
          <input placeholder="搜索会话" aria-label="搜索会话" />
        </div>
        <nav className="liked-sidebar__list" aria-label="会话">
          <SessionGroups
            tone="m3"
            sessions={props.sessions}
            collapsed={props.collapsed}
            onToggle={props.onToggle}
          />
        </nav>
      </aside>

      <section className="liked-main">
        <header className="liked-topbar">
          <div>
            <strong>整理本周阅读笔记</strong>
            <small>想着 · 做着 · 交替可见</small>
          </div>
        </header>

        <div className="liked-transcript">
          <div className="liked-col">
            <div className="liked-user">
              帮我把这周三篇长文压成可执行笔记，并核对有没有互相矛盾的结论。
            </div>

            <ol className="liked-trace" aria-label="思考与工具交替">
              <li className="liked-trace__item thinking">
                <span className="liked-trace__node" aria-hidden="true" />
                <div className="liked-think">
                  <header>
                    <Sparkles size={13} />
                    <span>思考</span>
                    <small>进行中</small>
                  </header>
                  <p>先抽出主张，再对照证据链，最后标出冲突点。发布节奏可能是分歧来源。</p>
                </div>
              </li>
              <li className="liked-trace__item tool">
                <span className="liked-trace__node diamond" aria-hidden="true" />
                <div className={`liked-tool ${toolOpen ? "open" : ""}`}>
                  <button type="button" className="liked-tool__head" aria-expanded={toolOpen} onClick={() => setToolOpen((v) => !v)}>
                    <Wrench size={14} />
                    <strong>retrieve_notes</strong>
                    <span className="liked-tool__state">完成</span>
                    <ChevronDown size={14} className={toolOpen ? "open" : undefined} />
                  </button>
                  {toolOpen ? (
                    <div className="liked-tool__body">
                      <div className="liked-tool__kv">
                        <span>week</span>
                        <code>2026-W34</code>
                      </div>
                      <div className="liked-tool__kv">
                        <span>sources</span>
                        <code>3</code>
                      </div>
                      <pre>matched: 3 · conflicts: 1</pre>
                    </div>
                  ) : null}
                </div>
              </li>
              <li className="liked-trace__item thinking">
                <span className="liked-trace__node" aria-hidden="true" />
                <div className="liked-think done">
                  <header>
                    <Sparkles size={13} />
                    <span>思考</span>
                    <small>已折叠</small>
                  </header>
                  <p>冲突只在发布节奏；建议用切片发布并保留周五复盘。</p>
                </div>
              </li>
            </ol>

            <div className="liked-answer">
              <AssistantDoc />
            </div>
            <div className="liked-actions">
              <button type="button"><Copy size={14} />复制</button>
              <button type="button"><Check size={14} />已读</button>
            </div>
          </div>
        </div>

        <div className="liked-dock">
          <div className={`liked-composer ${props.text.trim() ? "has-text" : "empty"}`}>
            <textarea
              rows={1}
              value={props.text}
              onChange={(e) => props.onText(e.target.value)}
              placeholder="继续布置任务…"
              aria-label="消息输入"
            />
            <div className="liked-composer__bar">
              <ModelPopover model={props.model} onModel={props.onModel} />
              <div className="liked-composer__trail">
                <button type="button" className="liked-attach" aria-label="添加附件"><Paperclip size={16} /></button>
                <MaterialIconButton
                  variant="filled"
                  label="发送消息"
                  disabled={!props.text.trim()}
                  onClick={() => undefined}
                >
                  <ArrowUp size={18} aria-hidden="true" />
                </MaterialIconButton>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

function ModelPopover({
  model,
  onModel,
}: {
  model: (typeof DEMO_MODELS)[number]["id"];
  onModel: (id: (typeof DEMO_MODELS)[number]["id"]) => void;
}) {
  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState<"favorites" | "opencode" | "codex" | "openrouter">("opencode");
  const [query, setQuery] = useState("");
  const root = useRef<HTMLDivElement>(null);
  const current = DEMO_MODELS.find((item) => item.id === model) ?? DEMO_MODELS[0];

  useEffect(() => {
    if (!open) return;
    const onDoc = (event: MouseEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  const rows = DEMO_MODELS.filter((item) => {
    if (query.trim()) {
      return `${item.name} ${item.provider}`.toLowerCase().includes(query.trim().toLowerCase());
    }
    if (tab === "favorites") return item.favorite;
    return item.provider === tab;
  });

  return (
    <div className="liked-model" ref={root}>
      <button
        type="button"
        className={`liked-model__trigger ${open ? "open" : ""}`}
        aria-haspopup="dialog"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        <span className="liked-model__mark">{current.provider.slice(0, 1).toUpperCase()}</span>
        <span className="liked-model__name">{current.name}</span>
        <ChevronDown size={12} />
      </button>
      {open ? (
        <div className="liked-model__pop" role="dialog" aria-label="选择模型">
          <div className="liked-model__rail" aria-label="来源">
            <button type="button" className={tab === "favorites" ? "active" : undefined} onClick={() => setTab("favorites")} aria-label="收藏">
              <Star size={16} />
            </button>
            <span className="liked-model__sep" />
            {(["opencode", "codex", "openrouter"] as const).map((id) => (
              <button
                key={id}
                type="button"
                className={tab === id ? "active" : undefined}
                onClick={() => setTab(id)}
                aria-label={id}
              >
                {id.slice(0, 1).toUpperCase()}
              </button>
            ))}
          </div>
          <div className="liked-model__pane">
            <label className="liked-model__search">
              <Search size={14} />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="搜索模型"
                aria-label="搜索模型"
              />
            </label>
            <div className="liked-model__list">
              {rows.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className={item.id === model ? "active" : undefined}
                  onClick={() => {
                    onModel(item.id);
                    setOpen(false);
                  }}
                >
                  <span>
                    <strong>{item.name}</strong>
                    <small>{item.provider}</small>
                  </span>
                  {item.id === model ? <Check size={14} /> : item.favorite ? <Star size={12} /> : null}
                </button>
              ))}
              {!rows.length ? <p className="liked-model__empty">无匹配</p> : null}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

/* ——— Waku clone ——— */

function WakuFrame(props: FrameProps) {
  return (
    <div className="waku-root">
      <aside className="waku-sidebar">
        <header className="waku-sidebar__head">
          <span className="waku-mark" aria-hidden="true" />
          <button type="button" className="waku-icon" aria-label="隐藏侧栏"><PanelLeft size={16} /></button>
        </header>
        <div className="waku-sidebar__actions">
          <button type="button" className="waku-action"><Plus size={15} />新任务</button>
          <button type="button" className="waku-action"><Search size={15} />搜索</button>
        </div>
        <nav className="waku-sidebar__list" aria-label="任务">
          <SessionGroups tone="waku" sessions={props.sessions} collapsed={props.collapsed} onToggle={props.onToggle} />
        </nav>
        <footer className="waku-sidebar__foot">
          <button type="button" className="waku-action muted">设置</button>
        </footer>
      </aside>
      <section className="waku-main">
        <div className="waku-transcript">
          <div className="waku-col">
            <div className="waku-user-wrap">
              <div className="waku-user">帮我把这周三篇长文压成可执行笔记，并核对有没有互相矛盾的结论。</div>
            </div>
            <ActivityCard open={props.openTool} onToggle={props.onToggleTool} />
            <AssistantDoc />
          </div>
        </div>
        <div className="waku-dock">
          <div className="waku-composer">
            <textarea rows={1} value={props.text} onChange={(e) => props.onText(e.target.value)} placeholder="继续布置任务…" aria-label="消息输入" />
            <div className="waku-composer__row">
              <button type="button" className="waku-pill">deepseek-v4-flash</button>
              <div className="waku-composer__trail">
                <button type="button" className="waku-icon" aria-label="附件"><Paperclip size={15} /></button>
                <button type="button" className="waku-send" aria-label="发送" disabled={!props.text.trim()}><ArrowUp size={16} /></button>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

/* ——— Grok / Sand clone ——— */

function GrokFrame(props: FrameProps) {
  return (
    <div className="sand-root">
      <aside className="sand-sidebar">
        <header className="sand-sidebar__head">
          <strong>会话</strong>
          <button type="button" className="sand-icon" aria-label="新聊天"><MessageSquarePlus size={16} /></button>
        </header>
        <nav className="sand-sidebar__list" aria-label="会话">
          <SessionGroups tone="sand" sessions={props.sessions} collapsed={props.collapsed} onToggle={props.onToggle} />
        </nav>
      </aside>
      <section className="sand-main">
        <header className="sand-topbar">
          <span className="sand-avatar">整</span>
          <div>
            <strong>整理本周阅读笔记</strong>
            <small>运行中 · retrieve_notes</small>
          </div>
        </header>
        <div className="sand-transcript">
          <article className="sand-row sand-row--user">
            <div className="sand-bubble sand-bubble--user">帮我把这周三篇长文压成可执行笔记，并核对有没有互相矛盾的结论。</div>
          </article>
          <article className="sand-row">
            <button type="button" className="sand-process" onClick={props.onToggleTool} aria-expanded={props.openTool}>
              <ChevronDown size={14} className={props.openTool ? "open" : undefined} />
              过程 · 思考与工具
            </button>
            {props.openTool ? (
              <div className="sand-tool">
                <Wrench size={14} />
                <code>retrieve_notes</code>
                <span>matched 3 · conflicts 1</span>
              </div>
            ) : null}
            <div className="sand-bubble sand-bubble--agent"><AssistantDoc /></div>
          </article>
        </div>
        <div className="sand-dock">
          <div className="sand-prompt">
            <textarea rows={1} value={props.text} onChange={(e) => props.onText(e.target.value)} placeholder="Message…" aria-label="消息输入" />
            <div className="sand-prompt__row">
              <button type="button" className="sand-round" aria-label="附件"><Paperclip size={14} /></button>
              <button type="button" className="sand-round sand-round--send" aria-label="发送" disabled={!props.text.trim()}><ArrowUp size={14} /></button>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

/* ——— M3 previous draft ——— */

function M3Frame(props: FrameProps & {
  model: (typeof DEMO_MODELS)[number]["id"];
  onModel: (id: (typeof DEMO_MODELS)[number]["id"]) => void;
}) {
  return (
    <div className="m3-root">
      <aside className="m3-sidebar">
        <div className="m3-sidebar__new">
          <MaterialIconButton variant="tonal" label="新会话" onClick={() => undefined}>
            <MessageSquarePlus size={18} aria-hidden="true" />
          </MaterialIconButton>
          <span>新会话</span>
        </div>
        <div className="m3-sidebar__search">
          <Search size={14} />
          <input placeholder="搜索会话" aria-label="搜索会话" />
        </div>
        <nav className="m3-sidebar__list" aria-label="会话">
          <SessionGroups tone="m3" sessions={props.sessions} collapsed={props.collapsed} onToggle={props.onToggle} />
        </nav>
      </aside>
      <section className="m3-main">
        <header className="m3-topbar">
          <div>
            <strong>整理本周阅读笔记</strong>
            <small>工具进行中</small>
          </div>
          <span className="m3-chip-static">{props.model}</span>
        </header>
        <div className="m3-transcript">
          <div className="m3-col">
            <div className="m3-user">帮我把这周三篇长文压成可执行笔记，并核对有没有互相矛盾的结论。</div>
            <button type="button" className="m3-activity" aria-expanded={props.openTool} onClick={props.onToggleTool}>
              <LoaderCircle size={14} className="spin" />
              <strong>retrieve_notes</strong>
              <span>匹配 3 · 冲突 1</span>
              <ChevronDown size={14} className={props.openTool ? "open" : undefined} />
            </button>
            {props.openTool ? (
              <div className="m3-activity-body">
                <code>{`{ "week": "2026-W34", "sources": 3 }`}</code>
              </div>
            ) : null}
            <div className="m3-answer"><AssistantDoc /></div>
          </div>
        </div>
        <div className="m3-dock">
          <div className="m3-composer">
            <textarea rows={2} value={props.text} onChange={(e) => props.onText(e.target.value)} placeholder="继续布置任务…" aria-label="消息输入" />
            <div className="m3-composer__row">
              <button type="button" className="m3-attach" aria-label="添加附件"><Paperclip size={16} /></button>
              <MaterialIconButton variant="filled" label="发送消息" disabled={!props.text.trim()} onClick={() => undefined}>
                <ArrowUp size={18} aria-hidden="true" />
              </MaterialIconButton>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

type FrameProps = {
  sessions: DemoSession[];
  collapsed: Set<DemoSession["group"]>;
  onToggle: (group: DemoSession["group"]) => void;
  openTool: boolean;
  onToggleTool: () => void;
  text: string;
  onText: (value: string) => void;
};

function SessionGroups({
  tone,
  sessions,
  collapsed,
  onToggle,
}: {
  tone: "waku" | "sand" | "m3";
  sessions: DemoSession[];
  collapsed: Set<DemoSession["group"]>;
  onToggle: (group: DemoSession["group"]) => void;
}) {
  const groups = useMemo(
    () => (["today", "week", "older"] as const).map((group) => ({
      group,
      items: sessions.filter((s) => s.group === group),
    })),
    [sessions],
  );

  return (
    <>
      {groups.map(({ group, items }) => (
        <section key={group} className={`sg sg--${tone}`}>
          <button type="button" className="sg__head" aria-expanded={!collapsed.has(group)} onClick={() => onToggle(group)}>
            <span>{GROUP_LABEL[group]}</span>
            {tone === "sand" ? <span className="sg__count">{items.length}</span> : null}
            {collapsed.has(group) ? <ChevronRight size={12} /> : <ChevronDown size={12} />}
          </button>
          {!collapsed.has(group)
            ? items.map((session) => (
              <button key={session.id} type="button" className={`sg__row sg__row--${tone} ${session.active ? "active" : ""}`}>
                {tone === "waku" ? <StatusGlyph status={session.status} /> : null}
                {tone === "sand" ? <span className={`sand-avatar sm status-${session.status}`}>{session.title.slice(0, 1)}</span> : null}
                {tone === "m3" ? <StatusDot status={session.status} /> : null}
                <span className="sg__body">
                  <span className="sg__title">
                    <strong>{session.title}</strong>
                    <time>{session.time}</time>
                  </span>
                  <small>{session.preview}</small>
                </span>
                {tone === "sand" ? <StatusDot status={session.status} /> : null}
              </button>
            ))
            : null}
        </section>
      ))}
    </>
  );
}

function StatusGlyph({ status }: { status: DemoSession["status"] }) {
  if (status === "running") return <LoaderCircle size={13} className="spin status-run" />;
  if (status === "waiting") return <AlertCircle size={13} className="status-wait" />;
  return <span className="status-idle" aria-hidden="true" />;
}

function StatusDot({ status }: { status: DemoSession["status"] }) {
  return <span className={`dot status-${status}`} />;
}

function ActivityCard({ open, onToggle }: { open: boolean; onToggle: () => void }) {
  return (
    <div className="waku-activity">
      <button type="button" className="waku-activity__head" aria-expanded={open} onClick={onToggle}>
        <LoaderCircle size={13} className="spin" />
        <strong>retrieve_notes</strong>
        <span>匹配 3 · 冲突 1</span>
        <ChevronDown size={13} className={open ? "open" : undefined} />
      </button>
      {open ? (
        <div className="waku-activity__body">
          <code>{`{ "week": "2026-W34", "sources": 3 }`}</code>
          <p>先抽出主张，再对照证据链，最后标出冲突点。</p>
        </div>
      ) : null}
    </div>
  );
}

function AssistantDoc() {
  return (
    <div className="doc">
      <h2>本周可读结论</h2>
      <p>三篇材料在「先验证再扩展」上一致，分歧只在发布节奏。</p>
      <ul>
        <li><b>共识</b>：工具轨迹要可复盘，不能只看最终答案</li>
        <li><b>分歧</b>：A 主张每周一次大版本；B 主张按风险切片发布</li>
        <li><b>建议</b>：本周用切片发布，但保留周五复盘窗口</li>
      </ul>
      <pre><code>{`type Note = { claim: string; evidence: string[]; conflict?: string };`}</code></pre>
    </div>
  );
}

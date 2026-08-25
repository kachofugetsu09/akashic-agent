import { useMemo, useState } from "react";
import "./paper-shell-showcase.css";

type ShellId = "now" | "lshape" | "arena" | "spokes";

type Surface = "chat" | "plugins" | "settings" | "runtime";

type Session = { id: string; title: string; preview: string; active?: boolean };

const SESSIONS: Session[] = [
  { id: "s1", title: "纸感壳层对照", preview: "L 形 / arena / spoke…", active: true },
  { id: "s2", title: "模型与认证", preview: "Provider · API Key" },
  { id: "s3", title: "张力.gif 渲染", preview: "正文自然比例图" },
  { id: "s4", title: "过程轨收起", preview: "底部横条收起" },
];

const PRODUCTS: { id: Surface; label: string }[] = [
  { id: "chat", label: "对话" },
  { id: "runtime", label: "知识与运行" },
  { id: "plugins", label: "插件" },
  { id: "settings", label: "设置" },
];

const SHELLS: { id: ShellId; label: string; angle: string; blurb: string }[] = [
  {
    id: "now",
    label: "现在（对照）",
    angle: "双竖栏",
    blurb: "Dashboard 轨 + Chat 会话栏并排——难受的根因。",
  },
  {
    id: "lshape",
    label: "L 形纸面 ★",
    angle: "正交导航",
    blurb: "产品走顶带，会话走唯一竖栏；同一张纸，不叠两根轨。",
  },
  {
    id: "arena",
    label: "Overworld → Arena",
    angle: "时间分层",
    blurb: "进对话 = 换区：产品轨退场，只留会话竖栏；回家再回 Overworld。",
  },
  {
    id: "spokes",
    label: "Hub + Spoke tabs",
    angle: "轴分工",
    blurb: "只留产品竖轨；会话变纸顶横向 tab，多了进 codex。",
  },
];

export function PaperShellShowcase() {
  const [shell, setShell] = useState<ShellId>("lshape");
  const [surface, setSurface] = useState<Surface>("chat");
  const [sessions, setSessions] = useState(SESSIONS);
  const [zone, setZone] = useState<"overworld" | "arena">("arena");
  const [codexOpen, setCodexOpen] = useState(false);

  const active = sessions.find((s) => s.active) ?? sessions[0];

  const activate = (id: string) => {
    setSessions((prev) => prev.map((s) => ({ ...s, active: s.id === id })));
    setSurface("chat");
    if (shell === "arena") setZone("arena");
  };

  const openProduct = (id: Surface) => {
    setSurface(id);
    if (shell === "arena") {
      setZone(id === "chat" ? "arena" : "overworld");
    }
  };

  const stageCopy = useMemo(() => {
    if (surface === "plugins") return { title: "插件工作台", body: "竖栏（或顶带）换成插件列表；主区是纸面工作台，不是第二张浮卡。" };
    if (surface === "settings") return { title: "设置", body: "设置分区进同一竖栏槽位；主区是分区正文。" };
    if (surface === "runtime") return { title: "知识与运行", body: "记忆 / MCP / 定时任务——仍是同一张羊皮纸上的分区。" };
    return {
      title: active.title,
      body: "消息流与输入条落在纸心。切换壳层方案时，注意左侧是否出现「轨旁边再一根轨」。",
    };
  }, [active.title, surface]);

  return (
    <div className="pss" data-shell={shell} data-zone={zone}>
      <header className="pss-hero">
        <p className="pss-path">/chat/?preview=paper-shell</p>
        <h1>纸感壳层 · 双导航对照</h1>
        <p>
          难点不是换颜色，而是 Dashboard 产品轨与 Chat 会话栏如何共存而不变成「侧栏右边再一个侧栏」。下面四个可点完的交互草案；★ 为 ADHD 收敛的首选。
        </p>
      </header>

      <nav className="pss-tabs" aria-label="壳层方案">
        {SHELLS.map((item) => (
          <button
            key={item.id}
            type="button"
            className={shell === item.id ? "is-active" : undefined}
            onClick={() => {
              setShell(item.id);
              setCodexOpen(false);
              if (item.id === "arena") setZone(surface === "chat" ? "arena" : "overworld");
            }}
          >
            <strong>{item.label}</strong>
            <em>{item.angle}</em>
            <span>{item.blurb}</span>
          </button>
        ))}
      </nav>

      <div className="pss-stage-wrap">
        {shell === "now" ? (
          <NowShell surface={surface} sessions={sessions} onProduct={openProduct} onSession={activate} stage={stageCopy} />
        ) : null}
        {shell === "lshape" ? (
          <LShapeShell
            surface={surface}
            sessions={sessions}
            onProduct={openProduct}
            onSession={activate}
            stage={stageCopy}
          />
        ) : null}
        {shell === "arena" ? (
          <ArenaShell
            zone={zone}
            surface={surface}
            sessions={sessions}
            onProduct={openProduct}
            onSession={activate}
            onHome={() => setZone("overworld")}
            onEnterChat={() => {
              setSurface("chat");
              setZone("arena");
            }}
            stage={stageCopy}
          />
        ) : null}
        {shell === "spokes" ? (
          <SpokesShell
            surface={surface}
            sessions={sessions}
            codexOpen={codexOpen}
            onProduct={openProduct}
            onSession={activate}
            onCodex={() => setCodexOpen((v) => !v)}
            stage={stageCopy}
          />
        ) : null}
      </div>

      <section className="pss-notes">
        <h2>怎么选</h2>
        <ul>
          <li>
            <strong>L 形 ★</strong>：始终只有一根竖栏；产品与会话用正交轴拆开——最贴「一张纸」且改动可渐进。
          </li>
          <li>
            <strong>Arena</strong>：双轨永不并存；适合「进对话就要沉浸」；代价是区切换与 Esc/焦点要严谨。
          </li>
          <li>
            <strong>Spokes</strong>：会话横向化，产品轨可极窄；会话一多必须有 codex，否则 tab 先崩。
          </li>
        </ul>
      </section>
    </div>
  );
}

function Stage({ title, body }: { title: string; body: string }) {
  return (
    <div className="pss-paper">
      <div className="pss-thread">
        <p className="pss-msg user">把双侧栏做成纸感，但不要轨旁边再一根轨。</p>
        <p className="pss-msg bot">
          <strong>{title}</strong>
          <span>{body}</span>
        </p>
      </div>
      <div className="pss-composer" aria-hidden="true">
        <span>继续布置任务…</span>
        <i />
      </div>
    </div>
  );
}

function ProductBand({
  surface,
  onProduct,
}: {
  surface: Surface;
  onProduct: (id: Surface) => void;
}) {
  return (
    <div className="pss-band" role="navigation" aria-label="产品">
      <span className="pss-mark">Akashic</span>
      {PRODUCTS.map((p) => (
        <button
          key={p.id}
          type="button"
          className={surface === p.id ? "is-active" : undefined}
          onClick={() => onProduct(p.id)}
        >
          {p.label}
        </button>
      ))}
    </div>
  );
}

function SessionRail({
  sessions,
  onSession,
  heading,
}: {
  sessions: Session[];
  onSession: (id: string) => void;
  heading: string;
}) {
  return (
    <aside className="pss-rail" aria-label={heading}>
      <div className="pss-rail-head">
        <strong>{heading}</strong>
        <button type="button">新会话</button>
      </div>
      <div className="pss-rail-search">搜索会话</div>
      <div className="pss-rail-list">
        {sessions.map((s) => (
          <button
            key={s.id}
            type="button"
            className={s.active ? "is-active" : undefined}
            onClick={() => onSession(s.id)}
          >
            <strong>{s.title}</strong>
            <span>{s.preview}</span>
          </button>
        ))}
      </div>
      <div className="pss-rail-foot">
        <button type="button">模型与认证</button>
        <button type="button">主题 · 纸感</button>
      </div>
    </aside>
  );
}

function IconRail({
  surface,
  onProduct,
  faint,
}: {
  surface: Surface;
  onProduct: (id: Surface) => void;
  faint?: boolean;
}) {
  return (
    <nav className={`pss-icons ${faint ? "is-faint" : ""}`} aria-label="产品轨">
      <span className="pss-mark-sq">A</span>
      {PRODUCTS.map((p) => (
        <button
          key={p.id}
          type="button"
          className={surface === p.id ? "is-active" : undefined}
          onClick={() => onProduct(p.id)}
          title={p.label}
        >
          {p.label.slice(0, 1)}
        </button>
      ))}
    </nav>
  );
}

function NowShell({
  surface,
  sessions,
  onProduct,
  onSession,
  stage,
}: {
  surface: Surface;
  sessions: Session[];
  onProduct: (id: Surface) => void;
  onSession: (id: string) => void;
  stage: { title: string; body: string };
}) {
  return (
    <div className="pss-frame is-now">
      <IconRail surface={surface} onProduct={onProduct} />
      <SessionRail sessions={sessions} onSession={onSession} heading="会话" />
      <Stage title={stage.title} body={stage.body} />
      <p className="pss-callout">两根竖栏并排 = 难受的对照。不要优化它，要换结构。</p>
    </div>
  );
}

function LShapeShell({
  surface,
  sessions,
  onProduct,
  onSession,
  stage,
}: {
  surface: Surface;
  sessions: Session[];
  onProduct: (id: Surface) => void;
  onSession: (id: string) => void;
  stage: { title: string; body: string };
}) {
  const heading =
    surface === "chat" ? "会话" : surface === "plugins" ? "插件" : surface === "settings" ? "设置" : "运行";
  return (
    <div className="pss-frame is-lshape">
      <ProductBand surface={surface} onProduct={onProduct} />
      <div className="pss-l-body">
        {surface === "chat" ? (
          <SessionRail sessions={sessions} onSession={onSession} heading={heading} />
        ) : (
          <aside className="pss-rail" aria-label={heading}>
            <div className="pss-rail-head">
              <strong>{heading}</strong>
            </div>
            <div className="pss-rail-list">
              {(surface === "plugins"
                ? ["工作台", "已安装", "市场"]
                : surface === "settings"
                  ? ["外观", "模型", "隐私"]
                  : ["记忆", "MCP", "定时"]
              ).map((label) => (
                <button key={label} type="button">
                  <strong>{label}</strong>
                  <span>同一竖栏槽位 · 不同 surface</span>
                </button>
              ))}
            </div>
          </aside>
        )}
        <Stage title={stage.title} body={stage.body} />
      </div>
    </div>
  );
}

function ArenaShell({
  zone,
  surface,
  sessions,
  onProduct,
  onSession,
  onHome,
  onEnterChat,
  stage,
}: {
  zone: "overworld" | "arena";
  surface: Surface;
  sessions: Session[];
  onProduct: (id: Surface) => void;
  onSession: (id: string) => void;
  onHome: () => void;
  onEnterChat: () => void;
  stage: { title: string; body: string };
}) {
  if (zone === "overworld") {
    return (
      <div className="pss-frame is-arena-overworld">
        <IconRail surface={surface} onProduct={onProduct} />
        <div className="pss-overworld">
          <h2>Overworld</h2>
          <p>产品轨在此。点「对话」进入 arena——轨退场，只剩会话竖栏。</p>
          <button type="button" className="pss-enter" onClick={onEnterChat}>
            进入对话 Arena
          </button>
          <div className="pss-overworld-grid">
            {PRODUCTS.filter((p) => p.id !== "chat").map((p) => (
              <button key={p.id} type="button" onClick={() => onProduct(p.id)}>
                {p.label}
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="pss-frame is-arena">
      <button type="button" className="pss-home-glyph" onClick={onHome} aria-label="返回 Overworld">
        A
      </button>
      <SessionRail sessions={sessions} onSession={onSession} heading="会话" />
      <Stage title={stage.title} body={stage.body} />
    </div>
  );
}

function SpokesShell({
  surface,
  sessions,
  codexOpen,
  onProduct,
  onSession,
  onCodex,
  stage,
}: {
  surface: Surface;
  sessions: Session[];
  codexOpen: boolean;
  onProduct: (id: Surface) => void;
  onSession: (id: string) => void;
  onCodex: () => void;
  stage: { title: string; body: string };
}) {
  return (
    <div className="pss-frame is-spokes">
      <IconRail surface={surface} onProduct={onProduct} />
      <div className="pss-spoke-stage">
        {surface === "chat" ? (
          <div className="pss-spoke-strip" role="tablist" aria-label="会话">
            {sessions.map((s) => (
              <button
                key={s.id}
                type="button"
                role="tab"
                aria-selected={!!s.active}
                className={s.active ? "is-active" : undefined}
                onClick={() => onSession(s.id)}
              >
                {s.title}
              </button>
            ))}
            <button type="button" className="pss-codex-btn" onClick={onCodex}>
              全部…
            </button>
          </div>
        ) : (
          <div className="pss-spoke-strip is-muted">
            <span>{PRODUCTS.find((p) => p.id === surface)?.label} · 无会话 spoke</span>
          </div>
        )}
        <Stage title={stage.title} body={stage.body} />
        {codexOpen ? (
          <div className="pss-codex" role="dialog" aria-label="全部会话">
            <header>
              <strong>Session codex</strong>
              <button type="button" onClick={onCodex}>
                关闭
              </button>
            </header>
            <input placeholder="搜索会话…" />
            <ul>
              {sessions.map((s) => (
                <li key={s.id}>
                  <button
                    type="button"
                    onClick={() => {
                      onSession(s.id);
                      onCodex();
                    }}
                  >
                    {s.title}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>
    </div>
  );
}

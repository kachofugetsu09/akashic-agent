import { Brain, Pause, Play, RotateCcw, Wrench } from "lucide-react";
import { useEffect, useState } from "react";
import "./trace-motion-showcase.css";

const PHASE_DURATION_MS = 1_800;
const LAST_PHASE = 4;

const steps = [
  { kind: "thinking", title: "分析请求与现有上下文", detail: "确认目标、边界和当前实现。" },
  { kind: "tool", title: "codegraph_explore", detail: "定位消息轨迹与动效样式。" },
  { kind: "thinking", title: "比较证据并收敛方案", detail: "判断哪种节奏更像持续思考。" },
  { kind: "tool", title: "frontend_preview", detail: "渲染候选并检查视觉反馈。" },
] as const;

const candidates = [
  {
    id: "hybrid",
    number: "02+03",
    name: "流光回响",
    summary: "微光先沿轨迹落入当前节点，抵达后再扩散成双环涟漪；用先后关系表达“接力完成，开始处理”。",
    tag: "融合候选",
  },
  {
    id: "breathe",
    number: "01",
    name: "呼吸萤火",
    summary: "核心轻微起伏，外圈像呼吸一样舒张。安静、连续，最贴近“正在思考”。",
    tag: "推荐",
  },
  {
    id: "flow",
    number: "02",
    name: "能量下行",
    summary: "一道微光沿轨迹落入当前节点，强调 thinking 与工具调用之间的接力。",
    tag: "叙事最强",
  },
  {
    id: "echo",
    number: "03",
    name: "涟漪回声",
    summary: "节点发出克制的双环脉冲，生命感最明显，也最容易吸引注意。",
    tag: "呼吸最强",
  },
  {
    id: "spring",
    number: "04",
    name: "柔性点火",
    summary: "切换阶段时短促蓄力再点亮，反馈明确，随后保持稳定。",
    tag: "动作最轻快",
  },
  {
    id: "scan",
    number: "05",
    name: "静默扫描",
    summary: "节点稳定发光，只让当前内容掠过一层柔光，克制且偏工具感。",
    tag: "最不打扰",
  },
] as const;

export function TraceMotionShowcase() {
  const [phase, setPhase] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [run, setRun] = useState(0);

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(
      () => setPhase((current) => current >= LAST_PHASE ? 0 : current + 1),
      PHASE_DURATION_MS,
    );
    return () => window.clearInterval(timer);
  }, [playing, run]);

  const replay = () => {
    setPhase(0);
    setPlaying(true);
    setRun((current) => current + 1);
  };

  return (
    <main className="trace-motion-showcase" data-phase={phase}>
      <header className="trace-motion-header">
        <div className="trace-motion-heading">
          <span className="trace-motion-eyebrow">AKASHIC · PROCESS TRACE MOTION STUDY</span>
          <h1>让思考轨迹真正“呼吸”</h1>
          <p>
            融合方案与原五个候选共享同一段 <b>thinking → 工具调用 → thinking → 工具调用</b>，
            只改变光与运动的语言。当前生产实现保持不变。
          </p>
        </div>
        <div className="trace-motion-controls" aria-label="动画播放控制">
          <button type="button" onClick={() => setPlaying((current) => !current)}>
            {playing ? <Pause size={17} aria-hidden="true" /> : <Play size={17} aria-hidden="true" />}
            {playing ? "暂停" : "继续"}
          </button>
          <button className="secondary" type="button" onClick={replay}>
            <RotateCcw size={17} aria-hidden="true" />
            重播
          </button>
        </div>
      </header>

      <div className="trace-motion-sequence" aria-label="当前演示阶段">
        {steps.map((step, index) => (
          <span className={phase === index ? "active" : phase > index || phase === LAST_PHASE ? "complete" : ""} key={step.title}>
            {index + 1}
          </span>
        ))}
        <p aria-live="polite">
          {phase === LAST_PHASE ? "本轮完成，准备重新开始" : `正在演示：${steps[phase].title}`}
        </p>
      </div>

      <section className="trace-motion-grid" aria-label="六个轨迹动效候选">
        {candidates.map((candidate) => (
          <article className={`motion-card motion-card--${candidate.id}`} key={candidate.id}>
            <header className="motion-card-header">
              <div>
                <span className="motion-card-number">{candidate.number}</span>
                <h2>{candidate.name}</h2>
              </div>
              <span className="motion-card-tag">{candidate.tag}</span>
            </header>
            <p className="motion-card-summary">{candidate.summary}</p>
            <div className="motion-stage">
              <div className="motion-rail" aria-hidden="true" />
              <ol className="motion-list">
                {steps.map((step, index) => {
                  const active = phase === index;
                  const complete = phase > index || phase === LAST_PHASE;
                  return (
                    <li
                      className={`${step.kind} ${active ? "active" : ""} ${complete ? "complete" : ""}`}
                      key={step.title}
                    >
                      <span className="motion-node" aria-hidden="true" />
                      <div className="motion-copy">
                        <div className="motion-title-row">
                          {step.kind === "thinking"
                            ? <Brain size={15} aria-hidden="true" />
                            : <Wrench size={15} aria-hidden="true" />}
                          <strong>{step.title}</strong>
                          <span>{active ? "进行中" : complete ? "已完成" : "等待"}</span>
                        </div>
                        <p>{step.detail}</p>
                      </div>
                    </li>
                  );
                })}
              </ol>
            </div>
          </article>
        ))}
      </section>
    </main>
  );
}

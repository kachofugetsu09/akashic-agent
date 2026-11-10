import { useEffect, useState } from "react";
import { useReducedMotion } from "motion/react";

// 到达节奏不稳（中转站成批转发、服务端事件循环停顿）时，按观测到的到达间隔把积压均匀放出：
// 下一批预计到达时刚好放完，屏幕上不再"停几秒、跳一大段"。链路平稳时间隔约 30ms，放字几乎没有延迟。
const PACE = {
  /** 积压很少时的保底字速，保证尾巴不拖。 */
  minCharsPerSecond: 40,
  /** 放完积压的目标时长下限与上限；上限防止一次长停顿后放字过慢。 */
  minLatencyMs: 60,
  maxLatencyMs: 1500,
  /** 积压超过这么多字时按追赶时长放出，长文本不越拖越远。 */
  catchUpChars: 600,
  catchUpLatencyMs: 200,
  /** 到达间隔变短时的指数平均权重；首批到达前的假定间隔。 */
  gapWeight: 0.3,
  firstGapMs: 200,
};

/** 按到达间隔自适应放字；不活动、减少动效或文本被替换时立即给出全文。 */
export class StreamPacer {
  private target: string;
  private shown: number;
  private budget = 0;
  private gapMs = 0;
  /** 每次到达时定下的匀速字速：积压在预计的下一次到达前刚好放完。 */
  private rate = 0;
  private lastArrival = 0;
  private lastTick = 0;

  // 挂载时已有的文字直接显示，不从头重放。
  constructor(initial: string) {
    this.target = initial;
    this.shown = initial.length;
  }

  // 记录新的权威全文；返回此刻应显示的长度。
  update(text: string, now: number): number {
    if (!text.startsWith(this.target.slice(0, this.shown))) this.reset();
    if (text.length > this.target.length) {
      // 1. 首次到达没有间隔样本，先按短间隔估计；间隔变长立即跟上（下一批多半同样慢），变短按指数平均回落。
      const gap = this.lastArrival ? Math.min(PACE.maxLatencyMs, now - this.lastArrival) : PACE.firstGapMs;
      this.gapMs = !this.lastArrival || gap > this.gapMs ? gap : this.gapMs + (gap - this.gapMs) * PACE.gapWeight;
      this.lastArrival = now;
      if (!this.lastTick) this.lastTick = now;
      // 2. 匀速放字：积压在预计的下一次到达前放完；积压太多时改用追赶时长。
      const backlog = text.length - this.shown;
      const latency = backlog > PACE.catchUpChars ? PACE.catchUpLatencyMs
        : Math.min(PACE.maxLatencyMs, Math.max(PACE.minLatencyMs, this.gapMs * 1.1));
      this.rate = Math.max(PACE.minCharsPerSecond, backlog * 1000 / latency);
    }
    // 挂载时已有的文字也算一次到达，下一批才有真实间隔可量。
    if (!this.lastArrival && text) this.lastArrival = now;
    this.target = text;
    this.shown = Math.min(this.shown, text.length);
    return this.shown;
  }

  // 每帧推进；返回显示长度，积压放完后 settled 为真。
  tick(now: number): { shown: number; settled: boolean } {
    const backlog = this.target.length - this.shown;
    if (backlog <= 0) {
      this.lastTick = 0;
      this.budget = 0;
      return { shown: this.shown, settled: true };
    }
    const dt = Math.min(100, Math.max(0, now - (this.lastTick || now)));
    this.lastTick = now;
    this.budget += Math.max(PACE.minCharsPerSecond, this.rate) * dt / 1000;
    const step = Math.min(backlog, Math.floor(this.budget));
    if (step > 0) {
      this.budget -= step;
      this.shown = surrogateSafe(this.target, this.shown + step);
    }
    return { shown: this.shown, settled: this.shown >= this.target.length };
  }

  flush(): number {
    this.shown = this.target.length;
    this.budget = 0;
    return this.shown;
  }

  // 文本被替换（换了一次尝试）：清空后按新文本从头放。
  private reset(): void {
    this.target = "";
    this.shown = 0;
    this.budget = 0;
    this.gapMs = 0;
    this.rate = 0;
    this.lastArrival = 0;
    this.lastTick = 0;
  }
}

// 不把代理对劈成两半显示。
function surrogateSafe(text: string, end: number): number {
  const code = text.charCodeAt(end - 1);
  return code >= 0xd800 && code <= 0xdbff && end < text.length ? end + 1 : end;
}

/** 流式时返回按节奏放出的前缀；结束时一次把剩余全部刷出。 */
export function usePacedText(text: string, active: boolean): string {
  const reducedMotion = useReducedMotion();
  const pacing = active && !reducedMotion;
  const [pacer] = useState(() => new StreamPacer(text));
  const [shown, setShown] = useState(text.length);
  // 必要 effect：放字跟随显示帧（requestAnimationFrame），只在新文本到达后挂一个帧循环。
  useEffect(() => {
    pacer.update(text, performance.now());
    if (!pacing) {
      pacer.flush();
      return;
    }
    let frame = 0;
    const step = (now: number) => {
      const next = pacer.tick(now);
      setShown(next.shown);
      frame = next.settled ? 0 : requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [pacer, text, pacing]);
  return pacing ? text.slice(0, Math.min(shown, text.length)) : text;
}

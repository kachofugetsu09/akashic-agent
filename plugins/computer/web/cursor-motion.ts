// Agent 光标的运动模型：移植自 cua-driver 默认风格 signature_arc（MIT，Copyright (c) 2025 Cua AI, Inc.）。
// 只有几何与时间，不碰 DOM；渲染层按帧取样即可，换皮肤或换风格都不需要改这里的调用方。

export interface Vec {
  readonly x: number;
  readonly y: number;
}

/** 某一时刻的光标姿态；速度单位为 px/s。 */
export interface Pose extends Vec {
  readonly vx: number;
  readonly vy: number;
  readonly speed: number;
  readonly done: boolean;
}

export interface Glow extends Vec {
  readonly alpha: number;
  readonly scale: number;
}

export interface Ripple {
  readonly radius: number;
  readonly width: number;
  readonly alpha: number;
}

const ARC_SIZE = 0.16;
const ARC_FLOW = 0.15;
const HANDLE = 0.3;
const FOLLOW_AMOUNT = 0.018;
const FOLLOW_MAX_PX = 8;
const FOLLOW_AT = 0.82;
const DURATION_SCALE = 1.1;
const SAMPLES = 128;
export const RIPPLE_MS = 520;
const SQUISH = 0.12;

const minJerk = (t: number): number => t * t * t * (10 - 15 * t + 6 * t * t);

// 在 min-jerk 上叠一个越过终点再回位的鼓包，峰值落在 at 处。
function followThrough(over: number, at: number): (t: number) => number {
  const a = Math.max(at * 10, 1.5);
  const b = Math.max((1 - at) * 10, 1.5);
  const peak = (a / (a + b)) ** a * (b / (a + b)) ** b;
  return t => minJerk(t) + over * (t ** a * (1 - t) ** b) / peak;
}

/** Fitts 定律时长：150 + 120·log2(D/W + 1) ms，限制在 300–1000 后乘风格系数。 */
export function fittsMs(distance: number, targetSize: number): number {
  const raw = 150 + 120 * Math.log2(distance / Math.max(targetSize, 4) + 1);
  return Math.min(Math.max(raw, 300), 1000) * DURATION_SCALE;
}

/** 一次从 from 到 to 的移动：Cua 三次贝塞尔 + 弧长表，速度曲线作用在距离上。 */
export class Motion {
  readonly from: Vec;
  readonly to: Vec;
  readonly durationMs: number;
  private readonly lengths = new Float64Array(SAMPLES + 1);
  private readonly length: number;
  private readonly c1: Vec;
  private readonly c2: Vec;
  private readonly tangent: Vec;
  private readonly profile: (t: number) => number;

  constructor(from: Vec, to: Vec, targetSize = 24) {
    this.from = from;
    this.to = to;
    // 1. 控制点：水平移动无论向左向右都向上拱（cua naturalSide），峰值略偏终点。
    const dx = to.x - from.x, dy = to.y - from.y;
    const distance = Math.hypot(dx, dy);
    const chord = Math.max(distance, 1);
    const side = dx >= 0 ? -1 : 1;
    const px = -dy / chord, py = dx / chord;
    const deflection = chord * ARC_SIZE * side;
    const flow = (ARC_FLOW + 1) / 2;
    const d1 = deflection * (1 - 0.5 * flow), d2 = deflection * (1 - 0.5 * (1 - flow));
    this.c1 = { x: from.x + dx * HANDLE + px * d1, y: from.y + dy * HANDLE + py * d1 };
    this.c2 = { x: to.x - dx * HANDLE + px * d2, y: to.y - dy * HANDLE + py * d2 };
    // 2. 弧长表，让 min-jerk 描述的是走过的距离而不是参数 u。
    let previous = from, total = 0;
    for (let i = 1; i <= SAMPLES; i++) {
      const point = this.curve(i / SAMPLES);
      total += Math.hypot(point.x - previous.x, point.y - previous.y);
      this.lengths[i] = total;
      previous = point;
    }
    this.length = total;
    const end = this.curve(1 - 1e-3);
    const tail = Math.max(Math.hypot(to.x - end.x, to.y - end.y), 1e-9);
    this.tangent = { x: (to.x - end.x) / tail, y: (to.y - end.y) / tail };
    // 3. 越过量按距离折算，短距离不会抖。
    const over = Math.min(FOLLOW_AMOUNT, FOLLOW_MAX_PX / Math.max(chord, 1));
    this.profile = followThrough(over, FOLLOW_AT);
    this.durationMs = distance < 1 ? 0 : fittsMs(distance, targetSize);
  }

  private curve(u: number): Vec {
    const v = 1 - u, a = v * v * v, b = 3 * v * v * u, c = 3 * v * u * u, d = u * u * u;
    return { x: a * this.from.x + b * this.c1.x + c * this.c2.x + d * this.to.x,
      y: a * this.from.y + b * this.c1.y + c * this.c2.y + d * this.to.y };
  }

  // 按走过的距离比例取点；超过 1 时沿终点切线外推（越过终点的那一小段）。
  private atFraction(fraction: number): Vec {
    if (this.length < 1e-9) return this.to;
    if (fraction >= 1) {
      const extra = (fraction - 1) * this.length;
      return { x: this.to.x + this.tangent.x * extra, y: this.to.y + this.tangent.y * extra };
    }
    const target = Math.max(fraction, 0) * this.length;
    let lo = 0, hi = SAMPLES;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (this.lengths[mid] < target) lo = mid;
      else hi = mid;
    }
    const span = this.lengths[hi] - this.lengths[lo] || 1;
    return this.curve((lo + (target - this.lengths[lo]) / span) / SAMPLES);
  }

  /** elapsedMs 时的姿态；速度用 1/120 s 的差分估计。 */
  poseAt(elapsedMs: number): Pose {
    if (this.durationMs <= 0 || elapsedMs >= this.durationMs)
      return { ...this.to, vx: 0, vy: 0, speed: 0, done: true };
    const t = Math.max(elapsedMs / this.durationMs, 0);
    const dt = 1 / 120;
    const here = this.atFraction(this.profile(t));
    const next = this.atFraction(this.profile(Math.min(t + dt * 1000 / this.durationMs, 1)));
    const vx = (next.x - here.x) / dt, vy = (next.y - here.y) / dt;
    return { ...here, vx, vy, speed: Math.hypot(vx, vy), done: false };
  }
}

/** 速度光晕：落在运动反方向，越快越亮越大。 */
export function glowFor(pose: Pose): Glow | null {
  const alpha = Math.min(pose.speed * 0.00014, 0.42);
  if (alpha <= 0.02) return null;
  const offset = Math.min(pose.speed * 0.009, 18);
  return { x: pose.x - pose.vx / pose.speed * offset, y: pose.y - pose.vy / pose.speed * offset,
    alpha, scale: 1 + Math.min(pose.speed * 0.00024, 0.44) };
}

/** 点击涟漪：0.52 s 内半径 8→52，线宽与透明度同步衰减。 */
export function rippleAt(ageMs: number): Ripple | null {
  const k = ageMs / RIPPLE_MS;
  if (k < 0 || k >= 1) return null;
  return { radius: 8 + 44 * (1 - (1 - k) ** 3), width: 4 * (1 - k) + 1, alpha: 0.75 * (1 - k) };
}

/** 点击挤压：0.05 s 压到底，0.09 s 后用衰减余弦弹回。 */
export function squishAt(ageMs: number): number {
  const age = ageMs / 1000;
  if (age < 0) return 0;
  if (age < 0.09) return SQUISH * Math.min(age / 0.05, 1);
  const after = (age - 0.09) / 0.22;
  return SQUISH * Math.max(Math.cos(Math.min(after, 1) * Math.PI * 1.5), 0) * Math.max(1 - after, 0);
}

import assert from "node:assert/strict";
import test from "node:test";
import { Motion, fittsMs, glowFor, rippleAt, squishAt } from "./cursor-motion.ts";

test("移动从起点出发、到终点停下，时长遵循 Fitts 且有上下限", () => {
  const motion = new Motion({ x: 0, y: 0 }, { x: 400, y: 0 });
  assert.deepEqual({ x: motion.poseAt(0).x, y: motion.poseAt(0).y }, { x: 0, y: 0 });
  const end = motion.poseAt(motion.durationMs);
  assert.deepEqual([end.x, end.y, end.done], [400, 0, true]);
  assert.equal(fittsMs(1, 24), 330);
  assert.equal(fittsMs(100000, 4), 1100);
});

test("水平移动无论向左向右都向上拱", () => {
  const right = new Motion({ x: 0, y: 0 }, { x: 400, y: 0 });
  const left = new Motion({ x: 400, y: 0 }, { x: 0, y: 0 });
  assert.ok(right.poseAt(right.durationMs / 2).y < -10);
  assert.ok(left.poseAt(left.durationMs / 2).y < -10);
});

test("到达前沿切线越过终点，最多 8px", () => {
  const motion = new Motion({ x: 0, y: 0 }, { x: 1000, y: 0 });
  let farthest = 0;
  for (let ms = 0; ms < motion.durationMs; ms += 4) farthest = Math.max(farthest, motion.poseAt(ms).x);
  assert.ok(farthest > 1000 && farthest <= 1008.5, String(farthest));
});

test("原地不动不产生动画", () => {
  const motion = new Motion({ x: 5, y: 5 }, { x: 5, y: 5 });
  assert.equal(motion.durationMs, 0);
  assert.equal(motion.poseAt(0).done, true);
});

test("光晕只在快速移动时出现并落后于运动方向；涟漪和挤压会结束", () => {
  assert.equal(glowFor({ x: 0, y: 0, vx: 10, vy: 0, speed: 10, done: false }), null);
  const glow = glowFor({ x: 100, y: 0, vx: 3000, vy: 0, speed: 3000, done: false })!;
  assert.ok(glow.x < 100 && glow.alpha === 0.42);
  assert.equal(rippleAt(0)!.radius, 8);
  assert.equal(rippleAt(520), null);
  assert.equal(squishAt(50), 0.12);
  assert.equal(squishAt(400), 0);
});

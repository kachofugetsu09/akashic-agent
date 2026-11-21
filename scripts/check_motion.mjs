/** 静态检查前端动效合同，规则见 packages/akashic-theme/design.md §2.3：动画只许落在合成线程能独立播放的属性上。 */
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import postcss from "postcss";

const repoRoot = dirname(dirname(fileURLToPath(import.meta.url)));
// 合成线程能独立播放的属性：主线程卡住时照样逐帧推进。
const COMPOSITED = new Set(["transform", "translate", "scale", "rotate", "opacity"]);
// 只重绘不重排、面积通常很小的颜色属性：主线程忙时最多晚一帧变色，不产生位移跳变。
const PAINT_ONLY = new Set(["color", "background-color", "border-color", "outline-color", "text-decoration-color", "caret-color",
  "fill", "stroke", "border-top-color", "border-right-color", "border-bottom-color", "border-left-color",
  "border-block-color", "border-inline-color", "border-block-start-color", "border-block-end-color",
  "border-inline-start-color", "border-inline-end-color"]);
// 离散切换（配合 allow-discrete）只在端点翻一次，没有逐帧插值。
const DISCRETE = new Set(["display", "visibility", "overlay", "content-visibility"]);
const NOT_A_PROPERTY = /^(none|initial|inherit|unset|revert|ease|ease-in|ease-out|ease-in-out|linear|step-start|step-end|normal|allow-discrete|[\d.]+m?s)$/;
// SVG 元素上的独立 rotate/scale/translate 属性无法合成（Chromium 的 kSVGTargetHasIndependentTransformProperty）。
// 静态检查认不出元素类型，按选择器末段的命名约定（图标、箭头、svg）提示；落在 HTML 元素上的写豁免说明。
const SVG_LIKE = /(^|[\s>+~])svg\b|chevron|caret|lucide|icon/i;
const INDEPENDENT_TRANSFORMS = new Set(["rotate", "scale", "translate"]);
const EXEMPT = /^motion-exempt:\s*\S/;

function trackedFiles(patterns) {
  const output = execFileSync("git", ["ls-files", "--", ...patterns], { cwd: repoRoot, encoding: "utf8" });
  return output.split("\n").filter(Boolean);
}

// 按顶层逗号拆开，括号内的逗号不算。
function splitTopLevel(value) {
  const parts = [];
  let depth = 0;
  let current = "";
  for (const char of value) {
    if (char === "(") depth += 1;
    if (char === ")") depth -= 1;
    if (char === "," && depth === 0) {
      parts.push(current.trim());
      current = "";
    } else current += char;
  }
  return [...parts, current.trim()].filter(Boolean);
}

// transition 简写每一层里找属性名：去掉时长、曲线、函数后剩下的那个词；什么都不剩就是 all。
function shorthandProperty(layer) {
  const words = layer.replace(/[\w-]+\([^()]*(\([^()]*\)[^()]*)*\)/g, " ").split(/\s+/).filter(Boolean);
  const names = words.filter((word) => !NOT_A_PROPERTY.test(word));
  return names[0] ?? "all";
}

function transitionedProperties(decl) {
  const prop = decl.prop.toLowerCase();
  if (prop === "transition-property") return splitTopLevel(decl.value.replace(/!important/, "")).map((name) => name.trim().toLowerCase()).filter((name) => name !== "none");
  if (prop !== "transition") return [];
  if (/^\s*none\s*(!important)?\s*$/i.test(decl.value)) return [];
  return splitTopLevel(decl.value.replace(/!important/, "")).map(shorthandProperty);
}

function propertyProblem(name) {
  if (COMPOSITED.has(name) || PAINT_ONLY.has(name) || DISCRETE.has(name)) return null;
  if (name === "all") return "transition 覆盖 all，会把布局与重绘属性一起带进动画";
  if (name.startsWith("--")) return `自定义属性 ${name} 每帧在主线程插值`;
  return `${name} 不能在合成线程播放`;
}

function hasExemptComment(node) {
  const previous = node.prev();
  return previous?.type === "comment" && EXEMPT.test(previous.text.trim());
}

// 每个选择器的最后一段复合选择器；:is()、:has() 等括号里的逗号和组合器不拆。
function lastCompound(selector) {
  return splitTopLevel(selector).map((part) => {
    let depth = 0;
    let start = 0;
    for (let i = 0; i < part.length; i += 1) {
      const char = part[i];
      if (char === "(") depth += 1;
      if (char === ")") depth -= 1;
      if (depth === 0 && /[\s>+~]/.test(char)) start = i + 1;
    }
    return part.slice(start);
  }).join(",");
}

function checkStyles(file, problems) {
  const root = postcss.parse(readFileSync(resolve(repoRoot, file), "utf8"), { from: file });
  root.walkDecls((decl) => {
    if (hasExemptComment(decl)) return;
    const where = `${file}:${decl.source?.start?.line ?? 0}`;
    for (const name of transitionedProperties(decl)) {
      const problem = propertyProblem(name);
      if (problem) problems.push(`${where} ${problem}`);
      const selector = decl.parent?.type === "rule" ? decl.parent.selector : "";
      if (INDEPENDENT_TRANSFORMS.has(name) && SVG_LIKE.test(lastCompound(selector))) {
        problems.push(`${where} 图标上的独立 ${name} 属性：目标是 SVG 时无法合成，改用 transform`);
      }
    }
  });
  root.walkAtRules(/keyframes$/i, (rule) => {
    if (hasExemptComment(rule)) return;
    const names = new Set();
    rule.walkDecls((decl) => { if (!hasExemptComment(decl) && decl.prop !== "offset") names.add(decl.prop.toLowerCase()); });
    for (const name of names) {
      if (name.startsWith("animation-")) continue;
      const problem = propertyProblem(name);
      if (problem) problems.push(`${file}:${rule.source?.start?.line ?? 0} @keyframes ${rule.params}：${problem}`);
    }
  });
}

const kebab = (name) => name.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`);
// motion（framer）组件的变换简写与 opacity 一样作为 transform 输出；height、filter 等照常判定。
const MOTION_TRANSFORM_KEYS = new Set(["x", "y", "z", "rotate-x", "rotate-y", "rotate-z", "scale-x", "scale-y", "skew", "skew-x", "skew-y", "transform-perspective"]);
const KEYFRAME_META = new Set(["offset", "easing", "composite", "transform-origin", "transition"]);

// 从 index 处的 [ 或 { 起，按括号配对取出完整参数；取不到返回 null。
function balanced(text, index) {
  const open = text[index];
  if (open !== "[" && open !== "{") return null;
  let depth = 0;
  for (let i = index; i < text.length; i += 1) {
    if (text[i] === "[" || text[i] === "{" || text[i] === "(") depth += 1;
    if (text[i] === "]" || text[i] === "}" || text[i] === ")") depth -= 1;
    if (depth === 0) return text.slice(index, i + 1);
  }
  return null;
}

// 只取对象字面量的顶层键（嵌套对象里的键不算，例如 transition 配置）。
function topLevelKeys(segment) {
  const keys = [];
  let depth = 0;
  for (let i = 0; i < segment.length; i += 1) {
    const char = segment[i];
    if (char === "{" || char === "[" || char === "(") { depth += 1; continue; }
    if (char === "}" || char === "]" || char === ")") { depth -= 1; continue; }
    const inFrame = segment[0] === "[" ? depth === 2 : depth === 1;
    if (!inFrame || !/[{,\s]/.test(segment[i - 1] ?? "{")) continue;
    const match = segment.slice(i).match(/^["']?([a-zA-Z][\w-]*)["']?\s*:/);
    if (match) { keys.push(match[1]); i += match[0].length - 1; }
  }
  return keys;
}

// 脚本里的动画：style.transition 写入的属性、element.animate() 与 motion 组件的关键帧键、Tailwind 的 transition 类。
function checkScript(file, problems) {
  const text = readFileSync(resolve(repoRoot, file), "utf8");
  const lines = text.split("\n");
  const lineOf = (index) => text.slice(0, index).split("\n").length;
  // 豁免写在上一行或同一行；motion 组件的 initial/animate/exit 常连写，往上最多看 16 行。
  const exempt = (line, lookback) => lines.slice(Math.max(0, line - 1 - lookback), line).some((text) => /motion-exempt:\s*\S/.test(text));
  const report = (line, label, names, lookback = 1) => {
    if (exempt(line, lookback)) return;
    for (const name of names) {
      const problem = propertyProblem(name);
      if (problem) problems.push(`${file}:${line} ${label}：${problem}`);
    }
  };
  for (const match of text.matchAll(/\.style\.transition(?:Property)?\s*=\s*(["'`])(.*?)\1|setProperty\(\s*["']transition(?:-property)?["']\s*,\s*(["'`])(.*?)\3/g)) {
    const value = (match[2] ?? match[4]).replace(/\$\{[^}]*\}/g, " ");
    if (!value.trim() || /^\s*none\s*$/.test(value)) continue;
    report(lineOf(match.index), "style.transition", splitTopLevel(value).map(shorthandProperty));
  }
  for (const match of text.matchAll(/\.animate\(\s*/g)) {
    const segment = balanced(text, match.index + match[0].length);
    if (!segment) continue;
    report(lineOf(match.index), "element.animate()", topLevelKeys(segment).map(kebab).filter((name) => !KEYFRAME_META.has(name)));
  }
  for (const match of text.matchAll(/\b(initial|animate|exit|whileHover|whileTap|whileInView)=\{\s*/g)) {
    const segment = balanced(text, match.index + match[0].length);
    if (!segment?.startsWith("{")) continue;
    const names = topLevelKeys(segment).map(kebab).filter((name) => !KEYFRAME_META.has(name) && !MOTION_TRANSFORM_KEYS.has(name));
    report(lineOf(match.index), `motion ${match[1]}`, names, 16);
  }
  for (const match of text.matchAll(/(?<![\w-])transition-(all|\[[^\]\s]+\])(?![\w-])/g)) {
    const names = match[1] === "all" ? ["all"] : match[1].slice(1, -1).split(",").map((name) => name.replace(/_/g, "-"));
    report(lineOf(match.index), `Tailwind ${match[0]}`, names);
  }
}

function main() {
  const styles = trackedFiles(["frontend/*.css", "plugins/*/web/*.css", "packages/*.css"]);
  const scripts = trackedFiles(["frontend/*.ts", "frontend/*.tsx", "plugins/*/web/*.ts", "plugins/*/web/*.tsx", "packages/*.ts", "packages/*.tsx"])
    .filter((file) => !/\.d\.ts$/.test(file));
  const problems = [];
  for (const file of styles) checkStyles(file, problems);
  for (const file of scripts) checkScript(file, problems);
  if (problems.length) {
    console.error(`${problems.join("\n")}\n\n共 ${problems.length} 处。改法与例外规则见 packages/akashic-theme/design.md §2.3；确需例外时在声明前写 /* motion-exempt: 理由 */（脚本里 // motion-exempt: 理由）。`);
    process.exit(1);
  }
  console.log(`动效合同检查通过：${styles.length} 个样式表，${scripts.length} 个脚本。`);
}

main();

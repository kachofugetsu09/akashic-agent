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
const SVG_LIKE = /(^|[\s>+~])svg\b|chevron|lucide|icon/i;
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
  if (prop === "transition-property") return splitTopLevel(decl.value).map((name) => name.toLowerCase());
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

function lastCompound(selector) {
  return selector.split(",").map((part) => part.trim().split(/\s+|>|\+|~/).filter(Boolean).at(-1) ?? "").join(",");
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

// 脚本里的动画：style.transition 写入的属性，以及 element.animate() 关键帧里的键。
function checkScript(file, problems) {
  const lines = readFileSync(resolve(repoRoot, file), "utf8").split("\n");
  const exempt = (index) => EXEMPT.test((lines[index - 1] ?? "").trim().replace(/^\/\/\s*|^\/\*\s*|\s*\*\/$/g, ""))
    || /motion-exempt:\s*\S/.test(lines[index]);
  lines.forEach((line, index) => {
    const assigned = line.match(/\.style\.transition(?:Property)?\s*=\s*(["'`])(.*?)\1/) ?? line.match(/setProperty\(\s*["']transition["']\s*,\s*(["'`])(.*?)\1/);
    if (assigned && !exempt(index)) {
      const value = assigned[2].replace(/\$\{[^}]*\}/g, " ");
      if (value.trim() && !/^\s*none\s*$/.test(value)) {
        for (const name of splitTopLevel(value).map(shorthandProperty)) {
          const problem = propertyProblem(name);
          if (problem) problems.push(`${file}:${index + 1} style.transition：${problem}`);
        }
      }
    }
    if (/\.animate\(\s*\[/.test(line) && !exempt(index)) {
      const window = lines.slice(index, index + 6).join(" ");
      const frames = window.slice(window.indexOf(".animate("), window.indexOf("]", window.indexOf(".animate(")) + 1);
      for (const match of frames.matchAll(/[{,]\s*["']?([a-zA-Z][\w-]*)["']?\s*:/g)) {
        const name = kebab(match[1]);
        if (["offset", "easing", "composite", "transform-origin"].includes(name)) continue;
        const problem = propertyProblem(name);
        if (problem) problems.push(`${file}:${index + 1} element.animate()：${problem}`);
      }
    }
  });
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

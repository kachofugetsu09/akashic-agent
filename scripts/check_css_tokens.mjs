/** 静态检查前端源码 CSS 的外观合同与依赖方向，规则见 frontend/design.md §2.2。不认识任何具体插件，只按目录约定判断 owner。 */
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import postcss from "postcss";

const repoRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const HOST = "host";
const PUBLIC_TOKEN = /^--(ak|md|chat)-/;
// 小于最小刻度的是细线、刻度、数据标记这类微小形状，不属于形状轴。
const MICRO_RADIUS_LIMIT_PX = 4;
const KEYWORD_RADII = new Set(["0", "50%", "100%", "inherit", "initial", "none", "unset"]);
const EXEMPT = /^token-exempt:\s*\S/;
// 脚本里真正写入变量的形式：setProperty("--x")、内联样式键 "--x": 和样式串 --x:；只读取的字符串不算定义。
const SCRIPT_DEFINITION = /setProperty\(\s*["'`](--[\w-]+)|["'`](--[\w-]+)["'`]\s*[:\]]|[`"';\s{](--[\w-]+):\s*\$?\{?/g;

// 只看入库的源码，构建产物由这些源码生成。git pathspec 的 * 可跨目录，写成 **/ 反而要求多一层目录。
function trackedFiles(patterns) {
  const output = execFileSync("git", ["ls-files", "--", ...patterns], { cwd: repoRoot, encoding: "utf8" });
  return output.split("\n").filter(Boolean);
}

function ownerOf(file) {
  return (file.match(/^frontend\/plugins\/([^/]+)\//) ?? file.match(/^plugins\/([^/]+)\/web\//))?.[1] ?? HOST;
}

function kebab(role) {
  return role.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`);
}

// theme-runtime 按目录角色在运行时生成的颜色变量：名字从目录和别名表推出，不靠前缀放行。
function runtimeTokens() {
  const catalog = JSON.parse(readFileSync(resolve(repoRoot, "frontend/theme/src/theme-catalog.json"), "utf8"));
  const theme = catalog.themes[0];
  const names = [
    ...Object.keys(theme.material).map((role) => `--md-sys-color-${kebab(role)}`),
    ...Object.keys(theme.domain).map((role) => `--ak-sys-color-${kebab(role)}`),
    ...Object.keys(theme.chart).map((role) => `--ak-chart-${role}`),
  ];
  const runtime = readFileSync(resolve(repoRoot, "frontend/theme/src/theme-runtime.ts"), "utf8");
  const aliases = runtime.slice(runtime.indexOf("LEGACY_COLOR_ALIASES"), runtime.indexOf("satisfies Record"));
  for (const match of aliases.matchAll(/^\s*(\w+): \[/gm)) names.push(`--ak-color-${kebab(match[1])}`);
  return names.flatMap((name) => [name, `${name}-rgb`]);
}

function isExempt(decl) {
  const previous = decl.prev();
  return previous?.type === "comment" && EXEMPT.test(previous.text.trim());
}

function radiusProblem(value) {
  if (KEYWORD_RADII.has(value.replace(/\s*!important$/, ""))) return null;
  if (/var\(\s*--ak-radius-[\w-]+\s*,/.test(value)) return "--ak-radius-* 不写回退值";
  if (value.includes("var(")) return null;
  for (const match of value.matchAll(/([\d.]+)(px|rem|em)/g)) {
    const pixels = match[2] === "px" ? Number(match[1]) : Number(match[1]) * 16;
    if (pixels >= MICRO_RADIUS_LIMIT_PX) return `裸圆角 ${match[0]}，请用 --ak-radius-*`;
  }
  return null;
}

// 按顶层逗号拆开多层阴影；color-mix() 里的逗号不算。
function shadowLayers(value) {
  const layers = [];
  let depth = 0;
  let current = "";
  for (const char of value) {
    if (char === "(") depth += 1;
    if (char === ")") depth -= 1;
    if (char === "," && depth === 0) {
      layers.push(current.trim());
      current = "";
    } else current += char;
  }
  return [...layers, current.trim()];
}

// 内描边和焦点环（模糊为 0）是结构线，只有带模糊的外投影才属于高度轴。
function shadowProblem(value) {
  if (value === "none") return null;
  for (const layer of shadowLayers(value)) {
    if (layer.startsWith("var(") || /\binset\b/.test(layer)) continue;
    const lengths = layer.replace(/[\w-]*\([^]*\)/, "").match(/-?[\d.]+(?:px|rem|em)?/g) ?? [];
    if (Number.parseFloat(lengths[2] ?? "0") > 0) return `裸投影 ${layer.slice(0, 40)}，请用 --ak-elevation-*`;
  }
  return null;
}

// 变量只能解析到自己或宿主：插件之间不共享私有变量，宿主不依赖任何插件。
function dependencyProblem({ name, owner }, definers) {
  const owners = definers.get(name);
  if (!owners) return `变量 ${name} 没有定义`;
  if (owners.has(owner) || owners.has(HOST)) return null;
  return `${owner === HOST ? "宿主" : `插件 ${owner}`} 依赖了 ${[...owners].join("、")} 私有的 ${name}`;
}

// 1. 脚本写入的变量算定义 2. 逐条声明记录定义和引用，并查形状与投影
function scan(styles, scripts, definers) {
  const define = (name, owner) => definers.set(name, (definers.get(name) ?? new Set()).add(owner));
  for (const name of runtimeTokens()) define(name, HOST);
  for (const file of scripts) {
    for (const match of readFileSync(resolve(repoRoot, file), "utf8").matchAll(SCRIPT_DEFINITION)) {
      define(match[1] ?? match[2] ?? match[3], ownerOf(file));
    }
  }
  const references = [];
  const problems = [];
  for (const file of styles) {
    const owner = ownerOf(file);
    postcss.parse(readFileSync(resolve(repoRoot, file), "utf8"), { from: file }).walkDecls((decl) => {
      const where = `${file}:${decl.source.start.line}`;
      const exempt = isExempt(decl);
      if (decl.prop.startsWith("--")) define(decl.prop, owner);
      for (const match of decl.value.matchAll(/var\(\s*(--[\w-]+)/g)) {
        if (!exempt) references.push({ name: match[1], owner, where });
      }
      if (exempt) return;
      const problem = /^border(-[a-z]+)*-radius$/.test(decl.prop) ? radiusProblem(decl.value)
        : decl.prop === "box-shadow" ? shadowProblem(decl.value) : null;
      if (problem) problems.push(`${where} ${problem}`);
    });
  }
  return { references, problems };
}

// 1. 扫描 2. 查依赖方向 3. 查插件占用公开命名空间
function main() {
  const styles = trackedFiles(["frontend/*.css", "plugins/*/web/*.css", "packages/*.css"]);
  const scripts = trackedFiles(["frontend/*.ts", "frontend/*.tsx", "plugins/*/web/*.ts", "plugins/*/web/*.tsx"]);
  const definers = new Map();
  const { references, problems } = scan(styles, scripts, definers);
  for (const reference of references) {
    const problem = dependencyProblem(reference, definers);
    if (problem) problems.push(`${reference.where} ${problem}`);
  }
  for (const [name, owners] of definers) {
    if (PUBLIC_TOKEN.test(name) && !owners.has(HOST)) problems.push(`${[...owners].join("、")} 占用了公开命名空间 ${name}，插件私有变量请用自己的前缀`);
  }
  if (problems.length) {
    console.error(`${problems.join("\n")}\n\n共 ${problems.length} 处。确需例外时在声明前写 /* token-exempt: 理由 */，规则见 frontend/design.md §2.2。`);
    process.exit(1);
  }
  console.log(`外观合同检查通过：${styles.length} 个样式表，${references.length} 处变量引用。`);
}

main();

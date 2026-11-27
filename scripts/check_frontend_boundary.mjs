/**
 * 前端依赖边界快扫：按目录把前端代码分成单元（宿主、SDK、各插件），扫出跨单元的显式耦合。
 * 原则是依赖倒置：父只经自己声明的抽象认识子，子只用父公开的东西，每个约定只有一个属主。
 * 规则只抓显而易见的形状；核对过确实没问题的一行，在本行或上一行写 `boundary-ok: <理由>` 屏蔽。
 *
 *   node scripts/check_frontend_boundary.mjs [--plugins-dir <dir> ...]
 */
import { readFileSync, readdirSync, statSync, existsSync } from "node:fs";
import { basename, dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import postcss from "postcss";

const repoRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const EXEMPT = /boundary-ok:\s*\S/;
const CODE = /\.(ts|tsx|js|mjs|jsx|html)$/;
const SKIP_DIR = new Set(["node_modules", "dist", "static", "assets", "release", "tests", "test", "__snapshots__", "fixtures", ".generated"]);
const TEST_FILE = /\.(test|spec|e2e|stories)\.[cm]?[jt]sx?$/;
// 状态修饰类（is-*/has-*）是通用约定，各单元各自写在自己的组件类后面，不算借用。
const STATE_CLASS = /^(is|has)-[a-z0-9-]+$/;
// 第三方库生成的类（lucide 图标的 lucide-<name>），不归任何单元。
const LIBRARY_CLASS = /^lucide(-[a-z0-9-]+)?$/;
const OVERLAY = /role=\\?["']?(dialog|alertdialog|menu)\b|aria-modal|dialog\[open\]/;
const RULES = {
  "window-event": "跨单元 window/document 自定义事件：约定没有属主，改为宿主上下文上的类型化接口",
  "overlay-selector": "自写整页浮层选择器：改用 @akashic/web-ui-v1 的 foregroundOverlay()",
  "foreign-class": "输出了别的单元私有 CSS 定义的类：样式应归输出方，或由 SDK 公开组件提供",
  "foreign-style": "给别的单元输出的类写样式：改由属主公开标记（如 data-* 约定）或扩展点",
  "foreign-query": "查询别的单元私有的类或 data 属性：改由属主公开接口",
  "host-names-plugin": "宿主或 SDK 写死了插件拥有的名字：宿主只经自己声明的抽象认识插件",
};

// 1. 单元划分：宿主两层、SDK 包（公开）、各插件的前端目录。
function units(pluginDirs) {
  const out = [
    { name: "sdk:web-ui", host: true, public: true, roots: ["frontend/plugins/ui/dashboard/src/design", "frontend/plugins/ui/dashboard/public/sdk"] },
    { name: "root", host: true, public: false, roots: ["frontend/plugins/ui/dashboard/src", "frontend/plugins/ui/dashboard/index.html"] },
    { name: "chat", host: true, public: false, roots: ["frontend/plugins/akashic_clients/chat/src"] },
  ];
  for (const pkg of readdirSync(join(repoRoot, "packages"))) {
    out.push({ name: `sdk:${pkg}`, host: true, public: true, roots: [`packages/${pkg}`] });
  }
  for (const dir of pluginDirs) {
    for (const plugin of readdirSync(dir).sort()) {
      const base = join(dir, plugin);
      if (!statSync(base).isDirectory()) continue;
      // 有 web/ 源码目录时，插件根下的 web_module.* 是构建产物；没有时它就是手写源码。
      const roots = existsSync(join(base, "web")) ? [join(base, "web")]
        : readdirSync(base).filter((f) => /^web_module\.(js|css)$/.test(f)).map((f) => join(base, f));
      out.push({ name: plugin, host: false, public: false, roots: roots.filter(existsSync), pythonRoot: base });
    }
  }
  return out;
}

function walk(path, files, claimed) {
  if (!existsSync(path) || claimed.has(path)) return;
  const stat = statSync(path);
  if (stat.isFile()) { claimed.add(path); files.push(path); return; }
  for (const entry of readdirSync(path)) {
    if (SKIP_DIR.has(entry) || entry.startsWith(".")) continue;
    walk(join(path, entry), files, claimed);
  }
}

function load(unitList) {
  // 2. 先登记更具体的根（sdk:web-ui 的 design/），同一文件只归一个单元。
  const claimed = new Set();
  for (const unit of unitList) {
    const files = [];
    for (const root of unit.roots) walk(resolve(repoRoot, root), files, claimed);
    unit.code = files.filter((f) => CODE.test(f) && !TEST_FILE.test(basename(f)) && !(f.endsWith(".d.ts") && !f.endsWith("index.d.ts")))
      .map((f) => ({ path: f, text: readFileSync(f, "utf8") }));
    unit.css = files.filter((f) => f.endsWith(".css")).map((f) => ({ path: f, text: readFileSync(f, "utf8") }));
    unit.docs = files.filter((f) => f.endsWith(".md")).map((f) => readFileSync(f, "utf8"));
  }
}

let externalRoots = [];
function display(path) {
  const outside = externalRoots.find((root) => path.startsWith(`${root}/`));
  return outside ? `${basename(dirname(outside))}/${relative(dirname(outside), path)}` : relative(repoRoot, path);
}

function lineOf(text, index) {
  return text.slice(0, index).split("\n").length;
}

function exempted(text, line) {
  const lines = text.split("\n");
  return EXEMPT.test(lines[line - 1] ?? "") || EXEMPT.test(lines[line - 2] ?? "");
}

function stripComments(text) {
  return text.replace(/\/\*[\s\S]*?\*\//g, (m) => m.replace(/[^\n]/g, " ")).replace(/(^|[^:\\])\/\/[^\n]*/g, (m, p) => p + " ".repeat(m.length - p.length));
}

// 标记里输出的类名（className/class/classList/cn/clsx 里的字面量），带出现位置。
function emittedClasses(text) {
  const out = [];
  const push = (body, index) => {
    for (const token of body.split(/[\s$]+/)) if (/^[a-zA-Z][\w-]*$/.test(token)) out.push({ name: token, index });
  };
  for (const m of text.matchAll(/className\s*=\s*\{?\s*["'`]([^"'`]+)["'`]/g)) push(m[1], m.index);
  for (const m of text.matchAll(/\bclass=\\?["']([^"'\\]+)/g)) push(m[1], m.index);
  for (const m of text.matchAll(/classList\.(?:add|toggle)\(\s*["']([\w-]+)/g)) push(m[1], m.index);
  for (const m of text.matchAll(/\b(?:cn|clsx)\(([^)]*)\)/g)) for (const s of m[1].matchAll(/["']([a-zA-Z][\w-]*)["']/g)) push(s[1], m.index);
  return out;
}

function emittedAttributes(text) {
  const out = new Set();
  for (const m of text.matchAll(/(?<![\w[-])(data-[a-z0-9-]+)(?=\s*=|\s*\/?>|\s+[a-z])/g)) out.add(m[1]);
  for (const m of text.matchAll(/setAttribute\(\s*["'](data-[a-z0-9-]+)/g)) out.add(m[1]);
  for (const m of text.matchAll(/\.dataset\.([a-zA-Z]+)\s*=(?!=)|\.dataset\[\s*["']([a-zA-Z]+)/g)) {
    out.add(`data-${(m[1] ?? m[2]).replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`)}`);
  }
  return out;
}

function queriedSelectors(text) {
  const out = [];
  for (const m of text.matchAll(/(?:querySelector(?:All)?|closest|matches)(?:<[^>]*>)?\(\s*(["'`])/g)) {
    const end = text.indexOf(m[1], m.index + m[0].length);
    if (end > 0) out.push({ selector: text.slice(m.index + m[0].length, end), index: m.index });
  }
  return out;
}

function classesInSelector(selector) {
  return [...selector.replace(/url\([^)]*\)|\d+\.\d+/g, "").matchAll(/\.(-?[_a-zA-Z][\w-]*)/g)].map((m) => m[1]);
}

// 3. 汇总每个单元输出与定义了哪些类、属性。CSS Modules 的类本来就是私有的，不参与。
function index(unitList) {
  const emit = new Map(); const style = new Map(); const attrs = new Map();
  const add = (map, key, unit) => { if (!map.has(key)) map.set(key, new Set()); map.get(key).add(unit.name); };
  for (const unit of unitList) {
    unit.rules = [];
    for (const file of unit.css) {
      if (file.path.endsWith(".module.css")) continue;
      let root;
      try { root = postcss.parse(file.text); } catch { continue; }
      root.walkRules((rule) => {
        if (rule.parent?.type === "atrule" && /keyframes/i.test(rule.parent.name)) return;
        const classes = classesInSelector(rule.selector);
        for (const c of classes) add(style, c, unit);
        const attributes = [...rule.selector.matchAll(/\[(data-[a-z0-9-]+)/g)].map((m) => m[1]);
        unit.rules.push({ file, line: rule.source?.start?.line ?? 1, classes, attributes });
      });
    }
    for (const file of unit.code) {
      file.clean = stripComments(file.text);
      for (const { name } of emittedClasses(file.clean)) add(emit, name, unit);
      for (const a of emittedAttributes(file.clean)) add(attrs, a, unit);
    }
  }
  // 契约包（index.d.ts、README、公开样式）里写明的 data-* 是公开约定。
  const declared = new Set();
  for (const unit of unitList.filter((u) => u.public)) {
    for (const text of [...unit.code.map((f) => f.text), ...unit.css.map((f) => f.text), ...unit.docs]) {
      for (const m of text.matchAll(/data-[a-z0-9-]+/g)) declared.add(m[0]);
    }
  }
  return { emit, style, attrs, declared };
}

// 插件拥有的名字：它往 ChatKernel 提供的服务、它在后端登记的维度。插件目录名多是通用词，不参与。
function pluginNames(unitList) {
  const names = new Map();
  for (const unit of unitList.filter((u) => !u.host)) {
    for (const file of unit.code) for (const m of file.clean.matchAll(/services\.provide\(\s*["']([^"']+)/g)) names.set(m[1], unit.name);
    for (const py of pythonFiles(unit.pythonRoot)) {
      for (const m of readFileSync(py, "utf8").matchAll(/^DIMENSION\s*=\s*["']([^"']+)/gm)) names.set(m[1], unit.name);
    }
  }
  return names;
}

function pythonFiles(dir) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    if (entry.endsWith(".py")) out.push(path);
  }
  return out;
}

function check(unitList) {
  const findings = [];
  const { emit, style, attrs, declared } = index(unitList);
  const skipClass = (c) => STATE_CLASS.test(c) || LIBRARY_CLASS.test(c);
  const publicUnits = new Set(unitList.filter((u) => u.public).map((u) => u.name));
  const others = (map, key, unit) => [...(map.get(key) ?? [])].filter((n) => n !== unit.name && !publicUnits.has(n));
  const isPublic = (map, key) => [...(map.get(key) ?? [])].some((n) => publicUnits.has(n));
  const report = (rule, file, idx, detail, line = lineOf(file.text, idx)) => {
    if (!exempted(file.text, line)) findings.push({ rule, path: display(file.path), line, detail });
  };
  const names = pluginNames(unitList);
  const events = new Map();
  for (const unit of unitList) {
    for (const file of unit.code) {
      for (const m of file.clean.matchAll(/new CustomEvent(?:<[^>]*>)?\(\s*["'`](akashic:[^"'`]+)|(?:window|document)\.addEventListener\(\s*["'`](akashic:[^"'`]+)/g)) {
        const name = m[1] ?? m[2];
        if (!events.has(name)) events.set(name, []);
        events.get(name).push({ unit, file, index: m.index });
      }
    }
  }
  // window-event：同一个全局自定义事件被多个单元使用，就是一条没有属主的约定。
  for (const [name, uses] of events) {
    if (new Set(uses.map((use) => use.unit.name)).size < 2) continue;
    for (const use of uses) report("window-event", use.file, use.index, `${name}（${[...new Set(uses.map((u) => u.unit.name))].join("/")}）`);
  }
  for (const unit of unitList) {
    for (const file of unit.code) {
      const text = file.clean;
      // host-names-plugin：宿主与 SDK 的字符串字面量不该是插件拥有的名字。
      if (unit.host) {
        for (const m of text.matchAll(/["'`]([^"'`\n]{2,80})["'`]/g)) {
          const owner = names.get(m[1]);
          if (owner && owner !== unit.name) report("host-names-plugin", file, m.index, `"${m[1]}" ← ${owner}`);
        }
      }
      if (unit.public) continue;
      // overlay-selector：整页查询浮层只许经 SDK。
      for (const m of text.matchAll(/(?:document|ownerDocument)\.querySelector(?:All)?(?:<[^>]*>)?\(\s*(["'`])([^"'`]*)/g)) {
        if (OVERLAY.test(m[2])) report("overlay-selector", file, m.index, m[2]);
      }
      // foreign-class：自己输出、自己不写样式、别的非公开单元写了样式。
      for (const { name, index: idx } of emittedClasses(text)) {
        if (skipClass(name) || isPublic(style, name) || style.get(name)?.has(unit.name)) continue;
        const owner = others(style, name, unit);
        if (owner.length) report("foreign-class", file, idx, `${name} ← ${owner.join("/")}`);
      }
      // foreign-query：查询的类或属性只由别的非公开单元输出。
      for (const { selector, index: idx } of queriedSelectors(text)) {
        const hits = [];
        for (const c of classesInSelector(selector)) {
          if (skipClass(c) || isPublic(emit, c) || isPublic(style, c) || emit.get(c)?.has(unit.name)) continue;
          const owner = [...new Set([...others(emit, c, unit), ...others(style, c, unit)])];
          if (owner.length) hits.push(`.${c} ← ${owner.join("/")}`);
        }
        for (const [, a] of selector.matchAll(/\[(data-[a-z0-9-]+)/g)) {
          if (declared.has(a) || isPublic(attrs, a) || attrs.get(a)?.has(unit.name)) continue;
          const owner = others(attrs, a, unit);
          if (owner.length) hits.push(`[${a}] ← ${owner.join("/")}`);
        }
        if (hits.length) report("foreign-query", file, idx, hits.join(", "));
      }
    }
    // foreign-style：CSS 覆盖别的单元输出并自己定义样式的类。
    if (unit.public) continue;
    for (const { file, line, classes, attributes } of unit.rules) {
      for (const a of attributes) {
        if (declared.has(a) || isPublic(attrs, a) || attrs.get(a)?.has(unit.name)) continue;
        const owner = others(attrs, a, unit);
        if (owner.length) report("foreign-style", file, 0, `[${a}] ← ${owner.join("/")}`, line);
      }
      for (const c of classes) {
        if (skipClass(c) || isPublic(emit, c) || isPublic(style, c) || emit.get(c)?.has(unit.name)) continue;
        const owner = others(emit, c, unit).filter((n) => style.get(c)?.has(n));
        if (owner.length) report("foreign-style", file, 0, `.${c} ← ${owner.join("/")}`, line);
      }
    }
  }
  return findings;
}

function main() {
  const args = process.argv.slice(2);
  const pluginDirs = [resolve(repoRoot, "plugins")];
  for (let i = 0; i < args.length; i += 1) if (args[i] === "--plugins-dir") pluginDirs.push(resolve(args[++i]));
  externalRoots = pluginDirs.slice(1);
  const unitList = units(pluginDirs);
  load(unitList);
  const findings = check(unitList);
  const seen = new Set();
  for (const f of findings) {
    const key = `${f.path}:${f.line}:${f.rule}:${f.detail}`;
    if (seen.has(key)) continue;
    seen.add(key);
    console.log(`${f.path}:${f.line}: [${f.rule}] ${f.detail}`);
  }
  if (!seen.size) { console.log("frontend boundary: ok"); return; }
  console.log(`\n${seen.size} 处跨单元耦合。规则说明：`);
  for (const rule of new Set(findings.map((f) => f.rule))) console.log(`  ${rule}: ${RULES[rule]}`);
  console.log("核对过确实没问题的行，在本行或上一行写 `boundary-ok: <理由>`。");
  process.exitCode = 1;
}

main();

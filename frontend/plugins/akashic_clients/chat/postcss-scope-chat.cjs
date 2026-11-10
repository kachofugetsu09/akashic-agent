// 聊天与外壳同一文档（ADR 0126）：聊天的每条样式只作用于聊天根节点 .akashic-chat 及挂在其中的弹层。
// 前缀用 :where()，不增加特异性，聊天内部各规则之间的层叠关系保持原样。
// - html / body / #root 指的就是承载聊天的页面，改写为聊天根节点；
// - :root 上的主题变量（自定义属性）与外壳同源，保持全局；其余声明是页面级排版，移到聊天根节点；
// - 以 :root / html / body / .dark 开头再接后代的选择器（按主题切换），根节点插在第一段之后；
// - 已经锚定在聊天根节点上的（.akashic-chat 与其上的布局属性）不再加前缀；
// - 嵌套规则由外层决定，@keyframes 里的帧选择器不动；
// - 页面 HTML 里的内联 <style>（夹具页面自己的 html/body）属于承载页面，不改写。
const ROOT = ":where(.akashic-chat)";
const PAGE = /^(?:html|body|#root)$/u;
const PAGE_ANCESTOR = /^((?::root|html|body|\.dark)(?:[[:.][^\s>+~]*)?)\s+(.+)$/u;
const ANCHORED = /^(?:\.akashic-chat|:where\(\[data-chat-|\[data-chat-|\[data-shell-tools)/u;

function scopeSelector(selector) {
  const value = selector.trim();
  if (value.includes(ROOT)) return value;
  if (PAGE.test(value)) return ROOT;
  if (ANCHORED.test(value)) return value;
  const ancestor = PAGE_ANCESTOR.exec(value);
  if (ancestor) return `${ancestor[1]} ${ROOT} ${ancestor[2]}`;
  if (value.startsWith(":root")) return value;
  return `${ROOT} ${value}`;
}

function moveRootDeclarations(rule) {
  const own = rule.nodes.filter((node) => node.type === "decl" && !node.prop.startsWith("--"));
  if (!own.length || !rule.selectors.every((selector) => /^:root(?:\[[^\]]*\])*$/u.test(selector.trim()))) return;
  const moved = rule.clone({ selectors: rule.selectors.map((selector) => `${selector.trim()} ${ROOT}`), nodes: [] });
  for (const node of own) moved.append(node);
  rule.after(moved);
  if (!rule.nodes.length) rule.remove();
}

function insideRuleOrKeyframes(node) {
  for (let parent = node.parent; parent && parent.type !== "root"; parent = parent.parent) {
    if (parent.type === "rule") return true;
    if (parent.type === "atrule" && /keyframes$/iu.test(parent.name)) return true;
  }
  return false;
}

module.exports = () => ({
  postcssPlugin: "scope-chat-css",
  OnceExit(root) {
    if ((root.source?.input.file ?? "").includes("html-proxy")) return;
    root.walkRules((rule) => {
      if (insideRuleOrKeyframes(rule)) return;
      moveRootDeclarations(rule);
      rule.selectors = rule.selectors.map(scopeSelector);
    });
  },
});
module.exports.postcss = true;

// Mermaid 已下线。markstream-react 的 dist 仍含 import("mermaid") 懒加载路径，
// 构建期必须可解析；该路径在 MermaidAsCode 接管后不会再执行。
const unavailable = () => Promise.reject(new Error("Mermaid not available"));

export default {
  parse: unavailable,
  render: unavailable,
  initialize: () => undefined,
};

// 浏览器模块：只能同步导出 activate(ctx)，只能 import react / react-dom/client / @akashic/web-ui-v1。
const ICON = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/></svg>';

export function activate(ctx) {
  return ctx.ui.inject("shell.pages.v1", (mount) => mount.register({
    id: "hello-page",
    label: "示例页面",
    order: 100,
    route: "hello",
    iconSvg: ICON,
    render(host) {
      const page = document.createElement("section");
      page.className = "hello-page";
      const title = document.createElement("h1");
      title.textContent = "示例页面";
      const body = document.createElement("p");
      body.textContent = "插件已加载。";
      page.append(title, body);
      host.append(page);
      return () => page.remove();
    },
  }));
}

export function activate(ctx) {
  return ctx.ui.inject("shell.pages.v1", (mount) => mount.register({
    id: "workbench",
    label: "工作台",
    route: "workbench",
    order: 20,
    render(host) {
      const frame = document.createElement("iframe");
      frame.className = "workbench-page-frame";
      frame.title = "Akashic 工作台";
      frame.src = "/dashboard?surface=workbench-adapter";
      host.replaceChildren(frame);
      return () => host.replaceChildren();
    },
  }));
}

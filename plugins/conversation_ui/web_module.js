export function activate(ctx) {
  return ctx.ui.inject("shell.pages.v1", (mount) => mount.register({
    id: "conversation",
    label: "对话",
    route: "",
    order: 10,
    render(host) {
      const frame = document.createElement("iframe");
      frame.className = "conversation-page-frame";
      frame.title = "Akashic 对话";
      frame.src = "/chat?embedded=1";
      host.replaceChildren(frame);
      return () => host.replaceChildren();
    },
  }));
}

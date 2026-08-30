export function activate(ctx) {
  return ctx.ui.inject("shell.pages.v1", (mount) => mount.register({
    id: "conversation",
    label: "对话",
    icon: "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.75' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 6V2H8'/%3E%3Crect width='16' height='12' x='4' y='8' rx='2'/%3E%3Cpath d='M2 14h2m16 0h2m-7-3v2m-6-2v2'/%3E%3C/svg%3E",
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

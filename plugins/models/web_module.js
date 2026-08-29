export function activate(ctx) {
  return ctx.ui.inject("shell.pages.v1", (mount) => mount.register({
    id: "models",
    label: "模型",
    route: "models",
    order: 30,
    render(host) {
      const frame = document.createElement("iframe");
      frame.className = "models-page-frame";
      frame.title = "模型配置";
      frame.src = "/settings?embedded=1";
      host.replaceChildren(frame);
      const settingsApplied = (event) => {
        if (event.origin !== window.location.origin
          || event.source !== frame.contentWindow
          || !event.data
          || event.data.type !== "akashic.settings.applied") return;
        const base = `${window.location.pathname}${window.location.search}`;
        window.history.pushState(null, "", base);
        window.dispatchEvent(new PopStateEvent("popstate"));
      };
      window.addEventListener("message", settingsApplied);
      return () => {
        window.removeEventListener("message", settingsApplied);
        host.replaceChildren();
      };
    },
  }));
}

import { currentTheme, subscribeTheme } from "@akashic/web-ui-v1";

function syncFrameTheme(frame) {
  const send = () => frame.contentWindow?.postMessage(
    { type: "akashic.theme", themeId: currentTheme().id },
    window.location.origin,
  );
  frame.addEventListener("load", send);
  const unsubscribe = subscribeTheme(send);
  return () => {
    unsubscribe();
    frame.removeEventListener("load", send);
  };
}

export function activate(ctx) {
  return ctx.ui.inject("shell.pages.v1", (mount) => mount.register({
    id: "runtime",
    label: "知识与运行",
    iconSvg: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-book-open-text" aria-hidden="true"><path d="M12 5v16"></path><path d="M16 13h2"></path><path d="M16 9h2"></path><path d="M20.001 19A2 2 0 0022 17V5a2 2 0 00-1.999-2L16 3.002A5 5 0 0012 5a5 5 0 00-4-2H4a2 2 0 00-2 2v12a2 2 0 001.999 2H8a5 5 0 014 2 5 5 0 014-2z"></path><path d="M6 13h2"></path><path d="M6 9h2"></path></svg>',
    route: "runtime",
    order: 25,
    render(host) {
      const frame = document.createElement("iframe");
      frame.title = "知识与运行";
      frame.src = "/chat?embedded=1&surface=runtime";
      host.replaceChildren(frame);
      const stopThemeSync = syncFrameTheme(frame);
      return () => {
        stopThemeSync();
        host.replaceChildren();
      };
    },
  }));
}

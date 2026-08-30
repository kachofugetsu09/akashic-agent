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
    id: "conversation",
    label: "对话",
    iconSvg: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-bot" aria-hidden="true"><path d="M12 8V4H8"></path><rect width="16" height="12" x="4" y="8" rx="2"></rect><path d="M2 14h2"></path><path d="M20 14h2"></path><path d="M15 13v2"></path><path d="M9 13v2"></path></svg>',
    route: "",
    order: 10,
    render(host) {
      const frame = document.createElement("iframe");
      frame.className = "conversation-page-frame";
      frame.title = "Akashic 对话";
      frame.src = "/chat?embedded=1";
      host.replaceChildren(frame);
      const stopThemeSync = syncFrameTheme(frame);
      return () => {
        stopThemeSync();
        host.replaceChildren();
      };
    },
  }));
}

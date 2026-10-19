import { useDesktopChatController } from "./use-desktop-chat-controller";
import { DesktopChatView } from "./desktop-chat-view";
import { PaperTooltipLayer } from "./paper-tooltip";
import "./styles.css";
import "./message-view.css";

export function DesktopChatApp() {
  return <>
    <DesktopChatView controller={useDesktopChatController()} />
    <PaperTooltipLayer />
  </>;
}

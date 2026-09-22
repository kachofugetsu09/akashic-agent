import { useDesktopChatController } from "./use-desktop-chat-controller";
import { DesktopChatView } from "./desktop-chat-view";
import "./styles.css";
import "./message-view.css";

interface DesktopChatAppProps {
  embeddedShell: boolean;
}

export function DesktopChatApp(props: DesktopChatAppProps) {
  return <DesktopChatView {...props} controller={useDesktopChatController()} />;
}

import { useDesktopChatController } from "./use-desktop-chat-controller";
import { DesktopChatView } from "./desktop-chat-view";
import "./styles.css";
import "./message-view.css";

interface DesktopChatAppProps {
  embeddedShell: boolean;
  embeddedRuntime: boolean;
}

export function DesktopChatApp(props: DesktopChatAppProps) {
  return <DesktopChatView {...props} controller={useDesktopChatController()} />;
}

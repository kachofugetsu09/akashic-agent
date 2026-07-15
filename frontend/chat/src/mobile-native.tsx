import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./mobile-native.css";

export interface MobileSnapshot {
  protocolVersion: 1;
  connection: {
    label: string;
    status: "connecting" | "ready" | "degraded" | "reconnecting" | "disconnected";
    notice: string | null;
  };
  selectedSessionId: string | null;
  messages: unknown[];
}

interface NativeBridge {
  requestSnapshot(): void;
}

declare global {
  interface Window {
    AkashicNative?: NativeBridge;
    AkashicMobile?: {
      receiveSnapshot(snapshot: MobileSnapshot): void;
    };
  }
}

function MobileNativeApp() {
  const [snapshot, setSnapshot] = useState<MobileSnapshot | null>(null);

  useEffect(() => {
    window.AkashicMobile = { receiveSnapshot: setSnapshot };
    window.AkashicNative?.requestSnapshot();
    return () => {
      delete window.AkashicMobile;
    };
  }, []);

  return (
    <main className="mobile-native-bootstrap">
      <div className="mobile-native-bootstrap__status" aria-live="polite">
        {snapshot?.connection.label ?? "正在准备对话"}
      </div>
      <div className="mobile-native-bootstrap__count">
        {snapshot ? `已载入 ${snapshot.messages.length} 条消息` : "正在连接本地数据"}
      </div>
    </main>
  );
}

const root = document.getElementById("root");
if (!root) throw new Error("Mobile Web root 不存在");
createRoot(root).render(<MobileNativeApp />);

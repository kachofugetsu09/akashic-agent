import "./mobile-polyfills";
import "./mobile-native.css";

import React from "react";
import { createRoot } from "react-dom/client";
import { AlertCircle, RefreshCw } from "lucide-react";

import { initializeTheme } from "../../theme/src/theme-runtime";
import { MobileNativeApp } from "./mobile-native";

class MobileErrorBoundary extends React.Component<React.PropsWithChildren, { message: string | null }> {
  state: { message: string | null } = { message: null };

  static getDerivedStateFromError(error: unknown) {
    return { message: error instanceof Error ? error.message : "会话界面发生未知错误" };
  }

  componentDidCatch(error: unknown) {
    console.error("[mobile] render failed", error);
  }

  render() {
    if (!this.state.message) return this.props.children;
    return (
      <main className="mobile-fatal" role="alert">
        <AlertCircle className="mobile-fatal__mark" size={28} />
        <h1>会话界面没有正常载入</h1>
        <p>{this.state.message}</p>
        <button type="button" onClick={() => window.location.reload()}>
          <RefreshCw size={18} />
          重新载入
        </button>
      </main>
    );
  }
}

/** 初始化生产或 Lab transport 后，挂载同一棵 Mobile React 树。 */
export function startMobileNativeApp(installTransport: () => void): void {
  // 1. Activity 的 adjustResize 已拥有 IME 高度，避免 visualViewport 重复扣除键盘。
  const syncViewportHeight = () => {
    const viewportHeight = Math.max(1, Math.round(window.innerHeight));
    document.documentElement.style.setProperty("--mobile-viewport-height", `${viewportHeight}px`);
  };
  syncViewportHeight();
  window.addEventListener("resize", syncViewportHeight);

  // 2. 主题和 transport 就绪后，再把真实应用挂到页面根节点。
  initializeTheme();
  installTransport();
  const root = document.getElementById("root");
  if (!root) throw new Error("Mobile Web root 不存在");
  createRoot(root).render(
    <MobileErrorBoundary>
      <MobileNativeApp />
    </MobileErrorBoundary>,
  );
}

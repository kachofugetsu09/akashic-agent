import React from "react";

interface State { error: Error | null }

/** Replace an unrecoverable render or lazy-load failure with an explicit reload action. */
export class WebUiErrorBoundary extends React.Component<React.PropsWithChildren, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error("WebUI 界面加载失败", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return <main className="webui-fatal-error" role="alert">
      <div>
        <h1>界面加载失败</h1>
        <p>本次界面资源没有完整载入。重新加载会重新获取资源，不会删除对话或设置。</p>
        <button type="button" onClick={() => window.location.reload()}>重新加载</button>
      </div>
    </main>;
  }
}

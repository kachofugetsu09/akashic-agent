import { Component, useEffect, useLayoutEffect, useRef } from "react";
import type { ReactNode } from "react";
import type { PluginConfig, PluginDispatch } from "./types";

class PluginErrorBoundary extends Component<{
  pluginId: string;
  slot: string;
  children: ReactNode;
}, { error: Error | null }> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error): { error: Error } {
    return { error };
  }

  componentDidCatch(error: Error): void {
    console.error(`[dashboard] ${this.props.pluginId} ${this.props.slot} failed`, error);
  }

  render(): ReactNode {
    if (!this.state.error) return this.props.children;
    return (
      <div className="plugin-entry-error" role="alert">
        <strong>{this.props.pluginId} 无法显示</strong>
        <span>{this.state.error.message}</span>
      </div>
    );
  }
}

export function mountPluginDom(
  host: HTMLElement,
  pluginId: string,
  slot: string,
  mount: () => void | (() => void),
): void | (() => void) {
  try {
    const dispose = mount();
    return () => {
      try {
        dispose?.();
      } catch (error) {
        console.error(`[dashboard] ${pluginId} ${slot} cleanup failed`, error);
      }
    };
  } catch (error) {
    console.error(`[dashboard] ${pluginId} ${slot} failed`, error);
    host.replaceChildren();
    const alert = document.createElement("div");
    alert.className = "plugin-entry-error";
    alert.setAttribute("role", "alert");
    const title = document.createElement("strong");
    title.textContent = `${pluginId} 无法显示`;
    const detail = document.createElement("span");
    detail.textContent = error instanceof Error ? error.message : String(error);
    alert.append(title, detail);
    host.append(alert);
  }
}

export function PluginDetail(props: {
  plugin: PluginConfig;
  item: Record<string, unknown> | null;
  dispatch?: PluginDispatch;
}): React.ReactElement {
  const ref = useRef<HTMLDivElement>(null);
  const Detail = props.plugin.Detail;

  useLayoutEffect(() => ref.current ? props.plugin.applyStyle(ref.current) : undefined, [props.plugin]);

  // 1. React-native plugins compose straight into the host tree (shared React).
  // 必要 effect：legacy 插件 DOM render 契约（renderDetail 直接操作 ref 节点），不可改为渲染期计算
  useEffect(() => {
    if (Detail) return;
    if (ref.current && props.plugin.renderDetail) {
      const host = ref.current;
      return mountPluginDom(host, props.plugin.id, "detail", () => (
        props.plugin.renderDetail!(props.item, host, props.dispatch)
      ));
    } else if (ref.current) {
      ref.current.innerHTML = "";
    }
  }, [Detail, props.item, props.plugin, props.dispatch]);

  // 2. Otherwise fall back to the legacy DOM render contract.
  if (Detail) {
    const rowKey = String(props.item?.[props.plugin.rowKey] ?? "empty");
    return (
      <div ref={ref} className="plugin-workbench-root" data-akashic-plugin={props.plugin.id}>
        <PluginErrorBoundary key={`${props.plugin.id}:${rowKey}`} pluginId={props.plugin.id} slot="detail">
          <Detail item={props.item} dispatch={props.dispatch} />
        </PluginErrorBoundary>
      </div>
    );
  }
  return <div ref={ref} data-akashic-plugin={props.plugin.id} />;
}

export function PluginMain(props: {
  plugin: PluginConfig;
  dispatch: PluginDispatch;
}): React.ReactElement {
  const ref = useRef<HTMLDivElement>(null);
  const Main = props.plugin.Main;
  const dispatchRef = useRef(props.dispatch);

  useLayoutEffect(() => ref.current ? props.plugin.applyStyle(ref.current) : undefined, [props.plugin]);

  useEffect(() => {
    dispatchRef.current = props.dispatch;
  }, [props.dispatch]);

  // 必要 effect：legacy renderMain 自己拥有 DOM、timer、listener；宿主只更新 dispatch，不可改为渲染期计算
  useEffect(() => {
    if (Main) return;
    // legacy renderMain 自己拥有 DOM、timer、listener；宿主只更新 dispatch。
    if (ref.current && props.plugin.renderMain) {
      const host = ref.current;
      return mountPluginDom(host, props.plugin.id, "main", () => (
        props.plugin.renderMain!(host, dispatchRef.current)
      ));
    }
  }, [Main, props.plugin]);

  if (Main) {
    return (
      <div ref={ref} className="plugin-workbench-root" data-akashic-plugin={props.plugin.id}>
        <PluginErrorBoundary key={props.plugin.id} pluginId={props.plugin.id} slot="main">
          <Main dispatch={props.dispatch} />
        </PluginErrorBoundary>
      </div>
    );
  }
  return <div className="plugin-workbench-root" ref={ref} data-akashic-plugin={props.plugin.id} />;
}

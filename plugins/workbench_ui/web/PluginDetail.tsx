import { useEffect, useLayoutEffect, useRef } from "react";
import type { PluginConfig, PluginDispatch } from "./types";

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
  dispatch: PluginDispatch;
}): React.ReactElement {
  const ref = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => ref.current ? props.plugin.applyStyle(ref.current) : undefined, [props.plugin]);

  useEffect(() => {
    if (ref.current && props.plugin.renderDetail) {
      const host = ref.current;
      return mountPluginDom(host, props.plugin.id, "detail", () => (
        props.plugin.renderDetail!(props.item, host, props.dispatch)
      ));
    } else if (ref.current) {
      ref.current.innerHTML = "";
    }
  }, [props.item, props.plugin, props.dispatch]);

  return <div ref={ref} className="plugin-workbench-root" data-akashic-plugin={props.plugin.id} />;
}

export function PluginMain(props: {
  plugin: PluginConfig;
  dispatch: PluginDispatch;
}): React.ReactElement {
  const ref = useRef<HTMLDivElement>(null);
  const dispatchRef = useRef(props.dispatch);

  useLayoutEffect(() => ref.current ? props.plugin.applyStyle(ref.current) : undefined, [props.plugin]);

  useEffect(() => {
    dispatchRef.current = props.dispatch;
  }, [props.dispatch]);

  useEffect(() => {
    if (ref.current && props.plugin.renderMain) {
      const host = ref.current;
      return mountPluginDom(host, props.plugin.id, "main", () => (
        props.plugin.renderMain!(host, dispatchRef.current)
      ));
    }
  }, [props.plugin]);

  return <div className="plugin-workbench-root" ref={ref} data-akashic-plugin={props.plugin.id} />;
}

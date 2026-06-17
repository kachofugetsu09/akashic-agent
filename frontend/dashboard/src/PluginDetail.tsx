import { useEffect, useRef } from "react";
import type { PluginConfig, PluginDispatch } from "./types";

export function PluginDetail(props: {
  plugin: PluginConfig;
  item: Record<string, unknown> | null;
  dispatch?: PluginDispatch;
}): React.ReactElement {
  const ref = useRef<HTMLDivElement>(null);
  const Detail = props.plugin.Detail;

  // 1. React-native plugins compose straight into the host tree (shared React).
  useEffect(() => {
    if (Detail) return;
    if (ref.current && props.plugin.renderDetail) {
      props.plugin.renderDetail(props.item, ref.current, props.dispatch);
    } else if (ref.current) {
      ref.current.innerHTML = "";
    }
  }, [Detail, props.item, props.plugin, props.dispatch]);

  // 2. Otherwise fall back to the legacy DOM render contract.
  if (Detail) {
    return <Detail item={props.item} dispatch={props.dispatch} />;
  }
  return <div ref={ref} />;
}

export function PluginMain(props: {
  plugin: PluginConfig;
  dispatch: PluginDispatch;
}): React.ReactElement {
  const ref = useRef<HTMLDivElement>(null);
  const Main = props.plugin.Main;

  useEffect(() => {
    if (Main) return;
    if (ref.current && props.plugin.renderMain) {
      props.plugin.renderMain(ref.current, props.dispatch);
    }
  }, [Main, props.plugin, props.dispatch]);

  if (Main) {
    return <div className="plugin-workbench-root"><Main dispatch={props.dispatch} /></div>;
  }
  return <div className="plugin-workbench-root" ref={ref} />;
}

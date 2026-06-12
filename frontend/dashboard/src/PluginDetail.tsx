import { useEffect, useRef } from "react";
import type { PluginConfig, PluginDispatch } from "./types";

export function PluginDetail(props: {
  plugin: PluginConfig;
  item: Record<string, unknown> | null;
  dispatch?: PluginDispatch;
}): React.ReactElement {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (ref.current && props.plugin.renderDetail) {
      props.plugin.renderDetail(props.item, ref.current, props.dispatch);
    } else if (ref.current) {
      ref.current.innerHTML = "";
    }
  }, [props.item, props.plugin, props.dispatch]);

  return <div ref={ref} />;
}

export function PluginMain(props: {
  plugin: PluginConfig;
  dispatch: PluginDispatch;
}): React.ReactElement {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (ref.current && props.plugin.renderMain) {
      props.plugin.renderMain(ref.current, props.dispatch);
    }
  }, [props.plugin, props.dispatch]);

  return <div className="plugin-workbench-root" ref={ref} />;
}

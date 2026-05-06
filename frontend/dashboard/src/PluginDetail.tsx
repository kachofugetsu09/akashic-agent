import { useEffect, useRef } from "react";
import type { PluginConfig, PluginDispatch } from "./types";

export function PluginDetail(props: {
  plugin: PluginConfig;
  item: Record<string, unknown> | null;
  dispatch?: PluginDispatch;
}): React.ReactElement {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (ref.current) {
      props.plugin.renderDetail(props.item, ref.current, props.dispatch);
    }
  }, [props.item, props.plugin, props.dispatch]);

  return <div ref={ref} />;
}

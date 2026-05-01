import { useEffect, useRef } from "react";
import type { PluginConfig } from "./types";

export function PluginDetail(props: {
  plugin: PluginConfig;
  item: Record<string, unknown> | null;
}): React.ReactElement {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (ref.current) {
      props.plugin.renderDetail(props.item, ref.current);
    }
  }, [props.item, props.plugin]);

  return <div ref={ref} />;
}

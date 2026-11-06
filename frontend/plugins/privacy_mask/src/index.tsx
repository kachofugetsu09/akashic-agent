import type { WebHostContextV1, WebUiDisposer } from "@akashic/web-ui-v1";
import { acquirePrivacyMaskEngine } from "./privacy-engine";
import "./style.css";

const RAIL_ACTIONS_MOUNT = "shell.rail-actions.v1";

const EYE_OFF_ICON =
  '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10.733 5.076a10.744 10.744 0 0 1 11.205 6.575 1 1 0 0 1 0 .696 10.747 10.747 0 0 1-1.444 2.49"/><path d="M14.084 14.158a3 3 0 0 1-4.242-4.242"/><path d="M17.479 17.499a10.75 10.75 0 0 1-15.417-5.151 1 1 0 0 1 0-.696 10.75 10.75 0 0 1 4.446-5.143"/><path d="m2 2 20 20"/></svg>';

export function activate(ctx: WebHostContextV1): WebUiDisposer {
  const engine = acquirePrivacyMaskEngine();
  const stopRail = ctx.ui.inject(RAIL_ACTIONS_MOUNT, (mount) =>
    mount.register({
      id: "privacy_mask.toggle",
      order: 90,
      label: "截屏打码",
      iconSvg: EYE_OFF_ICON,
      onActivate: () => engine.toggle(),
    }),
  );
  return () => {
    stopRail();
    engine.release();
  };
}

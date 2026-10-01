import type {WebHostContextV1, WebUiDisposer} from "@akashic/web-ui-v1";
export function activate(ctx: WebHostContextV1): WebUiDisposer;
export function createDiscoveryOwner(): {
  start(fingerprint: string): {signal: AbortSignal; isCurrent(fingerprint: string): boolean};
  invalidate(): void;
  close(): void;
};

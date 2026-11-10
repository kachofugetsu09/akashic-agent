// Host ReactDOM for plugin modules that portal or flush (Radix and similar libraries import "react-dom").
const D = window.__akashicRuntime.ReactDOM;
export default D.default ?? D;
export const {
  createPortal, flushSync, preconnect, prefetchDNS, preinit, preinitModule, preload, preloadModule,
  requestFormReset, unstable_batchedUpdates, useFormState, useFormStatus, version,
} = D;

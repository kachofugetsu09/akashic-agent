// Host ReactDOM client API for plugins that need to mount their own roots.
const D = window.__akashicRuntime.ReactDOMClient;
export const createRoot = D.createRoot;
export const hydrateRoot = D.hydrateRoot;
export default D.default ?? D;

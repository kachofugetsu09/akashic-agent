// Host React's automatic JSX runtime, for plugins compiled with jsx: automatic.
const J = window.__akashicRuntime.ReactJSXRuntime;
export const jsx = J.jsx;
export const jsxs = J.jsxs;
export const Fragment = J.Fragment;

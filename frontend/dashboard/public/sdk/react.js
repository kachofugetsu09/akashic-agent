// Re-exports the host's React instance to plugin modules (resolved via the
// import map). Keeps a single React so hooks work across host + plugins.
const R = window.__akashicRuntime.React;
export default R.default ?? R;
export const {
  useState, useEffect, useRef, useMemo, useCallback, useContext, useReducer,
  useLayoutEffect, useImperativeHandle, useId, useTransition, useDeferredValue,
  createContext, createElement, cloneElement, forwardRef, memo, lazy, Suspense,
  Fragment, StrictMode, Children, isValidElement, startTransition,
} = R;

// Re-exports the host's React instance to plugin modules (resolved via the
// import map). Keeps a single React so hooks work across host + plugins.
const R = window.__akashicRuntime.React;
export default R.default ?? R;
export const {
  Activity, Children, Component, Fragment, Profiler, PureComponent, StrictMode,
  Suspense, act, cache, cacheSignal, captureOwnerStack, cloneElement,
  createContext, createElement, createRef, forwardRef, isValidElement, lazy,
  memo, startTransition, unstable_useCacheRefresh, use, useActionState,
  useCallback, useContext, useDebugValue, useDeferredValue, useEffect,
  useEffectEvent, useId, useImperativeHandle, useInsertionEffect,
  useLayoutEffect, useMemo, useOptimistic, useReducer, useRef, useState,
  useSyncExternalStore, useTransition, version,
} = R;

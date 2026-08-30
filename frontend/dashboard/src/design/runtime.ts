import * as React from "react";
import * as ReactJSXRuntime from "react/jsx-runtime";
import * as ReactDOMClient from "react-dom/client";
import {
  currentTheme,
  cycleTheme,
  subscribeTheme,
  themes,
  useTheme,
} from "../../../theme/src/theme-runtime";
import {
  MaterialButton,
  MaterialFilterChip,
  MaterialIconButton,
} from "../../../theme/src/material-react";

const WebUi = {
  currentTheme,
  cycleTheme,
  subscribeTheme,
  themes,
  useTheme,
  MaterialButton,
  MaterialFilterChip,
  MaterialIconButton,
};

// The shared runtime handed to dynamically-imported plugin modules. The static
// shim files under /assets/sdk/*.js read these off window so that plugins and
// the host resolve React and the small Web UI runtime to one instance.
export interface AkashicRuntime {
  React: typeof React;
  ReactJSXRuntime: typeof ReactJSXRuntime;
  ReactDOMClient: typeof ReactDOMClient;
  WebUi: typeof WebUi;
}

declare global {
  interface Window {
    __akashicRuntime?: AkashicRuntime;
  }
}

// Publish the runtime before any plugin is imported.
export function exposeRuntime(): void {
  window.__akashicRuntime = { React, ReactJSXRuntime, ReactDOMClient, WebUi };
}

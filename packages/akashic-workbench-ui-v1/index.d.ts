import type { WebEntry, WebEntryView, WebUiDisposer } from "@akashic/web-ui-v1";

export type WorkbenchPanelEntry = Omit<WebEntry, "render"> & {
  label: string;
  render(host: HTMLElement, view: WebEntryView): WebUiDisposer;
};

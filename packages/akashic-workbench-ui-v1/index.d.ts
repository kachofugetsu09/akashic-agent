import type { WebEntry, WebEntryView, WebUiDisposer } from "@akashic/web-ui-v1";

export const contractId: "workbench.panels.v1";
export const contractSha256: "724b282c22c4b3f3a36967ab664c4dfd8bce4257665f99459000306938caf527";

export type WorkbenchPanelEntry = Omit<WebEntry, "render"> & {
  label: string;
  render(host: HTMLElement, view: WebEntryView): WebUiDisposer;
};

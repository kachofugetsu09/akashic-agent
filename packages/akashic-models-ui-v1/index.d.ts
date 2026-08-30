import type { WebEntry, WebEntryView, WebUiDisposer } from "@akashic/web-ui-v1";

export interface ModelConnectionSummary {
  id: string;
  name: string;
  driverId: string;
  authIdentity: string;
  availability: string;
}

export interface ModelSummary {
  id: string;
  connectionId: string;
  kind: "chat" | "embedding";
  model: string;
  availability: string;
}

export interface ProviderState {
  readonly connection: ModelConnectionSummary | null;
  readonly models: readonly ModelSummary[];
  readonly template: ModelProviderTemplate | null;
}

export interface ModelProviderTemplate {
  id: string;
  label: string;
  detail: string;
  icon?: `data:image/svg+xml,${string}`;
  order?: number;
  defaults?: Readonly<Record<string, unknown>>;
}

export interface ManualConnectionInput {
  name: string;
  endpoint: string;
  credential: Record<string, string>;
  driverConfig: Record<string, unknown>;
  model: Record<string, unknown>;
}

export interface ConnectionUpdateInput {
  name: string;
  endpoint: string | null;
  credential: Record<string, string> | null;
  driverConfig: Record<string, unknown> | null;
}

export interface ProviderActions {
  createManual(input: ManualConnectionInput): Promise<void>;
  update(input: ConnectionUpdateInput): Promise<void>;
  startAuth(input: Record<string, string>): Promise<Record<string, unknown>>;
  finishAuth(attemptId: string): Promise<Record<string, unknown>>;
  cancelAuth(attemptId: string): Promise<void>;
  sync(): Promise<void>;
}

export interface ProviderProps {
  readonly state: ProviderState;
  readonly actions: ProviderActions;
  close(): void;
  changed(message: string): void;
}

export type ModelProviderEntry = Omit<WebEntry, "render"> & {
  label: string;
  detail: string;
  icon?: `data:image/svg+xml,${string}`;
  connectionIcon?: `data:image/svg+xml,${string}`;
  editTemplateId?: string;
  templates?: readonly ModelProviderTemplate[];
  /** Build the dialog with the public settings-dialog-* form classes. */
  render(host: HTMLElement, view: WebEntryView, props: ProviderProps): WebUiDisposer;
};

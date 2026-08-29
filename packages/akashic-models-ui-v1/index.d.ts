import type { WebEntry, WebEntryView, WebUiDisposer } from "@akashic/web-ui-v1";

export const contractId: "models.connection-types.v1";
export const contractSha256: "258bc92f1a3f7e15c8d5421c787a31f2fa1c76f2d9166c0e806f858e23853266";

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
  render(host: HTMLElement, view: WebEntryView, props: ProviderProps): WebUiDisposer;
};

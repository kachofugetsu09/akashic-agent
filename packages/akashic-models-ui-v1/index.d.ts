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
  discover(input: Omit<ManualConnectionInput, "model">, signal?: AbortSignal): Promise<readonly Record<string, unknown>[]>;
  discoverSaved(signal?: AbortSignal): Promise<readonly Record<string, unknown>[]>;
  addModel(input: Record<string, unknown>): Promise<void>;
  verifyModel(modelId: string): Promise<void>;
  /**
   * Write user-declared chat capabilities for a saved model.
   * The patch carries the full target state of the three fields;
   * null clears a token limit to unknown. Embedding models reject it.
   */
  updateModel(modelId: string, patch: {
    context_window: number | null;
    max_output_tokens: number | null;
    image_input: boolean;
  }): Promise<void>;
  removeModel(modelId: string): Promise<void>;
  /** Discover, choose and save models; false means cancelled before saving; failed or interrupted saves reject. */
  selectModels(): Promise<boolean>;
  disableConnection(): Promise<void>;
  createManual(input: ManualConnectionInput): Promise<void>;
  update(input: ConnectionUpdateInput): Promise<void>;
  startAuth(input: Record<string, string>): Promise<Record<string, unknown>>;
  finishAuth(attemptId: string): Promise<Record<string, unknown>>;
  cancelAuth(attemptId: string): Promise<void>;
  /** Refresh capabilities of selected models; never adopt unseen candidates. */
  sync(): Promise<void>;
}

export interface ModelPickOptions {
  /** Sheet title, for example `目录 · ${connectionName}`. */
  title?: string;
  /** Extra hint line under the title; keep it to one sentence. */
  hint?: string;
  /** Candidate `model` names that start checked. */
  checked?: readonly string[];
  /** Candidate `model` names already saved in the catalog; the sheet marks them 已有. */
  present?: readonly string[];
  /** Candidate `model` names whose checkbox is locked checked (in-use models). */
  locked?: readonly string[];
  confirmLabel?: string;
}

export interface ProviderUi {
  /**
   * Host-owned candidate sheet with checkbox selection.
   * Resolves the selected candidate subset, or null when the user cancels.
   * Selection is not adoption: the caller still verifies/saves via actions.
   */
  pickModels(candidates: readonly Record<string, unknown>[], options?: ModelPickOptions): Promise<readonly Record<string, unknown>[] | null>;
}

export interface ProviderProps {
  readonly state: ProviderState;
  readonly actions: ProviderActions;
  readonly ui: ProviderUi;
  /** Report unsaved changes; the host owns close and navigation checks. */
  dirty(value: boolean): void;
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
  /** Direct API-key connection with an actual embedding dimension probe. */
  embeddingApiKey?: boolean;
  /** Driver supports catalog refresh for selected models; saved configuration owns verified purposes. */
  catalogSync?: boolean;
  /** Build the dialog with the public settings-dialog-* form classes. */
  render(host: HTMLElement, view: WebEntryView, props: ProviderProps): WebUiDisposer;
};

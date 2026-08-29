export type WebUiDisposer = () => void;
export type WebMountCardinality = "single" | "list";

export interface WebMountDefinition {
  id: string;
  cardinality: WebMountCardinality;
}

export interface WebEntry {
  id: string;
  order?: number;
  children?: WebMountDefinition[];
  render(host: HTMLElement, view: WebEntryView): void | WebUiDisposer;
  [key: string]: unknown;
}

export interface WebMountView {
  readonly entries: readonly WebEntry[];
  render(entryId: string, host: HTMLElement): WebUiDisposer;
}

export interface WebEntryView {
  child(mountId: string): WebMountView;
}

export interface WebMountRegistration {
  register(entry: WebEntry): WebUiDisposer;
}

export interface WebHostContextV1 {
  readonly catalogId: string;
  readonly pluginId: string;
  readonly generationId: string;
  readonly sourceRevision: string;
  readonly http: {
    request(path: string, init?: RequestInit): Promise<Response>;
  };
  readonly ui: {
    inject(
      mountId: string,
      connect: (mount: WebMountRegistration) => WebUiDisposer,
    ): WebUiDisposer;
  };
}

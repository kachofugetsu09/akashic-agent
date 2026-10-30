declare module "*.svg" {
  const src: string;
  export default src;
}

interface ImportMeta {
  readonly env: { readonly DEV: boolean };
}

declare module "*.html?raw" {
  const content: string;
  export default content;
}

declare module "*?worker&inline" {
  const WorkerFactory: { new (): Worker };
  export default WorkerFactory;
}

declare module "*?url" {
  const url: string;
  export default url;
}

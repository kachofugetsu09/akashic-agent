declare module "*.svg" {
  const src: string;
  export default src;
}

interface ImportMeta {
  readonly env: { readonly DEV: boolean };
}

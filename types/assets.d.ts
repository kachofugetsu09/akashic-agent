declare module "*.svg" {
  const src: string;
  export default src;
}

declare module "*.png" {
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

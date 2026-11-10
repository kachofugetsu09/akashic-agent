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

// 设计编辑器包用 esbuild 的 file 加载器引入 WASM 与字体，导入值是构建后的地址。
declare module "*.wasm" {
  const url: string;
  export default url;
}

declare module "*.ttf" {
  const url: string;
  export default url;
}

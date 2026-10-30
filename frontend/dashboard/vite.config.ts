import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import { emitThemeCatalog } from "../theme/src/vite-theme-plugin";
import { injectEntrySplash } from "./vite-splash-plugin";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");

// 导入映射里的 SDK 垫片是稳定 URL：每次打开都要逐个等一次 304，慢链路上是外壳挂载前最后一批往返。
// 构建时给它们带上内容版本，服务端对带版本的地址永久缓存（plugins/ui/dashboard_app.py）。
function versionSdkImports(): Plugin {
  return {
    name: "version-sdk-imports",
    apply: "build",
    transformIndexHtml(html) {
      return html.replace(/"\/dashboard\/assets\/sdk\/([\w-]+\.js)"/g, (_match, file: string) => {
        const digest = createHash("sha256").update(readFileSync(resolve(here, "public/sdk", file))).digest("hex").slice(0, 12);
        return `"/dashboard/assets/sdk/${file}?v=${digest}"`;
      });
    },
  };
}

// The FastAPI dashboard serves index.html at "/" and mounts the build output
// dir at "/assets" behind the public "/dashboard" route. We emit flat hashed
// files so every public asset URL resolves under /dashboard/assets/<hash>.
export default defineConfig({
  root: here,
  base: "/dashboard/assets/",
  plugins: [react(), emitThemeCatalog(), injectEntrySplash(), versionSdkImports()],
  build: {
    outDir: resolve(repoRoot, "static", "dashboard"),
    emptyOutDir: true,
    assetsDir: "",
    sourcemap: false,
    rollupOptions: {
      output: {
        entryFileNames: "[name]-[hash].js",
        chunkFileNames: "[name]-[hash].js",
        assetFileNames: "[name]-[hash][extname]",
      },
    },
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:2236",
      "/plugins": "http://127.0.0.1:2236",
    },
  },
});

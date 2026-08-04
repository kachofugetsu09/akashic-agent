import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { emitThemeCatalog } from "../theme/src/vite-theme-plugin";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");
const outDir = process.env.AKASHIC_MOBILE_WEB_OUT_DIR
  ? resolve(process.env.AKASHIC_MOBILE_WEB_OUT_DIR)
  : resolve(repoRoot, "dist", "mobile-web");

export default defineConfig({
  root: here,
  base: "./",
  plugins: [react(), emitThemeCatalog()],
  resolve: {
    alias: {
      "@": resolve(here, "src"),
    },
  },
  build: {
    outDir,
    emptyOutDir: true,
    assetsDir: "assets",
    sourcemap: false,
    rollupOptions: {
      input: resolve(here, "mobile.html"),
      output: {
        entryFileNames: "assets/[name]-[hash].js",
        chunkFileNames: "assets/[name]-[hash].js",
        assetFileNames: "assets/[name]-[hash][extname]",
      },
    },
  },
});

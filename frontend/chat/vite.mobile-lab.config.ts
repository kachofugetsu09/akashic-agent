import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { emitThemeCatalog } from "../theme/src/vite-theme-plugin";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");

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
    outDir: resolve(repoRoot, "dist", "mobile-web-lab"),
    emptyOutDir: true,
    assetsDir: "assets",
    sourcemap: false,
    rollupOptions: {
      input: {
        lab: resolve(here, "mobile-lab.html"),
        frame: resolve(here, "mobile-lab-frame.html"),
      },
      output: {
        entryFileNames: "assets/[name]-[hash].js",
        chunkFileNames: "assets/[name]-[hash].js",
        assetFileNames: "assets/[name]-[hash][extname]",
      },
    },
  },
});

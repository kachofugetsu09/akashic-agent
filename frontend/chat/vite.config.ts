import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");

export default defineConfig({
  root: here,
  base: "/assets/",
  plugins: [react()],
  resolve: {
    alias: {
      "@": resolve(here, "src"),
    },
  },
  build: {
    outDir: resolve(repoRoot, "static", "chat"),
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
      "/api": "http://127.0.0.1:6322",
      "/ws": {
        target: "ws://127.0.0.1:6322",
        ws: true,
      },
    },
  },
});

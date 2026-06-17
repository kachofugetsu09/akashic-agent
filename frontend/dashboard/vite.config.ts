import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");

// The FastAPI dashboard serves index.html at "/" and mounts the build output
// dir at "/assets". We emit flat hashed files (assetsDir: "") into
// static/dashboard so every asset URL resolves under /assets/<hash>.
export default defineConfig({
  root: here,
  base: "/assets/",
  plugins: [react()],
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

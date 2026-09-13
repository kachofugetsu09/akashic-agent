import { existsSync, createReadStream } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";
import { emitThemeCatalog } from "../theme/src/vite-theme-plugin";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");
const apiOrigin = process.env.AKASHIC_API_ORIGIN ?? "http://127.0.0.1:2236";
const wsOrigin = apiOrigin.replace(/^http/, "ws");

// AKASHIC_LOCAL_PLUGIN_UI=akasha 时，插件 UI 资源不再代理到远端，
// 改由本仓库 plugins/<id>/ 下的源码提供，便于本地改样式看远端数据。
// 本地 message_ui.js 落后于远端版本（缺 slot 挂载），只拦截 stylesheet。
const localPluginIds = (process.env.AKASHIC_LOCAL_PLUGIN_UI ?? "")
  .split(",").map((id) => id.trim()).filter(Boolean);
const localPluginFiles: Record<string, Record<string, string>> = {
  akasha: {
    stylesheet: resolve(repoRoot, "plugins", "akasha", "message_ui.css"),
  },
};

function serveLocalPluginUi(): Plugin {
  return {
    name: "serve-local-plugin-ui",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url?.startsWith("/api/chat/plugin-ui/asset")) return next();
        const params = new URL(req.url, "http://local").searchParams;
        const pluginId = params.get("plugin_id") ?? "";
        const kind = params.get("kind") ?? "";
        const file = localPluginIds.includes(pluginId) ? localPluginFiles[pluginId]?.[kind] : undefined;
        if (!file || !existsSync(file)) return next();
        res.setHeader("content-type", kind === "stylesheet" ? "text/css; charset=utf-8" : "text/javascript; charset=utf-8");
        res.setHeader("cache-control", "no-store");
        createReadStream(file).pipe(res);
      });
    },
  };
}

export default defineConfig({
  root: here,
  base: "/assets/",
  plugins: [react(), emitThemeCatalog(), serveLocalPluginUi()],
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
      "/api": apiOrigin,
      "/ws": {
        target: wsOrigin,
        ws: true,
      },
    },
  },
});

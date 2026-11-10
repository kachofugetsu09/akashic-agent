import { createReadStream, existsSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import autoprefixer from "autoprefixer";
import tailwindcss from "tailwindcss";
import { defineConfig, type Plugin } from "vite";
import { emitThemeCatalog } from "../theme/src/vite-theme-plugin";

// 聊天的独立夹具页面：只给本地开发（vite dev 连真实后端）与夹具脚本用，不进发行包、服务端也不提供。
// 和外壳里一样经 mountChat 挂到 .akashic-chat 根节点；React 与主题打进本页，插件模块经页面 import map 取共享运行时。
const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const repoRoot = resolve(here, "..", "..");
const apiOrigin = process.env.AKASHIC_API_ORIGIN ?? "http://127.0.0.1:2236";

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
  root: resolve(here, "harness"),
  base: "/assets/",
  // 插件共享运行时的垫片归外壳，夹具页面直接用同一份。
  publicDir: resolve(repoRoot, "frontend", "dashboard", "public"),
  plugins: [react(), emitThemeCatalog(), serveLocalPluginUi()],
  css: { postcss: { plugins: [tailwindcss(require("./tailwind.config.cjs")), autoprefixer(), require("./postcss-scope-chat.cjs")()] } },
  resolve: {
    dedupe: ["@pierre/diffs"],
    alias: {
      "@": resolve(here, "src"),
      mermaid: resolve(here, "src", "mermaid-stub.ts"),
    },
  },
  worker: { format: "es" },
  build: {
    outDir: resolve(repoRoot, "node_modules", ".cache", "chat-harness"),
    emptyOutDir: true,
    assetsDir: "",
    sourcemap: false,
    // 与发布构建同名的样式文件：挂载入口据此认出页面已带样式，不再重复插入。
    cssCodeSplit: false,
    rollupOptions: {
      output: {
        entryFileNames: "[name]-[hash].js",
        chunkFileNames: "[name]-[hash].js",
        assetFileNames: (asset) => asset.names.includes("style.css") ? "chat-app.css" : "[name]-[hash][extname]",
      },
    },
  },
  server: {
    proxy: {
      "/api": apiOrigin,
      "/ws": { target: apiOrigin.replace(/^http/, "ws"), ws: true },
    },
  },
});

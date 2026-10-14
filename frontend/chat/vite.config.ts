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

// 阅读字体此前要等 CSS 下载解析后才被发现，首屏文本先以系统字体渲染再 swap。
// 构建期把已成形的字体资产以 preload 注入 HTML，让字体下载与 JS 并行开始。
// 只预载首屏必用的三份：Source Serif 4 正常体、Noto Serif SC 首片（ASCII/标点/假名）、JetBrains Mono；
// 其余 Noto 分片与斜体仍按需加载，不为可能不渲染的字形提前占带宽。
function preloadReadingFonts(): Plugin {
  const wanted = [
    /^source-serif(?!-italic)-[A-Za-z0-9_-]+\.woff2$/u,
    /^noto-serif-sc-0-[A-Za-z0-9_-]+\.woff2$/u,
    /^JetBrainsMono-[A-Za-z0-9_-]+\.woff2$/u,
  ];
  return {
    name: "preload-reading-fonts",
    apply: "build",
    transformIndexHtml: {
      order: "post",
      handler(_html, ctx) {
        return Object.keys(ctx.bundle ?? {})
          .filter((key) => wanted.some((re) => re.test(key)))
          .map((key) => ({
            tag: "link",
            attrs: { rel: "preload", href: `/assets/${key}`, as: "font", type: "font/woff2", crossorigin: true },
            injectTo: "head-prepend" as const,
          }));
      },
    },
  };
}

export default defineConfig({
  root: here,
  base: "/assets/",
  plugins: [react(), emitThemeCatalog(), serveLocalPluginUi(), preloadReadingFonts()],
  resolve: {
    alias: {
      "@": resolve(here, "src"),
      mermaid: resolve(here, "src", "mermaid-stub.ts"),
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
        // 稳定的框架代码单独成块，发版时浏览器只失效应用层那一小块。
        // 不做兜底 vendor 桶：懒加载依赖必须留在自然 chunk 里，避免被打包进首屏。
        manualChunks(id) {
          if (!id.includes("node_modules")) return;
          if (/[/\\](react|react-dom|scheduler)[/\\]/u.test(id)) return "vendor-react";
          if (id.includes("/effect/")) return "vendor-effect";
        },
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

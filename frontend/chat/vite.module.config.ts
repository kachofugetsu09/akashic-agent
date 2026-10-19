import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

// 聊天的同文档挂载模块：conversation_ui 在外壳页面里 import 它。React、外壳运行时与聊天公开运行时
// 都经外壳 import map 取同一份，不打进本包；产物与 iframe 版同目录发布，入口和样式用稳定文件名。
const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");
const themeRuntime = resolve(repoRoot, "frontend", "theme", "src", "theme-runtime.ts");
const shellTheme = resolve(here, "src", "shell-theme.ts");

// 任何解析到主题运行时的导入都改用外壳那一份，避免同一文档里两份主题状态。
function useShellTheme(): Plugin {
  return {
    name: "use-shell-theme",
    enforce: "pre",
    async resolveId(source, importer, options) {
      if (!source.includes("theme-runtime") || importer === shellTheme) return null;
      const resolved = await this.resolve(source, importer, { ...options, skipSelf: true });
      return resolved?.id === themeRuntime ? shellTheme : null;
    },
  };
}

export default defineConfig({
  root: here,
  base: "/assets/",
  plugins: [react(), useShellTheme()],
  resolve: {
    dedupe: ["@pierre/diffs"],
    alias: {
      "@": resolve(here, "src"),
      mermaid: resolve(here, "src", "mermaid-stub.ts"),
    },
  },
  worker: { format: "es" },
  // 公共目录（iframe 页的 sdk 垫片）由 iframe 版构建发布，这里不重复复制。
  publicDir: false,
  build: {
    outDir: resolve(repoRoot, "static", "chat"),
    emptyOutDir: false,
    assetsDir: "",
    sourcemap: false,
    // 样式合成一份稳定文件名的 chat-app.css，由挂载入口自己插入页面。
    cssCodeSplit: false,
    // 不用库模式：库模式会把字体等资源全部内联进样式表。普通构建保留入口导出，资源各自成文件。
    rollupOptions: {
      input: { "chat-app": resolve(here, "src", "mount.tsx") },
      preserveEntrySignatures: "exports-only",
      external: ["react", "react/jsx-runtime", "react-dom", "react-dom/client", "@akashic/web-ui-v1", "@akashic/chat-ui-v1"],
      output: {
        entryFileNames: "[name].js",
        chunkFileNames: "chat-app-[name]-[hash].js",
        // 合并出的样式在 Vite 里叫 style.css；按 ?url 引用的样式（如 katex）保留带哈希的文件名，由使用处自己加载。
        assetFileNames: (asset) => asset.names.includes("style.css") ? "chat-app.css" : "[name]-[hash][extname]",
      },
    },
  },
});

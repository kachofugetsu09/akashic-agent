import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import autoprefixer from "autoprefixer";
import tailwindcss from "tailwindcss";
import { defineConfig, type Plugin } from "vite";
import { emitThemeCatalog } from "../theme/src/vite-theme-plugin";


// 聊天的发布构建：挂载模块 chat-app.js 由 conversation_ui 经 Host SDK @akashic/chat-app-v1 在外壳页面里 import。
// React、外壳运行时与聊天公开运行时都经外壳 import map 取同一份，不打进本包；入口和样式用稳定文件名。
// 开发与夹具脚本用的独立页面见 vite.harness.config.ts。
const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const repoRoot = resolve(here, "..", "..");
const shellOwned = new Set(["theme-runtime.ts", "material-react.tsx"].map((name) => resolve(repoRoot, "frontend", "theme", "src", name)));
const shellTheme = resolve(here, "src", "shell-theme.ts");

// 主题状态与 Material 组件（自定义元素全文档只能登记一次）归外壳所有：解析到它们的导入都改用外壳运行时那一份。
function useShellTheme(): Plugin {
  return {
    name: "use-shell-theme",
    enforce: "pre",
    async resolveId(source, importer, options) {
      if (!/theme-runtime|material-react/u.test(source) || importer === shellTheme) return null;
      const resolved = await this.resolve(source, importer, { ...options, skipSelf: true });
      return resolved && shellOwned.has(resolved.id) ? shellTheme : null;
    },
  };
}

// 构建版本：参与构建的全部源文件（相对路径与内容）的摘要，写进挂载代码并另存 chat-app.build；服务端读后者，页面对照前者，
// 不同就提示刷新。在 renderChunk 里替换：Rollup 之后才按最终内容算文件名哈希，所在分块会随版本换名，不会命中旧的永久缓存。
const BUILD_PLACEHOLDER = "__AKASHIC_CHAT_BUILD__";

function stampChatBuild(): Plugin {
  let build = "";
  let stamped = 0;
  return {
    name: "stamp-chat-build",
    apply: "build",
    // watch 模式复用同一插件实例，每轮输出从零计数。
    renderStart() {
      stamped = 0;
    },
    buildEnd() {
      const hash = createHash("sha256");
      for (const id of [...this.getModuleIds()].map((id) => id.split("?")[0]).sort()) {
        hash.update(relative(repoRoot, id)).update(existsSync(id) ? readFileSync(id) : this.getModuleInfo(id)?.code ?? "");
      }
      build = hash.digest("hex").slice(0, 16);
    },
    renderChunk(code) {
      if (!code.includes(BUILD_PLACEHOLDER)) return null;
      stamped += 1;
      return { code: code.replaceAll(BUILD_PLACEHOLDER, build), map: null };
    },
    generateBundle() {
      if (stamped !== 1) throw new Error(`构建版本占位应只出现在一个分块里，实际 ${stamped} 个`);
      this.emitFile({ type: "asset", fileName: "chat-app.build", source: build });
    },
  };
}

export default defineConfig({
  root: here,
  base: "/assets/",
  plugins: [react(), useShellTheme(), stampChatBuild(), emitThemeCatalog()],
  // 与外壳同一文档：样式只作用于聊天根节点（见 postcss-scope-chat.cjs）。
  css: { postcss: { plugins: [tailwindcss(require("./tailwind.config.cjs")), autoprefixer(), require("./postcss-scope-chat.cjs")()] } },
  resolve: {
    dedupe: ["@pierre/diffs"],
    alias: {
      "@": resolve(here, "src"),
      mermaid: resolve(here, "src", "mermaid-stub.ts"),
    },
  },
  worker: { format: "es" },
  // 插件共享运行时的垫片归外壳（frontend/dashboard/public/sdk），聊天不另发一份。
  publicDir: false,
  build: {
    outDir: resolve(repoRoot, "static", "chat"),
    emptyOutDir: true,
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
        // 入口唯一的静态依赖分块（mount.tsx 本体）用稳定文件名：外壳 index.html 直接 modulepreload 它
        // （浏览器的 modulepreload 不抓依赖分块，带哈希的名字外壳无从得知）。缓存合同与入口/样式一致
        // （ETag 校验，见 plugins/akashic_clients/static.py 的 _STABLE_ENTRY）。懒加载分块仍带内容哈希。
        chunkFileNames: (chunk) => chunk.name === "mount" ? "chat-app-mount.js" : "chat-app-[name]-[hash].js",
        // 合并出的样式在 Vite 里叫 style.css；按 ?url 引用的样式（如 katex）保留带哈希的文件名，由使用处自己加载。
        assetFileNames: (asset) => asset.names.includes("style.css") ? "chat-app.css" : "[name]-[hash][extname]",
      },
    },
  },
});

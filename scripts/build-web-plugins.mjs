import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import autoprefixer from "autoprefixer";
import tailwindcss from "tailwindcss";
import { build } from "vite";

const repoRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const tailwindConfig = resolve(repoRoot, "frontend/dashboard/tailwind.config.ts");
const modules = [
  ["shell_ui", "shell.tsx"],
  ["workbench_ui", "workbench.tsx"],
];

const trimGeneratedLines = {
  name: "trim-generated-lines",
  generateBundle(_options, bundle) {
    for (const output of Object.values(bundle)) {
      if (output.type === "chunk") {
        output.code = output.code.replace(/[ \t]+$/gmu, "");
      }
    }
  },
};

for (const [plugin, entry] of modules) {
  await build({
    configFile: false,
    root: repoRoot,
    plugins: [react(), trimGeneratedLines],
    define: {
      "process.env.NODE_ENV": JSON.stringify("production"),
    },
    css: {
      postcss: {
        plugins: [tailwindcss({ config: tailwindConfig }), autoprefixer()],
      },
    },
    build: {
      outDir: resolve(repoRoot, "plugins", plugin),
      emptyOutDir: false,
      minify: "esbuild",
      sourcemap: false,
      lib: {
        entry: resolve(repoRoot, "frontend/dashboard/src/plugin-modules", entry),
        formats: ["es"],
        fileName: "web_module",
        cssFileName: "web_module",
      },
      rollupOptions: {
        external: ["react", "react/jsx-runtime", "react-dom/client", "@akashic/dashboard-ui"],
        output: {
          inlineDynamicImports: true,
          entryFileNames: "web_module.js",
          assetFileNames: "web_module[extname]",
        },
      },
    },
  });
}

for (const [plugin, entry] of [
  ["akasha", "dashboard_panel_inspector.ts"],
  ["wake", "dashboard_panel.ts"],
]) {
  await build({
    configFile: false,
    root: repoRoot,
    define: {
      "process.env.NODE_ENV": JSON.stringify("production"),
    },
    build: {
      outDir: resolve(repoRoot, "plugins", plugin),
      emptyOutDir: false,
      minify: "esbuild",
      sourcemap: false,
      lib: {
        entry: resolve(repoRoot, "plugins", plugin, entry),
        formats: ["es"],
        fileName: "web_module",
        cssFileName: "web_module",
      },
      rollupOptions: {
        output: {
          inlineDynamicImports: true,
          entryFileNames: "web_module.js",
          assetFileNames: "web_module[extname]",
        },
      },
    },
  });
}

import { access, readdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import autoprefixer from "autoprefixer";
import { build as bundle, transform } from "esbuild";
import tailwindcss from "tailwindcss";
import { build } from "vite";
import webUiPreset from "../packages/akashic-web-ui-v1/tailwind-preset.mjs";

const repoRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const pluginsRoot = resolve(repoRoot, "plugins");
const pluginNames = (await readdir(pluginsRoot, { withFileTypes: true }))
  .filter((entry) => entry.isDirectory())
  .map((entry) => entry.name)
  .sort();
const sourceRoots = new Map();
const modules = (await Promise.all(pluginNames.map(async (plugin) => {
  for (const sourceRoot of [resolve(repoRoot, "frontend/plugins", plugin, "src"), resolve(pluginsRoot, plugin, "web")]) {
    try {
      await access(resolve(sourceRoot, "index.tsx"));
      sourceRoots.set(plugin, sourceRoot);
      return plugin;
    } catch (error) {
      if (!(error && typeof error === "object" && "code" in error && error.code === "ENOENT")) throw error;
    }
  }
  return null;
}))).filter((plugin) => plugin !== null);

const compactGeneratedModules = {
  name: "compact-generated-modules",
  async generateBundle(_options, bundle) {
    for (const output of Object.values(bundle)) {
      if (output.type === "chunk") {
        const result = await transform(output.code, {
          format: "esm",
          legalComments: "none",
          loader: "js",
          minify: true,
          target: "es2022",
        });
        output.code = `${result.code.replace(/[ \t]+$/gmu, "").trimEnd()}\n`;
      }
    }
  },
};

for (const plugin of modules) {
  const sourceRoot = sourceRoots.get(plugin);
  await build({
    configFile: false,
    root: repoRoot,
    plugins: [react(), compactGeneratedModules],
    define: {
      "process.env.NODE_ENV": JSON.stringify("production"),
    },
    css: {
      postcss: {
        plugins: [tailwindcss({
          config: {
            content: [resolve(sourceRoot, "**/*.{ts,tsx}")],
            presets: [webUiPreset],
          },
        }), autoprefixer()],
      },
    },
    build: {
      target: plugin === "computer" ? "es2022" : undefined,
      outDir: resolve(repoRoot, "plugins", plugin),
      emptyOutDir: false,
      minify: "esbuild",
      sourcemap: false,
      lib: {
        entry: resolve(sourceRoot, "index.tsx"),
        formats: ["es"],
        fileName: "web_module",
        cssFileName: "web_module",
      },
      rollupOptions: {
        external: ["react", "react/jsx-runtime", "react-dom/client", "@akashic/web-ui-v1"],
        output: {
          inlineDynamicImports: true,
          entryFileNames: "web_module.js",
          assetFileNames: "web_module[extname]",
        },
      },
    },
  });
}

// 新 Message Inspector 的源文件与其他前端一样保存在 frontend 下。
await bundle({
  entryPoints: [resolve(repoRoot, "frontend/plugins/akasha/src/mobile.js")],
  bundle: true,
  format: "esm",
  target: "es2022",
  minify: true,
  outfile: resolve(pluginsRoot, "akasha/message_ui.js"),
});

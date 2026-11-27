import { execFileSync } from "node:child_process";
import { access, readdir, readFile } from "node:fs/promises";
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
const pluginKitRoot = resolve(repoRoot, "packages", "akashic-plugin-kit");
const pluginKitPkg = JSON.parse(await readFile(resolve(pluginKitRoot, "package.json"), "utf8"));
const pluginKitExports = Object.fromEntries(
  Object.entries(pluginKitPkg.exports ?? {}).map(([subpath, target]) => {
    const cleanKey = subpath.replace(/^\.\//, "");
    const resolved = resolve(pluginKitRoot, String(target));
    if (!resolved.startsWith(`${pluginKitRoot}/`)) {
      throw new Error(`@akashic/plugin-kit export 越界: ${subpath} -> ${target}`);
    }
    return [cleanKey, resolved];
  }),
);
const themeSrc = resolve(repoRoot, "packages", "akashic-theme", "src");

const exists = (path) => access(path).then(() => true, (error) => {
  if (error && typeof error === "object" && "code" in error && error.code === "ENOENT") return false;
  throw error;
});

const pluginNames = (await readdir(pluginsRoot, { withFileTypes: true }))
  .filter((entry) => entry.isDirectory())
  .map((entry) => entry.name)
  .sort();

// 统一的构建期 SDK 解析：严格按 @akashic/plugin-kit/package.json 的 exports 白名单映射，防止拼写漂移或路径穿越。
const esbuildSdkAlias = {
  name: "akashic-sdk-alias",
  setup(buildApi) {
    buildApi.onResolve({ filter: /^@akashic\/plugin-kit\/.+/ }, (args) => {
      const subpath = args.path.slice("@akashic/plugin-kit/".length);
      const target = pluginKitExports[subpath];
      if (!target) {
        return { errors: [{ text: `@akashic/plugin-kit 未导出子路径: ${args.path}` }] };
      }
      return { path: target };
    });
    buildApi.onResolve({ filter: /^@akashic\/theme\/.+/ }, async (args) => {
      const subpath = args.path.slice("@akashic/theme/".length);
      const target = resolve(themeSrc, subpath);
      if (!target.startsWith(`${themeSrc}/`)) {
        return { errors: [{ text: `@akashic/theme 路径越界: ${args.path}` }] };
      }
      for (const candidate of [target, `${target}.tsx`, `${target}.ts`, `${target}.js`, `${target}.mjs`, `${target}.css`]) {
        if (await exists(candidate)) return { path: candidate };
      }
      return { errors: [{ text: `@akashic/theme 子路径不存在: ${args.path}` }] };
    });
  },
};

const viteSdkAlias = [
  ...Object.entries(pluginKitExports).map(([subpath, replacement]) => ({
    find: `@akashic/plugin-kit/${subpath}`,
    replacement,
  })),
  { find: /^@akashic\/theme\/(.+)$/, replacement: `${themeSrc}/$1` },
];

// 每个插件唯一的前端源码根目录约定：plugins/<name>/web[/src]。
function pluginFrontendRoots(plugin) {
  return {
    webDir: resolve(pluginsRoot, plugin, "web"),
    webSrc: resolve(pluginsRoot, plugin, "web", "src"),
  };
}

async function resolveUniqueFile(plugin, label, candidates) {
  const hits = [];
  for (const candidate of candidates) {
    if (await exists(candidate)) hits.push(candidate);
  }
  if (hits.length > 1) {
    throw new Error(`插件 ${plugin} 的 ${label} 存在重复入口：${hits.join(", ")}`);
  }
  return hits[0] ?? null;
}

async function ensureLocalPluginDeps(plugin, dir) {
  const pkg = resolve(dir, "package.json");
  const mods = resolve(dir, "node_modules");
  if (await exists(pkg) && !(await exists(mods))) {
    throw new Error(`插件 ${plugin} 缺少本地依赖，请先执行：npm ci --prefix ${dir.slice(repoRoot.length + 1)} --ignore-scripts`);
  }
  return (await exists(mods)) ? mods : null;
}

const compactGeneratedModules = {
  name: "compact-generated-modules",
  async generateBundle(_options, outputBundle) {
    for (const output of Object.values(outputBundle)) {
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

// 1. 识别自包含构建钩子（plugins/<name>/web/build.mjs 或 build-extra.mjs）。
const selfManaged = new Map();
const extraBuilders = new Map();
for (const plugin of pluginNames) {
  const { webDir } = pluginFrontendRoots(plugin);
  const fullScript = resolve(webDir, "build.mjs");
  if (await exists(fullScript)) {
    await ensureLocalPluginDeps(plugin, webDir);
    selfManaged.set(plugin, fullScript);
    continue;
  }
  const extraScript = resolve(webDir, "build-extra.mjs");
  if (await exists(extraScript)) {
    await ensureLocalPluginDeps(plugin, webDir);
    extraBuilders.set(plugin, extraScript);
  }
}

// 2. 构建标准 Dashboard web_module（入口为 plugins/<name>/web[/src]/index.tsx）。
const webModuleEntries = new Map();
for (const plugin of pluginNames) {
  if (selfManaged.has(plugin)) continue;
  const { webDir, webSrc } = pluginFrontendRoots(plugin);
  const entry = await resolveUniqueFile(plugin, "index.tsx", [
    resolve(webSrc, "index.tsx"),
    resolve(webDir, "index.tsx"),
  ]);
  if (entry) webModuleEntries.set(plugin, entry);
}

for (const [plugin, entryPath] of webModuleEntries.entries()) {
  const sourceRoot = dirname(entryPath);
  const { webDir } = pluginFrontendRoots(plugin);
  const localModules = await ensureLocalPluginDeps(plugin, webDir);
  await build({
    configFile: false,
    root: repoRoot,
    plugins: [react(), compactGeneratedModules],
    define: {
      "process.env.NODE_ENV": JSON.stringify("production"),
    },
    resolve: {
      alias: viteSdkAlias,
      ...(localModules ? { paths: [localModules] } : {}),
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
      target: "es2022",
      outDir: resolve(repoRoot, "plugins", plugin),
      emptyOutDir: false,
      minify: "esbuild",
      sourcemap: false,
      lib: {
        entry: entryPath,
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

// 3. 构建标准聊天插槽模块（plugin-ui / runner-ui / message-ui），与宿主共用 React 与 @akashic/chat-ui-v1。
const chatSlotModule = (source, outfile, localModules) => bundle({
  entryPoints: [source],
  bundle: true,
  format: "esm",
  target: "es2022",
  minify: true,
  jsx: "automatic",
  external: ["react", "react/jsx-runtime", "react-dom/client", "@akashic/chat-ui-v1"],
  plugins: [esbuildSdkAlias],
  ...(localModules ? { nodePaths: [localModules] } : {}),
  outfile: resolve(pluginsRoot, outfile),
});

const slotEntries = [["plugin-ui", "plugin_ui"], ["runner-ui", "runner_ui"], ["message-ui", "message_ui"]];
for (const plugin of pluginNames) {
  if (selfManaged.has(plugin)) {
    execFileSync(process.execPath, [selfManaged.get(plugin)], { cwd: repoRoot, stdio: "inherit" });
    continue;
  }
  const { webDir, webSrc } = pluginFrontendRoots(plugin);
  const localModules = await ensureLocalPluginDeps(plugin, webDir);
  for (const [source, output] of slotEntries) {
    const candidates = [];
    for (const dir of [webSrc, webDir]) {
      for (const extension of ["tsx", "ts", "js"]) {
        candidates.push(resolve(dir, `${source}.${extension}`));
      }
    }
    const hit = await resolveUniqueFile(plugin, `${source}.*`, candidates);
    if (hit) await chatSlotModule(hit, `${plugin}/${output}.js`, localModules);
  }
  if (extraBuilders.has(plugin)) {
    execFileSync(process.execPath, [extraBuilders.get(plugin)], { cwd: repoRoot, stdio: "inherit" });
  }
}


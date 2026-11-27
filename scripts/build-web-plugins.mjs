import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { access, cp, mkdir, readdir, copyFile, readFile, rename, rm, writeFile } from "node:fs/promises";
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

// 每个插件的前端源码候选目录（按新布局 plugins/<name>/web[/src] 与旧布局 frontend/plugins/<name>/src 双路径兼容，冲突报错）。
function pluginFrontendRoots(plugin) {
  return {
    webDir: resolve(pluginsRoot, plugin, "web"),
    webSrc: resolve(pluginsRoot, plugin, "web", "src"),
    legacyDir: resolve(repoRoot, "frontend", "plugins", plugin),
    legacySrc: resolve(repoRoot, "frontend", "plugins", plugin, "src"),
  };
}

async function resolveUniqueFile(plugin, label, candidates) {
  const hits = [];
  for (const candidate of candidates) {
    if (await exists(candidate)) hits.push(candidate);
  }
  if (hits.length > 1) {
    throw new Error(`插件 ${plugin} 的 ${label} 在新旧目录同时存在，请清理重复入口：${hits.join(", ")}`);
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

// 1. 识别完全接管构建的插件（web/build.mjs 或 frontend/plugins/<name>/build.mjs）。
const selfManaged = new Map();
const extraBuilders = new Map();
for (const plugin of pluginNames) {
  const { webDir, legacyDir } = pluginFrontendRoots(plugin);
  const fullScript = await resolveUniqueFile(plugin, "build.mjs", [
    resolve(webDir, "build.mjs"),
    resolve(legacyDir, "build.mjs"),
  ]);
  if (fullScript) {
    await ensureLocalPluginDeps(plugin, dirname(fullScript));
    selfManaged.set(plugin, fullScript);
    continue;
  }
  const extraScript = await resolveUniqueFile(plugin, "build-extra.mjs", [
    resolve(webDir, "build-extra.mjs"),
    resolve(legacyDir, "build-extra.mjs"),
  ]);
  if (extraScript) {
    await ensureLocalPluginDeps(plugin, dirname(extraScript));
    extraBuilders.set(plugin, extraScript);
  }
}

// 2. 构建标准 Dashboard web_module（入口为 index.tsx）。
const webModuleEntries = new Map();
for (const plugin of pluginNames) {
  if (selfManaged.has(plugin)) continue;
  const { webDir, webSrc, legacySrc } = pluginFrontendRoots(plugin);
  const entry = await resolveUniqueFile(plugin, "index.tsx", [
    resolve(webSrc, "index.tsx"),
    resolve(webDir, "index.tsx"),
    resolve(legacySrc, "index.tsx"),
  ]);
  if (entry) webModuleEntries.set(plugin, entry);
}

for (const [plugin, entryPath] of webModuleEntries.entries()) {
  const sourceRoot = dirname(entryPath);
  const { webDir, legacyDir } = pluginFrontendRoots(plugin);
  const localModules = (await ensureLocalPluginDeps(plugin, webDir)) ?? (await ensureLocalPluginDeps(plugin, legacyDir));
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
      target: plugin === "computer" ? "es2022" : undefined,
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
const legacyCustomBuilt = new Set(["visuals"]);

for (const plugin of pluginNames) {
  if (selfManaged.has(plugin)) {
    execFileSync(process.execPath, [selfManaged.get(plugin)], { cwd: repoRoot, stdio: "inherit" });
    continue;
  }
  if (legacyCustomBuilt.has(plugin)) continue;
  const { webDir, webSrc, legacyDir, legacySrc } = pluginFrontendRoots(plugin);
  const localModules = (await ensureLocalPluginDeps(plugin, webDir)) ?? (await ensureLocalPluginDeps(plugin, legacyDir));
  for (const [source, output] of slotEntries) {
    const candidates = [];
    for (const dir of [webSrc, webDir, legacySrc]) {
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

// 4. 过渡期兼容：仍未迁出到各自 build.mjs 的 visuals / design / file_viewer 专用构建段（待 PR D/E 迁出后删除）。
const visualsRoot = await resolveUniqueFile("visuals", "charts.ts", [
  resolve(pluginsRoot, "visuals/web/src/charts.ts"),
  resolve(pluginsRoot, "visuals/web/charts.ts"),
  resolve(repoRoot, "frontend/plugins/visuals/src/charts.ts"),
]);
const designEditorMain = await resolveUniqueFile("design", "editor/main.ts", [
  resolve(pluginsRoot, "design/web/editor/main.ts"),
  resolve(repoRoot, "frontend/plugins/design/editor/main.ts"),
]);
const fileViewerMain = await resolveUniqueFile("file_viewer", "viewer/main.ts", [
  resolve(pluginsRoot, "file_viewer/web/viewer/main.ts"),
  resolve(repoRoot, "frontend/plugins/file_viewer/viewer/main.ts"),
]);

const hasVisuals = Boolean(visualsRoot) && !selfManaged.has("visuals");
const hasDesign = Boolean(designEditorMain) && !selfManaged.has("design") && !extraBuilders.has("design");
const hasFileViewer = Boolean(fileViewerMain) && !selfManaged.has("file_viewer") && !extraBuilders.has("file_viewer");

if (hasVisuals) {
  const visualsDir = dirname(visualsRoot);
  const visualCharts = await bundle({
    entryPoints: [visualsRoot],
    bundle: true, format: "iife", target: "es2022", minify: true, write: false,
  });
  const visualFrameSource = await readFile(resolve(visualsDir, "frame.html"), "utf8");
  if (visualFrameSource.split("/*visual-charts*/").length !== 2) throw new Error("visuals frame.html 必须恰有一处 /*visual-charts*/ 占位符");
  const visualFrame = visualFrameSource
    .replace("/*visual-charts*/", () => visualCharts.outputFiles[0].text.trim().replaceAll("</script", "<\\/script"));
  await bundle({
    entryPoints: [resolve(visualsDir, "plugin-ui.ts")],
    bundle: true, format: "esm", target: "es2022", minify: true,
    plugins: [esbuildSdkAlias, { name: "frame-template", setup(b) {
      b.onResolve({ filter: /\.html\?raw$/ }, args => ({ path: resolve(args.resolveDir, args.path.slice(0, -4)), namespace: "frame" }));
      b.onLoad({ filter: /.*/, namespace: "frame" }, async args => ({
        contents: args.path.endsWith("frame.html") ? visualFrame : await readFile(args.path, "utf8"), loader: "text" }));
    } }],
    outfile: resolve(pluginsRoot, "visuals/message_ui.js"),
  });
  await writeFile(resolve(pluginsRoot, "visuals/frame.html"), visualFrame);
  await copyFile(resolve(visualsDir, "bootstrap.html"), resolve(pluginsRoot, "visuals/bootstrap.html"));
}

const themeBuild = await bundle({ entryPoints: [resolve(repoRoot, "packages/akashic-theme/src/theme-runtime.ts")],
  bundle: true, format: "esm", platform: "node", write: false });
const themeModule = await import(`data:text/javascript;base64,${Buffer.from(themeBuild.outputFiles[0].contents).toString("base64")}`);
if (hasVisuals) {
  await writeFile(resolve(pluginsRoot, "visuals/preview-theme.css"), themeModule.themeCss());
  await writeFile(
    resolve(pluginsRoot, "visuals/.akashic-generated-assets.json"),
    `${JSON.stringify(["frame.html", "bootstrap.html", "preview-theme.css"], null, 2)}\n`,
  );
}

if (hasDesign) {
  const designEditorDir = dirname(designEditorMain);
  const designAssets = resolve(pluginsRoot, "design/static");
  await rm(designAssets, { recursive: true, force: true });
  const designOptions = {
    bundle: true, format: "esm", target: "es2022", minify: true, metafile: true,
    outdir: designAssets, entryNames: "[name]-[hash]", assetNames: "[name]-[hash]",
    publicPath: "/design/assets/", legalComments: "none",
    define: { "process.env.NODE_ENV": JSON.stringify("production"), __VUE_OPTIONS_API__: "true",
      __VUE_PROD_DEVTOOLS__: "false", __VUE_PROD_HYDRATION_MISMATCH_DETAILS__: "false" },
    external: ["node:*", "fs", "path", "module", "url", "worker_threads"],
    loader: { ".wasm": "file", ".ttf": "file", ".woff2": "file" },
  };
  const designName = (p) => p.slice(p.lastIndexOf("/") + 1);
  const openPencilCore = resolve(repoRoot, "node_modules/@open-pencil/core/dist");
  const designWorkerBuild = await bundle({ ...designOptions, entryPoints: {
    "export-worker": resolve(openPencilCore, "io/formats/fig/export-worker.js"),
    "fig-worker": resolve(openPencilCore, "kiwi/fig/session/worker.js"),
  } });
  const designWorkers = Object.fromEntries(Object.entries(designWorkerBuild.metafile.outputs)
    .filter(([, output]) => output.entryPoint)
    .map(([p, output]) => [output.entryPoint.endsWith("export-worker.js") ? "export-worker" : "worker", `/design/assets/${designName(p)}`]));
  const designWorkerUrls = {
    name: "design-worker-urls",
    setup(b) {
      b.onLoad({ filter: /@open-pencil[\\/]core[\\/]dist[\\/](io[\\/]formats[\\/]fig[\\/]export|kiwi[\\/]fig[\\/]session[\\/]client)\.js$/ }, async (args) => ({
        loader: "js",
        contents: (await readFile(args.path, "utf8")).replace(/new URL\("\.\/(export-worker|worker)\.ts", import\.meta\.url\)/gu,
          (_, name) => `new URL(${JSON.stringify(designWorkers[name])}, import.meta.url)`),
      }));
    },
  };
  let designJSXPatched = 0;
  const designJSXEval = {
    name: "design-jsx-sandbox",
    setup(b) {
      b.onLoad({ filter: /@open-pencil[\\/]design-jsx[\\/]dist[\\/]chunks[\\/]fields\.js$/ }, async (args) => {
        designJSXPatched += 1;
        const source = await readFile(args.path, "utf8");
        const patched = source
          .replace("const element = createElement(buildComponent(jsxString, helperWarnings), null);\n\tconst tree = resolveToTree(element);",
            "const tree = await __akashicEvaluateJSX(jsxString, helperWarnings);")
          .replace('new Function("React", "__helpers", code)(mini_react_exports, designJSXHelpers(warnings))',
            '(() => { throw new Error("设计 JSX 只在沙箱里求值"); })()');
        if (patched.split("__akashicEvaluateJSX").length !== 2 || patched.includes("new Function(")) {
          throw new Error(`@open-pencil/design-jsx 的 fields.js 已变化，无法确认 JSX 沙箱补丁（${args.path}）`);
        }
        const client = JSON.stringify(resolve(designEditorDir, "jsx.ts"));
        return { loader: "js", contents: `import { evaluateJSX as __akashicEvaluateJSX } from ${client};\n${patched}` };
      });
    },
  };
  const designSandbox = await bundle({ ...designOptions, write: false, metafile: false,
    entryPoints: [resolve(designEditorDir, "jsx-sandbox.ts")] });
  const designSandboxScript = designSandbox.outputFiles.find((file) => file.path.endsWith(".js")).text.replaceAll("</script", "<\\/script");
  await writeFile(resolve(designAssets, "jsx.html"),
    `<!doctype html><meta charset="utf-8"><title>JSX</title><script type="module">${designSandboxScript}</script>\n`);
  await writeFile(resolve(designAssets, "preview.html"), await readFile(resolve(designEditorDir, "preview.html"), "utf8"));
  const designBuild = await bundle({ ...designOptions, plugins: [designWorkerUrls, designJSXEval],
    entryPoints: { editor: designEditorMain, worker: resolve(designEditorDir, "headless.ts") } });
  if (designJSXPatched !== 1) throw new Error(`design-jsx-sandbox 补丁命中 ${designJSXPatched} 次，应为 1 次`);
  const designOutputs = Object.entries(designBuild.metafile.outputs);
  const designEntry = designOutputs.find(([, output]) => output.entryPoint?.endsWith("editor/main.ts"));
  const designWorkerEntry = designOutputs.find(([, output]) => output.entryPoint?.endsWith("editor/headless.ts"));
  if (!designEntry || !designWorkerEntry) throw new Error("设计编辑器没有产出入口");
  await writeFile(resolve(designAssets, "worker.html"),
    `<!doctype html><meta charset="utf-8"><title>设计</title><script type="module" src="/design/assets/${designName(designWorkerEntry[0])}"></script>\n`);
  const { CORE_TOOLS } = await import("@open-pencil/core");
  const { toJsonSchema } = await import("@valibot/to-json-schema");
  const { JSX_REFERENCE } = await import("@open-pencil/design-jsx");
  const DESIGN_EXCLUDED = new Set(["stock_photo", "eval", "get_selection", "viewport_zoom_to_fit"]);
  await writeFile(resolve(designAssets, "tools.json"), JSON.stringify(CORE_TOOLS.filter((tool) => !DESIGN_EXCLUDED.has(tool.name))
    .map((tool) => ({ name: tool.name, description: tool.name === "render" ? `${tool.description}\n\n${JSX_REFERENCE}` : tool.description,
      parameters: toJsonSchema(tool.input, { errorMode: "ignore" }) }))));
  const designTheme = themeModule.themeCss() + await readFile(resolve(repoRoot, "packages/akashic-theme/src/brand-tokens.css"), "utf8");
  const designStyles = designOutputs.filter(([p]) => p.endsWith(".css")).map(([p]) => designName(p));
  await writeFile(resolve(designAssets, "editor.html"), `<!doctype html>
<html lang="zh-CN" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>设计</title>
<style>${designTheme}</style>
${designStyles.map((name) => `<link rel="stylesheet" href="/design/assets/${name}">`).join("\n")}
<script type="module" src="/design/assets/${designName(designEntry[0])}"></script>
</head>
<body><div id="design-root"></div></body>
</html>
`);
}

if (hasFileViewer) {
  const readerAssets = resolve(pluginsRoot, "file_viewer/static");
  const stagingAssets = resolve(pluginsRoot, "file_viewer/.static-staging");
  await rm(stagingAssets, { recursive: true, force: true });
  await mkdir(stagingAssets, { recursive: true });

  const copyHashed = async (sourcePath, stem, ext) => {
    const bytes = await readFile(sourcePath);
    const hash = createHash("sha256").update(bytes).digest("hex").slice(0, 8);
    const filename = `${stem}-${hash}${ext}`;
    await writeFile(resolve(stagingAssets, filename), bytes);
    return `/file_viewer/assets/${filename}`;
  };

  const pdfWorkerUrl = await copyHashed(resolve(repoRoot, "node_modules/pdfjs-dist/build/pdf.worker.min.mjs"), "pdf.worker", ".mjs");
  const docxWasmUrl = await copyHashed(resolve(repoRoot, "node_modules/@silurus/ooxml/dist/docx_parser_bg.wasm"), "docx_parser_bg", ".wasm");
  const xlsxWasmUrl = await copyHashed(resolve(repoRoot, "node_modules/@silurus/ooxml/dist/xlsx_parser_bg.wasm"), "xlsx_parser_bg", ".wasm");
  const pptxWasmUrl = await copyHashed(resolve(repoRoot, "node_modules/@silurus/ooxml/dist/pptx_parser_bg.wasm"), "pptx_parser_bg", ".wasm");

  await cp(resolve(repoRoot, "node_modules/pdfjs-dist/cmaps"), resolve(stagingAssets, "cmaps"), { recursive: true });
  await cp(resolve(repoRoot, "node_modules/pdfjs-dist/standard_fonts"), resolve(stagingAssets, "standard_fonts"), { recursive: true });

  const readerBuild = await bundle({
    entryPoints: { viewer: fileViewerMain },
    bundle: true, format: "esm", target: "es2022", minify: true, metafile: true,
    outdir: stagingAssets, entryNames: "[name]-[hash]", assetNames: "[name]-[hash]",
    publicPath: "/file_viewer/assets/", legalComments: "none",
    define: {
      "process.env.NODE_ENV": JSON.stringify("production"),
      "__AKASHIC_PDF_WORKER_URL__": JSON.stringify(pdfWorkerUrl),
      "__AKASHIC_DOCX_WASM_URL__": JSON.stringify(docxWasmUrl),
      "__AKASHIC_XLSX_WASM_URL__": JSON.stringify(xlsxWasmUrl),
      "__AKASHIC_PPTX_WASM_URL__": JSON.stringify(pptxWasmUrl),
      "__AKASHIC_PDF_CMAP_URL__": JSON.stringify("/file_viewer/assets/cmaps/"),
      "__AKASHIC_PDF_FONT_URL__": JSON.stringify("/file_viewer/assets/standard_fonts/"),
    },
    external: ["node:*", "fs", "path", "module", "url", "worker_threads"],
    loader: { ".wasm": "file", ".woff2": "file" },
  });
  const assetName = (p) => p.slice(p.lastIndexOf("/") + 1);
  const readerOutputs = Object.entries(readerBuild.metafile.outputs);
  const readerEntry = readerOutputs.find(([, output]) => output.entryPoint?.endsWith("viewer/main.ts"));
  if (!readerEntry) throw new Error("文档阅读器没有产出入口");
  const readerStyles = readerOutputs.filter(([p]) => p.endsWith(".css")).map(([p]) => assetName(p));
  const readerTheme = themeModule.themeCss() + await readFile(resolve(repoRoot, "packages/akashic-theme/src/brand-tokens.css"), "utf8");
  await writeFile(resolve(stagingAssets, "viewer.html"), `<!doctype html>
<html lang="zh-CN" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>文档阅读器</title>
<style>${readerTheme}</style>
${readerStyles.map((name) => `<link rel="stylesheet" href="/file_viewer/assets/${name}">`).join("\n")}
<script type="module" src="/file_viewer/assets/${assetName(readerEntry[0])}"></script>
</head>
<body><div id="reader-root"></div></body>
</html>
`);
  await rm(readerAssets, { recursive: true, force: true });
  await rename(stagingAssets, readerAssets);
}

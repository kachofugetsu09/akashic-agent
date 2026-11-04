import { access, readdir, copyFile, readFile, rm, writeFile } from "node:fs/promises";
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


// 改动摘要经回答尾注插槽挂载，与“改动”标签的 Web 模块分开打包。
await bundle({
  entryPoints: [resolve(repoRoot, "frontend/plugins/workspace_changes/src/message-ui.js")],
  bundle: true, format: "esm", target: "es2022", minify: true,
  outfile: resolve(pluginsRoot, "workspace_changes/message_ui.js"),
});

// 聊天插槽模块与宿主共用 React 与 @akashic/chat-ui-v1，由页面 import map 解析。
const chatSlotModule = (source, outfile) => bundle({
  entryPoints: [resolve(repoRoot, source)], bundle: true, format: "esm", target: "es2022", minify: true, jsx: "automatic",
  external: ["react", "react/jsx-runtime", "react-dom/client", "@akashic/chat-ui-v1"], outfile: resolve(pluginsRoot, outfile),
});
await chatSlotModule("frontend/plugins/host_execution/src/plugin-ui.tsx", "host_execution/plugin_ui.js");
await chatSlotModule("frontend/plugins/working_directory/src/plugin-ui.tsx", "working_directory/plugin_ui.js");
await chatSlotModule("frontend/plugins/skills/src/plugin-ui.tsx", "skills/plugin_ui.js");
await chatSlotModule("frontend/plugins/commands/src/plugin-ui.tsx", "commands/plugin_ui.js");
await chatSlotModule("frontend/plugins/jobs/src/plugin-ui.tsx", "jobs/plugin_ui.js");
await chatSlotModule("frontend/plugins/subagent/src/runner-ui.tsx", "subagent/runner_ui.js");
// 命令执行结果的命令块，只渲染本插件注册的工具结果。
await chatSlotModule("frontend/plugins/shell/src/plugin-ui.tsx", "shell/plugin_ui.js");
// Models 的回答尾注与输入框模型胶囊经插件界面插槽挂载，与设置页 Web 模块分开打包。
await chatSlotModule("frontend/plugins/models/src/plugin-ui.tsx", "models/plugin_ui.js");
// 项目的侧栏分组、新建表单与目录选择。
await chatSlotModule("frontend/plugins/projects/src/plugin-ui.tsx", "projects/plugin_ui.js");
// Akasha 的每轮检索卡片与项目记忆设置共用一个模块。
await chatSlotModule("frontend/plugins/akasha/src/plugin-ui.js", "akasha/message_ui.js");

// 可视化使用同一份 frame 模板完成服务端预检与浏览器展示；图表工具包内联进模板，不另开脚本入口。
const visualCharts = await bundle({
  entryPoints: [resolve(repoRoot, "frontend/plugins/visuals/src/charts.ts")],
  bundle: true, format: "iife", target: "es2022", minify: true, write: false,
});
const visualFrameSource = await readFile(resolve(repoRoot, "frontend/plugins/visuals/src/frame.html"), "utf8");
if (visualFrameSource.split("/*visual-charts*/").length !== 2) throw new Error("visuals frame.html 必须恰有一处 /*visual-charts*/ 占位符");
const visualFrame = visualFrameSource
  .replace("/*visual-charts*/", () => visualCharts.outputFiles[0].text.trim().replaceAll("</script", "<\\/script"));
await bundle({
  entryPoints: [resolve(repoRoot, "frontend/plugins/visuals/src/plugin-ui.ts")],
  bundle: true, format: "esm", target: "es2022", minify: true,
  plugins: [{ name: "frame-template", setup(build) {
    build.onResolve({ filter: /\.html\?raw$/ }, args => ({ path: resolve(args.resolveDir, args.path.slice(0, -4)), namespace: "frame" }));
    build.onLoad({ filter: /.*/, namespace: "frame" }, async args => ({
      contents: args.path.endsWith("frame.html") ? visualFrame : await readFile(args.path, "utf8"), loader: "text" }));
  } }],
  outfile: resolve(pluginsRoot, "visuals/message_ui.js"),
});
await writeFile(resolve(pluginsRoot, "visuals/frame.html"), visualFrame);
await copyFile(resolve(repoRoot, "frontend/plugins/visuals/src/bootstrap.html"), resolve(pluginsRoot, "visuals/bootstrap.html"));

const themeBuild = await bundle({ entryPoints: [resolve(repoRoot, "frontend/theme/src/theme-runtime.ts")],
  bundle: true, format: "esm", platform: "node", write: false });
const themeModule = await import(`data:text/javascript;base64,${Buffer.from(themeBuild.outputFiles[0].contents).toString("base64")}`);
await writeFile(resolve(pluginsRoot, "visuals/preview-theme.css"), themeModule.themeCss());

// 设计编辑器（Vue + open-pencil + CanvasKit WASM + 字体）单独打包成一个入口，按内容哈希命名放进不入库的 static/（发布时随插件制品一起打包），
// 标签模块第一次打开时建 iframe 加载 editor.html；Node 内置模块只在 open-pencil 的无头分支里用到，留作外部。
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
const designName = (path) => path.slice(path.lastIndexOf("/") + 1);
// open-pencil 按 Vite 的 new URL("./x.ts", import.meta.url) 写法起 Worker：先单独打包两个 Worker，再把编辑器里的地址换成产物地址。
const openPencilCore = resolve(repoRoot, "node_modules/@open-pencil/core/dist");
const designWorkerBuild = await bundle({ ...designOptions, entryPoints: {
  "export-worker": resolve(openPencilCore, "io/formats/fig/export-worker.js"),
  "fig-worker": resolve(openPencilCore, "kiwi/fig/session/worker.js"),
} });
const designWorkers = Object.fromEntries(Object.entries(designWorkerBuild.metafile.outputs)
  .filter(([, output]) => output.entryPoint)
  .map(([path, output]) => [output.entryPoint.endsWith("export-worker.js") ? "export-worker" : "worker", `/design/assets/${designName(path)}`]));
const designWorkerUrls = {
  name: "design-worker-urls",
  setup(build) {
    build.onLoad({ filter: /@open-pencil[\\/]core[\\/]dist[\\/](io[\\/]formats[\\/]fig[\\/]export|kiwi[\\/]fig[\\/]session[\\/]client)\.js$/ }, async (args) => ({
      loader: "js",
      contents: (await readFile(args.path, "utf8")).replace(/new URL\("\.\/(export-worker|worker)\.ts", import\.meta\.url\)/gu,
        (_, name) => `new URL(${JSON.stringify(designWorkers[name])}, import.meta.url)`),
    }));
  },
};
// Agent 写的 JSX 是可执行代码：renderJSX 改为交给不透明源沙箱求值（editor/jsx.ts），编辑器页面里的 new Function 一律拒绝。
// 依赖换了文件布局时 onLoad 根本不会被调用：构建结束后再核对补丁恰好打上一次。
let designJSXPatched = 0;
const designJSXEval = {
  name: "design-jsx-sandbox",
  setup(build) {
    build.onLoad({ filter: /@open-pencil[\\/]design-jsx[\\/]dist[\\/]chunks[\\/]fields\.js$/ }, async (args) => {
      designJSXPatched += 1;
      const source = await readFile(args.path, "utf8");
      const patched = source
        .replace("const element = createElement(buildComponent(jsxString, helperWarnings), null);\n\tconst tree = resolveToTree(element);",
          "const tree = await __akashicEvaluateJSX(jsxString, helperWarnings);")
        .replace('new Function("React", "__helpers", code)(mini_react_exports, designJSXHelpers(warnings))',
          '(() => { throw new Error("设计 JSX 只在沙箱里求值"); })()');
      if (patched.split("__akashicEvaluateJSX").length !== 2 || patched.includes("new Function"))
        throw new Error("open-pencil 的 JSX 求值代码变了，需要更新 design-jsx-sandbox 补丁");
      const client = JSON.stringify(resolve(repoRoot, "frontend/plugins/design/editor/jsx.ts"));
      return { loader: "js", contents: `import { evaluateJSX as __akashicEvaluateJSX } from ${client};\n${patched}` };
    });
  },
};
// 沙箱页自带脚本（内联），由 dashboard 按文件名给它单独的 CSP：只放行 eval，不连网、不取同源。
const designSandbox = await bundle({ ...designOptions, write: false, metafile: false,
  entryPoints: [resolve(repoRoot, "frontend/plugins/design/editor/jsx-sandbox.ts")] });
const designSandboxScript = designSandbox.outputFiles.find((file) => file.path.endsWith(".js")).text.replaceAll("</script", "<\\/script");
await writeFile(resolve(designAssets, "jsx.html"),
  `<!doctype html><meta charset="utf-8"><title>JSX</title><script type="module">${designSandboxScript}</script>\n`);
// 网页预览的运行页：原样拷贝，dashboard 按文件名给它网页沙箱的 CSP。
await writeFile(resolve(designAssets, "preview.html"), await readFile(resolve(repoRoot, "frontend/plugins/design/editor/preview.html"), "utf8"));
const designBuild = await bundle({ ...designOptions, plugins: [designWorkerUrls, designJSXEval],
  entryPoints: { editor: resolve(repoRoot, "frontend/plugins/design/editor/main.ts"),
    worker: resolve(repoRoot, "frontend/plugins/design/editor/headless.ts") } });
if (designJSXPatched !== 1) throw new Error(`design-jsx-sandbox 补丁命中 ${designJSXPatched} 次，应为 1 次`);
const designOutputs = Object.entries(designBuild.metafile.outputs);
const designEntry = designOutputs.find(([, output]) => output.entryPoint?.endsWith("editor/main.ts"));
const designWorkerEntry = designOutputs.find(([, output]) => output.entryPoint?.endsWith("editor/headless.ts"));
if (!designEntry || !designWorkerEntry) throw new Error("设计编辑器没有产出入口");
// 服务端无头画布的入口页：后端在无头浏览器里打开它执行 Agent 的设计工具，不经过 dashboard 路由。
await writeFile(resolve(designAssets, "worker.html"),
  `<!doctype html><meta charset="utf-8"><title>设计</title><script type="module" src="/design/assets/${designName(designWorkerEntry[0])}"></script>\n`);
// Agent 的设计工具表：名字、描述、参数 schema 取自 open-pencil 的核心工具，与画布里执行的是同一份定义。
// 去掉要联网取图库的 stock_photo，以及在用户页面里执行任意代码的 eval。
const { CORE_TOOLS } = await import("@open-pencil/core");
const { toJsonSchema } = await import("@valibot/to-json-schema");
// render 的描述后面接上 open-pencil 自己的 JSX 写法参考，与它内置 AI 拿到的说明一致。
const { JSX_REFERENCE } = await import("@open-pencil/design-jsx");
const DESIGN_EXCLUDED = new Set(["stock_photo", "eval"]);
await writeFile(resolve(designAssets, "tools.json"), JSON.stringify(CORE_TOOLS.filter((tool) => !DESIGN_EXCLUDED.has(tool.name))
  .map((tool) => ({ name: tool.name, description: tool.name === "render" ? `${tool.description}\n\n${JSX_REFERENCE}` : tool.description,
    parameters: toJsonSchema(tool.input, { errorMode: "ignore" }) }))));
// iframe 拿不到外壳页面的主题变量：与可视化预览一样内联主题 CSS 与品牌变量，按 data-theme 切换明暗。
const designTheme = themeModule.themeCss() + await readFile(resolve(repoRoot, "frontend/theme/src/brand-tokens.css"), "utf8");
const designStyles = designOutputs.filter(([path]) => path.endsWith(".css")).map(([path]) => designName(path));
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

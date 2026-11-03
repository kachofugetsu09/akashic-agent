# Design

让 agent 用**结构化设计工具**产出原型，而不是手写 HTML，再经 `sites` 发到用户右侧栏。

本目录当前只验证**投影链**（scene → 可发布的 HTML）。工具合同、发布和画布 UI 不在这一步。

## 已验证的投影链

```
SceneGraph  ──sceneGraphToDesignDocument──▶  DesignDocument
            ──exportHTMLBundle({html:'standalone'})──▶  { entrypoint: "index.html", files: [...] }
```

`entry.content` 是自包含单文件 HTML（1091 B 起），含 `<style>`、文本与圆角等样式，
正是 `plugins/sites/snapshot.py` 接受的入口形态。

## 为什么投影必须发生在浏览器侧

| 镜像 | 运行期有 Node |
|------|---------------|
| `docker/host-runtime/Dockerfile` | 有，但构建后 `rm -rf node_modules` |
| `docker/host-runtime/Dockerfile.distribution` | **没有** |

`@open-pencil/dom-css` 是 JS 包，发行镜像无法在运行期执行它。
发行镜像装了 Chromium（`playwright install --only-shell chromium`），
因此投影在**浏览器 JS 环境**里跑，Python 只负责喂 scene、收 HTML。
`plugins/visuals/preview.py` 已有 Playwright 浏览器池的现成范式。

## 预构建 bundle

`projection/entry.js` 用 esbuild 打成单个 ESM 文件（实测 484.5 KB）：

```sh
npx esbuild plugins/design/projection/entry.js --bundle --format=esm --platform=browser --minify \
  --alias:tailwindcss=plugins/design/projection/tw-stub.js \
  --alias:node:fs/promises=plugins/design/projection/fs-stub.js \
  --outfile=<构建输出>/project.mjs
```

两处打桩的原因：
- `tailwindcss`：只有 `style:'tailwind'` 才需要编译 Tailwind，本投影用 `style:'inline'`，不需要；
- `node:fs/promises`：来自 dom-css 的 **HTML 导入**路径（`chunks/import/html.js`），导出路径不调用它。

## 验证

```sh
node plugins/design/projection/verify.mjs
```

期望输出：`HTML 长度` 非零、`是完整文档: true`、`含文本: true`、`含圆角: true`。

## 未完成

- `design_*` 工具（`tools.v1` 合同）与 scene 的持久化形态；
- Playwright 投影客户端（`plugins/design/projection.py`）；
- 经 `sites` 的 `akashic.site.v1` 附件发布与右侧栏标签；
- 画布 UI（React + 同一份 scene）。

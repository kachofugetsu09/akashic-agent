# 模型连接 · 交互原型

无依赖单页 showcase，演示「连接 → 探测 → 开放 → 可见 → 会话」整条模型配置交互链。
综合 deepseek-harness 的凭据/目录流程与 magpie 的厂商布局，落到 Akashic 纸张品牌 token 上；
插件组合边界（`models.connection-types.v1`）保持不变：凭据表单与目录应答由来源插件渲染。

## 运行

```sh
cd showcase/model-config
python3 -m http.server 8471
# 打开 http://127.0.0.1:8471/
```

三个视图：连接页（管理面）、对话选择器（composer 胶囊）、设计说明（与 dsh/magpie 的对照表）。
顶栏可切换「首跑态」（整页即引导）与暗色纸面。

数据全部为演示桩，不接真实 API。

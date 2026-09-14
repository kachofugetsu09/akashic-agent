---
name: develop-akashic-plugin
description: 创建、编写、修改并验证 Akashic 插件及插件内 Skill/MCP。用户要求创建插件、加入能力、安装候选或热更新时使用。
---

# 开发 Akashic 插件

只修改插件源码，不编辑安装 cache、runtime pointer 或正式 plugin-data。先确定范围和恢复点。

## 阅读入口

1. 读取目标仓库的 docs/INDEX.md 和 docs/WORKFLOW.md。
2. 读取 [编写参考](references/plugin-authoring.md)，并打开其中指定的当前能力手册与决策。
3. 以目标 checkout 的真实接口为准。旧 self-validation 和 runtime-diagnostics 文档仅作历史调查线索，不能据此恢复已删除的协议。

## 实现

- 唯一入口是 apply(ctx)，配置从 ctx.config 取得并由插件自行解析。
- 功能启用条件写在 apply 内，不导出 is_active。已选插件的 inject 仍是硬依赖；整包禁用由组合选择决定。
- 依赖通过 ServiceKey 组合；资源、监听和任务必须有清楚的 Scope owner。
- 插件负责自己的持久数据。代码回退不表示数据已回退。
- import 不启动进程、开放端口或发送消息；正式接纳和候选隔离服从当前宿主合同。
- 不重新引入第二份运行描述、静态业务自测或固定参数名检查。

## 验证与交付

按当前用户授权及仓库工作流确定是否运行测试、安装或部署；用户禁止时不得执行。
说明实际运行过的检查、未验证边界和恢复点。提交 PR 不表示已完成安装、换代或业务验证。
安装和发布必须走当前管理入口，不手改 cache 或 stable。行为证据应来自实际插件能力和持久回执，
不能只凭 catalog、最终回复文字或健康检查宣称成功。

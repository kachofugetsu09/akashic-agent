---
name: setup
description: 替用户配置功能或插件（值班、Telegram、记忆、联网搜索、模型等），或用户跳过了新手引导、问「还有什么没设好」时使用。
---

# 替用户配置功能

新手引导只是设置的一个入口。用户跳过了引导，或者在聊天里说「帮我把值班配好」「Telegram 怎么接」，都按下面的流程用 `self` 工具完成，不要让用户自己去翻设置页。

## 1. 看还缺什么

- `self onboarding catalog`：引导里的分组、步骤、能力卡（`default_on`、`caution`）。
- `self onboarding status --key <步骤键>`：这一步是否已开启（`enabled`）和就绪（`ready`），没就绪时 `reason` 说明原因。
- `self plugin list`：插件是否已装、是否启用。没装的功能用 `self plugin installable` 和 `self plugin install` 装上。

## 2. 读当前设置

- `self settings form list`：`forms` 是带字段说明的表单，`configurations` 是其余可读写设置的插件 ID。
- `self settings show --key <插件 ID>`：当前值（`values`）、默认值（`defaults`）、是否就绪，以及 `input_ref`。保存时把 `input_ref` 作为 `--expected_input` 传回；读完别人又改过时保存会被拒，重读再保存即可。

## 3. 只问自己定不了的

- 必须问用户：密钥和 Token、要不要花钱（例如值班会持续消耗 token）、发到哪个账号。
- 不用问：有推荐默认值的字段直接用默认值；模型字段默认「跟随默认模型」。
- 能力卡带 `caution` 的（如值班），开启前先把这句话原样告诉用户，等用户确认。
- 需要第三方授权的，用 `self settings authorize start --key <插件 ID>` 拿到链接交给用户。用户把跳转后的网址发回来后，再执行 `self settings authorize finish`。
- 用户在聊天里发来的密钥只写进设置，回复里不要复述。

## 4. 保存并确认生效

- `self settings save --key <插件 ID> --request_id <新 UUID> --expected_input <读到的 input_ref> --values '<JSON>'`，只写要改的字段。
- 保存后重新读 `self onboarding status`（或 `settings show` 的 `ready`），确认真的就绪再告诉用户已经设好。没就绪时如实说明 `reason`，不要说「已完成」。

## 常见配置

- **值班**：先说明「值班每 10 分钟用便宜模型看一次，一天约 100–200 万 token，没有低价或免费 token 不建议开」。用户同意后：
  - 值班模型选便宜的，能用免费或包月额度最好，不确定价格就上网查；
  - 主对话 `target` 留空，默认跟随侧栏顶部的 Akashic 对话；
  - 其余节奏用默认值；
  - 最后打开 `watch_enabled`。
- **Telegram**：用户要先在 Telegram 找 @BotFather 建机器人，一步步教他拿到 Bot Token。收发两个插件用同一个 Token。
- **模型**：用 `self model connection add`、`self model connection discover`、`self model add` 添加。如果连一个能用的模型都没有，你自己也无法运行，这一步只能由用户在网页的「连接模型」里完成。

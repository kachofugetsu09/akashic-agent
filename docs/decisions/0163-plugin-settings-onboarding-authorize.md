# 0163 · 声明式设置进引导，并提供粘贴回调式授权

- 状态：proposed / implemented for review
- 日期：2026-11-20
- 关联：ADR 0130

## 背景

外置插件（steam、fitbit、calendar）的凭据和身份都放在数据目录里的私有文件，新用户只能手改文件、`docker exec` 跑授权脚本。
ADR 0130 让它们可以声明设置字段，但引导里看不到它们，也不覆盖 OAuth 授权。

## 决定

1. `SettingsForm` 可带 `setup=Setup(分组, 标题, 能力卡)`；插件在注入了 `ONBOARDING` 的可选子 Fiber 里调用 `declare_setup`，
   引导出现一步，内嵌的就是这张表单（route `plugin-settings/<插件 ID>`）。
2. `declare_settings`/`declare_setup` 接受插件自己的 `status`：返回的 `ready`/`reason` 覆盖表单默认的"已就绪"，
   必填项和外部状态（是否已授权）由插件说明。
3. Shell 的 route 可带 `/后缀`：找不到完整 route 时按前缀找分节，后缀作为 `subroute` 交给分节解释；深链同理。
4. `Authorize(label, hint, start, finish)`：表单下方一张授权卡。`start` 给授权页地址，用户同意后把浏览器停下的完整网址交给 `finish`。
   路由在贡献插件的运行作用域里执行，`ValueError` 显示给用户。

## 取舍

- **粘贴回调网址，而不是直接回调到 Akashic。** 自托管实例的公网地址各不相同，还要在 Google 控制台登记；
  回调到 localhost 再粘贴回来对任何部署都成立，代价是多一步复制粘贴。
- **授权状态由插件服务自己保存。** 进行中的授权（含 PKCE verifier）只在插件服务进程内存里，进程重启后要重新开始。
- **一个插件一张表单、一个引导分组。** 分组键由插件自选，重复时注册失败，不做合并。

# 0111 删除没有消费者的业务端口

状态：proposed / implemented for review
关联：#1179；PLG、STA

## 决定

删除 `drift.proposals.v2` 的发布、适配器及无人触发的 `drift.changed` 订阅。
删除 `shell.owners.v1` 的发布；Shell 清理仍直接使用同一个 ShellOwners。
删除 Core 中没有消费者的 `feed_fetcher` HTTP profile 和专用连接池。

判断范围是当前主仓库源码、动态 key 字符串和 Fleet 当前 14 个维护仓库的远端源码。
没有实际消费者，不能以未来可能接入为由保留端口。
能力目录另识别 `bindings.bind/open` 的 ServiceKey 参数，避免误删通过 binding 消费的能力。

## 可观察影响与恢复

未受维护的外部代码若请求上述两个 key，将收到缺失依赖；不提供旧 key 别名。
Drift 的既有持久提案仍由 Wake 和 Delivery 读取、选择和结算；不删除 SQLite 行。
离线迁移直接使用 DriftStore，不通过被删除的发布接口。今后新增提案来源应同时提交
真实来源消费者及提供方公开合同，不恢复隐藏于实现模块的 key。

没有数据迁移或数据缩减。恢复点为父提交 8b8e0945；恢复源码后可重新发布旧端口。

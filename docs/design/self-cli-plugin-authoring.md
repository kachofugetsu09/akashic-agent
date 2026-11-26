# 写插件时让助手与用户能做同样的事

用户在界面上能做的每件事，助手经 `self` 也能做；反过来，助手的每个操作都对应一个用户入口。CI 的 `scripts/check_interaction_parity.py` 会检查这一点（ADR 0168），新入口必须在同一个 PR 里归属。

## 门禁认哪些用户入口

从插件源码 AST 采集，不看运行时自报：

| 入口 | ID 形状 | 来自 |
|---|---|---|
| HTTP/WS 路由 | `http:<插件>:<动词> <路径>` | `app.get(...)` / `router.post(...)` 等 |
| 一个路由里按 `type`/`kind` 分出的动作 | `http:<插件>:<动词> <路径>#<值>` | 处理函数里对字面量的比较或判别联合 |
| 聊天 WS 帧 | `ws:<插件>/<帧类型>` | `frame_type == "..."` |
| Plugin UI 查询 | `rpc:<插件>/<方法>` | UI 插件里 `method == "..."` |
| Slash 命令 | `slash:<名>`；声明了子动作时 `slash:<名>#<子动作>` | `CommandDefinition(name=..., subcommands=(...))` |
| Gateway RPC | `gateway:<方法>` | `RpcMethod.key("...")` |

## 默认做法：镜像为 operation

界面入口与 operation 调**同一段领域函数**，不准 HTTP 自调用，也不准复制业务逻辑。

Plugin UI 查询最省事：把同一个 `query` 函数交给 `QueryMirror`。

```python
MIRRORS = (
    QueryMirror("note.create", ("note", "create"), "新建一条笔记",
                (OptionSpec("title", "标题", required=True),)),
)

async def register(child: Context) -> None:
    await register_query_mirrors(child, child.require(OPERATIONS_V1), query=query, mirrors=MIRRORS)

_ = await ctx.inject((OPERATIONS_V1,), register, name="operations")  # 软挂载：没有 operations 也能运行
```

HTTP 路由、WS 帧或 Slash 命令要写 `OperationSpec`，用 `mirrors=` 指明它对应的入口：

```python
OperationSpec(
    operation_id="demo.note.create", path=("note", "create"), summary="新建一条笔记",
    handler=ctx.entrypoint(create_note),          # 与路由共用；entrypoint 让它在本插件的作用域里运行
    mirrors=("http:demo:POST /api/demo/notes",),
    options=(OptionSpec("title", "标题", required=True),),
)
```

`operation_id` 和 `mirrors` 都要写字面量，门禁才能读到。一个入口只能被一个 operation 镜像；一个 operation 可以镜像多个入口，例如同一读数在 HTTP 和 RPC 各有一个入口。

## 依赖别的插件的设置：扩展参数

如果你的插件给另一个插件的创建动作附加设置（例如 akasha 给新项目附加记忆策略），就用 `ExtensionSpec` 挂到对方声明的扩展点上，选项会以 `--<命名空间>.<选项>` 出现在那条命令里。需要原子性时，`stage` 返回 `OwnerWrite`，由主操作在 `OwnerState.transact_joint` 里与主记录一起提交。界面侧在 `scope.fields` 里用 `fields.contribute(namespace, read)` 只登记取值，不要自己先写。

## 只有四类可以豁免

写进 `scripts/interaction_parity_manifest.json`（Core）或 `interaction_parity.extra.json`（extra）的 `explicit_exemption`，并写明理由：

- `transport`：心跳、订阅流、视频或信令、静态资产、设备通道；
- `view_only`：页面壳，或只改一个客户端视图的状态；
- `user_only`：代表人本人的动作，或既有决定禁止模型去做（如 ADR 0122 交互终端）；更好的做法是照样镜像，并把 `audience` 设成 `user_only`，这样终端 `akashic self` 仍能用；
- `auth_security`：口令、登录票据、OAuth 回填。

不允许通配，也没有欠账清单。

## 本地自查

```bash
python scripts/check_interaction_parity.py                 # 与 CI 相同
python scripts/check_interaction_parity.py scaffold        # 为每个未归属入口打印可粘贴的修法
python scripts/check_interaction_parity.py runtime --workspace <W>   # 对账运行中实例：运行期注册了、静态采集看不到的 operation 判失败
akashic self help <组>                                     # 以用户身份试调自己的命令
```

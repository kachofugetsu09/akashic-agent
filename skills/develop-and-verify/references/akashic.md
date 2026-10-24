# Akashic 自身的开发实例

在 `develop-and-verify` 的流程里，Akashic 特有的几步。

## 起实例

`scripts/start.py` 只构建**已提交**的 HEAD：改完先 commit（迭代用 `git commit --amend`），再重启实例。

```sh
cd "$DEV/src"
python3 scripts/start.py --state "$DEV/state" --port <端口> --no-browser
```

- 用 shell 的后台运行托管它。日志出现 `WebUI 已就绪` 才算起好；首次要构建界面，几分钟正常。
- `--state` 必须是本任务的新目录，不要指向 `~/.akashic` 或生产数据目录。同一个 `--state` 已在运行会直接报错。
- 构建产物在 `$DEV/src/.akashic-start/<commit>/`：其中 `env/bin/python` 和 `core/main.py` 是这个实例自己的解释器和入口，下面的命令都用它们（记作 `$PY`、`$MAIN`）。

## 接模型

新实例没有模型。接上本 skill 的假模型：

```sh
python3 <skill>/scripts/mock_model.py --port 44099 --steps 6      # 用 run_in_background 托管，不要加 &
python3 <skill>/scripts/use_mock_model.py --state "$DEV/state" --url http://localhost:<端口>
```

自己调 dashboard 接口时注意：`/api/dashboard/<插件>/...` 要带 bootstrap 给出的模块身份请求头
（`x-akashic-web-snapshot/-catalog/-module/-generation`），否则一律 404，不是插件没加载。
`use_mock_model.py` 就是现成写法。接口鉴权用 `Authorization: Bearer $(cat "$DEV/state/workspace/web-auth/secret")`。

## 登录

不需要初始密码：

```sh
AKASHIC_PLUGIN_HOME="$DEV/state/plugin-home" $PY $MAIN web-login --url http://localhost:<端口> \
  --config "$DEV/state/config.toml" --workspace "$DEV/state/workspace"
```

在 computer 里打开打印出的 `/_auth/login#t=...`。链接只能用一次，用过再生成。
全新实例会先要求设密码：填一个至少 10 位的临时密码（记在 `$DEV/created.txt`），保存进入。

## 验收要点

- 改聊天界面：真实发一条消息，等回复完成再看；多步过程用 `--steps`。
- 改插件面板：从侧栏真实打开，手机宽度下从抽屉进入再试一次。
- 截图推到 `claude/ux-shots` 分支的单独目录，PR 里用 raw 链接引用。

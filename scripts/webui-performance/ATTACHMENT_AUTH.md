# 文本附件鉴权验收

复用[附件基准](ATTACHMENTS.md)的构建与隔离规则，但必须新建带真实web-auth的state，不复用性能实例。账号和凭据只属于本次隔离状态，不读正式凭据、不注入cookie、不伪造登录或响应。

```sh
.venv/bin/python scripts/webui-performance/serve-attachment-runtime.py --state /tmp/attachment-auth-1 --port 24720 --auth
```

另一个终端，在同一worktree指定两份独立制品和浏览器路径：

```sh
BASELINE_CHAT=/path/to/baseline/chat
CANDIDATE_CHAT=/path/to/candidate/chat
BROWSER=/path/to/chromium
node scripts/webui-performance/check-attachment-auth.mjs --state /tmp/attachment-auth-1 --baseline "$BASELINE_CHAT" --candidate "$CANDIDATE_CHAT" --browser "$BROWSER" --output /tmp/attachment-auth-report-1
```

登录与首次采样前同时冻结两侧完整文件清单；每次安装、HTTP字节和最终恢复都对照这份清单，不能在后续轮次接受源变化。两侧都须通过：无凭据GET返回401→真实登录表单（首次通过真实表单改密）→打开并逐行核对正文、精确核对描述段落总行数、关闭→真实POST `/_auth/logout`→原页面热缓存再次点击返回401且不显示正文→新tab真实登录→原页面重新打开恢复全文。每次核对原页面timeOrigin未变，不能刷新来掩盖热缓存行为。

只读取新fixture中0600的initial-password；路径、PID、监听端口和marker必须一致。新密码只留在进程内存；报告不写密码、cookie或认证请求体，错误脱敏。认证owner按正常协议更新本次账号与session，消息和附件事实仍须完全不变。

关闭浏览器失败也继续尝试恢复baseline和核对完整性，错误分别保留。结束后在服务终端Ctrl-C正常停机，保留全部state、日志、报告。失败后使用全新state重跑，不重置既有账号；此脚本不提供重复使用的测试账号。鉴权、性能、正确性三份报告都完成才覆盖全部附件合同。

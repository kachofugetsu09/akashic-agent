# Computer 真实浏览器验收

需要 Docker、仓库既有 Python/Node 依赖与独立 Chromium 可执行文件。默认使用
`plugins/computer/plugin.py` 声明的发布镜像，不挂载驱动源码；没有镜像时先 `docker pull`。

```bash
COMPUTER_E2E_CHROMIUM=/path/to/chrome node scripts/computer-e2e/run.mjs
node scripts/computer-e2e/idle.mjs
```

开发尚未发布的驱动时，明确设置 `COMPUTER_E2E_SOURCE=1`；`COMPUTER_E2E_IMAGE`
可选择已下载的基础镜像。结果记录镜像和是否挂载源码，不能把开发挂载当作发布镜像验收。

脚本创建独立 Docker 容器、临时数据目录、loopback 随机端口和独立用户浏览器 Context。
实际生产 Computer/Conversation/Shell UI、Dashboard adapter、gateway、driver、Chromium 与
Selkies 参与验收。主桌面观看检查真实 H.264 帧；匿名输入检查页面自己的 DOM 与粘贴 ACK。
错误恢复、接管断连、OpenCLI 暂停、资源关闭与 320px 浏览器交互各有可观察回执。

`idle.mjs` 把同一真实闲置时钟缩短到 1.2 秒，验证操作、接管、观看和 workload 回收。
它没有替换进程或伪造成功，不声称实际等待了默认的十分钟。

结束会移除本次容器、关闭浏览器和 fixture 服务；日志、截图、结果与临时 profile 留在输出的
`/tmp/akashic-computer-*` 目录，便于诊断。失败同样保存截图和容器日志。

fixture 只提供 UI 挂载目录、主题接口、聊天占位和设置正文，不启动完整 Core，也不读写正式
workspace。该验收不代表生产热更新、generation 切换或真实 Android WebView 已验证。
开发容器允许 Chromium 创建沙箱所需的系统调用；正式 workload 的 seccomp 接入另行验收。

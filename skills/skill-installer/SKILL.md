---
name: skill-installer
description: 从 GitHub 仓库、子树链接或本地目录安装、更新、卸载 Akashic 技能（SKILL.md），无需编写插件即可热生效。当用户要求"安装这个 skill"、"给我装个技能"、"从 GitHub 安装技能"、"卸载某个 skill"、"怎么安装 skill"时使用。触发词：安装技能, 安装 skill, install skill, 装个 skill, 更新 skill, 删除 skill, 卸载 skill, skill 目录。
metadata: {"akashic": {"always": false, "requires": {"bins": ["git"]}}}
---

# 安装、更新与卸载 Akashic 技能（Skill）

Akashic 的 `skills` 插件按四层来源热扫描技能，**普通技能不需要编写或安装插件**，把含 `SKILL.md` 的目录放入本地技能目录即可立即被 `load_skill` 热读取。

## 1. 四层来源与优先级（高 → 低）

| 优先级 | 来源 (`source`) | 目录路径 | 适用场景 |
| --- | --- | --- | --- |
| 1（最高） | `workspace` | `$AKASHIC_WORKSPACE/skills/<skill-name>/SKILL.md` | **默认安装位置**。随工作区持久化，容器/服务重启后不丢失 |
| 2 | `user` | `~/.akashic/skills/<skill-name>/SKILL.md` | 宿主用户级共享技能（注意容器部署时需确认 `HOME` 是否挂载） |
| 3 | `builtin` | `<Core 根>/skills/<skill-name>/SKILL.md` | 随 Akashic Core 仓库发布的内置通用技能 |
| 4（最低） | `plugin` | 插件 `INSTALLED_ASSETS.register` 声明的 `skills/` | 与特定插件工具强绑定的随包技能（需走 `plugin-system` 安装） |

- **同名覆盖**：高层同名技能静默覆盖低层；目录名 `<skill-name>` 即技能的唯一标识（对应 `load_skill(skill="<skill-name>")`）。
- **热加载机制**：`SkillCatalogParser` 按目录项与 `SKILL.md` 的 `(mtime, size)` 自动失效缓存。复制或修改文件后，**无需重启服务或重装插件**，当轮即可直接调用 `load_skill(skill="<skill-name>")` 验证。

## 2. 确定工作区技能目录

优先使用环境变量 `$AKASHIC_WORKSPACE/skills`；若当前 Shell 未注入该变量，从提示词的「工作区 → 根目录」或运行进程参数 `--workspace` 取得工作区路径（例如 `/srv/data/services/akashic/state/workspace/skills`）：

```sh
SKILL_ROOT="${AKASHIC_WORKSPACE:-/srv/data/services/akashic/state/workspace}/skills"
mkdir -p "$SKILL_ROOT"
```

## 3. 从 GitHub 仓库或本地路径安装

外部技能仓库通常有两种目录形态（兼容 Claude Code / Codex / OpenClaw 等常见 SKILL.md 仓库）：
- **单技能仓库**：仓库根目录下直接是 `SKILL.md`
- **多技能集合仓库**：`skills/<skill-name>/SKILL.md` 或 `<skill-name>/SKILL.md`（例如 `https://github.com/<owner>/<repo>/tree/<branch>/skills/<skill-name>`）

### 标准安装步骤

1. **克隆到隔离临时目录**（不要直接 clone 到 `skills/` 根目录，避免残留 `.git` 或嵌套层级错误）：
   ```sh
   TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/akashic-skill.XXXXXX")"
   git clone --depth 1 https://github.com/<owner>/<repo>.git "$TMP_DIR/repo"
   ```
2. **定位 `SKILL.md` 所在的技能目录并检查内容**：
   - 找到目标 `<skill-dir>/SKILL.md`，确认 frontmatter 至少包含 `name` 和 `description`。
   - 若外部 `SKILL.md` 带有 `metadata` 字段，注意 Akashic 解析规则：当 `metadata` 为字符串时必须能被 `json.loads` 解析为 JSON 对象（或直接写成 YAML 映射对象）；若原文件使用了非 JSON 的自由文本 `metadata`，安装后需用 `edit_file` 修正或移除该字段，否则解析会报错。
   - 检查 `metadata.akashic.requires`（或 `metadata.requires`）中的 `bins` / `env` 在本机是否满足。
3. **复制到 `$SKILL_ROOT/<skill-name>` 并清理临时目录**：
   ```sh
   rm -rf "$SKILL_ROOT/<skill-name>"
   cp -R "$TMP_DIR/repo/skills/<skill-name>" "$SKILL_ROOT/<skill-name>"
   rm -rf "$TMP_DIR/.git" "$SKILL_ROOT/<skill-name>/.git" "$TMP_DIR"
   ```
   > 注意：若本机启用了 `shell_restore`（`rm` 被改写为 `mv`），优先把临时目录建在 `~/.cache/akashic-dev/skill-install/` 并按目录清理，避免把无关工作目录移入回收站。

## 4. 立即验证（必做）

文件落盘后，**必须在当轮直接调用 `load_skill` 工具**核对：

```json
{"skill": "<skill-name>"}
```

- 返回 `status="success"` 且包含 `base_directory` 与 `instructions` 正文 → 安装成功且当前可用。
- 返回 `技能不可用：...；缺少依赖：CLI: xxx / ENV: xxx` → 如实告知用户缺少哪个二进制或环境变量，或按授权安装依赖后再验。
- 返回 `技能不存在` → 检查目录层级是否为 `$SKILL_ROOT/<skill-name>/SKILL.md`（不能多套一层父目录）。

## 5. 更新与卸载

- **更新**：重新拉取上游最新仓库，覆盖 `$SKILL_ROOT/<skill-name>/` 后再次调用 `load_skill(skill="<skill-name>")` 核对新正文。
- **卸载**：删除 `$SKILL_ROOT/<skill-name>/` 目录即可；若该名称在 `builtin` 或 `plugin` 层存在同名技能，删除工作区覆盖后会回退到低层版本。

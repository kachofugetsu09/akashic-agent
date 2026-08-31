---
name: opencli
description: Use OpenCLI from the ordinary shell for site adapters, structured web data, and logged-in browser sessions.
---

# OpenCLI

OpenCLI is a command-line program, not the Browser Use tool. Run it with the ordinary `shell` tool; the Computer plugin connects OpenCLI's standard local port to its persistent browser. Use `browser_observe` and `browser_action` when the task needs direct page interaction.

## Rules

1. Run `opencli` through `shell`; never send OpenCLI arguments to a Browser tool.
2. Prefer a site adapter. Run `opencli list -f json` and the leaf command's `--help -f json` before guessing commands or flags.
3. Request `-f json` for structured adapter output. `PUBLIC` and `LOCAL` adapters need no browser; `COOKIE`, `INTERCEPT`, and `UI` adapters need the Browser Bridge.
4. Browser-backed site adapters use `--window background --site-session persistent --keep-tab true` so one logged-in session can be reused.
5. Before browser-dependent work, run `opencli doctor`. Daemon, Extension, and Connectivity must all be ready.
6. Preserve the exact shell exit code, stdout, stderr, and error. Do not silently fall back to raw HTTP or another browser profile.
7. Never start another browser, close the persistent Chromium process, delete its profile, or remove `Singleton*` files. The Computer plugin owns that lifecycle.

## Discover and diagnose

```text
shell({"command":"opencli list -f json","description":"列出 OpenCLI 能力"})
shell({"command":"opencli bilibili history --help -f json","description":"查看站点命令帮助"})
shell({"command":"opencli doctor","description":"检查 OpenCLI 连接"})
shell({"command":"opencli auth status","description":"检查登录状态"})
```

## Site adapters

Put persistent browser options after the leaf command:

```text
shell({"command":"opencli bilibili history -f json --window background --site-session persistent --keep-tab true","description":"读取哔哩哔哩历史"})
shell({"command":"opencli github whoami -f json --window background --site-session persistent --keep-tab true","description":"读取 GitHub 身份"})
```

The Computer plugin refreshes known login sessions every 12 hours and retries a failed refresh after 15 minutes. If a site has logged out, ask the user to log in once through the visible Computer panel.

## Choosing the right capability

```text
Site adapter or structured data → OpenCLI through shell
Direct page navigation or element interaction → browser_observe / browser_action
Login dialog or whole-desktop fallback → computer_observe / computer_action
```

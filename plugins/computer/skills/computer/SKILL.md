---
name: computer
description: Use the persistent Computer plugin for browser commands, visual browser work, login screens, or desktop interaction.
---

# Computer

Use `browser` first for web work. It runs OpenCLI inside one persistent Chromium profile, so cookies and login state survive plugin restarts.

## Rules

1. Pass the argument array that normally follows the `opencli` executable to `browser`.
2. Prefer a site adapter. Run `list -f json` and the leaf command's `--help -f json` instead of guessing commands or flags.
3. Request `-f json` for structured adapter output. `PUBLIC` and `LOCAL` adapters need no browser; `COOKIE`, `INTERCEPT`, and `UI` adapters need the Browser Bridge.
4. Every browser-backed site adapter uses `--window background --site-session persistent --keep-tab true`. This keeps one reusable logged-in site session.
5. Before browser-dependent work, run `doctor`. Daemon, Extension, and Connectivity must all be ready.
6. For an ad-hoc page, use one stable browser session name. Observe with `state` or `find` before acting, and observe again after navigation or layout changes.
7. Prefer a numeric ref from `state` or `find`. If a ref is stale or missing, get a new state instead of guessing a selector.
8. Use `computer_observe` and `computer_action` only for login screens, native dialogs, CAPTCHA, or visual-coordinate fallback. Screenshots are 1280 by 800.
9. Never start another browser, close the persistent Chromium process, delete its profile, or remove `Singleton*` files. The plugin owns those lifecycle details.

## Discover and diagnose

```text
browser({"args":["list","-f","json"]})
browser({"args":["bilibili","history","--help","-f","json"]})
browser({"args":["doctor"]})
browser({"args":["auth","status"]})
```

If a browser-backed command fails, preserve the exact error. Do not silently fall back to raw HTTP or a second profile. `PUBLIC` and `LOCAL` commands may still work when the Browser Bridge is unavailable.

## Site adapters

For a browser-backed site command, put the persistent options after the leaf command:

```text
browser({"args":["bilibili","history","-f","json","--window","background","--site-session","persistent","--keep-tab","true"]})
browser({"args":["github","whoami","-f","json","--window","background","--site-session","persistent","--keep-tab","true"]})
```

The plugin refreshes known login sessions every 12 hours and retries a failed refresh after 15 minutes. If a site has logged out, the user logs in once through the visible Computer panel; later calls reuse that profile.

## Ad-hoc browser work

The session name comes immediately after `browser`. Keep it stable across the flow:

```text
browser({"args":["browser","work","--window","background","open","https://example.com"]})
browser({"args":["browser","work","state"]})
browser({"args":["browser","work","find","--role","button","--name","Submit"]})
browser({"args":["browser","work","click","3"]})
browser({"args":["browser","work","wait","text","Done","--timeout","15000"]})
```

Useful read commands are `state`, `find`, `get`, `extract`, `network`, and `screenshot`. Useful write commands are `click`, `fill`, `type`, `select`, `keys`, `scroll`, and `upload`. Prefer semantic `--role` and `--name` lookup before raw CSS. Write actions must identify one target.

Use `network` when the page already fetches structured data. Use `extract` for long-form content. Use screenshots only when visual position matters.

Release an owned one-off session with `browser <session> close` when the task is done. Do not close the persistent site sessions used for login refresh.

## Visual fallback

```text
computer_observe({"observe":"screenshot"})
computer_action({"action":"click","x":420,"y":310})
computer_action({"action":"drag","x":420,"y":310,"to_x":760,"to_y":310})
computer_action({"action":"type","text":"hello"})
computer_action({"action":"key","key":"Enter"})
computer_action({"action":"scroll","amount":-3})
```

Re-observe after every visual action that can change layout. Never enter secrets unless the user explicitly provided them for this task.

Examples:

```text
browser({"args":["hackernews","top","-f","json","--limit","10"]})
browser({"args":["browser","research","extract"]})
```

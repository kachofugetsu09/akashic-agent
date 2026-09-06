# Chromium Browser Troubleshooting
## General guidance
- Use the selected browser family for every diagnostic command: `chrome` for Google Chrome or `edge` for Microsoft Edge.
- If communication with the ChatGPT browser extension ultimately fails, do not attempt to complete the request with AppleScript, shell automation, or another scripting substitute.
- Do not install or repair the native host yourself. If native-host setup appears broken, tell the user to reinstall the Browser plugin from the ChatGPT plugin UI.
- These checks diagnose extension and native-host transport. They do not change Chrome DevTools Protocol behavior.

## Browser extension checks
On the first extension-backed browser task in a session, try a lightweight browser-client call such as listing open tabs. If it fails, wait two seconds and retry that call once. Any non-error response means the extension is working.

If browser-client still cannot communicate with the selected browser, run these commands from the plugin root with the matching family:

```text
scripts/chrome-is-running.js --browser edge --check
scripts/installed-browsers.js --json
scripts/check-extension-installed.js --browser edge --json
scripts/check-native-host-manifest.js --json
```

Use `--browser chrome` for Google Chrome. The filenames remain stable for compatibility; their behavior comes from the generated Chromium diagnostics in `scripts/extension-ids.json`.

### 1. The selected browser is not installed
Keep the first response short and non-technical. Explain that the selected browser is unavailable and ask whether the user wants to use another supported installed browser.

### 2. The selected browser is not running
Ask whether the user wants you to launch the selected browser, and wait for permission before doing so.

### 3. The native-host manifest is missing or invalid
Do not install or repair it yourself. Tell the user to reinstall the Browser plugin from the ChatGPT plugin UI.

### 4. The ChatGPT browser extension is missing or disabled
Tell the user:

`Cannot communicate with the ChatGPT browser extension. Confirm that the extension is installed and enabled in the selected browser.`

Read the selected family's `storeUrl` and `extensionManagementUrl` from `scripts/extension-ids.json`. Ask permission before opening either page. Never invent a store URL when `storeUrl` is `null`; explain that the extension listing is not yet published for that browser.

If the extension is disabled by browser or enterprise policy, report that state without attempting to override the policy.

### 5. The checks pass but communication still fails
Ask permission to open a window for the selected browser profile. If the user agrees, run:

```text
scripts/open-chrome-window.js --browser edge
```

Use `--browser chrome` for Google Chrome. Wait two seconds, then retry browser-client setup once. If it still fails, tell the user to reinstall the Browser plugin from the ChatGPT plugin UI. Never import or run `scripts/installManifest.mjs` yourself.

## Commands
### installed-browsers.js
Reports supported installed browsers:

```text
scripts/installed-browsers.js --json
```

### chrome-is-running.js
Checks whether the selected browser is running. It exits `0` when running, `1` when not running, and `2` for usage or runtime errors.

```text
scripts/chrome-is-running.js --browser chrome --check
scripts/chrome-is-running.js --browser edge --json
```

### open-chrome-window.js
Opens `about:blank` in the profile selected by the extension check. Use it only after the user gives permission. Dry-run output verifies the generated launch command without opening a browser:

```text
scripts/open-chrome-window.js --browser edge --dry-run --json
```

### check-extension-installed.js
Checks every usable profile for any configured extension ID for the selected family. The top-level status and exit code reflect the selected profile: `0` means installed and enabled, `1` means installed but disabled, `2` means not installed, and `3` means a usage or runtime error.

```text
scripts/check-extension-installed.js --browser edge --json
```

Use `CODEX_CHROMIUM_USER_DATA_DIR` to override the profile root or `CODEX_CHROMIUM_PREFERENCES_PATH` to select one profile. The legacy `CODEX_CHROME_*` overrides remain supported for Google Chrome.

### check-native-host-manifest.js
Checks the shared native-host manifest in every configured Chromium browser destination and, on Windows, its shared generated `NativeMessagingHosts` registry root. It also verifies the shared native-host name and every configured extension origin. It exits `0` when every destination is correct, `1` when any destination is missing or incorrect, and `2` for usage or runtime errors.

```text
scripts/check-native-host-manifest.js --json
```

Use `--browser chrome` or `--browser edge` to inspect only one browser destination. Use `CODEX_CHROMIUM_NATIVE_HOST_MANIFEST_PATH` to check an explicit manifest file. The legacy Chrome override remains supported for Google Chrome.

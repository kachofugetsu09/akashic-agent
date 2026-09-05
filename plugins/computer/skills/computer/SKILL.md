---
name: computer
description: Use the container browser and Linux desktop with persistent JavaScript bindings, accessible elements, locators, and screenshots.
---

# Computer

Use the `computer` tool for browser and desktop work. It runs JavaScript inside the existing Computer
container, using its logged-in Chromium profile. `browser`, `agent`, `sky`, and `nodeRepl` are ready.
Bindings persist within this Akashic Session; a timeout, error, reset, or workload restart invalidates them.

```js
var tabs = await browser.tabs.list();
nodeRepl.write(tabs);
var tab = await browser.tabs.get(tabs[0].id);
await tab.ax.write();
```

Read the API when needed: `nodeRepl.write(await browser.documentation())`. Read additional documents
with `agent.documentation.get(name)` using names listed in that guidance. The bundled API is the reference Browser API; optional capabilities must be
listed before use. macOS application AX and optional desktop audio are unavailable here.

1. Inspect the current page before acting. Prefer `tab.ax` element indices or `tab.playwright` locators.
   Derive indices from the latest AX state; derive selectors from observed page structure.
2. Use `fill()` for typed controls such as dates and range sliders, and `selectOption()` for native
   selects. Do not type an ISO date into Chromium's segmented date editor.
3. Batch related actions and their resulting observation in one call. The driver waits after input;
   use page state or locator waits when loading is asynchronous.
4. Use `sky` for native dialogs, browser chrome, and coordinate actions. Coordinates refer to the
   container's 1280 × 800 desktop. It supports click, move, drag, scroll, press_key, type_text and screenshot.
5. New tabs belong to this Turn and close at its end unless marked deliverable or handoff. Preserve
   useful output with `await tab.markDeliverable()`. Never close a human tab without explicitly claiming it.
6. Keep `drag_handle()` inside one call and use `try/finally` with `end()`. The driver releases remaining
   input when the call ends. A cancelled action may already have changed the page; observe before retrying.
7. Do not start another Chromium, alter profile files, or bypass the Workload owner. Use the Computer
   panel for human login and takeover. File chooser paths refer to the container, not the Akashic host.

```js
await tab.playwright.getByRole("textbox", {name: "Name"}).fill("花月");
await tab.playwright.locator('input[type="date"]').fill("2026-09-05");
await tab.playwright.getByRole("combobox").selectOption({label: "China"});
await tab.ax.write();
```

```js
await sky.click({x: 640, y: 400, mouse_button: "right"});
await nodeRepl.emitImage((await sky.get_screenshot())[0].bytes);
```

Images are saved by the existing screenshot owner; inspect the returned path with `read_file`.
It supplies image content to the current multimodal model. Use `read_image_vision` only if `read_file`
reports that the current model cannot accept images.
`sky.get_screenshot()` returns bytes and a data URL, without a separate temporary filepath. AX state is
currently full text; its formatting and compression are not claimed identical to the original WASM.

The four legacy `browser_observe/action` and `computer_observe/action` tools remain compatibility entrypoints.
Prefer `computer` for new work. OpenCLI remains a separate ordinary shell command and uses the same browser.

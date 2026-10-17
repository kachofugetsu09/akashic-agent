---
name: computer
description: Read or operate web pages in the user's logged-in browser when a task needs interaction, login state, visual inspection, or content that web search/fetch cannot obtain.
---

# Computer

Prefer `web_search` / `web_fetch` for information that can be retrieved without operating UI.
Use `computer` when the task needs browser interaction, an existing login or tab,
visual inspection, or when search/fetch fails or returns insufficient content. If UI interaction is
clearly required, start here without a redundant search/fetch attempt. Prefer a suitable dedicated
skill, connector, API, or CLI when available. Respect an explicit user request to use the browser.

Use the `computer` tool for browser work. It runs JavaScript inside the existing Computer
container, using its logged-in Chromium profile. `browser`, `agent`, and `nodeRepl` are ready.
There is no desktop: only one Chromium, whose tabs the user watches in the Browser panel.
If the tool is not visible, call `load_tools` with `{"plugin":"computer"}`. Its exact tool name is
`computer`.

Bindings persist within this Akashic Session while Computer is awake; a timeout, cancellation, reset,
idle sleep, or workload restart invalidates them. First use wakes the saved personal profile.
An active Turn prevents sleep. After the Turn and identity refresh finish, 10 minutes without
actual input stops the browser. A panel left open does not keep it alive.
After a script error, bindings and pages remain if input release was confirmed. Read the error and
API, inspect the current page, and reuse the existing browser. Acquire fresh bindings only when the
receipt says they were reset or the browser was closed.
Earlier actions may have taken effect: inspect the current page before retrying an action.
Correct an invalid method and continue; do not treat a script error as task completion.

Keep each call focused: observe, choose a small action, then check its visible effect. Once one
reliable UI signal confirms the requested result, finish. A second view is useful only when the
first is unclear or contradicted. Do not keep inspecting a completed task, look up its implementation,
or repeat it to improve a displayed score. A highlighted row alone does not prove its checkbox or
icon changed; inspect the specific control's selected/checked state.

Before the first browser action, read the installed API in a separate call:

```js
nodeRepl.write(await browser.documentation());
```

Use only methods shown by that API. Create a tab with `await browser.tabs.new()`;
`browser.tabs.create()` does not exist. For example:

```js
var tab = await browser.tabs.new();
await tab.goto("about:blank");
await tab.ax.write();
```

To inspect an existing tab:

```js
var tabs = await browser.tabs.list();
nodeRepl.write(tabs);
var tab = await browser.tabs.get(tabs[0].id);
await tab.ax.write();
```

Read additional documents
with `agent.documentation.get(name)` using names listed in that guidance. The bundled API is the reference Browser API; optional capabilities must be
listed before use. macOS application AX and desktop audio are unavailable here.

This is a bounded Browser API, not the full Playwright library. Locators do not have `inputValue()`
or `boundingBox()`. Read a visible input's current value or an element's rectangle through the
documented read-only `evaluate`, for example:

```js
var field = await tab.playwright.getByRole("textbox", {name: "Name"}).evaluate(el => ({
  value: el.value, readOnly: el.readOnly, rect: el.getBoundingClientRect()
}));
nodeRepl.write(field);
```

Keep page evaluation to short DOM reads. Do calculations and timing loops in the outer REPL;
the read-only page scope does not expose every browser global. Use `Number(text)` for numeric
conversion. Do not discover APIs by reflection, source reads or guessing many method/key names.
Use `var` for bindings reused across calls; do not redeclare top-level `const` or `let` names.
Output only what is needed with `nodeRepl.write` or `nodeRepl.emitImage`. Assignments and returned
values do not print automatically. Keep screenshot bytes in a variable and emit the image;
never write a byte array as JSON. Methods such as `tab.ax.write()` already emit their output.

1. Inspect the current page before acting. Prefer `tab.ax` element indices or `tab.playwright` locators.
   Derive indices from the latest AX state; derive selectors from observed page structure.
2. Use `fill()` for typed controls such as dates and range sliders, and `selectOption()` for native
   selects. Do not type an ISO date into Chromium's segmented date editor.
   If an input is read-only, open its picker and use the visible controls instead of retrying fill.
3. Verify the resulting UI state before claiming success; partial content does not prove completeness.
   Batch related actions and their resulting observation in one call. The driver waits after input;
   use page state or locator waits when loading is asynchronous.
   If the UI confirms there are no matching items to change, complete the remaining requested
   confirmation or submission; do not invent a target or search for hidden state.
4. Use `tab.cua` for coordinate actions inside the page viewport: click, double_click, move, drag,
   scroll, keypress and type. There is no browser chrome or desktop to click.
5. New tabs belong to this Turn and close at its end unless marked deliverable or handoff. Preserve
   useful output with `await tab.markDeliverable()`. Never close a human tab without explicitly claiming it.
6. The driver releases remaining input when the call ends. A cancelled action may already have changed the page; observe before retrying.
7. Do not launch Chromium yourself, alter profile files, or bypass the Workload owner. Use the Computer
   panel for human login and takeover. File chooser paths refer to the container, not the Akashic host.

```js
await tab.playwright.getByRole("textbox", {name: "Name"}).fill("花月");
await tab.playwright.locator('input[type="date"]').fill("2026-09-05");
await tab.playwright.getByRole("combobox").selectOption({label: "China"});
await tab.ax.write();
```

```js
await tab.cua.click({x: 640, y: 400, button: 3});
await nodeRepl.emitImage(await tab.screenshot({}));
```

Computer saves a screenshot path and also attaches the image to the tool result. If the image pixels
are available to the current model, inspect them directly; a second `read_file` is unnecessary.
`read_file` remains the general entry point for saved images. If the tool result says the current model
does not receive images, use `read_image_vision` with that path and a specific visual question. It uses
the configured vision model and returns text. If that role is unavailable, report the missing capability;
do not keep taking screenshots or claim to have seen pixels. Never infer visual access from a path alone.
AX state is currently full text; its formatting and compression are not claimed identical to the original WASM.
The returned screenshot path belongs to Akashic, not the Computer container's filesystem.
Do not open it with `require('fs')` inside Computer. For a small visual
target, emit a clipped screenshot with `tab.screenshot({clip:{x,y,width,height}})` and inspect it
before probing image libraries. Coordinates are relative to the tab viewport. Refresh geometry after
scrolling and bring a target into view before a coordinate action.
`tab.screenshot` uses CSS coordinates and returns CSS-pixel images.

OpenCLI remains a separate ordinary shell command and uses the same browser.

For anonymous E2E, use the managed extension `await agent.browsers.create()`. It returns the same
Browser API with an empty, isolated BrowserContext inside the same Chromium. It never sees the
personal profile's cookies, storage, login state, or OpenCLI extension; the user can watch its tabs too. `browser` still refers to the main browser.
At most eight anonymous browsers can be open. Close them with `await testBrowser.close()`;
ending the Turn, an interrupted call, Session reset, or sleep also closes them. Ordinary script errors
keep them when cleanup was confirmed. Another Turn cannot use them. Each instance has a stable
`browserId`: `await agent.browsers.get(testBrowser.browserId)` returns that same browser; use
`await agent.browsers.list()` to find live instances before creating another. Reuse the browser and tab
across tool calls instead of restarting the E2E for every step. An instance idle for 10 minutes is
released; running operations and explicit human takeover prevent idle release. Watching does not.
A closed or expired ID fails explicitly and never selects the main browser.
Closing a Context discards its cookies and storage. A browser restart closes all anonymous Contexts;
acquire fresh bindings. Saved screenshots keep their normal owner.

```js
var testBrowser = await agent.browsers.create();
var testTab = await testBrowser.tabs.new();
await testTab.goto("file:///data/pages/demo/index.html");
await testTab.ax.write();
await testBrowser.close();
```

## Local pages

The browser runs in its own container. `localhost` there is that container, not the machine where
your shell commands run, and the host firewall usually drops connections from the container to host
ports. A server you start on the host is therefore normally not reachable, even through
`host.docker.internal`; `goto` fails within a few seconds and says so.

To show or check a page you generated, write it (with its relative assets) into the shared pages
directory named in the `computer` tool description, then open it with a `file://` URL:

```js
var tab = await browser.tabs.new();
await tab.goto("file:///data/pages/demo/index.html");
await tab.ax.write();
```

Create the directory first if it does not exist. Use a subdirectory per page and remove it when it is
no longer needed. A dev server that needs a live port works only if the user allows that port; ask
instead of retrying.

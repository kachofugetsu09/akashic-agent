# Accessibility
Use the accessibility API (`ax` property on `Tab` objects) as the primary way to inspect and interact with webpages. Prefer accessibility state and actions targeting accessibility indices over inspecting the DOM, or actions using locators or coordinates.

## Workflow
Start by getting the state for the tab you want to use:

```js
await tab.ax.write();
```

After performing one or more UI actions, call `tab.ax.write()` before deciding what to do next. This keeps you in the current UI state and forces you to re-derive fresh `element_index` values from the latest accessibility text instead of reusing stale ones.

For token efficiency, when appropriate, the accessibility tree will be returned as a diff from the previous accessibility tree, listing only the elements that were removed, added, or changed. Prefer this default diff output; use `tab.ax.write("state", { disableDiffing: true })` only when you need a fresh full accessibility tree.

Batch as many actions as possible and the resulting `tab.ax.write()` into one Node REPL `js` call.

Calling `tab.ax.write()` automatically emits the latest AX state into the tool result; you do not need to call `nodeRepl.write(...)`. Use `tab.ax.get()` if you need to manipulate the state in JavaScript without emitting it.

If `tab.ax.write()` or `tab.ax.get()` reports no accessibility-tree change, do not immediately repeat it without an intervening action. Use `tab.ax.write("screenshot")`, `tab.ax.write("both")`, or `tab.ax.write("state", { disableDiffing: true })` only when you can identify missing context that representation should provide.

Prefer a directly relevant result already visible in the current state over opening a broader intermediate page such as “Show All.”

Once the requested task is complete, stop exploring and immediately respond or move on to the next task.

Perform one or more actions, and then fetch the latest state:

```js
await tab.ax.click(42);
await tab.ax.setValue(42, "openai.com");
await tab.ax.pressKey("Return");
await tab.ax.typeText("hello");
await tab.ax.scroll(42, "down", 1);
await tab.ax.scroll([640, 480], "down", 1);
await tab.ax.selectText(42, "hello");
await tab.ax.performSecondaryAction(42, "Expand");
await tab.ax.write();
```

* Prefer element index based actions over coordinate actions whenever an accessibility element is available. If AX actions are not available or not working, fall back to using screenshots and coordinate actions.
* `tab.ax.scroll()` accepts either an `element_index` or an `[x, y]` point. When working from a screenshot, use a point over the scrollable area you want to move.
* If the UI is not behaving as expected, call `tab.ax.write()` to make sure you have the latest context.
* The AX text of a tab always includes the title & url, so you don't need to call `tab.url()` and `tab.title()` separately.
* Prefer using accessibility text over screenshots for efficiency. Use `await tab.ax.write("screenshot")` when only visual context is needed, or `await tab.ax.write("both")` when you need fresh accessibility text and visual context together.
* `tab.ax.performSecondaryAction()` invokes a secondary action exposed for that element in the accessibility text. Do not guess action names.
* `tab.ax.selectText()` selects matching text in an editable element. Use `prefix` and `suffix` to disambiguate repeated matches, and `selectionType` to choose whether to select the text itself or place the cursor before or after it.
* `tab.ax.pressKey()` presses a key or key combination, including modifier and navigation keys. It supports xdotool-style key syntax. Examples: `"a"`, `"Return"`, `"Tab"`, `"super+c"`, `"Up"`, and `"KP_0"` for numpad `0`.
* It is usually not necessary to pause or delay between performing an action and getting the updated page state. The runtime automatically waits an appropriate amount of time before capturing the new state.

## Using other APIs
The accessibility API is the most efficient way to:

* Complete short tasks
* Complete tasks which lack repetition, even if it is longer

The other APIs are available in case:

* The accessibility API is not working or does not support the capability
* The specific task can be completed more efficiently with another API

For example, for certain tasks you can build locators with Playwright to batch more actions into a single call:

* Long and repetitive tasks, where element indices do not stay stable
* Testing sites you're developing, where you know the structure of the website

Playwright locators are more verbose to generate than the accessibility API, so ensure there are opportunities to reduce several calls to `tab.ax.write()` before using it.

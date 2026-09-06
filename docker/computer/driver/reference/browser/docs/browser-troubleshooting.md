# Browser Interaction Troubleshooting
- Do not inspect browser-use source code or switch to an unrelated control mechanism before using the selected browser's documented API.
- A stale or missing tab, an empty `browser.tabs.list()` result, or an unavailable Playwright injected helper is not evidence that the selected browser disconnected. Empty tab lists are normal after tab cleanup. Keep the existing browser binding, obtain or create a fresh tab in that browser, and use its documented non-Playwright alternatives. Do not reselect the browser or reread its documentation for these errors.
- If an error explicitly reports that the selected browser disconnected, obtain a fresh browser and fresh tabs, then read that fresh browser's complete documentation.
- If a documented API is unavailable on the selected browser, use the alternatives that its effective API and capabilities expose rather than guessing hidden methods.

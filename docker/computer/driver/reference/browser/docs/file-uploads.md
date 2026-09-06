# File Uploads
Handle file inputs and uploads through the file chooser flow:

```js
const chooserPromise = tab.playwright.waitForEvent("filechooser", { timeoutMs: 10000 });
await tab.playwright.locator('input[type="file"]').click();
const chooser = await chooserPromise;
await chooser.setFiles(["/absolute/path/to/file.txt"]);
```

- Start `waitForEvent("filechooser")` before clicking the file input or its associated upload control.
- Prefer the actual `input[type="file"]` when available. Click a visible button or label only when it opens the chooser.
- Use absolute paths for `setFiles(...)`.
- Use `chooser.isMultiple()` before passing multiple files when needed.
- Do not look for `locator.setInputFiles(...)`; uploads are exposed through the chooser object.
- Try the file chooser flow before falling back to a native picker.
- If an upload fails, use any browser-specific upload troubleshooting listed in the selected browser's documentation catalog.

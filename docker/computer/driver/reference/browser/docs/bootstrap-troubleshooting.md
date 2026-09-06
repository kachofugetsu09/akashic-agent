# Browser Runtime Troubleshooting
- If browser setup completed but discovery or selection fails, reuse the existing `agent`; do not reset the JavaScript session or import another browser runtime.
- Inspect `await agent.browsers.list()` once to see which browser types are available. Do not assume that a missing requested browser can be replaced with another backend when the user explicitly named it.
- If a requested backend has specific troubleshooting documentation in the skill's setup catalog, read it before retrying.
- If the requested browser remains unavailable, report that plainly instead of controlling it through an unrelated browser tool or source-code workaround.

# WebMCP
Browser notifications may list page-defined tools. Prefer WebMCP when one
covers the requested action:

```js
const webmcp = await tab.capabilities.get("webmcp");
const tools = await webmcp.fetchTools();
await tools.call("tool_name", input);
```

If no current notification lists the tools, print `tools.description()`. Call
only listed tools. Reuse the same tool handle while on the same page. Fetch again
only if a call reports a stale or invalid handle, or a notification says the
page’s available tools changed.

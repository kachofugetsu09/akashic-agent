# Tab Capability: pageAssets
Asset inventory and bundling for the current rendered page state. Use `list()` to inspect assets already observed in the tab's current state. If lazy-loaded content or another UI state matters, load that state first, then call `list()` again so the inventory reflects what is currently observable. Use `bundle()` to export discovered file assets into a temporary local artifact directory. Prefer `kinds` for broad acquisition and `assetIds` for narrow follow-up. Do not navigate directly to asset URLs just to fetch them.

```ts
const capability = await tab.capabilities.get("pageAssets");

interface PageAssetsTabCapability {
  bundle(options: { assetIds?: Array<string>; inventoryId: string; kinds?: Array<"font" | "image" | "stylesheet" | "video"> }): Promise<{ assets: Array<{ contentType: null | string; id: string; kind: "font" | "image" | "stylesheet" | "video"; name: string; path: string; url: string }>; directoryPath: string; failures: Array<{ contentType: null | string; id: string; name: string; reason: string; url: string }>; manifestPath: string; summary: { downloadedCount: number; elapsedMs: number; failedCount: number; requestedCount: number } }>; // Export file assets from a prior inventory into a local artifact directory.
  list(): Promise<{ assets: Array<{ id: string; kind: "script" | "font" | "image" | "stylesheet" | "video" | "other"; name: string; sources: Array<{ kind: "attribute" | "computedStyle" | "resource"; nodeId?: number; property?: string }>; url: string }>; id: string; inlineSvgs: Array<{ id: string; markup: string; name: string }>; pageUrl: null | string; summary: { byKind: Partial<Record<"script" | "font" | "image" | "stylesheet" | "video" | "other", number>>; inlineSvgCount: number; totalCount: number } }>; // Inventory file assets and inline SVGs observed in the current page state.
}
```

export const MOBILE_PLUGIN_RESULT_CACHE_MAX_ENTRIES = 128;
export const MOBILE_PLUGIN_RESULT_CACHE_MAX_BYTES = 8 * 1024 * 1024;

interface CachedPluginResult {
  value: string;
  bytes: number;
}

/** 按最近使用顺序限制 WebView 内不可变插件结果的条数和总字节。 */
export class MobilePluginResultCache {
  private readonly entries = new Map<string, CachedPluginResult>();
  private readonly maxEntries: number;
  private readonly maxBytes: number;
  private totalBytes = 0;

  constructor(
    maxEntries = MOBILE_PLUGIN_RESULT_CACHE_MAX_ENTRIES,
    maxBytes = MOBILE_PLUGIN_RESULT_CACHE_MAX_BYTES,
  ) {
    if (!Number.isSafeInteger(maxEntries) || maxEntries <= 0) {
      throw new Error("插件结果缓存条数预算无效");
    }
    if (!Number.isSafeInteger(maxBytes) || maxBytes <= 0) {
      throw new Error("插件结果缓存字节预算无效");
    }
    this.maxEntries = maxEntries;
    this.maxBytes = maxBytes;
  }

  get size() {
    return this.entries.size;
  }

  get byteSize() {
    return this.totalBytes;
  }

  get(key: string): string | undefined {
    const cached = this.entries.get(key);
    if (!cached) return undefined;
    this.entries.delete(key);
    this.entries.set(key, cached);
    return cached.value;
  }

  set(key: string, value: string): void {
    if (!key) throw new Error("插件结果缓存身份为空");
    const bytes = new TextEncoder().encode(value).byteLength;

    // 1. 替换同一身份时先移除旧尺寸，避免重复计费
    const existing = this.entries.get(key);
    if (existing) {
      this.entries.delete(key);
      this.totalBytes -= existing.bytes;
    }

    // 2. 单项超过整个预算表示上游结果体积不变量已经失效
    if (bytes > this.maxBytes) throw new Error("插件结果缓存项超过总字节预算");
    this.entries.set(key, { value, bytes });
    this.totalBytes += bytes;

    // 3. 同时满足条数和总字节预算，最久未使用项先淘汰
    while (this.entries.size > this.maxEntries || this.totalBytes > this.maxBytes) {
      const oldestKey = this.entries.keys().next().value;
      if (oldestKey === undefined) throw new Error("插件结果缓存容量统计失配");
      const oldest = this.entries.get(oldestKey);
      if (!oldest) throw new Error("插件结果缓存 LRU 索引失配");
      this.entries.delete(oldestKey);
      this.totalBytes -= oldest.bytes;
    }
  }

  clear() {
    this.entries.clear();
    this.totalBytes = 0;
  }
}

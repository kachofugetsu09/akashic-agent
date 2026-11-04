import type { SceneGraph, SceneNode } from "@open-pencil/scene-graph";

// 乐观并发的合并：画布上有没存下的修改时，别处（Agent）又保存了新修订，不报冲突，而是把对方相对共同基线的改动
// 原地合进当前画布，再带着新修订号保存。图层按 .fig 里不变的 GUID 对齐，按字段合并；同一字段两边都改了，以本地为准。

// 派生出来的缓存、结构字段与界面状态不参与字段比较：结构单独按父子关系合并
const SKIPPED = new Set(["id", "parentId", "childIds", "source", "derivedLayout", "fillGeometry", "strokeGeometry",
  "textPicture", "derivedTextGlyphs", "textPathData", "textPathBox", "expanded"]);
// 引用别的图层的字段：三份文档里的图层 ID 编号不同，比较与搬运前要换算；别的字段原样比较，文字里的“12:30”不是 ID
const REFERENCES = new Set(["componentId", "overrideKey", "instanceOverrides"]);
const CONTENT = new Set(["text", "characters", "name"]);
const PATH = /^\d+:\d+(?:[/;]\d+:\d+)*$/u;

export interface MergeSummary { added: number; updated: number; moved: number; removed: number; kept: number; tables: number }

// 文档根没有 GUID，按“根”对齐；其余图层用读入时的 GUID，新建的图层用自己的 ID（保存时它就是 GUID）。
export function guidOf(graph: SceneGraph, node: SceneNode): string {
  if (node.id === graph.rootId) return "root";
  const saved = node.source.id;
  return saved && /^\d+:\d+$/u.test(saved) ? saved : node.id;
}

// 比较用的序列化：实例覆盖等字段里有 Map 与 Set，JSON 会把它们写成空对象
function serialize(value: unknown): string {
  return JSON.stringify(value ?? null, (_key, item: unknown) =>
    item instanceof Map ? { map: [...item] } : item instanceof Set ? { set: [...item] } : item);
}

// 引用字段里整串是图层 ID（或 ID 路径）的值和键换算一遍，覆盖里的文字等内容不碰
function swapIds(value: unknown, swap: (id: string) => string): unknown {
  const each = (text: string) => PATH.test(text) ? text.replace(/\d+:\d+/gu, swap) : text;
  const entry = ([key, item]: [unknown, unknown]): [unknown, unknown] => typeof key !== "string" ? [key, swapIds(item, swap)]
    : [each(key), CONTENT.has(key) ? item : swapIds(item, swap)];
  if (typeof value === "string") return each(value);
  if (Array.isArray(value)) return value.map((item) => swapIds(item, swap));
  if (value instanceof Map) return new Map([...value].map(entry));
  if (value === null || typeof value !== "object") return value;
  return Object.fromEntries(Object.entries(value).map(entry));
}

class Side {
  readonly byGuid = new Map<string, SceneNode>();
  readonly toGuid = new Map<string, string>();

  constructor(readonly graph: SceneGraph) {
    const seen = new Set<string>();
    for (const node of graph.nodes.values()) {
      const guid = guidOf(graph, node);
      // 复制出来的图层可能带着同一个 GUID：重复的不参与对齐，当作只在这一侧的图层
      if (seen.has(guid)) this.byGuid.delete(guid);
      else this.byGuid.set(guid, node);
      seen.add(guid);
      this.toGuid.set(node.id, guid);
    }
  }

  canon(key: string, value: unknown): string {
    return serialize(REFERENCES.has(key) ? swapIds(value, (id) => this.toGuid.get(id) ?? id) : value);
  }

  canonAll(fields: Record<string, unknown>): string {
    return Object.keys(fields).sort().map((key) => `${key}=${this.canon(key, fields[key])}`).join("\n");
  }

  fields(node: SceneNode): Record<string, unknown> {
    return Object.fromEntries(Object.entries(node).filter(([key]) => !SKIPPED.has(key)));
  }

  parentGuid(node: SceneNode): string | null {
    return node.parentId ? this.toGuid.get(node.parentId) ?? null : null;
  }

  children(node: SceneNode): string[] {
    return node.childIds.map((id) => this.toGuid.get(id) ?? id);
  }
}

// 对方的值搬进本地（复制一份）：引用字段里的图层 ID 换成本地的 ID
function localize(key: string, value: unknown, theirs: Side, live: Side): unknown {
  const copy = structuredClone(value ?? null);
  if (!REFERENCES.has(key)) return copy;
  return swapIds(copy, (id) => {
    const guid = theirs.toGuid.get(id);
    return guid ? live.byGuid.get(guid)?.id ?? id : id;
  });
}

// 本地动过这个图层或它的任何子孙（字段、父图层、子图层列表，或者新建了子图层）
function touched(local: SceneNode, live: Side, base: Side): boolean {
  const before = base.byGuid.get(live.toGuid.get(local.id) ?? local.id);
  if (!before) return true;
  if (live.canonAll(live.fields(local)) !== base.canonAll(base.fields(before)) || live.parentGuid(local) !== base.parentGuid(before)
      || live.children(local).join() !== base.children(before).join()) return true;
  return local.childIds.some((id) => {
    const child = live.graph.getNode(id);
    return child ? touched(child, live, base) : false;
  });
}

// 1. 先按本地原样算出能删的（对方删掉、本地连同子孙都没动过）；2. 新增（父图层先建，引用等全部建好再补）；
// 3. 字段：对方改了而本地没改的搬过来；4. 移动与排序；5. 删除；6. 图片与变量表
export function rebase(liveGraph: SceneGraph, baseGraph: SceneGraph, theirsGraph: SceneGraph): MergeSummary {
  const live = new Side(liveGraph);
  const base = new Side(baseGraph);
  const theirs = new Side(theirsGraph);
  const summary: MergeSummary = { added: 0, updated: 0, moved: 0, removed: 0, kept: 0, tables: 0 };
  const doomed = [...base.byGuid.keys()].filter((guid) => !theirs.byGuid.has(guid))
    .map((guid) => live.byGuid.get(guid)).filter((node): node is SceneNode => !!node && !!liveGraph.getNode(node.id));
  const remove = doomed.filter((node) => !touched(node, live, base));
  summary.kept = doomed.length - remove.length;
  const walk = (node: SceneNode, visit: (node: SceneNode) => void) => {
    visit(node);
    for (const id of node.childIds) {
      const child = theirsGraph.getNode(id);
      if (child) walk(child, visit);
    }
  };
  const root = theirsGraph.getNode(theirsGraph.rootId);
  if (!root) throw new Error("对方的文档没有根节点");
  const added: [SceneNode, SceneNode][] = [];
  walk(root, (node) => addNode(node, live, base, theirs, added));
  for (const [node, created] of added) liveGraph.updateNode(created.id, Object.fromEntries(Object.entries(theirs.fields(node))
    .filter(([key]) => REFERENCES.has(key)).map(([key, value]) => [key, localize(key, value, theirs, live)])) as Partial<SceneNode>);
  summary.added = added.length;
  walk(root, (node) => { if (updateNode(node, live, base, theirs)) summary.updated++; });
  walk(root, (node) => { if (placeNode(node, live, base, theirs)) summary.moved++; });
  for (const node of remove) {
    if (!liveGraph.getNode(node.id)) continue;
    liveGraph.deleteNode(node.id);
    summary.removed++;
  }
  summary.tables = mergeTables(liveGraph, baseGraph, theirsGraph);
  return summary;
}

function addNode(node: SceneNode, live: Side, base: Side, theirs: Side, added: [SceneNode, SceneNode][]): void {
  const guid = guidOf(theirs.graph, node);
  if (base.byGuid.has(guid) || live.byGuid.has(guid)) return;
  const parentGuid = theirs.parentGuid(node);
  const parent = parentGuid ? live.byGuid.get(parentGuid) : undefined;
  if (!parent) return;
  const fields = Object.fromEntries(Object.entries(theirs.fields(node)).filter(([key]) => !REFERENCES.has(key))
    .map(([key, value]) => [key, localize(key, value, theirs, live)])) as Partial<SceneNode>;
  // 用 GUID 当新图层的 ID：保存时它原样成为 GUID；ID 已被本地别的图层占用时换个 ID 建，GUID 记在来源里
  const free = /^\d+:\d+$/u.test(guid) && !live.graph.getNode(guid);
  const created = free ? live.graph.createNodeWithId(guid, node.type, parent.id, fields) : live.graph.createNode(node.type, parent.id, fields);
  if (!free) live.graph.updateNode(created.id, { source: { ...created.source, id: guid } });
  live.byGuid.set(guid, created);
  live.toGuid.set(created.id, guid);
  added.push([node, created]);
}

function updateNode(node: SceneNode, live: Side, base: Side, theirs: Side): boolean {
  const guid = guidOf(theirs.graph, node);
  const before = base.byGuid.get(guid);
  const local = live.byGuid.get(guid);
  if (!before || !local) return false;
  const mine = live.fields(local);
  const old = base.fields(before);
  const next = theirs.fields(node);
  if (theirs.canonAll(next) === base.canonAll(old)) return false;
  const changes: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(next)) {
    const was = base.canon(key, old[key]);
    if (theirs.canon(key, value) !== was && live.canon(key, mine[key]) === was) changes[key] = localize(key, value, theirs, live);
  }
  if (!Object.keys(changes).length) return false;
  live.graph.updateNode(local.id, changes as Partial<SceneNode>);
  return true;
}

// 对方挪了父图层而本地没挪就跟着挪；子图层顺序本地没改过就按对方的顺序排
function placeNode(node: SceneNode, live: Side, base: Side, theirs: Side): boolean {
  const guid = guidOf(theirs.graph, node);
  const local = live.byGuid.get(guid);
  if (!local) return false;
  const before = base.byGuid.get(guid);
  const parentGuid = theirs.parentGuid(node);
  const parent = parentGuid ? live.byGuid.get(parentGuid) : undefined;
  let moved = false;
  if (before && parent && parentGuid !== base.parentGuid(before) && live.parentGuid(local) === base.parentGuid(before)
      && parent.id !== local.parentId) {
    live.graph.reparentNode(local.id, parent.id);
    moved = true;
  }
  const was = before ? base.children(before) : [];
  const mine = live.children(local);
  const common = (list: string[], keep: string[]) => list.filter((item) => keep.includes(item));
  if (before && common(mine, was).join() !== common(was, mine).join()) return moved;
  theirs.children(node).map((child) => live.byGuid.get(child)).forEach((child, index) => {
    if (!child || child.parentId !== local.id || local.childIds.indexOf(child.id) === index) return;
    live.graph.reorderChild(child.id, local.id, index);
    moved = true;
  });
  return moved;
}

// 图片按内容哈希取，缺的补上；变量与集合按 ID：对方新增或改了而本地没改的搬过来，对方删了而本地没改的删掉
function mergeTables(live: SceneGraph, base: SceneGraph, theirs: SceneGraph): number {
  let changed = 0;
  for (const [hash, bytes] of theirs.images) if (!live.images.has(hash)) live.images.set(hash, bytes);
  for (const name of ["variables", "variableCollections"] as const) {
    const mine = live[name] as Map<string, unknown>;
    const old = base[name] as Map<string, unknown>;
    const next = theirs[name] as Map<string, unknown>;
    for (const [id, value] of next) {
      const was = serialize(old.get(id));
      if (serialize(value) === was || serialize(mine.get(id)) !== was) continue;
      mine.set(id, structuredClone(value));
      changed++;
    }
    for (const [id, value] of old) {
      if (next.has(id) || !mine.has(id) || serialize(mine.get(id)) !== serialize(value)) continue;
      mine.delete(id);
      changed++;
    }
  }
  return changed;
}

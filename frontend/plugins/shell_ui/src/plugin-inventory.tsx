import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { WebHostContextV1 } from "@akashic/web-ui-v1";

type Phase = "active" | "pending" | "failed" | "disabled" | "stopping" | "loading";

interface PluginRow {
  id: string;
  name: string;
  version: string | null;
  description: string;
  source: "builtin" | "external";
  enabled: boolean;
  phase: Phase;
  error: string | null;
  missing_services: string[];
  dependents: string[];
  locked: "management-required" | null;
}

type Http = WebHostContextV1["http"];
type Filter = "all" | "attention";
/** 行内的操作结果：进行中、需要重启，或失败原因。 */
type RowNote = { kind: "busy"; enabled: boolean } | { kind: "restart" } | { kind: "error"; text: string };

const PHASE_TEXT: Record<Exclude<Phase, "active">, string> = {
  pending: "等待依赖", failed: "启动失败", disabled: "已停用", stopping: "正在停用", loading: "正在启动",
};
// 需要用户处理：启动失败、等待依赖，或停用时资源没能关闭。
const needsAttention = (row: PluginRow) => row.phase === "failed" || row.phase === "pending" || (row.phase === "stopping" && !!row.error);
const CHEVRON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m9 18 6-6-6-6"/></svg>';

async function call<T>(http: Http, path: string, init?: RequestInit): Promise<T> {
  const response = await http.request(path, init);
  const body = await response.json().catch(() => ({})) as { detail?: unknown };
  if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `请求失败（${response.status}）`);
  return body as T;
}

// 插件列表：内置 / 外置两组，被其他插件依赖的内置插件收进“基础组件”子组；行展开才显示说明与开关。
export function PluginInventory({ http }: { http: Http }): React.ReactElement {
  const [rows, setRows] = useState<PluginRow[] | null>(null);
  const [loadError, setLoadError] = useState("");
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<Filter>("all");
  const [expanded, setExpanded] = useState<ReadonlySet<string>>(new Set());
  const [infraOpen, setInfraOpen] = useState(false);
  const [notes, setNotes] = useState<ReadonlyMap<string, RowNote>>(new Map());
  // 分组按本次打开时见过的依赖关系固定，停用基础组件后它不会跳到别的分组。
  const infra = useRef(new Set<string>());

  const load = useCallback(async () => {
    try {
      const body = await call<{ plugins: PluginRow[] }>(http, "/api/dashboard/plugins");
      for (const row of body.plugins) if (row.source === "builtin" && row.dependents.length > 0) infra.current.add(row.id);
      setRows(body.plugins);
      setLoadError("");
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : String(error));
    }
  }, [http]);

  useEffect(() => {
    void load();
    const timer = window.setInterval(() => void load(), 5000);
    return () => window.clearInterval(timer);
  }, [load]);

  const toggle = useCallback(async (row: PluginRow, enabled: boolean) => {
    const note = (value: RowNote | null) => setNotes((current) => {
      const next = new Map(current);
      if (value) next.set(row.id, value); else next.delete(row.id);
      return next;
    });
    note({ kind: "busy", enabled });
    try {
      const result = await call<{ publication_state?: string }>(http, `/api/dashboard/plugins/${encodeURIComponent(row.id)}/enabled`, {
        method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ enabled }),
      });
      note(result.publication_state === "restart_required" ? { kind: "restart" } : null);
    } catch (error) {
      note({ kind: "error", text: error instanceof Error ? error.message : String(error) });
    }
    await load();
  }, [http, load]);

  const all = rows ?? [];
  const attention = all.filter(needsAttention).length;
  const searching = query.trim() !== "" || filter !== "all";
  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return all.filter((row) => (filter === "all" || needsAttention(row))
      && (!needle || `${row.name} ${row.id} ${row.description}`.toLowerCase().includes(needle)));
  }, [all, query, filter]);

  if (rows === null) return <p className="shell-inventory-empty" role="status">{loadError || "正在读取插件…"}</p>;
  const builtin = visible.filter((row) => row.source === "builtin" && !infra.current.has(row.id));
  const base = visible.filter((row) => row.source === "builtin" && infra.current.has(row.id));
  const external = visible.filter((row) => row.source === "external");
  const flip = (id: string) => setExpanded((current) => {
    const next = new Set(current);
    if (!next.delete(id)) next.add(id);
    return next;
  });
  const list = (items: PluginRow[]) => <ul className="shell-inventory-rows">
    {items.map((row) => <InventoryRow key={row.id} row={row} rows={all} open={expanded.has(row.id)}
      note={notes.get(row.id)} flip={() => flip(row.id)} toggle={(enabled) => void toggle(row, enabled)} />)}
  </ul>;

  return <div className="shell-inventory">
    <div className="shell-inventory-toolbar">
      <input type="search" placeholder="搜索插件" aria-label="搜索插件" value={query} onChange={(event) => setQuery(event.target.value)} />
      {attention > 0 && <div className="shell-inventory-filter" role="group" aria-label="筛选">
        <button type="button" aria-pressed={filter === "all"} onClick={() => setFilter("all")}>全部</button>
        <button type="button" aria-pressed={filter === "attention"} onClick={() => setFilter("attention")}>需要处理 {attention}</button>
      </div>}
    </div>
    {loadError && <p className="shell-inventory-error" role="alert">{loadError}</p>}
    {builtin.length + base.length > 0 && <section className="shell-inventory-group">
      <h4>内置插件 <span>{builtin.length + base.length}</span></h4>
      {list(builtin)}
      {base.length > 0 && <div className="shell-inventory-sub">
        <button type="button" className="shell-inventory-sub-head" aria-expanded={searching || infraOpen} onClick={() => setInfraOpen(!infraOpen)}>
          <span className="shell-inventory-chevron" aria-hidden="true" dangerouslySetInnerHTML={{ __html: CHEVRON }} />
          基础组件 <span>{base.length}</span><small>通常不需要改动</small>
        </button>
        {(searching || infraOpen) && list(base)}
      </div>}
    </section>}
    {external.length > 0 && <section className="shell-inventory-group">
      <h4>外置插件 <span>{external.length}</span></h4>
      {list(external)}
    </section>}
    {visible.length === 0 && <div className="shell-inventory-empty">
      <p>{query.trim() ? `没有匹配“${query.trim()}”的插件` : "没有需要处理的插件"}</p>
      <button type="button" onClick={() => { setQuery(""); setFilter("all"); }}>显示全部插件</button>
    </div>}
  </div>;
}

// 一行插件：收起时只有名称和状态（运行中只显示一个点），展开后给说明、原因与开关。
function InventoryRow({ row, rows, open, note, flip, toggle }: {
  row: PluginRow;
  rows: readonly PluginRow[];
  open: boolean;
  note: RowNote | undefined;
  flip: () => void;
  toggle: (enabled: boolean) => void;
}): React.ReactElement {
  const busy = note?.kind === "busy";
  const status = busy ? (note.enabled ? "正在启用" : "正在停用")
    : note?.kind === "restart" ? "重启后生效" : row.phase === "active" ? null
    : row.phase === "stopping" && row.error ? "未能停用" : PHASE_TEXT[row.phase];
  const waiting = row.enabled ? row.dependents.map((id) => rows.find((item) => item.id === id)?.name ?? id) : [];
  const hint = waiting.length === 0 ? ""
    : `停用后，${waiting.slice(0, 2).join("、")}${waiting.length > 2 ? ` 等 ${waiting.length} 个插件` : ""}会等待依赖`;
  return <li className="shell-inventory-row" data-phase={row.phase === "stopping" && row.error ? "failed" : row.phase}>
    <button type="button" className="shell-inventory-head" aria-expanded={open} onClick={flip}>
      <strong>{row.name}</strong>
      {status
        ? <span className="shell-inventory-status">{status}</span>
        : <span className="shell-inventory-dot" role="img" aria-label="运行中" />}
      <span className="shell-inventory-chevron" aria-hidden="true" dangerouslySetInnerHTML={{ __html: CHEVRON }} />
    </button>
    {open && <div className="shell-inventory-body">
      {row.description && <p>{row.description}</p>}
      {row.error && <p className="shell-inventory-error">{row.error}</p>}
      {note?.kind === "error" && <p className="shell-inventory-error" role="alert">{note.text}</p>}
      {row.missing_services.length > 0 && <p>缺少 <code>{row.missing_services.join("、")}</code>，依赖恢复后自动运行。</p>}
      {row.locked
        ? <p className="shell-inventory-hint">插件管理依赖它，不能停用</p>
        : <label className="shell-inventory-switch">
          <input type="checkbox" role="switch" checked={busy ? note.enabled : row.enabled} disabled={busy}
            onChange={(event) => toggle(event.target.checked)} />
          启用
          {hint && <small>{hint}</small>}
        </label>}
      <details>
        <summary>技术信息</summary>
        <dl>
          <dt>ID</dt><dd><code>{row.id}</code></dd>
          {row.version && <><dt>版本</dt><dd><code>{row.version}</code></dd></>}
          <dt>运行</dt><dd><code>{row.phase}</code></dd>
        </dl>
      </details>
    </div>}
  </li>;
}

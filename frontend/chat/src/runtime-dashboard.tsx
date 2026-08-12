import {
  AlertCircle,
  ArrowLeft,
  BookOpenText,
  CheckCircle2,
  ChevronRight,
  Copy,
  RefreshCw,
  Server,
  Timer,
} from "lucide-react";
import { MessageResponse } from "@/components/ai-elements/message-response";
import {
  formatRuntimeSyncTime,
  runtimeDirectoryDescription,
  type RuntimeDirectoryItem,
  type RuntimeView,
} from "./runtime-dashboard-data";
import { useRuntimeDashboard } from "./use-runtime-dashboard";
import "./runtime-dashboard.css";

const viewMeta = {
  documents: { label: "文档", icon: BookOpenText },
  mcp: { label: "MCP 与 Skills", icon: Server },
  jobs: { label: "定时任务", icon: Timer },
} satisfies Record<RuntimeView, { label: string; icon: typeof BookOpenText }>;

const runtimeViews = Object.keys(viewMeta) as RuntimeView[];

/** Render the desktop read-only projection while its controller owns I/O and selection. */
export function RuntimeDashboard() {
  const runtime = useRuntimeDashboard();
  return (
    <section className={`runtime-dashboard runtime-dashboard--${runtime.view}`}>
      <header className="runtime-dashboard__topbar">
        <div>
          <h1>知识与运行</h1>
          <p>当前电脑的只读投影 · 与移动端使用相同的运行目录</p>
        </div>
        <div className="runtime-dashboard__sync">
          {runtime.syncedAt ? <span><CheckCircle2 size={17} />已同步 · {formatRuntimeSyncTime(runtime.syncedAt)}</span> : null}
          <button type="button" onClick={() => void runtime.refresh()} disabled={runtime.loading}>
            <RefreshCw size={17} className={runtime.loading ? "is-spinning" : ""} />
            {runtime.loading ? "正在刷新" : "刷新"}
          </button>
        </div>
      </header>

      <section className="runtime-dashboard__metrics" aria-label="运行摘要">
        <RuntimeMetric icon={<BookOpenText size={20} />} value={runtime.overview?.documents.length ?? 0} label="核心文档" tone="documents" />
        <RuntimeMetric icon={<Server size={20} />} value={runtime.overview?.capabilities.mcp_servers.reduce((sum, item) => sum + item.tool_count, 0) ?? 0} label={`MCP 工具 · ${runtime.overview?.capabilities.skills.length ?? 0} Skills`} tone="mcp" />
        <RuntimeMetric icon={<Timer size={20} />} value={runtime.overview?.jobs.filter((job) => job.enabled).length ?? 0} label="已启用定时任务" tone="jobs" />
      </section>

      <RuntimeTabs view={runtime.view} onSelect={runtime.selectView} />

      <div className="runtime-dashboard__notice-slot">
        {runtime.error ? <div className="runtime-dashboard__error" role="alert"><AlertCircle size={18} />{runtime.error}</div> : null}
        <span className="sr-only" role="status" aria-live="polite">{runtime.copyFeedback}</span>
      </div>

      <section
        id="runtime-directory-panel"
        className={`runtime-directory ${runtime.detailOpen ? "detail-open" : ""}`}
        role="tabpanel"
        aria-labelledby={`runtime-tab-${runtime.view}`}
      >
        <RuntimeDirectoryList
          view={runtime.view}
          items={runtime.items}
          selectedKey={runtime.selectedKey}
          loading={runtime.loading}
          description={runtimeDirectoryDescription(runtime.view, runtime.overview)}
          onSelect={runtime.selectItem}
        />
        <article className="runtime-detail">
          <header className="runtime-detail__header">
            <button className="runtime-detail__back" type="button" onClick={runtime.closeDetail}><ArrowLeft size={18} />返回{viewMeta[runtime.view].label}</button>
            <div><h2>{runtime.detail?.title ?? "选择一个项目"}</h2><p>{runtime.detail?.subtitle ?? "在左侧目录中选择要查看的内容。"}</p></div>
            {runtime.detail ? (
              <button className="runtime-detail__copy" type="button" onClick={() => void runtime.copyDetail()}>
                <Copy size={17} />{runtime.copyFeedback || "复制标识"}
              </button>
            ) : null}
          </header>
          <div className="runtime-detail__scroll">
            {runtime.detailLoading ? <p className="runtime-detail__loading"><RefreshCw size={22} className="is-spinning" />正在读取最新内容…</p> : null}
            {!runtime.detailLoading && runtime.detail ? <MessageResponse className="runtime-detail__markdown">{runtime.detail.markdown}</MessageResponse> : null}
            {!runtime.detailLoading && !runtime.detail ? <p className="runtime-detail__loading">详情将在这里显示。</p> : null}
          </div>
        </article>
      </section>
    </section>
  );
}

function RuntimeTabs({ view, onSelect }: { view: RuntimeView; onSelect: (view: RuntimeView) => void }) {
  return (
    <nav className="runtime-dashboard__tabs" role="tablist" aria-label="知识与运行目录">
      {runtimeViews.map((key) => {
        const Icon = viewMeta[key].icon;
        return (
          <button
            id={`runtime-tab-${key}`}
            key={key}
            type="button"
            role="tab"
            aria-controls="runtime-directory-panel"
            aria-selected={view === key}
            tabIndex={view === key ? 0 : -1}
            onClick={() => onSelect(key)}
            onKeyDown={(event) => moveRuntimeTab(event, key, onSelect)}
          >
            <Icon size={20} aria-hidden="true" />
            <span>{viewMeta[key].label}</span>
          </button>
        );
      })}
    </nav>
  );
}

function RuntimeDirectoryList({
  view,
  items,
  selectedKey,
  loading,
  description,
  onSelect,
}: {
  view: RuntimeView;
  items: RuntimeDirectoryItem[];
  selectedKey: string;
  loading: boolean;
  description: string;
  onSelect: (key: string) => void;
}) {
  return (
    <div className="runtime-directory__list">
      <header><h2>{viewMeta[view].label}</h2><p>{description}</p></header>
      {loading ? <p className="runtime-directory__empty">正在读取最新运行目录…</p> : null}
      {!loading && items.length === 0 ? <p className="runtime-directory__empty">当前目录没有可展示的项目。</p> : null}
      {items.map((item) => {
        const Icon = viewMeta[item.icon].icon;
        return (
          <button
            className="runtime-directory__item"
            type="button"
            key={item.key}
            aria-pressed={selectedKey === item.key}
            disabled={item.disabled}
            onClick={() => onSelect(item.key)}
          >
            <span className="runtime-directory__item-icon"><Icon size={item.icon === "documents" ? 19 : 20} aria-hidden="true" /></span>
            <span className="runtime-directory__item-copy"><strong>{item.title}</strong><small>{item.description}</small></span>
            {item.status ? <span className={`runtime-directory__status ${item.statusTone ?? ""}`}>{item.status}</span> : null}
            <ChevronRight size={18} aria-hidden="true" />
          </button>
        );
      })}
    </div>
  );
}

function RuntimeMetric({ icon, value, label, tone }: { icon: React.ReactNode; value: number; label: string; tone: RuntimeView }) {
  return <div className={`runtime-metric ${tone}`}><span>{icon}</span><div><small>{label}</small><strong>{value}</strong></div></div>;
}

function moveRuntimeTab(event: React.KeyboardEvent<HTMLButtonElement>, key: RuntimeView, onSelect: (view: RuntimeView) => void) {
  if (!(event.key === "ArrowLeft" || event.key === "ArrowRight" || event.key === "Home" || event.key === "End")) return;
  event.preventDefault();
  const current = runtimeViews.indexOf(key);
  const next = event.key === "Home"
    ? runtimeViews[0]
    : event.key === "End"
      ? runtimeViews[runtimeViews.length - 1]
      : runtimeViews[(current + (event.key === "ArrowRight" ? 1 : -1) + runtimeViews.length) % runtimeViews.length];
  onSelect(next);
  requestAnimationFrame(() => document.getElementById(`runtime-tab-${next}`)?.focus());
}

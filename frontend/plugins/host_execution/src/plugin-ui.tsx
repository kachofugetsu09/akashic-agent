import { useEffect } from "react";
import { X } from "lucide-react";
import { Notice, render, type HeaderMetaItem, type PluginUiContext, type PluginUiDefinition } from "@akashic/chat-ui-v1";
import { useValue, valueStore } from "../../../../packages/akashic-plugin-kit/src/meta-store";
import "./plugin-ui.css";

const IDLE_POLL_MS = 5000;
const DEPLOY_POLL_MS = 2000;
// 单次查询由宿主限时；15 秒宽限保证一次偶发失败不报警，持续失败约 15 秒后才提示未知。
const UNCONFIRMED_GRACE_MS = 15_000;
const UNCONFIRMED_NOTICE = "暂时无法确认宿主执行状态，正在重新检查。";
const RECENT_SUCCESS_WINDOW_MS = 10 * 60_000;
const RECENT_FAILURE_WINDOW_MS = 15 * 60_000;
const DISMISSED_STORAGE_KEY = "akashic.deploy.dismissed";

const DEPLOY_ICON_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><line x1="3" x2="9" y1="12" y2="12"/><line x1="15" x2="21" y1="12" y2="12"/></svg>';

type DeployStatusKind = "submitting" | "accepted" | "running" | "active" | "failed";

export interface DeployRecordView {
  requestId: string;
  status: DeployStatusKind;
  phase: string;
  targetCommit: string;
  previousCommit: string | null;
  commitSubject: string | null;
  submittedAt: string | null;
  startedAt: string | null;
  finishedAt: string | null;
  recordedAt: string | null;
  stale: boolean;
  degradedExternal: string[];
  error: string | null;
}

export interface RuntimeVersionView {
  commit: string | null;
  startedAt: string | null;
}

interface HostStateSnapshot {
  bridgeNotice: string;
  runtime: RuntimeVersionView;
  deploy: DeployRecordView | null;
}

const INITIAL_SNAPSHOT: HostStateSnapshot = {
  bridgeNotice: "",
  runtime: { commit: null, startedAt: null },
  deploy: null,
};

const hostStateStore = valueStore<HostStateSnapshot>(INITIAL_SNAPSHOT);
const bootCommitStore = valueStore<string | null>(null);
const watchedDeployIds = new Set<string>();
const changedInTabIds = new Set<string>();
const dismissedStore = valueStore<string>(readDismissedId());

function readDismissedId(): string {
  try {
    return window.localStorage.getItem(DISMISSED_STORAGE_KEY) ?? "";
  } catch {
    return "";
  }
}

function dismissDeployId(requestId: string): void {
  dismissedStore.set(requestId);
  try {
    window.localStorage.setItem(DISMISSED_STORAGE_KEY, requestId);
  } catch {
    // 隐私模式或只读存储仅保留会话内关闭态。
  }
}

function isDeployInProgress(deploy: DeployRecordView | null): boolean {
  return Boolean(deploy && !deploy.stale && (deploy.status === "submitting" || deploy.status === "accepted" || deploy.status === "running"));
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
}

function asString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function parseDeploy(raw: unknown): DeployRecordView | null {
  const row = asRecord(raw);
  if (!row) return null;
  const requestId = asString(row.requestId);
  const status = asString(row.status);
  const targetCommit = asString(row.targetCommit);
  if (!requestId || !targetCommit || !status || !["submitting", "accepted", "running", "active", "failed"].includes(status)) {
    return null;
  }
  const degraded = Array.isArray(row.degradedExternal)
    ? row.degradedExternal.filter((item): item is string => typeof item === "string" && item.length > 0)
    : [];
  return {
    requestId,
    status: status as DeployStatusKind,
    phase: asString(row.phase) ?? status,
    targetCommit,
    previousCommit: asString(row.previousCommit),
    commitSubject: asString(row.commitSubject),
    submittedAt: asString(row.submittedAt),
    startedAt: asString(row.startedAt),
    finishedAt: asString(row.finishedAt),
    recordedAt: asString(row.recordedAt),
    stale: row.stale === true,
    degradedExternal: degraded,
    error: asString(row.error),
  };
}

function parseRuntime(raw: unknown): RuntimeVersionView {
  const row = asRecord(raw);
  if (!row) return { commit: null, startedAt: null };
  return { commit: asString(row.commit), startedAt: asString(row.startedAt) };
}

function shortSha(commit: string | null): string {
  return commit ? commit.slice(0, 7) : "未知";
}

function parseTimeMs(iso: string | null): number | null {
  if (!iso) return null;
  const parsed = Date.parse(iso);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatDurationMs(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor(total / 60) % 60;
  const seconds = total % 60;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${seconds}s`;
  return `${seconds}s`;
}

function formatClock(iso: string | null): string {
  const ms = parseTimeMs(iso);
  if (ms === null) return "未知";
  return new Date(ms).toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function deployElapsedMs(deploy: DeployRecordView, nowMs: number): number | null {
  const start = parseTimeMs(deploy.startedAt ?? deploy.submittedAt);
  if (start === null) return null;
  const end = deploy.status === "active" || deploy.status === "failed"
    ? parseTimeMs(deploy.finishedAt ?? deploy.recordedAt) ?? nowMs
    : nowMs;
  return Math.max(0, end - start);
}

function phaseDescription(deploy: DeployRecordView): string {
  const short = shortSha(deploy.targetCommit);
  switch (deploy.phase) {
    case "submitting":
    case "accepted":
      return `准备部署 ${short}`;
    case "preparing":
      return `正在构建新版本 ${short}`;
    case "waiting_for_turn":
      return `新版本 ${short} 已就绪，等待本轮收尾后重启`;
    case "restarting":
      return `正在重启切换到 ${short}…`;
    default:
      return `正在部署 ${short}`;
  }
}

function phaseShortLabel(phase: string): string {
  switch (phase) {
    case "submitting":
    case "accepted":
      return "排队中";
    case "preparing":
      return "构建中";
    case "waiting_for_turn":
      return "等待重启";
    case "restarting":
      return "重启切换中";
    case "active":
      return "已上线";
    case "failed":
      return "失败";
    default:
      return phase;
  }
}

// 多个插槽共用同一条后台轮询链，避免 chat.notice 与 header.meta 重复请求。
type PollerContext = PluginUiContext<"chat.notice"> | PluginUiContext<"header.meta">;
const pollerSubscribers = new Map<symbol, { context: PollerContext; stopShown: () => void }>();
let stopPollerLoop: (() => void) | null = null;
let triggerPollerShown: (() => void) | null = null;
let triggerPollerNow: (() => void) | null = null;

function syncDeployMarker(deploy: DeployRecordView | null): void {
  if (typeof document === "undefined") return;
  if (isDeployInProgress(deploy) && deploy) {
    document.documentElement.dataset.akashicDeploying = shortSha(deploy.targetCommit);
  } else {
    delete document.documentElement.dataset.akashicDeploying;
  }
}

function latestPollerContext(): PollerContext | null {
  let latest: PollerContext | null = null;
  for (const entry of pollerSubscribers.values()) latest = entry.context;
  return latest;
}

function startSharedPoller(context: PollerContext): () => void {
  const token = Symbol("host-poller");
  const stopShown = context.onShown(() => { triggerPollerShown?.(); });
  pollerSubscribers.set(token, { context, stopShown });

  if (!stopPollerLoop) {
    let active = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let failingSince: number | null = null;
    let inflight = false;
    let pausedAt: number | null = null;
    let lastCompletedAt = 0;

    const refresh = async () => {
      const currentContext = latestPollerContext();
      if (!active || !currentContext) return;
      if (document.visibilityState === "hidden") {
        pausedAt ??= Date.now();
        return;
      }
      const startedAt = Date.now();
      inflight = true;
      let nextInterval = IDLE_POLL_MS;
      try {
        const body = await currentContext.query("bridge.state", {}, { cache: "none" });
        const bridgeNotice = noticeFor(body);
        const runtime = parseRuntime(body.runtime);
        const deploy = parseDeploy(body.deploy);
        failingSince = null;

        if (runtime.commit) {
          const initial = bootCommitStore.get();
          if (initial === null) {
            bootCommitStore.set(runtime.commit);
          } else if (initial !== runtime.commit && deploy?.status === "active" && deploy.targetCommit === runtime.commit) {
            changedInTabIds.add(deploy.requestId);
          }
        }
        if (isDeployInProgress(deploy) && deploy) {
          watchedDeployIds.add(deploy.requestId);
          nextInterval = DEPLOY_POLL_MS;
        }
        syncDeployMarker(deploy);
        if (active) hostStateStore.set({ bridgeNotice, runtime, deploy });
      } catch {
        if (!active) return;
        failingSince ??= startedAt;
        if (Date.now() - failingSince >= UNCONFIRMED_GRACE_MS) {
          const previous = hostStateStore.get();
          hostStateStore.set({ ...previous, bridgeNotice: UNCONFIRMED_NOTICE });
        }
      } finally {
        inflight = false;
        lastCompletedAt = Date.now();
      }
      if (active) timer = setTimeout(refresh, nextInterval);
    };

    triggerPollerShown = () => {
      if (!active || inflight || pausedAt === null || document.visibilityState === "hidden") return;
      if (Date.now() - pausedAt >= UNCONFIRMED_GRACE_MS) failingSince = null;
      pausedAt = null;
      clearTimeout(timer);
      void refresh();
    };

    triggerPollerNow = () => {
      if (!active || inflight || document.visibilityState === "hidden") return;
      if (Date.now() - lastCompletedAt < 500) return;
      clearTimeout(timer);
      void refresh();
    };

    void refresh();

    stopPollerLoop = () => {
      active = false;
      clearTimeout(timer);
      triggerPollerShown = null;
      triggerPollerNow = null;
      syncDeployMarker(null);
    };
  } else {
    triggerPollerNow?.();
  }

  return () => {
    const entry = pollerSubscribers.get(token);
    if (entry) {
      entry.stopShown();
      pollerSubscribers.delete(token);
    }
    if (pollerSubscribers.size === 0 && stopPollerLoop) {
      stopPollerLoop();
      stopPollerLoop = null;
    }
  };
}

function useHostState(context: PollerContext): HostStateSnapshot {
  const snapshot = useValue(hostStateStore);
  useEffect(() => startSharedPoller(context), [context]);
  return snapshot;
}

const clockStore = valueStore<number>(Date.now());
let clockSubscribers = 0;
let clockTimer: number | undefined;

function useLiveClock(active: boolean): number {
  const now = useValue(clockStore);
  useEffect(() => {
    if (!active) return;
    clockSubscribers += 1;
    if (clockSubscribers === 1) {
      clockStore.set(Date.now());
      clockTimer = window.setInterval(() => clockStore.set(Date.now()), 1000);
    }
    return () => {
      clockSubscribers -= 1;
      if (clockSubscribers <= 0 && clockTimer !== undefined) {
        window.clearInterval(clockTimer);
        clockTimer = undefined;
      }
    };
  }, [active]);
  return now;
}

function shouldShowOutcomeBanner(deploy: DeployRecordView | null, dismissedId: string, nowMs: number): boolean {
  if (!deploy || deploy.requestId === dismissedId) return false;
  if (deploy.status !== "active" && deploy.status !== "failed") return false;
  if (watchedDeployIds.has(deploy.requestId) || changedInTabIds.has(deploy.requestId)) return true;
  const finishedMs = parseTimeMs(deploy.finishedAt ?? deploy.recordedAt);
  if (finishedMs === null) return false;
  const windowMs = deploy.status === "failed" ? RECENT_FAILURE_WINDOW_MS : RECENT_SUCCESS_WINDOW_MS;
  return nowMs - finishedMs <= windowMs;
}

function DeployNoticeBanner({ deploy, nowMs, dismissedId }: {
  deploy: DeployRecordView | null;
  nowMs: number;
  dismissedId: string;
}) {
  if (!deploy) return null;

  if (isDeployInProgress(deploy)) {
    const elapsed = deployElapsedMs(deploy, nowMs);
    const subject = deploy.commitSubject ? ` · ${deploy.commitSubject}` : "";
    return (
      <div className="host-deploy-notice" data-tone="live" role="status" aria-live="polite">
        <span className="host-deploy-notice__main">
          <span className="host-deploy-notice__dot" aria-hidden="true" />
          <span className="host-deploy-notice__text" title={deploy.commitSubject ?? deploy.targetCommit}>
            {phaseDescription(deploy)}{subject}
          </span>
        </span>
        {elapsed !== null ? <span className="host-deploy-notice__meta">{formatDurationMs(elapsed)}</span> : null}
      </div>
    );
  }

  if (!shouldShowOutcomeBanner(deploy, dismissedId, nowMs)) return null;

  const short = shortSha(deploy.targetCommit);
  const elapsed = deployElapsedMs(deploy, nowMs);
  const durationNote = elapsed !== null ? `耗时 ${formatDurationMs(elapsed)}` : "";

  if (deploy.status === "active") {
    const degraded = deploy.degradedExternal;
    const tone = degraded.length > 0 ? "warning" : "success";
    const text = degraded.length > 0
      ? `已切换至 ${short}（外置插件降级：${degraded.join("、")}）`
      : `已部署上线 ${short}${deploy.commitSubject ? `（${deploy.commitSubject}）` : ""}`;
    return (
      <div className="host-deploy-notice" data-tone={tone} role="status">
        <span className="host-deploy-notice__main">
          <span className="host-deploy-notice__dot" aria-hidden="true" />
          <span className="host-deploy-notice__text" title={deploy.commitSubject ?? deploy.targetCommit}>{text}</span>
        </span>
        <span className="host-deploy-notice__right">
          {durationNote ? <span className="host-deploy-notice__meta">{durationNote}</span> : null}
          <button
            type="button"
            className="host-deploy-notice__dismiss"
            aria-label="关闭部署提示"
            onClick={() => dismissDeployId(deploy.requestId)}
          >
            <X size={13} aria-hidden="true" />
          </button>
        </span>
      </div>
    );
  }

  const rolledBack = Boolean(deploy.error && deploy.error.includes("已回到原版本"));
  const rollbackTag = rolledBack ? "（已回退）" : "";
  const detailText = deploy.error ? `：${deploy.error}` : "";
  return (
    <div className="host-deploy-notice" data-tone="error" role="status">
      <span className="host-deploy-notice__main">
        <span className="host-deploy-notice__dot" aria-hidden="true" />
        <span className="host-deploy-notice__text" title={deploy.error ?? undefined}>
          部署 {short} 失败{rollbackTag}{detailText}
        </span>
      </span>
      <span className="host-deploy-notice__right">
        {durationNote ? <span className="host-deploy-notice__meta">{durationNote}</span> : null}
        <button
          type="button"
          className="host-deploy-notice__dismiss"
          aria-label="关闭部署提示"
          onClick={() => dismissDeployId(deploy.requestId)}
        >
          <X size={13} aria-hidden="true" />
        </button>
      </span>
    </div>
  );
}

// 显示宿主执行连接状态与当前版本的自部署进度/结果。
function HostBridgeNotice({ context }: { context: PluginUiContext<"chat.notice"> }) {
  const { bridgeNotice, deploy } = useHostState(context);
  const dismissedId = useValue(dismissedStore);
  const deploying = isDeployInProgress(deploy);
  const nowMs = useLiveClock(deploying);
  return (
    <>
      <DeployNoticeBanner deploy={deploy} nowMs={nowMs} dismissedId={dismissedId} />
      {bridgeNotice ? <Notice>{bridgeNotice}</Notice> : null}
    </>
  );
}

function RuntimeDeployDetail() {
  const { runtime, deploy } = useValue(hostStateStore);
  const deploying = isDeployInProgress(deploy);
  const nowMs = useLiveClock(deploying);
  const elapsed = deploy ? deployElapsedMs(deploy, nowMs) : null;

  return (
    <div className="host-deploy-detail">
      {runtime.commit ? (
        <div className="host-deploy-detail__row">
          <span className="host-deploy-detail__label">运行提交</span>
          <code className="host-deploy-detail__value" title={runtime.commit}>{runtime.commit.slice(0, 12)}</code>
        </div>
      ) : null}
      {runtime.startedAt ? (
        <div className="host-deploy-detail__row">
          <span className="host-deploy-detail__label">本次启动</span>
          <span className="host-deploy-detail__value">{formatClock(runtime.startedAt)}</span>
        </div>
      ) : null}
      {deploy ? (
        <>
          <div className="host-deploy-detail__row">
            <span className="host-deploy-detail__label">最近部署</span>
            <span className="host-deploy-detail__badge" data-status={deploy.status}>
              {phaseShortLabel(deploy.phase)}
              {deploy.stale ? "（已超时）" : ""}
            </span>
          </div>
          <div className="host-deploy-detail__row">
            <span className="host-deploy-detail__label">目标变更</span>
            <code className="host-deploy-detail__value" title={deploy.targetCommit}>
              {deploy.previousCommit ? `${shortSha(deploy.previousCommit)} → ${shortSha(deploy.targetCommit)}` : shortSha(deploy.targetCommit)}
            </code>
          </div>
          {deploy.commitSubject ? <p className="host-deploy-detail__subject">{deploy.commitSubject}</p> : null}
          {elapsed !== null ? (
            <div className="host-deploy-detail__row">
              <span className="host-deploy-detail__label">部署耗时</span>
              <span className="host-deploy-detail__value">{formatDurationMs(elapsed)}</span>
            </div>
          ) : null}
          {deploy.finishedAt ? (
            <div className="host-deploy-detail__row">
              <span className="host-deploy-detail__label">完成时间</span>
              <span className="host-deploy-detail__value">{formatClock(deploy.finishedAt)}</span>
            </div>
          ) : null}
          {deploy.degradedExternal.length > 0 ? (
            <p className="host-deploy-detail__error">外置插件降级：{deploy.degradedExternal.join("、")}</p>
          ) : null}
          {deploy.error ? <p className="host-deploy-detail__error">{deploy.error}</p> : null}
        </>
      ) : null}
    </div>
  );
}

function RuntimeDeployMeta({ context }: { context: PluginUiContext<"header.meta"> }) {
  const { runtime, deploy } = useHostState(context);
  const dismissedId = useValue(dismissedStore);
  const deploying = isDeployInProgress(deploy);

  useEffect(() => {
    if (!runtime.commit && !deploy) {
      context.meta.publish([]);
      return;
    }
    const items: HeaderMetaItem[] = [];
    if (deploying && deploy) {
      items.push({
        id: "runtime-deploy",
        scope: "session",
        tone: "live",
        label: "版本部署",
        text: `部署 ${shortSha(deploy.targetCommit)}`,
        value: phaseShortLabel(deploy.phase),
        iconSvg: DEPLOY_ICON_SVG,
        weight: 20,
        detail: (host) => render(host, <RuntimeDeployDetail />),
      });
    } else if (deploy?.status === "failed" && shouldShowOutcomeBanner(deploy, dismissedId, Date.now())) {
      items.push({
        id: "runtime-deploy",
        scope: "account",
        tone: "alert",
        label: "部署失败",
        text: `部署 ${shortSha(deploy.targetCommit)} 失败`,
        value: "失败",
        iconSvg: DEPLOY_ICON_SVG,
        weight: 20,
        detail: (host) => render(host, <RuntimeDeployDetail />),
      });
    } else if (runtime.commit) {
      items.push({
        id: "runtime-deploy",
        scope: "account",
        tone: "info",
        label: "运行版本",
        value: shortSha(runtime.commit),
        iconSvg: DEPLOY_ICON_SVG,
        weight: 5,
        detail: (host) => render(host, <RuntimeDeployDetail />),
      });
    }
    context.meta.publish(items);
  }, [context, deploy, deploying, dismissedId, runtime.commit]);

  return null;
}

// 把服务端明确给出的状态换成提示文案；响应无效时抛错，按查询失败处理。
function noticeFor(body: Record<string, unknown>): string {
  if (!["disabled", "checking", "healthy", "degraded"].includes(String(body.state))) throw new Error("宿主状态响应无效");
  return body.state === "degraded"
    ? "宿主执行暂时不可用，正在恢复连接。依赖宿主的命令和文件操作可能失败；已有对话仍可阅读。"
    : body.state === "checking" ? "正在检查宿主执行连接…" : "";
}

const definition: PluginUiDefinition = {
  slots: {
    "chat.notice": { mount: (host, context) => render(host, <HostBridgeNotice context={context} />) },
    "header.meta": { mount: (host, context) => render(host, <RuntimeDeployMeta context={context} />) },
  },
};
export default definition;

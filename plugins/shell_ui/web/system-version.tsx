import { useCallback, useEffect, useState } from "react";
import type { WebHostContextV1 } from "@akashic/web-ui-v1";

type Http = WebHostContextV1["http"];

interface RuntimeInfo {
  commit: string | null;
  startedAt: string | null;
}

interface BridgeInfo {
  state: "disabled" | "checking" | "healthy" | "degraded";
  failures: number;
  code: string | null;
  egress: "ready" | "unavailable" | null;
}

interface DeployEntry {
  requestId: string;
  status: "submitting" | "accepted" | "running" | "active" | "failed";
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

interface VersionPayload {
  runtime: RuntimeInfo;
  bridge: BridgeInfo;
  deploy: DeployEntry | null;
  lastSuccess: DeployEntry | null;
  history: DeployEntry[];
}

const IDLE_POLL_MS = 5000;
const DEPLOY_POLL_MS = 2000;

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
}

function asString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function parseEntry(raw: unknown): DeployEntry | null {
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
    status: status as DeployEntry["status"],
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

function parsePayload(raw: unknown): VersionPayload {
  const body = asRecord(raw);
  if (!body) throw new Error("版本接口返回了无效格式");
  const runtimeRaw = asRecord(body.runtime);
  const bridgeRaw = asRecord(body.bridge);
  const historyRaw = Array.isArray(body.history) ? body.history : [];
  return {
    runtime: {
      commit: asString(runtimeRaw?.commit),
      startedAt: asString(runtimeRaw?.startedAt),
    },
    bridge: {
      state: (["disabled", "checking", "healthy", "degraded"].includes(String(bridgeRaw?.state))
        ? String(bridgeRaw?.state)
        : "disabled") as BridgeInfo["state"],
      failures: typeof bridgeRaw?.failures === "number" ? bridgeRaw.failures : 0,
      code: asString(bridgeRaw?.code),
      egress: bridgeRaw?.egress === "ready" || bridgeRaw?.egress === "unavailable" ? bridgeRaw.egress : null,
    },
    deploy: parseEntry(body.deploy),
    lastSuccess: parseEntry(body.lastSuccess),
    history: historyRaw.map(parseEntry).filter((item): item is DeployEntry => item !== null),
  };
}

function isInProgress(entry: DeployEntry | null): boolean {
  return Boolean(entry && !entry.stale && (entry.status === "submitting" || entry.status === "accepted" || entry.status === "running"));
}

function shortSha(commit: string | null): string {
  return commit ? commit.slice(0, 7) : "未知";
}

function parseMs(iso: string | null): number | null {
  if (!iso) return null;
  const parsed = Date.parse(iso);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatDateTime(iso: string | null): string {
  const ms = parseMs(iso);
  if (ms === null) return "未知";
  return new Date(ms).toLocaleString("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function formatRelative(iso: string | null, nowMs: number): string {
  const ms = parseMs(iso);
  if (ms === null) return "";
  const seconds = Math.max(0, Math.floor((nowMs - ms) / 1000));
  if (seconds < 60) return "刚刚";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} 分钟前`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} 小时前`;
  const days = Math.floor(hours / 24);
  return `${days} 天前`;
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

function entryDuration(entry: DeployEntry, nowMs: number): string | null {
  const start = parseMs(entry.startedAt ?? entry.submittedAt);
  if (start === null) return null;
  const end = entry.status === "active" || entry.status === "failed"
    ? parseMs(entry.finishedAt ?? entry.recordedAt) ?? nowMs
    : nowMs;
  return formatDurationMs(Math.max(0, end - start));
}

function phaseText(entry: DeployEntry): string {
  if (entry.stale && (entry.status === "submitting" || entry.status === "accepted" || entry.status === "running")) {
    return "已超时 / 状态未收尾";
  }
  switch (entry.phase) {
    case "submitting":
    case "accepted":
      return "排队等待构建";
    case "preparing":
      return "正在构建新版本";
    case "waiting_for_turn":
      return "等待本轮回复结束后重启";
    case "restarting":
      return "正在重启切换";
    case "active":
      return "已上线";
    case "failed":
      return entry.error && entry.error.includes("已回到原版本") ? "失败（已回退）" : "失败";
    default:
      return entry.phase;
  }
}

function bridgeText(bridge: BridgeInfo): string {
  switch (bridge.state) {
    case "healthy":
      return bridge.egress === "unavailable" ? "已连接（宿主外网出口不可用）" : "正常";
    case "checking":
      return "正在检查连接";
    case "degraded":
      return bridge.code ? `暂时不可用（${bridge.code}）` : "暂时不可用";
    case "disabled":
      return "未启用";
  }
}

export function SystemVersion({ http, active }: { http: Http; active: boolean }): React.ReactElement {
  const [data, setData] = useState<VersionPayload | null>(null);
  const [error, setError] = useState("");
  const [refreshing, setRefreshing] = useState(false);
  const [copied, setCopied] = useState(false);
  const [nowMs, setNowMs] = useState(() => Date.now());

  const load = useCallback(async (signal?: AbortSignal) => {
    setRefreshing(true);
    try {
      const response = await http.request("/api/dashboard/runtime-version", { signal, cache: "no-store" });
      if (!response.ok) {
        const body = await response.json().catch(() => ({})) as { detail?: unknown };
        throw new Error(typeof body.detail === "string" ? body.detail : `读取版本信息失败（${response.status}）`);
      }
      const next = parsePayload(await response.json());
      setData(next);
      setError("");
      setNowMs(Date.now());
    } catch (reason) {
      if (signal?.aborted) return;
      setError(reason instanceof Error ? reason.message : "读取版本信息失败");
    } finally {
      if (!signal?.aborted) setRefreshing(false);
    }
  }, [http]);

  const deploying = isInProgress(data?.deploy ?? null);

  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    void load(controller.signal);
    const interval = deploying ? DEPLOY_POLL_MS : IDLE_POLL_MS;
    const timer = window.setInterval(() => void load(controller.signal), interval);
    return () => {
      controller.abort();
      window.clearInterval(timer);
    };
  }, [active, deploying, load]);

  useEffect(() => {
    if (!active || !deploying) return;
    const timer = window.setInterval(() => setNowMs(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [active, deploying]);

  if (!data && error) {
    return (
      <div className="shell-version-page">
        <div className="shell-inventory-empty">
          <p>{error}</p>
          <button type="button" onClick={() => void load()}>重试</button>
        </div>
      </div>
    );
  }

  if (!data) {
    return <p className="shell-inventory-empty" role="status">正在读取系统版本与更新记录…</p>;
  }

  const { runtime, bridge, deploy, lastSuccess, history } = data;
  const currentCommit = runtime.commit;
  const activeUpdatedAt = lastSuccess?.finishedAt ?? lastSuccess?.recordedAt ?? runtime.startedAt;
  const activeRelative = formatRelative(activeUpdatedAt, nowMs);
  const activeDuration = lastSuccess ? entryDuration(lastSuccess, nowMs) : null;
  const currentSubject = (lastSuccess && (!currentCommit || lastSuccess.targetCommit === currentCommit) ? lastSuccess.commitSubject : null)
    ?? (deploy && (!currentCommit || deploy.targetCommit === currentCommit) ? deploy.commitSubject : null);

  const heroTone = deploying ? "live" : deploy?.status === "failed" ? "error" : "healthy";
  const heroTitle = deploying && deploy
    ? `正在更新至 ${shortSha(deploy.targetCommit)} · ${phaseText(deploy)}`
    : deploy?.status === "failed"
      ? `最近一次更新（${shortSha(deploy.targetCommit)}）未完成`
      : "当前版本运行正常";

  const copyCommit = () => {
    if (!currentCommit || !navigator.clipboard) return;
    void navigator.clipboard.writeText(currentCommit).then(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    }).catch(() => undefined);
  };

  return (
    <div className="shell-version-page">
      <section className="shell-version-hero" data-tone={heroTone}>
        <div className="shell-version-hero__head">
          <span className="shell-version-hero__status">
            <span className="shell-version-dot" data-tone={heroTone} aria-hidden="true" />
            <span>{heroTitle}</span>
          </span>
          <button type="button" className="shell-version-refresh" disabled={refreshing} onClick={() => void load()}>
            {refreshing ? "刷新中…" : "刷新"}
          </button>
        </div>

        <dl className="shell-version-grid">
          <div>
            <dt>版本号</dt>
            <dd className="shell-version-commit">
              <code title={currentCommit ?? undefined}>{shortSha(currentCommit)}</code>
              {currentCommit ? (
                <button type="button" className="shell-version-copy" onClick={copyCommit}>
                  {copied ? "已复制" : "复制"}
                </button>
              ) : null}
            </dd>
          </div>
          <div>
            <dt>更新时间</dt>
            <dd>
              <span>{formatDateTime(activeUpdatedAt)}</span>
              {activeRelative ? <span className="shell-version-muted"> · {activeRelative}</span> : null}
              {activeDuration ? <span className="shell-version-muted"> · 耗时 {activeDuration}</span> : null}
            </dd>
          </div>
          {currentSubject ? (
            <div>
              <dt>当前变更</dt>
              <dd>{currentSubject}</dd>
            </div>
          ) : null}
          {bridge.state !== "healthy" && bridge.state !== "disabled" ? (
            <div>
              <dt>宿主执行</dt>
              <dd>{bridgeText(bridge)}</dd>
            </div>
          ) : null}
        </dl>
      </section>

      {deploy && (deploying || deploy.status === "failed") ? (
        <section className="shell-version-callout" data-tone={deploying ? "live" : "error"}>
          <div className="shell-version-callout__top">
            <strong>
              {deploying ? `正在更新：${shortSha(deploy.targetCommit)}` : `更新未完成：${shortSha(deploy.targetCommit)}`}
            </strong>
            <span>{phaseText(deploy)}{entryDuration(deploy, nowMs) ? ` · ${entryDuration(deploy, nowMs)}` : ""}</span>
          </div>
          {deploy.commitSubject ? <p className="shell-version-callout__subject">{deploy.commitSubject}</p> : null}
          {deploy.error ? <p className="shell-version-callout__error">{deploy.error}</p> : null}
        </section>
      ) : null}

      {history.length > 0 ? (
        <section className="shell-version-history" aria-label="最近更新记录">
          <div className="shell-version-history__head">
            <h4>最近更新</h4>
          </div>
          <ul className="shell-version-list">
            {history.map((item) => {
              const duration = entryDuration(item, nowMs);
              const when = item.finishedAt ?? item.startedAt ?? item.submittedAt ?? item.recordedAt;
              const tone = item.status === "active" ? "healthy" : item.status === "failed" ? "error" : "live";
              return (
                <li key={item.requestId} className="shell-version-row" data-status={item.status}>
                  <span className="shell-version-dot" data-tone={tone} aria-hidden="true" />
                  <div className="shell-version-row__main">
                    <div className="shell-version-row__top">
                      <code className="shell-version-row__sha">
                        {item.previousCommit ? `${shortSha(item.previousCommit)} → ${shortSha(item.targetCommit)}` : shortSha(item.targetCommit)}
                      </code>
                      <span className="shell-version-row__status">{phaseText(item)}</span>
                    </div>
                    {item.commitSubject ? <p className="shell-version-row__subject">{item.commitSubject}</p> : null}
                    {item.degradedExternal.length > 0 ? (
                      <p className="shell-version-row__warn">外置插件降级：{item.degradedExternal.join("、")}</p>
                    ) : null}
                    {item.error ? <p className="shell-version-row__error">{item.error}</p> : null}
                  </div>
                  <div className="shell-version-row__meta">
                    <span>{formatDateTime(when)}</span>
                    {duration ? <span>耗时 {duration}</span> : null}
                  </div>
                </li>
              );
            })}
          </ul>
        </section>
      ) : null}
      {error ? <p className="shell-version-row__error" role="status">{error}</p> : null}
    </div>
  );
}

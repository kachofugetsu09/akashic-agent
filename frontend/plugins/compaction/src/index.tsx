import { useMemo, useState } from "react";
import { registerForm, ModelField, type FormProps, type Status } from "../../../../packages/akashic-plugin-kit/src/configuration";
import type { WebHostContextV1, WebUiDisposer } from "@akashic/web-ui-v1";
import "./style.css";

interface CompactionSessionOption {
  session_id: string;
  title: string;
  channel: string;
  project: string;
  message_count: number;
  updated_at: string;
  compaction_count: number;
  last_trigger: "idle" | "soft_limit" | "context_overflow" | null;
  last_tokens_before: number | null;
  last_tokens_after: number | null;
}

type IdleScope = "selected" | "all" | "off";

type CompactionStatus = Status & {
  sessions?: CompactionSessionOption[];
};

const KEEP_PRESETS: readonly { value: number; label: string }[] = [
  { value: 10_000, label: "10k 精简" },
  { value: 20_000, label: "20k 默认" },
  { value: 40_000, label: "40k 充裕" },
];

const MINUTES_PRESETS: readonly { value: number; label: string }[] = [
  { value: 30, label: "30 分钟" },
  { value: 60, label: "1 小时推荐" },
  { value: 120, label: "2 小时" },
];

const TOKENS_PRESETS: readonly { value: number; label: string }[] = [
  { value: 128_000, label: "128k 积极" },
  { value: 200_000, label: "200k 推荐" },
  { value: 300_000, label: "300k 保守" },
];

const SCOPE_OPTIONS: readonly { value: IdleScope; label: string }[] = [
  { value: "selected", label: "绑定对话" },
  { value: "all", label: "全部对话" },
  { value: "off", label: "关闭" },
];

function channelLabel(channel: string): string {
  if (channel === "akashic") return "Web";
  if (channel === "telegram") return "Telegram";
  if (channel === "qq") return "QQ";
  return channel || "Web";
}

function formatK(tokens: number): string {
  if (tokens >= 1000) return `${Math.round(tokens / 1000)}k`;
  return String(tokens);
}

function triggerLabel(trigger: CompactionSessionOption["last_trigger"]): string {
  if (trigger === "idle") return " · 空闲预压缩";
  if (trigger === "soft_limit") return " · 容量软水位";
  if (trigger === "context_overflow") return " · 容量触顶";
  return "";
}

function formatRelativeTime(iso: string): string {
  if (!iso) return "未知";
  const ts = Date.parse(iso);
  if (Number.isNaN(ts)) return iso;
  const diffMinutes = Math.max(0, Math.floor((Date.now() - ts) / 60_000));
  if (diffMinutes < 1) return "刚刚";
  if (diffMinutes < 60) return `${diffMinutes} 分钟前`;
  const diffHours = Math.floor(diffMinutes / 60);
  if (diffHours < 24) return `${diffHours} 小时前`;
  const diffDays = Math.floor(diffHours / 24);
  if (diffDays < 30) return `${diffDays} 天前`;
  const date = new Date(ts);
  return `${date.getMonth() + 1}月${date.getDate()}日`;
}

function readNumber(raw: unknown, fallback: number): number {
  return typeof raw === "number" && Number.isFinite(raw) ? raw : fallback;
}

function readSessions(raw: unknown): string[] {
  if (!Array.isArray(raw)) return [];
  return raw.filter((item): item is string => typeof item === "string" && item.trim().length > 0);
}

function PresetNumberField({
  label,
  name,
  hint,
  unit,
  min,
  step,
  value,
  fallback,
  presets,
  change,
}: {
  label: string;
  name: string;
  hint: string;
  unit: string;
  min: number;
  step: number;
  value: unknown;
  fallback: number;
  presets: readonly { value: number; label: string }[];
  change: (key: string, next: unknown) => void;
}) {
  const effective = readNumber(value, fallback);
  const inputText = value == null ? "" : String(value);
  return (
    <div className="config-field">
      <span>{label}</span>
      <small>{hint}</small>
      <div className="compaction-preset-control">
        <div className="compaction-pills" role="group" aria-label={`${label}预设档位`}>
          {presets.map((preset) => (
            <button
              key={preset.value}
              type="button"
              className="compaction-pill"
              aria-pressed={effective === preset.value}
              onClick={() => change(name, preset.value === fallback ? null : preset.value)}
            >
              {preset.label}
            </button>
          ))}
        </div>
        <div className="compaction-number-box">
          <input
            name={name}
            type="number"
            inputMode="numeric"
            min={min}
            step={step}
            value={inputText}
            placeholder={String(fallback)}
            aria-label={label}
            onChange={(event) => {
              const raw = event.target.value;
              change(name, raw === "" ? null : Number(raw));
            }}
          />
          <span className="compaction-number-unit">{unit}</span>
        </div>
      </div>
    </div>
  );
}

function CompactionFields({ values, change, status }: FormProps) {
  const compactionStatus = status as CompactionStatus;
  const defaults = status.defaults ?? {};
  const fallbackKeep = readNumber(defaults.keep_recent_tokens, 20_000);
  const fallbackMinutes = readNumber(defaults.idle_trigger_minutes, 60);
  const fallbackTokens = readNumber(defaults.idle_trigger_tokens, 200_000);
  const fallbackScope = (typeof defaults.idle_scope === "string" ? defaults.idle_scope : "selected") as IdleScope;

  const rawScope = values.idle_scope ?? fallbackScope;
  const scope: IdleScope = rawScope === "all" || rawScope === "off" ? rawScope : "selected";
  const boundIds = readSessions(values.idle_sessions ?? defaults.idle_sessions);
  const boundSet = useMemo(() => new Set(boundIds), [boundIds]);

  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState<"messages" | "recent">("messages");

  const candidates = compactionStatus.sessions ?? [];

  const filteredSessions = useMemo(() => {
    const q = search.trim().toLowerCase();
    const list = candidates.filter((item) => {
      if (!q) return true;
      return (
        item.title.toLowerCase().includes(q) ||
        channelLabel(item.channel).toLowerCase().includes(q) ||
        item.project.toLowerCase().includes(q) ||
        item.session_id.toLowerCase().includes(q)
      );
    });
    return [...list].sort((a, b) => {
      const aSelected = boundSet.has(a.session_id);
      const bSelected = boundSet.has(b.session_id);
      if (aSelected !== bSelected) return aSelected ? -1 : 1;
      if (sortBy === "recent") return b.updated_at.localeCompare(a.updated_at);
      if (b.message_count !== a.message_count) return b.message_count - a.message_count;
      return b.updated_at.localeCompare(a.updated_at);
    });
  }, [candidates, boundSet, search, sortBy]);

  const effectiveMinutes = readNumber(values.idle_trigger_minutes, fallbackMinutes);
  const effectiveTokens = readNumber(values.idle_trigger_tokens, fallbackTokens);

  const toggleSession = (sessionId: string, checked: boolean) => {
    const next = checked
      ? Array.from(new Set([...boundIds, sessionId]))
      : boundIds.filter((id) => id !== sessionId);
    change("idle_sessions", next);
  };

  const recommendLongSessions = () => {
    const longOnes = candidates
      .filter((item) => item.message_count >= 500 || item.compaction_count > 0)
      .map((item) => item.session_id);
    const picked = longOnes.length > 0
      ? longOnes
      : candidates.filter((item) => item.message_count > 0).slice(0, 3).map((item) => item.session_id);
    change("idle_sessions", Array.from(new Set([...boundIds, ...picked])));
  };

  const strategyText = (() => {
    if (scope === "off" || effectiveMinutes <= 0) {
      return "当前策略：已关闭空闲预压缩，仅在模型上下文触及 85% 容量软水位时自动压缩。";
    }
    if (scope === "all") {
      return `当前策略：对全部对话生效，闲置 ≥ ${effectiveMinutes} 分钟且上下文 ≥ ${effectiveTokens.toLocaleString()} tokens 时自动预压缩。`;
    }
    if (boundIds.length === 0) {
      return "当前策略：绑定对话模式下尚未勾选对话；请从上方列表勾选常聊长对话，或点击「一键勾选常聊长对话」。";
    }
    return `当前策略：仅对已绑定的 ${boundIds.length} 个常驻对话，在闲置 ≥ ${effectiveMinutes} 分钟且上下文 ≥ ${effectiveTokens.toLocaleString()} tokens 时自动预压缩。`;
  })();

  return (
    <>
      <ModelField
        label="摘要模型"
        name="model"
        value={values.model}
        change={change}
        status={status}
        hint="用来把早期对话浓缩成摘要；默认跟随当前对话，也可指定更便宜的 Flash 模型"
      />

      <PresetNumberField
        label="保留最近原文"
        name="keep_recent_tokens"
        hint="每次压缩后末尾至少留下的原文容量；当前输入和进行中的工具调用始终完整保留"
        unit="tokens"
        min={1}
        step={1000}
        value={values.keep_recent_tokens}
        fallback={fallbackKeep}
        presets={KEEP_PRESETS}
        change={change}
      />

      <div className="config-field">
        <span>空闲预压缩范围</span>
        <small>对话闲置超过缓存有效期后，赶在下次发消息前先把旧历史压成摘要，避免长上下文冷启动重算</small>
        <div className="compaction-scope-tabs" role="group" aria-label="空闲预压缩范围">
          {SCOPE_OPTIONS.map((option) => (
            <button
              key={option.value}
              type="button"
              className="compaction-scope-tab"
              aria-pressed={scope === option.value}
              onClick={() => change("idle_scope", option.value === fallbackScope ? null : option.value)}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      {scope !== "off" && (
        <>
          <PresetNumberField
            label="空闲等待时长"
            name="idle_trigger_minutes"
            hint="距上一轮回复结束超过此时长（KV 缓存已过期），新发消息时才允许触发预压缩"
            unit="分钟"
            min={0}
            step={5}
            value={values.idle_trigger_minutes}
            fallback={fallbackMinutes}
            presets={MINUTES_PRESETS}
            change={change}
          />

          <PresetNumberField
            label="触发上下文体积"
            name="idle_trigger_tokens"
            hint="当前请求累计达到该体积才在空闲后压缩；短对话不打断"
            unit="tokens"
            min={1}
            step={10_000}
            value={values.idle_trigger_tokens}
            fallback={fallbackTokens}
            presets={TOKENS_PRESETS}
            change={change}
          />
        </>
      )}

      {scope === "selected" && (
        <div className="compaction-sessions-panel">
          <div className="compaction-sessions-head">
            <div className="compaction-sessions-title">
              <strong>绑定开启空闲预压缩的对话</strong>
              <span className={`compaction-bound-count${boundIds.length === 0 ? " is-empty" : ""}`}>
                {boundIds.length > 0 ? `已绑定 ${boundIds.length} 个对话` : "未绑定对话"}
              </span>
            </div>
            <div className="compaction-sessions-actions">
              {candidates.length > 0 && (
                <button
                  type="button"
                  className="compaction-mini-btn"
                  onClick={recommendLongSessions}
                >
                  一键勾选常聊长对话
                </button>
              )}
              {boundIds.length > 0 && (
                <button
                  type="button"
                  className="compaction-mini-btn"
                  onClick={() => change("idle_sessions", [])}
                >
                  清空
                </button>
              )}
            </div>
          </div>

          {candidates.length > 0 && (
            <div className="compaction-sessions-toolbar">
              <input
                type="search"
                className="compaction-search"
                value={search}
                placeholder="搜索对话标题、渠道或项目…"
                aria-label="搜索可绑定的对话"
                onChange={(event) => setSearch(event.target.value)}
              />
              <select
                className="compaction-sort-select"
                value={sortBy}
                aria-label="对话排序方式"
                onChange={(event) => setSortBy(event.target.value as "messages" | "recent")}
              >
                <option value="messages">按消息规模排序</option>
                <option value="recent">按最近活跃排序</option>
              </select>
            </div>
          )}

          <div className="compaction-session-list" role="group" aria-label="可绑定的对话列表">
            {filteredSessions.length === 0 ? (
              <p className="compaction-session-empty">
                {candidates.length === 0 ? "暂无可绑定的对话记录" : "没有匹配的对话"}
              </p>
            ) : (
              filteredSessions.map((item) => {
                const selected = boundSet.has(item.session_id);
                const isLong = item.message_count >= 500 || item.compaction_count >= 3;
                const compactDetail =
                  item.compaction_count > 0
                    ? item.last_tokens_before != null && item.last_tokens_after != null
                      ? `已压缩 ${item.compaction_count} 次（上次 ${formatK(item.last_tokens_before)} → ${formatK(item.last_tokens_after)}${triggerLabel(item.last_trigger)}）`
                      : `已压缩 ${item.compaction_count} 次`
                    : "尚未压缩";
                return (
                  <label
                    key={item.session_id}
                    className={`compaction-session-row${selected ? " is-selected" : ""}`}
                  >
                    <input
                      type="checkbox"
                      checked={selected}
                      onChange={(event) => toggleSession(item.session_id, event.target.checked)}
                    />
                    <div className="compaction-session-main">
                      <div className="compaction-session-line1">
                        <span className="compaction-session-name" title={item.title}>
                          {item.title}
                        </span>
                        <span className="compaction-tag">{channelLabel(item.channel)}</span>
                        {item.project && <span className="compaction-tag">{item.project}</span>}
                        {isLong && <span className="compaction-tag is-recommend">常聊长对话</span>}
                      </div>
                      <div className="compaction-session-meta">
                        <span>{item.message_count.toLocaleString()} 条消息</span>
                        <span className="compaction-meta-dot">{compactDetail}</span>
                        {item.updated_at && (
                          <span className="compaction-meta-dot" title={item.updated_at}>
                            活跃于 {formatRelativeTime(item.updated_at)}
                          </span>
                        )}
                      </div>
                    </div>
                    <span className="compaction-state-pill">{selected ? "已绑定" : "未绑定"}</span>
                  </label>
                );
              })
            )}
          </div>
        </div>
      )}

      <p className="compaction-strategy-note" role="status">
        {strategyText}
      </p>
    </>
  );
}

export function activate(ctx: WebHostContextV1): WebUiDisposer {
  return registerForm(ctx, {
    id: "compaction",
    title: "上下文压缩",
    description: "对话接近模型窗口上限、或绑定的常驻对话在长空闲后重新发消息时，把较早的历史浓缩成摘要。",
    family: "memory",
    familyLabel: "记忆",
    switchable: false,
    fields: (props) => <CompactionFields {...props} />,
  });
}

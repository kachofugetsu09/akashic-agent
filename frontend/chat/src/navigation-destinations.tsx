import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import type { DestinationControls, DestinationForm, NavigationDestination } from "@akashic/chat-ui-v1";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { DestinationMount, destinationsPending, identityDestinationSource, type DestinationSource } from "./plugin-ui-runtime";
import { NavigationRowMenu, type NavigationRowAction } from "./navigation-row-menu";
import { SessionStatusMark, type SessionStatus } from "./session-status-mark";
import "./navigation-destinations.css";

/** 侧栏 identity 目的地的宿主状态；数据全部来自插件的 navigation.destinations。 */
export interface IdentityDestinationState {
  source: DestinationSource | null;
  /** 当前目的地；读到之前先用本机上次的结果，免得品牌行启动时跳一下。 */
  value: NavigationDestination | null;
  /** 已读到（或已确定没有数据源）；之前不显示提示。 */
  loaded: boolean;
  /** 每次重读递增；提示区据此重新读取。 */
  revision: number;
  reload(): void;
  /** 打开目的地；省略 sessionId 时开始新对话，首条消息后交回插件。 */
  open(sessionId?: string): void;
}

const SNAPSHOT_KEY = "akashic.chat.navigation-destination.v1";
const LOAD_TIMEOUT_MS = 20_000;

function checkDestination(value: unknown): NavigationDestination | null {
  if (value === null) return null;
  const row = typeof value === "object" ? value as Partial<Record<keyof NavigationDestination, unknown>> : null;
  const text = (item: unknown) => item === undefined || typeof item === "string";
  if (!row || typeof row.label !== "string" || !row.label || !text(row.sessionId) || !text(row.preview) || !text(row.hint)) {
    throw new Error("目的地记录无效");
  }
  return { label: row.label, ...(row.sessionId ? { sessionId: row.sessionId } : {}),
    ...(row.preview ? { preview: row.preview } : {}), ...(row.hint ? { hint: row.hint } : {}) };
}

function readSnapshot(): NavigationDestination | null {
  try { return checkDestination(JSON.parse(localStorage.getItem(SNAPSHOT_KEY) ?? "null")); }
  catch { return null; }
}

function writeSnapshot(value: NavigationDestination | null) {
  try { localStorage.setItem(SNAPSHOT_KEY, JSON.stringify(value)); }
  catch { /* 存储不可用时下次照常等网络 */ }
}

// 目录或重读请求变化时读取第一个 identity 数据源；失败只报一次错并保留上次结果。
export function useIdentityDestination(enabled: boolean, catalogVersion: number, reportError: (error: unknown) => void) {
  const [source, setSource] = useState<DestinationSource | null>(null);
  const [value, setValue] = useState<NavigationDestination | null>(readSnapshot);
  const [loaded, setLoaded] = useState(false);
  const [revision, setRevision] = useState(0);
  const report = useRef(reportError);
  useEffect(() => { report.current = reportError; }, [reportError]);
  // 1. 目录换代时对齐数据源；目录定了且没有数据源时品牌行回到只有 logo。
  useEffect(() => {
    const next = identityDestinationSource();
    setSource((current) => current?.key === next?.key ? current : next);
    if (!next && !destinationsPending()) {
      setValue(null);
      setLoaded(true);
      writeSnapshot(null);
    }
  }, [catalogVersion]);
  // 2. 有数据源且聊天就绪时读取；换代或重读时取消上一次。
  useEffect(() => {
    if (!source || !enabled) return;
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(new Error("读取超时")), LOAD_TIMEOUT_MS);
    void source.provider.load(source.context(controller.signal))
      .then((result) => {
        const next = checkDestination(result);
        if (controller.signal.aborted) return;
        setValue(next);
        setLoaded(true);
        writeSnapshot(next);
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted && controller.signal.reason?.message !== "读取超时") return;
        report.current(new Error(`侧栏顶部的对话读取失败：${error instanceof Error ? error.message : String(error)}`));
      })
      .finally(() => window.clearTimeout(timer));
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [source, enabled, revision]);
  const reload = useCallback(() => setRevision((current) => current + 1), []);
  const shown = source || !loaded ? value : null;
  return useMemo(() => ({ source, value: shown, loaded, revision, reload }), [source, shown, loaded, revision, reload]);
}

// 联系人行的时间：今天写钟点，一周内写星期，更早写日期。
export function contactTime(iso: string | undefined, now = new Date()): string {
  const at = iso ? new Date(iso) : null;
  if (!at || Number.isNaN(at.getTime())) return "";
  const day = (value: Date) => new Date(value.getFullYear(), value.getMonth(), value.getDate()).getTime();
  const days = Math.round((day(now) - day(at)) / 86_400_000);
  if (days <= 0) return at.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", hour12: false });
  if (days === 1) return "昨天";
  if (days < 7) return `周${"日一二三四五六"[at.getDay()]}`;
  return `${at.getMonth() + 1}月${at.getDate()}日`;
}

/** 品牌行变成的联系人行：头像、名字、时间或状态、最近一句；菜单与会话行同一套。 */
export function IdentityDestinationRow({ destination, mark, status, updatedAt, active, actions, onOpen }: {
  destination: NavigationDestination;
  mark: CSSProperties;
  status?: SessionStatus;
  updatedAt?: string;
  active: boolean;
  actions: NavigationRowAction[];
  onOpen(): void;
}) {
  const time = destination.sessionId ? contactTime(updatedAt) : "";
  const preview = destination.preview || destination.hint || "";
  return <NavigationRowMenu title={destination.label} actions={actions} className="destination-row">
    <button type="button" className="destination-row__open" data-active={active || undefined}
      aria-current={active ? "page" : undefined} data-unread={status === "unread" || undefined} onClick={onOpen}>
      <span className="destination-row__avatar" aria-hidden="true"><span className="chat-sidebar__mark" style={mark} /></span>
      <span className="destination-row__copy">
        <span className="destination-row__name">
          <strong>{destination.label}</strong>
          {status ? <SessionStatusMark status={status} /> : time ? <time dateTime={updatedAt}>{time}</time> : null}
        </span>
        {preview ? <span className="destination-row__preview">{preview}</span> : null}
      </span>
    </button>
  </NavigationRowMenu>;
}

/** 没有目的地时品牌行下面的插件提示。 */
export function DestinationNotice({ state, controls }: { state: IdentityDestinationState; controls: DestinationControls }) {
  const notice = state.source?.provider.notice;
  if (!state.source || !notice || state.value || !state.loaded) return null;
  return <div className="destination-notice">
    <DestinationMount source={state.source} renderer={notice} destination={controls} refreshToken={state.revision} />
  </div>;
}

/** 插件的目的地对话框：宿主给外框与标题，关闭后重读目的地。 */
export function DestinationFormDialog({ source, form, controls, onClose }: {
  source: DestinationSource;
  form: DestinationForm;
  controls: DestinationControls;
  onClose(): void;
}) {
  return <Dialog open onOpenChange={(open) => { if (!open) onClose(); }}>
    <DialogContent className="group-dialog destination-dialog" overlayClassName="group-dialog-overlay" aria-describedby={undefined}>
      <DialogHeader className="group-dialog__header"><DialogTitle>{form.title}</DialogTitle></DialogHeader>
      <DestinationMount source={source} renderer={form} destination={controls} close={onClose} />
    </DialogContent>
  </Dialog>;
}

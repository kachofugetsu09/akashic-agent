import "./mobile-polyfills";

import React, { type ReactNode, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  animate,
  motion,
  useMotionValue,
  useMotionValueEvent,
  useReducedMotion,
  useTransform,
} from "motion/react";
import { useStickToBottomContext } from "use-stick-to-bottom";
import {
  AlertCircle,
  ArrowLeft,
  Check,
  ChevronRight,
  ChevronDown,
  ChevronUp,
  Copy,
  Download,
  FileText,
  Menu,
  MessageSquarePlus,
  Paperclip,
  Puzzle,
  RefreshCw,
  Reply,
  RotateCcw,
  Search,
  SendHorizontal,
  Share2,
  Square,
  TimerReset,
  Wifi,
  WifiOff,
  X,
} from "lucide-react";
import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
} from "@/components/ai-elements/conversation";
import { TooltipProvider } from "@/components/ui/tooltip";
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { ChatMessageView } from "./message-view";
import {
  MobilePluginDashboard,
  MobilePluginSlot,
  receiveMobilePluginAssets,
  settleMobilePluginResponses,
  type MobilePluginAsset,
  type MobilePluginDashboardEntry,
  useMobilePluginDashboards,
} from "./mobile-plugin-runtime";
import {
  advanceMobileProjectionBaseline,
  advanceMobileUnreadTracking,
  allMobileAttachmentsReady,
  isMobileImageViewerHistoryState,
  normalizeMobileSearchText,
  updateMobileSearchIndex,
  type MobileSearchIndexEntry,
} from "./mobile-message-state";
import type { AgentBlock, ChatMessage } from "./main";
import {
  pushMobileSurface,
  readMobileSurfaceHistoryState,
  replaceMobileSurface,
  type MobileSurface,
} from "./mobile-surface-history";
import "./mobile-native.css";

type ConnectionStatus = "connecting" | "ready" | "degraded" | "reconnecting" | "disconnected";
type ProcessState = "completed" | "running" | "failed";

interface MobileAttachment {
  id: string;
  filename: string;
  contentType: string;
  sizeBytes: number;
  transferredBytes: number;
  state: string;
  canRemove?: boolean;
  contentUrl?: string;
}

interface MobileTransferStatus {
  title: string;
  detail: string;
  progressPercent: number;
  requiresMeteredApproval: boolean;
}

interface MobileProcessBlock {
  id: string;
  kind: "thinking" | "tool";
  title: string;
  detail: string;
  state: ProcessState;
  arguments?: SnapshotRecord;
  resultPreview?: string;
  durationMillis?: number;
}

interface MobileMessage {
  id: string;
  sessionId: string;
  role: "user" | "assistant";
  content: string;
  createdAt: number;
  searchRevision: number;
  replyable: boolean;
  reply?: MobileReply;
  deliveryLabel?: string;
  deliveryAction?: "retry" | "verify";
  blocks: MobileProcessBlock[];
  streaming: boolean;
  interrupted: boolean;
  durationSeconds?: number;
  attachments: MobileAttachment[];
}

interface MobileReply {
  messageId: string;
  role: "user" | "assistant";
  preview: string;
}

interface MobileUnreadState {
  firstMessageId?: string;
  anchorKey?: string;
  count: number;
}

interface MobileSession {
  id: string;
  title: string;
  lastMessagePreview?: string;
  lastMessageAt?: number;
  unreadCount: number;
  isRunning: boolean;
}

interface MobileReadingPosition {
  messageId: string;
  offsetPx: number;
}

interface MobileNavigationTarget {
  sessionId: string;
  messageId: string;
}

interface MobilePendingMessage {
  messageId: string;
  preview: string;
  createdAt: number;
}

export interface MobileSnapshot {
  protocolVersion: 3;
  connection: {
    label: string;
    status: ConnectionStatus;
    notice?: string;
    error?: string;
  };
  sessions: MobileSession[];
  selectedSessionId?: string;
  readingPosition?: MobileReadingPosition;
  navigationTarget?: MobileNavigationTarget;
  projectionGeneration: number;
  messages: MobileMessage[];
  pluginResponses: { requestId: string; resultJson?: string; error?: string }[];
  composer: {
    attachments: MobileAttachment[];
    pendingMessages: MobilePendingMessage[];
    transferStatus?: MobileTransferStatus;
    commands: { command: string; description: string }[];
    isStreaming: boolean;
    isResyncing: boolean;
    canResync: boolean;
    isStopping: boolean;
    canStop: boolean;
    canSend: boolean;
  };
}

interface NativeBridge {
  reportReady(): void;
  requestSnapshot(): void;
  selectSession(sessionId: string): void;
  createSession(): void;
  restartPairing(): void;
  reloadFromServer(): void;
  chooseAttachments(): void;
  removeAttachment(attachmentId: string): void;
  retryAttachment(attachmentId: string): void;
  continueMeteredTransfer(): void;
  retryFailedMessage(messageId: string): void;
  saveReadingPosition(sessionId: string, messageId: string, offsetPx: number): void;
  markSessionReadThrough(sessionId: string, readAtMillis: number): void;
  navigationTargetHandled(messageId: string): void;
  retryDownloadedAttachment(attachmentId: string): void;
  touchDownloadedAttachment(attachmentId: string): void;
  openDownloadedAttachment(attachmentId: string): void;
  shareDownloadedAttachment(attachmentId: string): void;
  saveDownloadedAttachment(attachmentId: string): void;
  setWebHistoryActive(active: boolean): void;
  dismissError(): void;
  sendMessage(requestId: string, text: string, replyToMessageId: string, attachmentIdsJson: string): void;
  copyText(text: string): void;
  performActionHaptic(): void;
  sendCommand(command: string): void;
  stopTurn(): void;
  callPluginUi(
    requestId: string,
    sessionId: string | null,
    turnId: string | null,
    pluginId: string,
    method: string,
    payloadJson: string,
  ): void;
  acknowledgePluginUiResponses(requestIdsJson: string): void;
}

type SnapshotRecord = Record<string, unknown>;

function requireRecord(value: unknown, label: string): SnapshotRecord {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} 不是对象`);
  return value as SnapshotRecord;
}

function requireString(value: unknown, label: string): string {
  if (typeof value !== "string") throw new Error(`${label} 不是字符串`);
  return value;
}

function requireBoolean(value: unknown, label: string): boolean {
  if (typeof value !== "boolean") throw new Error(`${label} 不是布尔值`);
  return value;
}

function requireNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${label} 不是有效数字`);
  return value;
}

function requireNonNegativeInteger(value: unknown, label: string): number {
  const parsed = requireNumber(value, label);
  if (!Number.isSafeInteger(parsed) || parsed < 0) throw new Error(`${label} 必须是非负安全整数`);
  return parsed;
}

function requireInteger(value: unknown, label: string): number {
  const parsed = requireNumber(value, label);
  if (!Number.isSafeInteger(parsed)) throw new Error(`${label} 必须是安全整数`);
  return parsed;
}

function optionalString(value: unknown, label: string): string | undefined {
  if (value === undefined || value === null) return undefined;
  return requireString(value, label);
}

function optionalRecord(value: unknown, label: string): SnapshotRecord | undefined {
  if (value === undefined || value === null) return undefined;
  return requireRecord(value, label);
}

function requireArray<T>(value: unknown, label: string, parse: (item: unknown, index: number) => T): T[] {
  if (!Array.isArray(value)) throw new Error(`${label} 不是数组`);
  return value.map(parse);
}

function optionalArray<T>(value: unknown, label: string, parse: (item: unknown, index: number) => T): T[] {
  return value === undefined ? [] : requireArray(value, label, parse);
}

function parseAttachment(value: unknown, index: number): MobileAttachment {
  const raw = requireRecord(value, `attachments[${index}]`);
  return {
    id: requireString(raw.id, `attachments[${index}].id`),
    filename: requireString(raw.filename, `attachments[${index}].filename`),
    contentType: requireString(raw.contentType, `attachments[${index}].contentType`),
    sizeBytes: requireNumber(raw.sizeBytes, `attachments[${index}].sizeBytes`),
    transferredBytes: requireNumber(raw.transferredBytes, `attachments[${index}].transferredBytes`),
    state: requireString(raw.state, `attachments[${index}].state`),
    canRemove: raw.canRemove === undefined ? false : requireBoolean(raw.canRemove, `attachments[${index}].canRemove`),
    contentUrl: optionalString(raw.contentUrl, `attachments[${index}].contentUrl`),
  };
}

function parseProcessBlock(value: unknown, index: number): MobileProcessBlock {
  const raw = requireRecord(value, `blocks[${index}]`);
  const kind = requireString(raw.kind, `blocks[${index}].kind`);
  const state = requireString(raw.state, `blocks[${index}].state`);
  if (kind !== "thinking" && kind !== "tool") throw new Error(`blocks[${index}].kind 不受支持`);
  if (state !== "completed" && state !== "running" && state !== "failed") {
    throw new Error(`blocks[${index}].state 不受支持`);
  }
  const durationMillis = raw.durationMillis === undefined || raw.durationMillis === null
    ? undefined
    : requireNumber(raw.durationMillis, `blocks[${index}].durationMillis`);
  if (durationMillis !== undefined && (!Number.isSafeInteger(durationMillis) || durationMillis < 0)) {
    throw new Error(`blocks[${index}].durationMillis 必须是非负安全整数`);
  }
  return {
    id: requireString(raw.id, `blocks[${index}].id`),
    kind,
    title: requireString(raw.title, `blocks[${index}].title`),
    detail: requireString(raw.detail, `blocks[${index}].detail`),
    state,
    arguments: optionalRecord(raw.arguments, `blocks[${index}].arguments`),
    resultPreview: optionalString(raw.resultPreview, `blocks[${index}].resultPreview`),
    durationMillis,
  };
}

function parseMessage(value: unknown, index: number): MobileMessage {
  const raw = requireRecord(value, `messages[${index}]`);
  const role = requireString(raw.role, `messages[${index}].role`);
  if (role !== "user" && role !== "assistant") throw new Error(`messages[${index}].role 不受支持`);
  const createdAt = requireNumber(raw.createdAt, `messages[${index}].createdAt`);
  if (createdAt <= 0) throw new Error(`messages[${index}].createdAt 必须是正数`);
  const reply = raw.reply === undefined || raw.reply === null
    ? undefined
    : parseReply(raw.reply, `messages[${index}].reply`);
  const deliveryAction = optionalString(raw.deliveryAction, `messages[${index}].deliveryAction`);
  if (deliveryAction !== undefined && deliveryAction !== "retry" && deliveryAction !== "verify") {
    throw new Error(`messages[${index}].deliveryAction 不受支持`);
  }
  return {
    id: requireString(raw.id, `messages[${index}].id`),
    sessionId: requireString(raw.sessionId, `messages[${index}].sessionId`),
    role,
    content: requireString(raw.content, `messages[${index}].content`),
    createdAt,
    searchRevision: requireNumber(raw.searchRevision, `messages[${index}].searchRevision`),
    replyable: requireBoolean(raw.replyable, `messages[${index}].replyable`),
    reply,
    deliveryLabel: optionalString(raw.deliveryLabel, `messages[${index}].deliveryLabel`),
    deliveryAction,
    blocks: optionalArray(raw.blocks, `messages[${index}].blocks`, parseProcessBlock),
    streaming: raw.streaming === undefined ? false : requireBoolean(raw.streaming, `messages[${index}].streaming`),
    interrupted: raw.interrupted === undefined ? false : requireBoolean(raw.interrupted, `messages[${index}].interrupted`),
    durationSeconds: raw.durationSeconds === undefined ? undefined : requireNumber(raw.durationSeconds, `messages[${index}].durationSeconds`),
    attachments: optionalArray(raw.attachments, `messages[${index}].attachments`, parseAttachment),
  };
}

function parseReply(value: unknown, label: string): MobileReply {
  const raw = requireRecord(value, label);
  const role = requireString(raw.role, `${label}.role`);
  if (role !== "user" && role !== "assistant") throw new Error(`${label}.role 不受支持`);
  return {
    messageId: requireString(raw.messageId, `${label}.messageId`),
    role,
    preview: requireString(raw.preview, `${label}.preview`),
  };
}

function parseTransferStatus(value: unknown): MobileTransferStatus {
  const raw = requireRecord(value, "composer.transferStatus");
  const progressPercent = requireNumber(raw.progressPercent, "composer.transferStatus.progressPercent");
  if (!Number.isInteger(progressPercent) || progressPercent < 0 || progressPercent > 100) {
    throw new Error("composer.transferStatus.progressPercent 必须是 0..100 的整数");
  }
  return {
    title: requireString(raw.title, "composer.transferStatus.title"),
    detail: requireString(raw.detail, "composer.transferStatus.detail"),
    progressPercent,
    requiresMeteredApproval: requireBoolean(
      raw.requiresMeteredApproval,
      "composer.transferStatus.requiresMeteredApproval",
    ),
  };
}

/** 在 native 协议边界校验完整快照，并只补齐 Kotlin 明确定义的默认字段。 */
function parseMobileSnapshot(value: unknown): MobileSnapshot {
  // 1. 校验协议版本与根对象
  const raw = requireRecord(value, "snapshot");
  if (raw.protocolVersion !== 3) throw new Error(`不支持的移动端协议版本: ${String(raw.protocolVersion)}`);
  const connection = requireRecord(raw.connection, "connection");
  const status = requireString(connection.status, "connection.status");
  if (!["connecting", "ready", "degraded", "reconnecting", "disconnected"].includes(status)) {
    throw new Error(`connection.status 不受支持: ${status}`);
  }

  // 2. 校验会话、消息和插件响应
  const sessions = requireArray(raw.sessions, "sessions", (item, index) => {
    const session = requireRecord(item, `sessions[${index}]`);
    const lastMessageAt = session.lastMessageAt === undefined || session.lastMessageAt === null
      ? undefined
      : requireNonNegativeInteger(session.lastMessageAt, `sessions[${index}].lastMessageAt`);
    return {
      id: requireString(session.id, `sessions[${index}].id`),
      title: requireString(session.title, `sessions[${index}].title`),
      lastMessagePreview: optionalString(session.lastMessagePreview, `sessions[${index}].lastMessagePreview`),
      lastMessageAt,
      unreadCount: requireNonNegativeInteger(session.unreadCount, `sessions[${index}].unreadCount`),
      isRunning: requireBoolean(session.isRunning, `sessions[${index}].isRunning`),
    };
  });
  const messages = requireArray(raw.messages, "messages", parseMessage);
  const pluginResponses = requireArray(raw.pluginResponses, "pluginResponses", (item, index) => {
    const response = requireRecord(item, `pluginResponses[${index}]`);
    return {
      requestId: requireString(response.requestId, `pluginResponses[${index}].requestId`),
      resultJson: optionalString(response.resultJson, `pluginResponses[${index}].resultJson`),
      error: optionalString(response.error, `pluginResponses[${index}].error`),
    };
  });

  // 3. 校验输入区状态并构造内部类型
  const composer = requireRecord(raw.composer, "composer");
  const projectionGeneration = requireNumber(raw.projectionGeneration, "projectionGeneration");
  if (!Number.isSafeInteger(projectionGeneration) || projectionGeneration < 0) {
    throw new Error("projectionGeneration 必须是非负安全整数");
  }
  const readingPosition = raw.readingPosition === undefined || raw.readingPosition === null
    ? undefined
    : (() => {
      const position = requireRecord(raw.readingPosition, "readingPosition");
      return {
        messageId: requireString(position.messageId, "readingPosition.messageId"),
        offsetPx: requireInteger(position.offsetPx, "readingPosition.offsetPx"),
      };
    })();
  const navigationTarget = raw.navigationTarget === undefined || raw.navigationTarget === null
    ? undefined
    : (() => {
      const target = requireRecord(raw.navigationTarget, "navigationTarget");
      return {
        sessionId: requireString(target.sessionId, "navigationTarget.sessionId"),
        messageId: requireString(target.messageId, "navigationTarget.messageId"),
      };
    })();
  return {
    protocolVersion: 3,
    connection: {
      label: requireString(connection.label, "connection.label"),
      status: status as ConnectionStatus,
      notice: optionalString(connection.notice, "connection.notice"),
      error: optionalString(connection.error, "connection.error"),
    },
    sessions,
    selectedSessionId: optionalString(raw.selectedSessionId, "selectedSessionId"),
    readingPosition,
    navigationTarget,
    projectionGeneration,
    messages,
    pluginResponses,
    composer: {
      attachments: requireArray(composer.attachments, "composer.attachments", parseAttachment),
      pendingMessages: requireArray(composer.pendingMessages, "composer.pendingMessages", (item, index) => {
        const pending = requireRecord(item, `composer.pendingMessages[${index}]`);
        return {
          messageId: requireString(pending.messageId, `composer.pendingMessages[${index}].messageId`),
          preview: requireString(pending.preview, `composer.pendingMessages[${index}].preview`),
          createdAt: requireNonNegativeInteger(pending.createdAt, `composer.pendingMessages[${index}].createdAt`),
        };
      }),
      transferStatus: composer.transferStatus === null || composer.transferStatus === undefined
        ? undefined
        : parseTransferStatus(composer.transferStatus),
      commands: requireArray(composer.commands, "composer.commands", (item, index) => {
        const command = requireRecord(item, `composer.commands[${index}]`);
        return {
          command: requireString(command.command, `composer.commands[${index}].command`),
          description: requireString(command.description, `composer.commands[${index}].description`),
        };
      }),
      isStreaming: requireBoolean(composer.isStreaming, "composer.isStreaming"),
      isResyncing: requireBoolean(composer.isResyncing, "composer.isResyncing"),
      canResync: requireBoolean(composer.canResync, "composer.canResync"),
      isStopping: requireBoolean(composer.isStopping, "composer.isStopping"),
      canStop: requireBoolean(composer.canStop, "composer.canStop"),
      canSend: requireBoolean(composer.canSend, "composer.canSend"),
    },
  };
}

declare global {
  interface Window {
    AkashicNative?: NativeBridge;
    AkashicMobile?: {
      receiveSnapshot(snapshot: unknown): void;
      receivePluginAssets(assets: MobilePluginAsset[]): void;
      receiveSendResult(requestId: string, accepted: boolean): void;
      navigateBack(): boolean;
    };
  }
}

function MobileNativeApp() {
  const pluginDashboards = useMobilePluginDashboards();
  const [snapshot, setSnapshot] = useState<MobileSnapshot | null>(null);
  const [surface, setSurface] = useState<MobileSurface>({ kind: "chat" });
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [commandsOpen, setCommandsOpen] = useState(false);
  const [queueOpen, setQueueOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchTargetId, setSearchTargetId] = useState<string | null>(null);
  const [highlightedMessageId, setHighlightedMessageId] = useState<string | null>(null);
  const [unreadState, setUnreadState] = useState<MobileUnreadState>({ count: 0 });
  const [unreadAnchorVisited, setUnreadAnchorVisited] = useState(false);
  const [input, setInput] = useState("");
  const [replyTarget, setReplyTarget] = useState<MobileMessage | null>(null);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [recoveringMessageIds, setRecoveringMessageIds] = useState<Set<string>>(() => new Set());
  const [stopRequested, setStopRequested] = useState(false);
  const [sendPending, setSendPending] = useState(false);
  const [sendScrollRequest, setSendScrollRequest] = useState(0);
  const [pluginLoadError, setPluginLoadError] = useState<string | null>(null);
  const [snapshotError, setSnapshotError] = useState<string | null>(null);
  const [searchIndex, setSearchIndex] = useState(new Map<string, MobileSearchIndexEntry>());
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const drawerToggleRef = useRef<HTMLButtonElement>(null);
  const searchButtonRef = useRef<HTMLButtonElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const searchOpenRef = useRef(false);
  const normalizedSearchQueryRef = useRef("");
  const messageElementsRef = useRef(new Map<string, HTMLDivElement>());
  const copiedTimerRef = useRef<number | null>(null);
  const searchFocusTimerRef = useRef<number | null>(null);
  const searchHighlightTimerRef = useRef<number | null>(null);
  const previousSessionIdRef = useRef<string | undefined>(undefined);
  const handledNavigationTargetRef = useRef<string | undefined>(undefined);
  const pendingSendRequestRef = useRef<string | null>(null);
  const surfaceRef = useRef<MobileSurface>({ kind: "chat" });

  useEffect(() => {
    let pending: MobileSnapshot | null = null;
    let frame: number | null = null;
    let requestTimer: number | null = null;
    let snapshotAccepted = false;
    const requestSnapshot = () => {
      if (snapshotAccepted) return;
      window.AkashicNative?.requestSnapshot();
      requestTimer = window.setTimeout(requestSnapshot, 250);
    };
    replaceMobileSurface(window.history, { kind: "chat" });
    const handlePopState = (event: PopStateEvent) => {
      const next = readMobileSurfaceHistoryState(event.state);
      surfaceRef.current = next;
      setSurface(next);
    };
    window.addEventListener("popstate", handlePopState);
    window.AkashicMobile = {
      receiveSnapshot(next) {
        try {
          const parsed = parseMobileSnapshot(next);
          settleMobilePluginResponses(parsed.pluginResponses);
          pending = parsed;
          setSnapshotError(null);
        } catch (error) {
          console.error("[mobile] rejected native snapshot", error);
          setSnapshotError(error instanceof Error ? error.message : "原生快照无效");
          return;
        }
        if (frame !== null) return;
        frame = requestAnimationFrame(() => {
          frame = null;
          const nextSnapshot = pending;
          if (nextSnapshot === null) throw new Error("移动端快照帧缺少待发布数据");
          pending = null;
          if (searchOpenRef.current && normalizedSearchQueryRef.current) {
            setSearchIndex((current) => updateMobileSearchIndex(
              current,
              nextSnapshot.messages,
              normalizedSearchQueryRef.current,
              false,
            ));
          }
          setSnapshot(nextSnapshot);
          snapshotAccepted = true;
          if (requestTimer !== null) window.clearTimeout(requestTimer);
        });
      },
      receivePluginAssets(assets) {
        void receiveMobilePluginAssets(assets).then(
          () => setPluginLoadError(null),
          (error: unknown) => {
            console.error("[mobile] failed to load plugin assets", error);
            setPluginLoadError(error instanceof Error ? error.message : "插件界面加载失败");
          },
        );
      },
      receiveSendResult(requestId, accepted) {
        if (pendingSendRequestRef.current !== requestId) return;
        pendingSendRequestRef.current = null;
        setSendPending(false);
        if (!accepted) return;
        setSendScrollRequest((current) => current + 1);
        setInput("");
        setReplyTarget(null);
        setCommandsOpen(false);
        textareaRef.current?.blur();
      },
      navigateBack() {
        const historyState = window.history.state;
        if (
          typeof historyState === "object" &&
          historyState !== null &&
          "akashicImageViewer" in historyState
        ) {
          window.history.back();
          return true;
        }
        if (surfaceRef.current.kind === "chat") return false;
        window.history.back();
        return true;
      },
    };
    window.AkashicNative?.reportReady();
    requestSnapshot();
    return () => {
      if (frame !== null) cancelAnimationFrame(frame);
      if (requestTimer !== null) window.clearTimeout(requestTimer);
      window.removeEventListener("popstate", handlePopState);
      delete window.AkashicMobile;
    };
  }, []);

  useEffect(() => {
    if (snapshot?.composer.isStopping || !snapshot?.composer.canStop || snapshot?.connection.error) {
      setStopRequested(false);
    }
  }, [snapshot?.composer.canStop, snapshot?.composer.isStopping, snapshot?.connection.error]);

  useEffect(() => {
    if (!replyTarget) return;
    const targetStillVisible = snapshot?.messages.some((message) => message.id === replyTarget.id) ?? false;
    if (replyTarget.sessionId !== snapshot?.selectedSessionId || !targetStillVisible) setReplyTarget(null);
  }, [replyTarget, snapshot?.messages, snapshot?.selectedSessionId]);

  useEffect(() => {
    const actionable = new Set(
      snapshot?.messages.filter((message) => message.deliveryAction).map((message) => message.id) ?? [],
    );
    setRecoveringMessageIds((current) => {
      const next = new Set([...current].filter((messageId) => actionable.has(messageId)));
      return next.size === current.size ? current : next;
    });
  }, [snapshot?.messages]);

  useEffect(() => {
    if (surface.kind !== "dashboard") return;
    if (!pluginDashboards.some((plugin) => plugin.id === surface.pluginId)) {
      setSurface({ kind: "plugins" });
    }
  }, [pluginDashboards, surface]);

  useEffect(() => () => {
    if (copiedTimerRef.current !== null) window.clearTimeout(copiedTimerRef.current);
    if (searchFocusTimerRef.current !== null) window.clearTimeout(searchFocusTimerRef.current);
    if (searchHighlightTimerRef.current !== null) window.clearTimeout(searchHighlightTimerRef.current);
  }, []);

  const messages = useMemo(
    () => snapshot?.messages.map(toChatMessage) ?? [],
    [snapshot?.messages],
  );
  const normalizedSearchQuery = normalizeMobileSearchText(searchQuery.trim());
  const searchResults = useMemo(() => {
    if (!normalizedSearchQuery || !snapshot) return [];
    const results: MobileMessage[] = [];
    snapshot.messages.forEach((message) => {
      const cached = searchIndex.get(message.id);
      if (cached?.revision === message.searchRevision && cached.matches) {
        results.push(message);
      }
    });
    return results;
  }, [normalizedSearchQuery, searchIndex, snapshot]);
  const searchTargetIndex = searchTargetId === null
    ? -1
    : searchResults.findIndex((message) => message.id === searchTargetId);

  const jumpToMessage = useCallback((messageId: string, focus = false) => {
    // 1. 按距离选择短平滑跳转或直接定位
    const element = messageElementsRef.current.get(messageId);
    if (!element) return;
    if (searchFocusTimerRef.current !== null) {
      window.clearTimeout(searchFocusTimerRef.current);
      searchFocusTimerRef.current = null;
    }
    const distance = Math.abs(element.getBoundingClientRect().top - window.innerHeight / 2);
    element.scrollIntoView({
      block: "center",
      behavior: distance <= window.innerHeight * 2 ? "smooth" : "auto",
    });

    // 2. 点亮目标状态层并恢复无障碍焦点
    setHighlightedMessageId(messageId);
    if (searchHighlightTimerRef.current !== null) window.clearTimeout(searchHighlightTimerRef.current);
    searchHighlightTimerRef.current = window.setTimeout(() => {
      setHighlightedMessageId((current) => current === messageId ? null : current);
      searchHighlightTimerRef.current = null;
    }, 1300);
    if (focus) {
      searchFocusTimerRef.current = window.setTimeout(() => {
        element.focus({ preventScroll: true });
        searchFocusTimerRef.current = null;
      }, 320);
    }
  }, []);

  useEffect(() => {
    const target = snapshot?.navigationTarget;
    if (!target || target.sessionId !== snapshot.selectedSessionId) return;
    const key = `${target.sessionId}\u001f${target.messageId}`;
    if (handledNavigationTargetRef.current === key) return;
    if (!snapshot.messages.some((message) => message.id === target.messageId)) return;
    handledNavigationTargetRef.current = key;
    jumpToMessage(target.messageId, true);
    window.AkashicNative?.navigationTargetHandled(target.messageId);
  }, [jumpToMessage, snapshot?.messages, snapshot?.navigationTarget, snapshot?.selectedSessionId]);

  useEffect(() => {
    if (!searchOpen || !normalizedSearchQuery || searchResults.length === 0) {
      setSearchTargetId(null);
      return;
    }
    if (searchTargetId !== null && searchResults.some((message) => message.id === searchTargetId)) return;
    setSearchTargetId(searchResults[searchResults.length - 1].id);
  }, [normalizedSearchQuery, searchOpen, searchResults, searchTargetId]);

  useEffect(() => {
    if (searchTargetId !== null) jumpToMessage(searchTargetId);
  }, [jumpToMessage, searchTargetId]);

  useEffect(() => {
    const sessionId = snapshot?.selectedSessionId;
    if (previousSessionIdRef.current === undefined) {
      previousSessionIdRef.current = sessionId;
      return;
    }
    if (sessionId === previousSessionIdRef.current) return;
    previousSessionIdRef.current = sessionId;
    if (searchFocusTimerRef.current !== null) {
      window.clearTimeout(searchFocusTimerRef.current);
      searchFocusTimerRef.current = null;
    }
    setSearchOpen(false);
    searchOpenRef.current = false;
    setSearchQuery("");
    normalizedSearchQueryRef.current = "";
    setSearchIndex(new Map());
    setSearchTargetId(null);
    setUnreadState({ count: 0 });
    setQueueOpen(false);
  }, [snapshot?.selectedSessionId]);

  useEffect(() => {
    setUnreadAnchorVisited(false);
  }, [unreadState.anchorKey]);
  if (!snapshot) {
    if (snapshotError) {
      return (
        <main className="mobile-fatal" role="alert">
          <AlertCircle className="mobile-fatal__mark" size={28} />
          <h1>会话数据没有通过校验</h1>
          <p>{snapshotError}</p>
          <button type="button" onClick={() => window.location.reload()}>
            <RefreshCw size={18} />
            重新载入
          </button>
        </main>
      );
    }
    return (
      <main className="mobile-loading" aria-live="polite">
        <span className="mobile-loading__mark" />
        <span>正在载入对话</span>
      </main>
    );
  }

  const send = () => {
    const text = input.trim();
    const attachmentsReady = allMobileAttachmentsReady(snapshot.composer.attachments);
    if (sendPending || !snapshot.composer.canSend || !attachmentsReady) return;
    if (!text && snapshot.composer.attachments.length === 0) return;
    const native = window.AkashicNative;
    if (!native) return;
    const requestId = `send-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    pendingSendRequestRef.current = requestId;
    setSendPending(true);
    native.sendMessage(
      requestId,
      text,
      replyTarget?.id ?? "",
      JSON.stringify(snapshot.composer.attachments.map((attachment) => attachment.id)),
    );
  };
  const stop = () => {
    if (!snapshot.composer.canStop || stopRequested) return;
    setStopRequested(true);
    window.AkashicNative?.stopTurn();
  };
  const closeDrawer = () => {
    setDrawerOpen(false);
    requestAnimationFrame(() => drawerToggleRef.current?.focus());
  };
  const toggleDrawer = () => {
    if (drawerOpen) closeDrawer();
    else setDrawerOpen(true);
  };
  const openSearch = () => {
    setDrawerOpen(false);
    setCommandsOpen(false);
    searchOpenRef.current = true;
    normalizedSearchQueryRef.current = "";
    setSearchIndex(new Map());
    setSearchOpen(true);
    requestAnimationFrame(() => searchInputRef.current?.focus());
  };
  const closeSearch = () => {
    if (searchFocusTimerRef.current !== null) {
      window.clearTimeout(searchFocusTimerRef.current);
      searchFocusTimerRef.current = null;
    }
    searchOpenRef.current = false;
    normalizedSearchQueryRef.current = "";
    setSearchOpen(false);
    setSearchQuery("");
    setSearchIndex(new Map());
    setSearchTargetId(null);
    setHighlightedMessageId(null);
    requestAnimationFrame(() => searchButtonRef.current?.focus());
  };
  const updateSearchQuery = (query: string) => {
    const normalized = normalizeMobileSearchText(query.trim());
    const queryChanged = normalized !== normalizedSearchQueryRef.current;
    normalizedSearchQueryRef.current = normalized;
    setSearchQuery(query);
    setSearchIndex((current) => updateMobileSearchIndex(
      current,
      snapshot.messages,
      normalized,
      queryChanged,
    ));
  };
  const moveSearch = (offset: number) => {
    if (searchResults.length === 0) return;
    const current = searchTargetIndex >= 0 ? searchTargetIndex : searchResults.length - 1;
    const next = Math.min(searchResults.length - 1, Math.max(0, current + offset));
    setSearchTargetId(searchResults[next].id);
  };
  const closeCommands = (restoreFocus = true) => {
    setCommandsOpen(false);
    if (restoreFocus) requestAnimationFrame(() => textareaRef.current?.focus());
  };
  const toggleCommands = () => {
    if (commandsOpen) closeCommands();
    else setCommandsOpen(true);
  };
  const copyMessage = (message: MobileMessage) => {
    window.AkashicNative?.copyText(message.content);
    setCopiedMessageId(message.id);
    if (copiedTimerRef.current !== null) window.clearTimeout(copiedTimerRef.current);
    copiedTimerRef.current = window.setTimeout(() => {
      setCopiedMessageId((current) => current === message.id ? null : current);
      copiedTimerRef.current = null;
    }, 1600);
  };
  const navigateToSurface = (next: Exclude<MobileSurface, { kind: "chat" }>) => {
    pushMobileSurface(window.history, next);
    surfaceRef.current = next;
    setSurface(next);
  };

  return (
    <TooltipProvider>
      <main className={`mobile-shell surface-${surface.kind}`}>
        {surface.kind === "chat" ? (
          <MobileTopBar
            status={snapshot.connection.status}
            label={snapshot.connection.label}
            activeTaskCount={snapshot.sessions.filter((session) => session.isRunning).length}
            drawerOpen={drawerOpen}
            searchOpen={searchOpen}
            searchQuery={searchQuery}
            toggleRef={drawerToggleRef}
            searchButtonRef={searchButtonRef}
            searchInputRef={searchInputRef}
            onToggleDrawer={toggleDrawer}
            onOpenSearch={openSearch}
            onCloseSearch={closeSearch}
            onSearchQuery={updateSearchQuery}
            onSearchSubmit={() => {
              if (searchTargetId !== null) jumpToMessage(searchTargetId, true);
            }}
          />
        ) : (
          <MobilePluginTopBar
            title={surface.kind === "plugins"
              ? "插件"
              : pluginDashboards.find((plugin) => plugin.id === surface.pluginId)?.label ?? "插件看板"}
            onBack={() => window.history.back()}
          />
        )}
        <MobileDrawer
          open={drawerOpen}
          snapshot={snapshot}
          pluginCount={pluginDashboards.length}
          onOpenPlugins={() => {
            navigateToSurface({ kind: "plugins" });
            closeDrawer();
          }}
          onClose={closeDrawer}
        />
        {(snapshot.connection.error || pluginLoadError) ? (
          <div className="mobile-surface-errors" aria-live="assertive">
            {snapshot.connection.error ? (
              <div className="mobile-snackbar" role="alert">
                <AlertCircle className="mobile-snackbar__mark" size={19} />
                <span>
                  <strong>连接出现问题</strong>
                  <small>{snapshot.connection.error}</small>
                </span>
                <button type="button" onClick={() => window.AkashicNative?.dismissError()}>关闭</button>
              </div>
            ) : null}
            {pluginLoadError ? (
              <div className="mobile-snackbar" role="alert">
                <AlertCircle className="mobile-snackbar__mark" size={19} />
                <span>
                  <strong>插件暂时无法读取</strong>
                  <small>{pluginLoadError}</small>
                </span>
                <button type="button" onClick={() => window.location.reload()}>重试</button>
              </div>
            ) : null}
          </div>
        ) : null}

        {surface.kind === "plugins" ? (
          <MobilePluginDirectory
            plugins={pluginDashboards}
            onOpen={(pluginId) => navigateToSurface({ kind: "dashboard", pluginId })}
          />
        ) : surface.kind === "dashboard" ? (
          <section className="mobile-plugin-dashboard" aria-label="插件看板">
            <MobilePluginDashboard pluginId={surface.pluginId} />
          </section>
        ) : (
        <div className={`mobile-main-content ${replyTarget ? "replying" : ""} ${searchOpen ? "searching" : ""} ${queueOpen && snapshot.composer.pendingMessages.length > 1 ? "queueing" : ""}`} inert={drawerOpen ? true : undefined}>
          <Conversation key={snapshot.selectedSessionId ?? "empty"} className="mobile-conversation">
            <ConversationContent className="mobile-conversation__content">
              {messages.length === 0 ? (
                <ConversationEmptyState className="mobile-empty">
                  <h1>开始一段新对话</h1>
                  <p>消息会通过电脑上的 Akashic 实时处理。</p>
                </ConversationEmptyState>
              ) : (
                messages.map((message, index) => {
                  const source = snapshot.messages[index];
                  const previous = snapshot.messages[index - 1];
                  const startsDay = !previous || !sameLocalDay(previous.createdAt, source.createdAt);
                  const canReply = source.replyable;
                  return (
                    <React.Fragment key={message.id}>
                      {startsDay ? <MessageDateDivider createdAt={source.createdAt} /> : null}
                      {source.id === unreadState.firstMessageId ? <MessageUnreadDivider count={unreadState.count} /> : null}
                      {!startsDay && previous?.role === source.role ? (
                        <div className={`mobile-role-divider ${source.role}`} />
                      ) : null}
                      <div
                        ref={(element) => {
                          if (element) messageElementsRef.current.set(source.id, element);
                          else messageElementsRef.current.delete(source.id);
                        }}
                        className={`mobile-message-anchor ${source.role} ${highlightedMessageId === source.id ? "search-target" : ""}`}
                        data-message-id={source.id}
                        tabIndex={-1}
                      >
                        <SwipeToReply disabled={!canReply} onReply={() => setReplyTarget(source)}>
                          <ChatMessageView
                            message={message}
                            leadingContent={source.reply ? <MessageReplyReference reply={source.reply} /> : undefined}
                            attachmentContent={<MobileMessageAttachments attachments={source.attachments} />}
                            processStartContent={isPluginTurnMessage(message) ? (
                              <MobilePluginSlot
                                name="turn.before_reasoning"
                                sessionId={source.sessionId}
                                messageId={message.id}
                                turnId={pluginTurnId(message)}
                              />
                            ) : undefined}
                            beforeProcessBlock={(block) => isPluginTurnMessage(message) && block.kind === "tool" ? (
                              <MobilePluginSlot
                                name="turn.before_tool"
                                sessionId={source.sessionId}
                                messageId={message.id}
                                turnId={pluginTurnId(message)}
                                block={block}
                              />
                            ) : null}
                            answerEndContent={isPluginTurnMessage(message) ? (
                              <MobilePluginSlot
                                name="turn.after_answer"
                                sessionId={source.sessionId}
                                messageId={message.id}
                                turnId={pluginTurnId(message)}
                              />
                            ) : undefined}
                          />
                          <MessageMeta
                            source={source}
                            copied={copiedMessageId === source.id}
                            canReply={canReply}
                            onCopy={() => copyMessage(source)}
                            onReply={() => setReplyTarget(source)}
                            deliveryActionBusy={recoveringMessageIds.has(source.id)}
                            onDeliveryAction={() => {
                              setRecoveringMessageIds((current) => new Set(current).add(source.id));
                              window.AkashicNative?.performActionHaptic();
                              window.AkashicNative?.retryFailedMessage(source.id);
                            }}
                          />
                        </SwipeToReply>
                      </div>
                    </React.Fragment>
                  );
                })
              )}
            </ConversationContent>
            <MobileConversationBehavior
              sessionId={snapshot.selectedSessionId}
              sourceMessages={snapshot.messages}
              chatMessages={messages}
              streaming={snapshot.composer.isStreaming}
              projectionGeneration={snapshot.projectionGeneration}
              resyncing={snapshot.composer.isResyncing}
              suspended={searchOpen}
              forceScrollToken={sendScrollRequest}
              readingPosition={snapshot.readingPosition}
              messageElementsRef={messageElementsRef}
              onUnreadChange={setUnreadState}
            />
            <MobileSearchScrollBehavior active={searchOpen} targetMessageId={searchTargetId} />
            <MobileSearchTextHighlight query={searchQuery} messageId={searchTargetId} />
            <MobileScrollButton
              unread={unreadState}
              unreadAnchorVisited={unreadAnchorVisited}
              onVisitUnread={(messageId) => {
                setUnreadAnchorVisited(true);
                jumpToMessage(messageId, false);
              }}
            />
          </Conversation>

          {searchOpen ? (
            <MobileSearchNavigator
              query={searchQuery}
              count={searchResults.length}
              currentIndex={searchTargetIndex}
              onPrevious={() => moveSearch(-1)}
              onNext={() => moveSearch(1)}
            />
          ) : (
            <MobileComposer
              snapshot={snapshot}
              input={input}
              textareaRef={textareaRef}
              commandsOpen={commandsOpen}
              queueOpen={queueOpen}
              stopRequested={stopRequested}
              sendPending={sendPending}
              replyTarget={replyTarget}
              onInput={setInput}
              onToggleCommands={toggleCommands}
              onToggleQueue={() => setQueueOpen((current) => !current)}
              onCloseCommands={closeCommands}
              onSend={send}
              onStop={stop}
              onCancelReply={() => setReplyTarget(null)}
            />
          )}
        </div>
        )}
      </main>
    </TooltipProvider>
  );
}

function MobilePluginTopBar({ title, onBack }: { title: string; onBack: () => void }) {
  return (
    <header className="mobile-topbar mobile-plugin-topbar">
      <button className="mobile-icon-button" type="button" onClick={onBack} aria-label="返回">
        <ArrowLeft size={24} />
      </button>
      <h1>{title}</h1>
    </header>
  );
}

function MobilePluginDirectory({
  plugins,
  onOpen,
}: {
  plugins: MobilePluginDashboardEntry[];
  onOpen: (pluginId: string) => void;
}) {
  return (
    <section className="mobile-plugin-directory" aria-labelledby="mobile-plugin-directory-title">
      <h2 className="mobile-plugin-directory__status" id="mobile-plugin-directory-title">
        运行中 · {plugins.length}
      </h2>
      {plugins.length ? (
        <div className="mobile-plugin-directory__list">
          {plugins.map((plugin) => (
            <button type="button" key={plugin.id} onClick={() => onOpen(plugin.id)}>
              <span className="mobile-plugin-directory__mark" aria-hidden="true">
                {plugin.label.replaceAll(" ", "").slice(0, 2).toUpperCase()}
              </span>
              <span>
                <strong>{plugin.label}</strong>
                <small>{plugin.description}</small>
              </span>
              <ChevronRight size={20} aria-hidden="true" />
            </button>
          ))}
        </div>
      ) : (
        <p className="mobile-plugin-directory__empty">当前没有声明移动看板的运行中插件。</p>
      )}
    </section>
  );
}

function MobileTopBar({
  status,
  label,
  activeTaskCount,
  drawerOpen,
  searchOpen,
  searchQuery,
  toggleRef,
  searchButtonRef,
  searchInputRef,
  onToggleDrawer,
  onOpenSearch,
  onCloseSearch,
  onSearchQuery,
  onSearchSubmit,
}: {
  status: ConnectionStatus;
  label: string;
  activeTaskCount: number;
  drawerOpen: boolean;
  searchOpen: boolean;
  searchQuery: string;
  toggleRef: React.RefObject<HTMLButtonElement | null>;
  searchButtonRef: React.RefObject<HTMLButtonElement | null>;
  searchInputRef: React.RefObject<HTMLInputElement | null>;
  onToggleDrawer: () => void;
  onOpenSearch: () => void;
  onCloseSearch: () => void;
  onSearchQuery: (query: string) => void;
  onSearchSubmit: () => void;
}) {
  if (searchOpen) {
    return (
      <header className="mobile-topbar search-mode">
        <button className="mobile-icon-button" type="button" onClick={onCloseSearch} aria-label="关闭消息搜索">
          <ArrowLeft size={24} />
        </button>
        <form className="mobile-search-field" onSubmit={(event) => {
          event.preventDefault();
          onSearchSubmit();
        }}>
          <Search size={18} aria-hidden="true" />
          <input
            ref={searchInputRef}
            value={searchQuery}
            type="search"
            enterKeyHint="search"
            autoComplete="off"
            placeholder="搜索这段对话"
            aria-label="搜索这段对话"
            onChange={(event) => onSearchQuery(event.target.value)}
          />
          {searchQuery ? (
            <button type="button" onClick={() => onSearchQuery("")} aria-label="清除搜索">
              <X size={18} />
            </button>
          ) : null}
        </form>
      </header>
    );
  }
  const online = status === "ready";
  return (
    <header className={`mobile-topbar ${drawerOpen ? "drawer-open" : ""}`}>
      <button ref={toggleRef} className="mobile-icon-button drawer-toggle" type="button" onClick={onToggleDrawer} aria-label={drawerOpen ? "收起会话" : "打开会话"} aria-expanded={drawerOpen}>
        {drawerOpen ? <X size={25} /> : <Menu size={25} />}
      </button>
      <div className={`connection-state ${status}`} aria-live="polite">
        {online ? <Wifi size={19} /> : status === "disconnected" ? <WifiOff size={19} /> : <RefreshCw className="connection-spinner" size={18} />}
        <span>{label}</span>
      </div>
      {activeTaskCount > 0 ? (
        <div className="agent-task-state" aria-live="polite">
          <i />
          <span>运行 {activeTaskCount}</span>
        </div>
      ) : null}
      <button ref={searchButtonRef} className="mobile-icon-button" type="button" onClick={onOpenSearch} aria-label="搜索消息">
        <Search size={22} />
      </button>
    </header>
  );
}

function MobileSearchNavigator({
  query,
  count,
  currentIndex,
  onPrevious,
  onNext,
}: {
  query: string;
  count: number;
  currentIndex: number;
  onPrevious: () => void;
  onNext: () => void;
}) {
  const position = currentIndex >= 0 ? currentIndex + 1 : 0;
  return (
    <nav className="mobile-search-navigator" aria-label="搜索结果导航">
      <span aria-live="polite">
        {!query.trim() ? "输入关键词" : count === 0 ? "没有结果" : `${position} / ${count}`}
      </span>
      <div>
        <button type="button" disabled={currentIndex <= 0} onClick={onPrevious} aria-label="上一个搜索结果">
          <ChevronUp size={22} />
        </button>
        <button type="button" disabled={currentIndex < 0 || currentIndex >= count - 1} onClick={onNext} aria-label="下一个搜索结果">
          <ChevronDown size={22} />
        </button>
      </div>
    </nav>
  );
}

function MobileDrawer({
  open,
  snapshot,
  pluginCount,
  onOpenPlugins,
  onClose,
}: {
  open: boolean;
  snapshot: MobileSnapshot;
  pluginCount: number;
  onOpenPlugins: () => void;
  onClose: () => void;
}) {
  const drawerRef = useRef<HTMLElement>(null);
  useEffect(() => {
    if (open) requestAnimationFrame(() => drawerRef.current?.focus());
  }, [open]);
  return (
    <div className={`mobile-drawer-layer ${open ? "open" : ""}`} aria-hidden={!open}>
      <button className="mobile-drawer-scrim" type="button" onClick={onClose} aria-label="关闭会话抽屉" tabIndex={open ? 0 : -1} />
      <aside ref={drawerRef} className="mobile-drawer" role="dialog" aria-modal="true" aria-label="会话列表" tabIndex={-1}>
        <div className="mobile-drawer__heading">会话</div>
        <button className="mobile-plugin-destination" type="button" onClick={onOpenPlugins}>
          <Puzzle size={20} aria-hidden="true" />
          <span>插件</span>
          <small>{pluginCount}</small>
          <ChevronRight size={18} aria-hidden="true" />
        </button>
        <nav className="mobile-session-list">
          {snapshot.sessions.map((session) => (
            <button
              className={`mobile-session-row ${session.id === snapshot.selectedSessionId ? "active" : ""}`}
              type="button"
              key={session.id}
              onClick={() => {
                window.AkashicNative?.selectSession(session.id);
                onClose();
              }}
            >
              <span className="mobile-session-row__copy">
                <span className="mobile-session-row__title">
                  <strong>{session.title || "未命名会话"}</strong>
                  {session.lastMessageAt ? <time>{formatDrawerTime(session.lastMessageAt)}</time> : null}
                </span>
                <small>{session.lastMessagePreview || "还没有消息"}</small>
              </span>
              <span className="mobile-session-row__state">
                {session.isRunning ? <span className="session-running" aria-label="Agent 正在处理" /> : null}
                {session.unreadCount > 0 ? (
                  <strong className="session-unread" aria-label={`${session.unreadCount} 条未读`}>
                    {session.unreadCount > 99 ? "99+" : session.unreadCount}
                  </strong>
                ) : session.id === snapshot.selectedSessionId ? <Check size={18} /> : null}
              </span>
            </button>
          ))}
        </nav>
        <MobilePluginSlot name="drawer.panel" sessionId={snapshot.selectedSessionId} />
        <div className="mobile-drawer__actions">
          <button className="drawer-action" type="button" disabled={!snapshot.composer.canResync} onClick={() => {
            if (window.confirm("清除本机已同步消息和附件缓存，并从电脑重新拉取？连接状态会保留。")) {
              window.AkashicNative?.reloadFromServer();
              onClose();
            }
          }}>
            <RotateCcw size={18} />
            <span>{snapshot.composer.isResyncing ? "正在重新同步" : "清理缓存并同步"}</span>
          </button>
          <button className="drawer-action" type="button" onClick={() => window.AkashicNative?.restartPairing()}>
            <RefreshCw size={18} />
            <span>重新扫码</span>
          </button>
          <button className="new-chat-action" type="button" onClick={() => {
            window.AkashicNative?.createSession();
            onClose();
          }}>
            <MessageSquarePlus size={18} />
            <span>新聊天</span>
          </button>
        </div>
      </aside>
    </div>
  );
}

function MobileComposer({
  snapshot,
  input,
  textareaRef,
  commandsOpen,
  queueOpen,
  stopRequested,
  sendPending,
  replyTarget,
  onInput,
  onToggleCommands,
  onToggleQueue,
  onCloseCommands,
  onSend,
  onStop,
  onCancelReply,
}: {
  snapshot: MobileSnapshot;
  input: string;
  textareaRef: React.RefObject<HTMLTextAreaElement | null>;
  commandsOpen: boolean;
  queueOpen: boolean;
  stopRequested: boolean;
  sendPending: boolean;
  replyTarget: MobileMessage | null;
  onInput: (value: string) => void;
  onToggleCommands: () => void;
  onToggleQueue: () => void;
  onCloseCommands: (restoreFocus?: boolean) => void;
  onSend: () => void;
  onStop: () => void;
  onCancelReply: () => void;
}) {
  const stopping = stopRequested || snapshot.composer.isStopping;
  const hasDraft = snapshot.composer.attachments.length > 0;
  const attachmentsReady = allMobileAttachmentsReady(snapshot.composer.attachments);
  const canSubmit = snapshot.composer.canSend && attachmentsReady && !sendPending && (!!input.trim() || hasDraft);

  return (
    <div className="mobile-composer-zone">
      <CommandSheet open={commandsOpen} commands={snapshot.composer.commands} onClose={onCloseCommands} />
      {snapshot.connection.notice ? <div className="connection-notice">{snapshot.connection.notice}</div> : null}
      {stopping ? <div className="stop-feedback" aria-live="polite">正在中止本轮处理…</div> : null}
      {snapshot.composer.transferStatus ? <TransferBanner status={snapshot.composer.transferStatus} /> : null}
      {hasDraft ? <DraftAttachments attachments={snapshot.composer.attachments} disabled={sendPending} /> : null}
      <div className={`mobile-composer-frame ${replyTarget ? "has-reply" : ""}`}>
        {snapshot.composer.pendingMessages.length > 1 ? (
          <PendingQueue
            open={queueOpen}
            messages={snapshot.composer.pendingMessages}
            onToggle={onToggleQueue}
          />
        ) : null}
        {replyTarget ? <ComposerReply target={replyTarget} onCancel={onCancelReply} /> : null}
        <div className="mobile-composer">
        <button className={`mobile-icon-button command-toggle ${commandsOpen ? "active" : ""}`} type="button" onClick={onToggleCommands} aria-label={commandsOpen ? "关闭快捷命令" : "打开快捷命令"}>
          {commandsOpen ? <X size={22} /> : <Menu size={22} />}
        </button>
        <textarea
          ref={textareaRef}
          rows={1}
          value={input}
          placeholder="输入消息"
          onChange={(event) => onInput(event.target.value)}
          onFocus={() => commandsOpen && onCloseCommands(false)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              onSend();
            }
          }}
        />
        <button className="mobile-icon-button" type="button" disabled={sendPending} onClick={() => window.AkashicNative?.chooseAttachments()} aria-label="添加附件">
          <Paperclip size={22} />
        </button>
        {snapshot.composer.canStop || stopping ? (
          <button className={`mobile-send-button stop ${stopping ? "pending" : ""}`} type="button" onClick={onStop} aria-label={stopping ? "正在中止" : "中止回答"} disabled={stopping}>
            <Square size={17} fill="currentColor" />
          </button>
        ) : (
          <button className="mobile-send-button" type="button" onClick={onSend} aria-label={sendPending ? "正在保存消息" : "发送消息"} disabled={!canSubmit}>
            <SendHorizontal size={22} />
          </button>
        )}
        </div>
      </div>
    </div>
  );
}

function PendingQueue({
  open,
  messages,
  onToggle,
}: {
  open: boolean;
  messages: MobilePendingMessage[];
  onToggle: () => void;
}) {
  return (
    <section className={`composer-queue ${open ? "open" : ""}`} aria-label="待发送消息">
      <button type="button" onClick={onToggle} aria-expanded={open}>
        <TimerReset size={18} />
        <span>{messages.length} 条消息等待连接</span>
        <ChevronDown className={open ? "open" : ""} size={18} />
      </button>
      <div className="composer-queue__disclosure">
        <div>
          {messages.map((message) => (
            <div className="composer-queue__item" key={message.messageId}>
              <span>{message.preview}</span>
              <time>{formatMessageTime(message.createdAt)}</time>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function TransferBanner({ status }: { status: MobileTransferStatus }) {
  return (
    <section className={`transfer-banner ${status.requiresMeteredApproval ? "paused" : ""}`} aria-live="polite">
      <div>
        <strong>{status.title}</strong>
        <span>{status.detail}</span>
      </div>
      <span className="transfer-banner__percent">{status.progressPercent}%</span>
      <div className="transfer-banner__track"><span style={{ inlineSize: `${status.progressPercent}%` }} /></div>
      {status.requiresMeteredApproval ? (
        <button type="button" onClick={() => window.AkashicNative?.continueMeteredTransfer()}>使用当前网络继续</button>
      ) : null}
    </section>
  );
}

function CommandSheet({ open, commands, onClose }: { open: boolean; commands: MobileSnapshot["composer"]["commands"]; onClose: (restoreFocus?: boolean) => void }) {
  const dispatchedRef = useRef(false);
  useEffect(() => {
    if (open) dispatchedRef.current = false;
  }, [open]);
  return (
    <section className={`command-sheet ${open ? "open" : ""}`} aria-label="快捷命令" aria-hidden={!open}>
      <div className="command-sheet__handle" />
      <div className="command-sheet__header">
        <div>
          <h2>快捷命令</h2>
          <p>选择后立即发送</p>
        </div>
        <button className="mobile-icon-button" type="button" onClick={() => onClose()} aria-label="关闭快捷命令" tabIndex={open ? 0 : -1}><ChevronDown size={22} /></button>
      </div>
      <div className="command-sheet__list">
        {commands.map((item) => (
          <button type="button" className="command-row" key={item.command} tabIndex={open ? 0 : -1} onClick={() => {
            if (dispatchedRef.current) return;
            dispatchedRef.current = true;
            window.AkashicNative?.sendCommand(`/${item.command}`);
            onClose();
          }}>
            <span className="command-row__description">{item.description}</span>
            <span className="command-row__name">/{item.command}</span>
          </button>
        ))}
      </div>
    </section>
  );
}

function DraftAttachments({ attachments, disabled }: { attachments: MobileAttachment[]; disabled: boolean }) {
  const [operations, setOperations] = useState(new Map<string, "retry" | "remove">());
  const operationsRef = useRef(operations);

  useEffect(() => {
    setOperations((current) => {
      const next = new Map(current);
      current.forEach((operation, attachmentId) => {
        const attachment = attachments.find((item) => item.id === attachmentId);
        const completed = !attachment ||
          (operation === "retry" && attachment.state !== "failed") ||
          (operation === "remove" && !attachment.canRemove);
        if (completed) next.delete(attachmentId);
      });
      operationsRef.current = next;
      return next.size === current.size ? current : next;
    });
  }, [attachments]);

  const dispatch = (attachmentId: string, operation: "retry" | "remove") => {
    if (operationsRef.current.has(attachmentId)) return;
    const next = new Map(operationsRef.current).set(attachmentId, operation);
    operationsRef.current = next;
    setOperations(next);
    if (operation === "retry") window.AkashicNative?.retryAttachment(attachmentId);
    else window.AkashicNative?.removeAttachment(attachmentId);
  };

  return (
    <div className="draft-attachments">
      {attachments.map((attachment) => {
        const progress = attachment.sizeBytes > 0
          ? Math.min(100, Math.round(attachment.transferredBytes / attachment.sizeBytes * 100))
          : 0;
        const busy = disabled || operations.has(attachment.id);
        return (
          <div className="draft-attachment" key={attachment.id} aria-busy={busy}>
            <FileText size={19} />
            <div className="draft-attachment__body">
              <div className="draft-attachment__name">{attachment.filename}</div>
              <div className={`draft-attachment__status ${attachment.state === "failed" ? "error" : ""}`}>
                {attachment.state === "ready"
                  ? "上传完成"
                  : attachment.state === "failed"
                    ? `上传失败 · 已保留 ${formatBytes(attachment.transferredBytes)}`
                    : attachment.state === "waiting"
                      ? `等待连接 · ${progress}%`
                      : attachment.state === "metered_paused"
                        ? `等待网络许可 · 已完成 ${progress}%`
                      : `上传中 ${progress}% · ${formatBytes(attachment.transferredBytes)} / ${formatBytes(attachment.sizeBytes)}`}
              </div>
              <div className="draft-attachment__track"><span style={{ inlineSize: `${progress}%` }} /></div>
            </div>
            <div className="draft-attachment__actions">
              {attachment.state === "failed" ? (
                <button type="button" disabled={busy} onClick={() => dispatch(attachment.id, "retry")}>重试</button>
              ) : null}
              {attachment.canRemove ? (
                <button type="button" disabled={busy} onClick={() => dispatch(attachment.id, "remove")} aria-label={`移除 ${attachment.filename}`}><X size={18} /></button>
              ) : null}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function MobileMessageAttachments({ attachments }: { attachments: MobileAttachment[] }) {
  if (attachments.length === 0) return null;
  return (
    <div className="mobile-message-attachments">
      {attachments.map((attachment) => (
        <MobileMessageAttachment attachment={attachment} key={attachment.id} />
      ))}
    </div>
  );
}

function MobileMessageAttachment({ attachment }: { attachment: MobileAttachment }) {
  const progress = attachment.sizeBytes > 0
    ? Math.min(100, Math.round(attachment.transferredBytes / attachment.sizeBytes * 100))
    : 0;
  const cached = attachment.state === "cached";
  const image = cached && attachment.contentType.startsWith("image/") && attachment.contentUrl;
  const [viewerOpen, setViewerOpen] = useState(false);
  const viewerOpenRef = useRef(false);

  useEffect(() => {
    const closeFromHistory = () => {
      if (!viewerOpenRef.current) return;
      viewerOpenRef.current = false;
      setViewerOpen(false);
      window.AkashicNative?.setWebHistoryActive(false);
    };
    window.addEventListener("popstate", closeFromHistory);
    return () => {
      window.removeEventListener("popstate", closeFromHistory);
      if (!viewerOpenRef.current) return;
      viewerOpenRef.current = false;
      window.AkashicNative?.setWebHistoryActive(false);
      if (isMobileImageViewerHistoryState(window.history.state, attachment.id)) window.history.back();
    };
  }, [attachment.id]);

  const changeViewer = (open: boolean) => {
    if (open) {
      if (viewerOpenRef.current) return;
      window.history.pushState({ akashicImageViewer: attachment.id }, "");
      viewerOpenRef.current = true;
      setViewerOpen(true);
      window.AkashicNative?.setWebHistoryActive(true);
      window.AkashicNative?.touchDownloadedAttachment(attachment.id);
      return;
    }
    if (isMobileImageViewerHistoryState(window.history.state, attachment.id)) {
      viewerOpenRef.current = false;
      setViewerOpen(false);
      window.AkashicNative?.setWebHistoryActive(false);
      window.history.back();
      return;
    }
    viewerOpenRef.current = false;
    setViewerOpen(false);
    window.AkashicNative?.setWebHistoryActive(false);
  };
  const status = attachment.state === "pending"
    ? "等待下载"
    : attachment.state === "downloading"
      ? `下载中 ${progress}%`
      : attachment.state === "remote"
        ? `${formatBytes(attachment.sizeBytes)} · 尚未下载`
      : attachment.state === "failed"
        ? "下载失败"
        : attachment.state === "evicted"
          ? "缓存已清理"
          : "已下载";

  return (
    <div className={`mobile-message-attachment ${attachment.state}`}>
      {image ? (
        <Dialog open={viewerOpen} onOpenChange={changeViewer}>
          <DialogTrigger asChild>
            <button className="message-attachment-preview" type="button" aria-label={`预览 ${attachment.filename}`}>
              <img alt="" src={attachment.contentUrl} />
            </button>
          </DialogTrigger>
          <ImageViewer attachment={attachment} onClose={() => changeViewer(false)} />
        </Dialog>
      ) : (
        <button
          className="message-attachment-main"
          type="button"
          disabled={!cached}
          onClick={() => window.AkashicNative?.openDownloadedAttachment(attachment.id)}
        >
          <FileText size={20} />
          <span>
            <strong>{attachment.filename}</strong>
            <small>{status}</small>
          </span>
        </button>
      )}
      {image ? (
        <div className="message-attachment-caption">
          <strong>{attachment.filename}</strong>
          <small>{status}</small>
        </div>
      ) : null}
      {attachment.state === "downloading" ? (
        <div className="message-attachment-progress" aria-label={status}><span style={{ inlineSize: `${progress}%` }} /></div>
      ) : null}
      {attachment.state === "remote" || attachment.state === "failed" || attachment.state === "evicted" ? (
        <button
          aria-label={`${attachment.state === "remote" ? "下载" : "重试"} ${attachment.filename}`}
          className="message-attachment-action"
          type="button"
          onClick={() => window.AkashicNative?.retryDownloadedAttachment(attachment.id)}
        >
          {attachment.state === "remote" ? "下载" : "重试"}
        </button>
      ) : cached ? (
        <button className="message-attachment-action icon" type="button" onClick={() => window.AkashicNative?.shareDownloadedAttachment(attachment.id)} aria-label={`分享 ${attachment.filename}`}><Share2 size={18} /></button>
      ) : null}
    </div>
  );
}

interface ViewerTransform {
  scale: number;
  x: number;
  y: number;
}

function ImageViewer({ attachment, onClose }: { attachment: MobileAttachment; onClose: () => void }) {
  const [transform, setTransform] = useState<ViewerTransform>({ scale: 1, x: 0, y: 0 });
  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const gesture = useRef({ distance: 1, centerX: 0, centerY: 0, transform });

  const beginGesture = (event: React.PointerEvent<HTMLDivElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    pointers.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    gesture.current = gestureSnapshot(pointers.current, transform);
  };

  const moveGesture = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!pointers.current.has(event.pointerId)) return;
    pointers.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    const points = [...pointers.current.values()];
    const start = gesture.current;
    if (points.length >= 2) {
      const currentDistance = pointDistance(points[0], points[1]);
      const currentCenter = pointCenter(points[0], points[1]);
      const nextScale = clampScale(start.transform.scale * currentDistance / start.distance);
      setTransform({
        scale: nextScale,
        x: start.transform.x + currentCenter.x - start.centerX,
        y: start.transform.y + currentCenter.y - start.centerY,
      });
    } else if (start.transform.scale > 1) {
      setTransform({
        ...start.transform,
        x: start.transform.x + points[0].x - start.centerX,
        y: start.transform.y + points[0].y - start.centerY,
      });
    }
  };

  const endGesture = (event: React.PointerEvent<HTMLDivElement>) => {
    pointers.current.delete(event.pointerId);
    gesture.current = gestureSnapshot(pointers.current, transform);
  };

  const zoomTo = (scale: number) => {
    const next = clampScale(scale);
    setTransform(next === 1 ? { scale: 1, x: 0, y: 0 } : { ...transform, scale: next });
  };

  return (
    <DialogContent className="image-viewer-dialog" showCloseButton={false}>
      <DialogTitle className="sr-only">{attachment.filename}</DialogTitle>
      <header className="image-viewer-toolbar">
        <button type="button" onClick={onClose} aria-label="返回会话"><ArrowLeft size={22} /></button>
        <strong title={attachment.filename}>{attachment.filename}</strong>
        <button type="button" onClick={() => window.AkashicNative?.saveDownloadedAttachment(attachment.id)} aria-label={`保存 ${attachment.filename}`}><Download size={21} /></button>
        <button type="button" onClick={() => window.AkashicNative?.shareDownloadedAttachment(attachment.id)} aria-label={`分享 ${attachment.filename}`}><Share2 size={21} /></button>
      </header>
      <div
        className="image-viewer-stage"
        onDoubleClick={() => zoomTo(transform.scale > 1 ? 1 : 2)}
        onPointerDown={beginGesture}
        onPointerMove={moveGesture}
        onPointerUp={endGesture}
        onPointerCancel={endGesture}
      >
        <img
          alt={attachment.filename}
          draggable={false}
          src={attachment.contentUrl}
          style={{ transform: `translate3d(${transform.x}px, ${transform.y}px, 0) scale(${transform.scale})` }}
        />
      </div>
      <div className="image-viewer-zoom" aria-label={`缩放 ${Math.round(transform.scale * 100)}%`}>
        <button type="button" onClick={() => zoomTo(transform.scale - 0.5)} aria-label="缩小">−</button>
        <span>{Math.round(transform.scale * 100)}%</span>
        <button type="button" onClick={() => zoomTo(transform.scale + 0.5)} aria-label="放大">＋</button>
      </div>
    </DialogContent>
  );
}

function gestureSnapshot(
  pointers: Map<number, { x: number; y: number }>,
  transform: ViewerTransform,
) {
  const points = [...pointers.values()];
  if (points.length >= 2) {
    const center = pointCenter(points[0], points[1]);
    return {
      distance: pointDistance(points[0], points[1]),
      centerX: center.x,
      centerY: center.y,
      transform,
    };
  }
  const point = points[0] ?? { x: 0, y: 0 };
  return { distance: 1, centerX: point.x, centerY: point.y, transform };
}

function pointDistance(first: { x: number; y: number }, second: { x: number; y: number }) {
  return Math.hypot(second.x - first.x, second.y - first.y);
}

function pointCenter(first: { x: number; y: number }, second: { x: number; y: number }) {
  return { x: (first.x + second.x) / 2, y: (first.y + second.y) / 2 };
}

function clampScale(scale: number) {
  return Math.min(4, Math.max(1, scale));
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MiB`;
}

function MessageMeta({
  source,
  copied,
  canReply,
  deliveryActionBusy,
  onCopy,
  onReply,
  onDeliveryAction,
}: {
  source: MobileMessage;
  copied: boolean;
  canReply: boolean;
  deliveryActionBusy: boolean;
  onCopy: () => void;
  onReply: () => void;
  onDeliveryAction: () => void;
}) {
  const deliveryActionLabel = deliveryActionBusy
    ? "处理中"
    : source.deliveryAction === "retry" ? "重试" : "核对";
  const deliveryActionAria = source.deliveryAction === "retry"
    ? "发送失败，重试消息"
    : "结果待确认，核对消息状态";
  return (
    <div className={`mobile-message-meta ${source.role}`}>
      <div className="mobile-message-meta__text">
        <time dateTime={new Date(source.createdAt).toISOString()}>{formatMessageTime(source.createdAt)}</time>
        {source.role === "user" && source.deliveryLabel ? (
          <span className={source.deliveryAction ? `delivery-state ${source.deliveryAction}` : undefined}>
            {source.deliveryLabel}
          </span>
        ) : null}
        {source.role === "user" && source.deliveryAction ? (
          <button
            className={`mobile-delivery-action ${source.deliveryAction}`}
            type="button"
            onClick={onDeliveryAction}
            aria-label={deliveryActionAria}
            aria-busy={deliveryActionBusy}
            disabled={deliveryActionBusy}
          >
            <RotateCcw size={14} aria-hidden="true" />
            <span>{deliveryActionLabel}</span>
          </button>
        ) : null}
        {source.interrupted ? <span className="interrupted-label">本轮已中止</span> : null}
      </div>
      <div className="mobile-message-actions">
        {canReply ? (
          <button type="button" onClick={onReply} aria-label="引用此消息">
            <Reply size={16} />
          </button>
        ) : null}
        {source.content ? (
          <button type="button" className={copied ? "copied" : ""} onClick={onCopy} aria-label={copied ? "已复制" : "复制消息"}>
            {copied ? <Check size={16} /> : <Copy size={16} />}
          </button>
        ) : null}
      </div>
    </div>
  );
}

function SwipeToReply({ disabled, onReply, children }: { disabled: boolean; onReply: () => void; children: ReactNode }) {
  const x = useMotionValue(0);
  const hapticFired = useRef(false);
  const reduceMotion = useReducedMotion();
  const iconOpacity = useTransform(x, [-80, -50, -12, 0], [1, 1, 0.12, 0]);
  const iconScale = useTransform(x, [-80, -50, 0], [1, 1, 0.72]);

  useMotionValueEvent(x, "change", (value) => {
    if (value > -50) {
      hapticFired.current = false;
      return;
    }
    if (value <= -50 && !hapticFired.current) {
      hapticFired.current = true;
      window.AkashicNative?.performActionHaptic();
    }
  });

  return (
    <div className={`swipe-reply ${disabled ? "disabled" : ""}`}>
      <motion.div className="swipe-reply__indicator" style={{ opacity: iconOpacity, scale: iconScale }} aria-hidden="true">
        <Reply size={20} />
      </motion.div>
      <motion.div
        className="swipe-reply__content"
        drag={disabled ? false : "x"}
        dragConstraints={{ left: -80, right: 0 }}
        dragElastic={0.03}
        dragMomentum={false}
        style={{ x }}
        onDragStart={() => {
          hapticFired.current = false;
        }}
        onDragEnd={() => {
          const shouldReply = x.get() <= -50;
          if (shouldReply) onReply();
          animate(x, 0, reduceMotion ? { duration: 0 } : {
            type: "spring",
            stiffness: 520,
            damping: 42,
            mass: 0.8,
            bounce: 0,
          });
        }}
      >
        {children}
      </motion.div>
    </div>
  );
}

function MessageReplyReference({ reply }: { reply: MobileReply }) {
  return (
    <div className="message-reply-reference">
      <span>{reply.role === "assistant" ? "Akashic" : "你"}</span>
      <p>{reply.preview}</p>
    </div>
  );
}

function ComposerReply({ target, onCancel }: { target: MobileMessage; onCancel: () => void }) {
  return (
    <div className="composer-reply" aria-label={`正在回复${target.role === "assistant" ? " Akashic" : "你的消息"}`}>
      <Reply size={18} />
      <div>
        <strong>回复 {target.role === "assistant" ? "Akashic" : "你"}</strong>
        <span>{replyPreview(target)}</span>
      </div>
      <button type="button" onClick={onCancel} aria-label="取消引用"><X size={19} /></button>
    </div>
  );
}

function MessageDateDivider({ createdAt }: { createdAt: number }) {
  return (
    <div className="message-date-divider" role="separator">
      <span />
      <time dateTime={new Date(createdAt).toISOString()}>{formatMessageDate(createdAt)}</time>
      <span />
    </div>
  );
}

function MessageUnreadDivider({ count }: { count: number }) {
  return (
    <div className="message-unread-divider" role="separator" aria-label={`${count} 条新消息`}>
      <span />
      <strong>{count} 条新消息</strong>
      <span />
    </div>
  );
}

const messageTimeFormatter = new Intl.DateTimeFormat("zh-CN", {
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});
const messageDateFormatter = new Intl.DateTimeFormat("zh-CN", {
  month: "long",
  day: "numeric",
  weekday: "short",
});

function formatMessageTime(value: number) {
  return messageTimeFormatter.format(new Date(value));
}

function formatDrawerTime(value: number) {
  const date = new Date(value);
  const now = new Date();
  if (sameLocalDay(value, now.getTime())) return formatMessageTime(value);
  const yesterday = new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1);
  if (sameLocalDay(value, yesterday.getTime())) return "昨天";
  return `${date.getMonth() + 1}/${date.getDate()}`;
}

function formatMessageDate(value: number) {
  const date = new Date(value);
  const today = new Date();
  if (sameLocalDay(date.getTime(), today.getTime())) return "今天";
  const yesterday = new Date(today.getFullYear(), today.getMonth(), today.getDate() - 1);
  if (sameLocalDay(date.getTime(), yesterday.getTime())) return "昨天";
  return messageDateFormatter.format(date);
}

function sameLocalDay(left: number, right: number) {
  const a = new Date(left);
  const b = new Date(right);
  return a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate();
}

function replyPreview(message: MobileMessage) {
  return message.content.trim().replace(/\s+/g, " ").slice(0, 512) || "[无文字消息]";
}

function MobileConversationBehavior({
  sessionId,
  sourceMessages,
  chatMessages,
  streaming,
  projectionGeneration,
  resyncing,
  suspended,
  forceScrollToken,
  readingPosition,
  messageElementsRef,
  onUnreadChange,
}: {
  sessionId?: string;
  sourceMessages: MobileMessage[];
  chatMessages: ChatMessage[];
  streaming: boolean;
  projectionGeneration: number;
  resyncing: boolean;
  suspended: boolean;
  forceScrollToken: number;
  readingPosition?: MobileReadingPosition;
  messageElementsRef: React.RefObject<Map<string, HTMLDivElement>>;
  onUnreadChange: (state: MobileUnreadState) => void;
}) {
  useMobileUnreadTracking(
    sessionId,
    sourceMessages,
    projectionGeneration,
    resyncing,
    suspended,
    onUnreadChange,
  );
  useMobileAutoScroll(sessionId, chatMessages, streaming, suspended, forceScrollToken);
  useMobileReadingPosition(sessionId, sourceMessages, readingPosition, messageElementsRef, suspended);
  return null;
}

/** 以可见消息为锚点持久化每个会话的阅读位置，并在回到会话时恢复。 */
function useMobileReadingPosition(
  sessionId: string | undefined,
  messages: MobileMessage[],
  readingPosition: MobileReadingPosition | undefined,
  messageElementsRef: React.RefObject<Map<string, HTMLDivElement>>,
  suspended: boolean,
) {
  const { isAtBottom, scrollRef, stopScroll } = useStickToBottomContext();
  const restoredSessionRef = useRef<string | undefined>(undefined);
  const saveTimerRef = useRef<number | null>(null);
  const lastSavedRef = useRef("");
  const lastReadAtRef = useRef(0);

  useLayoutEffect(() => {
    // 1. 每次进入会话只恢复一次已持久化锚点
    if (!sessionId || restoredSessionRef.current === sessionId || messages.length === 0) return;
    if (!readingPosition) {
      restoredSessionRef.current = sessionId;
      return;
    }
    const element = messageElementsRef.current.get(readingPosition.messageId);
    const scrollElement = scrollRef.current;
    if (!element || !scrollElement) return;
    restoredSessionRef.current = sessionId;
    stopScroll();
    element.scrollIntoView({ block: "start", behavior: "auto" });
    scrollElement.scrollTop -= readingPosition.offsetPx;
  }, [messageElementsRef, messages.length, readingPosition, scrollRef, sessionId, stopScroll]);

  useEffect(() => {
    // 2. 滚动停止后记录首个可见消息和相对偏移，避免按绝对 scrollTop 恢复后被乱序归位破坏
    const scrollElement = scrollRef.current;
    if (!sessionId || !scrollElement || suspended) return;
    const persist = () => {
      saveTimerRef.current = null;
      const viewportTop = scrollElement.getBoundingClientRect().top;
      const anchor = messages.find((message) => {
        const element = messageElementsRef.current.get(message.id);
        return element ? element.getBoundingClientRect().bottom > viewportTop + 1 : false;
      });
      if (!anchor) return;
      const element = messageElementsRef.current.get(anchor.id);
      if (!element) return;
      const offsetPx = Math.round(element.getBoundingClientRect().top - viewportTop);
      const key = `${sessionId}\u001f${anchor.id}\u001f${offsetPx}`;
      if (key === lastSavedRef.current) return;
      lastSavedRef.current = key;
      window.AkashicNative?.saveReadingPosition(sessionId, anchor.id, offsetPx);
    };
    const schedulePersist = () => {
      if (saveTimerRef.current !== null) window.clearTimeout(saveTimerRef.current);
      saveTimerRef.current = window.setTimeout(persist, 180);
    };
    scrollElement.addEventListener("scroll", schedulePersist, { passive: true });
    schedulePersist();
    return () => {
      scrollElement.removeEventListener("scroll", schedulePersist);
      if (saveTimerRef.current !== null) window.clearTimeout(saveTimerRef.current);
    };
  }, [messageElementsRef, messages, scrollRef, sessionId, suspended]);

  useEffect(() => {
    // 3. 真正回到底部时推进该会话的持久化已读水位
    const scrollElement = scrollRef.current;
    if (!sessionId || !isAtBottom || suspended || !scrollElement) return;
    const distanceFromBottom = scrollElement.scrollHeight - scrollElement.scrollTop - scrollElement.clientHeight;
    if (distanceFromBottom > 2) return;
    const readAt = messages.reduce(
      (latest, message) => message.role === "assistant" ? Math.max(latest, message.createdAt) : latest,
      0,
    );
    if (readAt <= lastReadAtRef.current) return;
    lastReadAtRef.current = readAt;
    window.AkashicNative?.markSessionReadThrough(sessionId, readAt);
  }, [isAtBottom, messages, scrollRef, sessionId, suspended]);
}

/** 按顶层助手消息追踪当前阅读位置之后的未读集合。 */
function useMobileUnreadTracking(
  sessionId: string | undefined,
  sourceMessages: MobileMessage[],
  projectionGeneration: number,
  resyncing: boolean,
  suspended: boolean,
  onUnreadChange: (state: MobileUnreadState) => void,
) {
  const { escapedFromLock, isAtBottom } = useStickToBottomContext();
  const trackedSessionRef = useRef<string | undefined>(undefined);
  const knownMessagesRef = useRef(new Map<string, MobileMessage>());
  const unseenMessageIdsRef = useRef<string[]>([]);
  const publishedUnreadKeyRef = useRef("");
  const projectionBaselineRef = useRef({ generation: projectionGeneration, rebuilding: false });
  const anchorMessageIdRef = useRef<string | undefined>(undefined);
  const anchorKeyRef = useRef<string | undefined>(undefined);
  const anchorOrdinalRef = useRef(0);

  useEffect(() => {
    const publishUnread = (ids: string[], migrations: ReadonlyMap<string, string>) => {
      const key = ids.join("\u001f");
      if (key === publishedUnreadKeyRef.current) return;
      publishedUnreadKeyRef.current = key;
      const firstMessageId = ids[0];
      if (!firstMessageId) {
        anchorMessageIdRef.current = undefined;
        anchorKeyRef.current = undefined;
      } else if (firstMessageId !== anchorMessageIdRef.current) {
        const migrated = anchorMessageIdRef.current
          ? migrations.get(anchorMessageIdRef.current)
          : undefined;
        if (migrated !== firstMessageId) {
          anchorOrdinalRef.current += 1;
          anchorKeyRef.current = `${sessionId}\u001f${anchorOrdinalRef.current}`;
        }
        anchorMessageIdRef.current = firstMessageId;
      }
      onUnreadChange({
        firstMessageId,
        anchorKey: anchorKeyRef.current,
        count: ids.length,
      });
    };

    // 1. 无会话或切换会话时，以当前完整历史建立已读基线
    if (!sessionId) {
      knownMessagesRef.current.clear();
      unseenMessageIdsRef.current = [];
      projectionBaselineRef.current = { generation: projectionGeneration, rebuilding: false };
      anchorMessageIdRef.current = undefined;
      anchorKeyRef.current = undefined;
      publishUnread([], new Map());
      return;
    }
    if (trackedSessionRef.current !== sessionId) {
      trackedSessionRef.current = sessionId;
      knownMessagesRef.current = new Map(sourceMessages.map((message) => [message.id, message]));
      unseenMessageIdsRef.current = [];
      projectionBaselineRef.current = { generation: projectionGeneration, rebuilding: false };
      anchorMessageIdRef.current = undefined;
      anchorKeyRef.current = undefined;
      publishedUnreadKeyRef.current = "";
      publishUnread([], new Map());
      return;
    }

    // 2. 只有原生声明的破坏性投影代际才重建已读基线
    const baseline = advanceMobileProjectionBaseline(
      projectionBaselineRef.current,
      projectionGeneration,
      resyncing,
    );
    projectionBaselineRef.current = baseline.state;
    const advanced = advanceMobileUnreadTracking(
      knownMessagesRef.current,
      unseenMessageIdsRef.current,
      sourceMessages,
      { escapedFromLock, isAtBottom, suspended },
      baseline.resetBaseline,
    );
    knownMessagesRef.current = advanced.knownMessages;
    unseenMessageIdsRef.current = advanced.unseenMessageIds;

    // 3. 仅在集合变化时发布，避免快照流触发无意义重绘
    publishUnread(unseenMessageIdsRef.current, advanced.messageIdMigrations);
  }, [
    escapedFromLock,
    isAtBottom,
    onUnreadChange,
    projectionGeneration,
    resyncing,
    sessionId,
    sourceMessages,
    suspended,
  ]);
}

/** 在用户未主动离开阅读位置时跟随新提问与流式回答。 */
function useMobileAutoScroll(
  sessionId: string | undefined,
  chatMessages: ChatMessage[],
  streaming: boolean,
  suspended: boolean,
  forceScrollToken: number,
) {
  const { escapedFromLock, isAtBottom, scrollToBottom } = useStickToBottomContext();
  const autoScrollSessionRef = useRef<string | undefined>(undefined);
  const handledForceScrollTokenRef = useRef(forceScrollToken);
  const lastMessage = chatMessages[chatMessages.length - 1];
  const lastBlock = lastMessage?.blocks[lastMessage.blocks.length - 1];
  const scrollKey = `${chatMessages.length}:${lastMessage?.id}:${lastMessage?.content.length}:${lastMessage?.blocks.length}:${lastBlock?.kind === "thinking" ? lastBlock.content.length : ""}`;

  useEffect(() => {
    // 1. 切换会话只建立快照基线，不制造一次假滚动
    if (autoScrollSessionRef.current !== sessionId) {
      autoScrollSessionRef.current = sessionId;
      handledForceScrollTokenRef.current = forceScrollToken;
      return;
    }

    // 2. 搜索接管阅读位置；否则发送动作强制到底，流式增量只跟随底部
    if (suspended) return;
    if (handledForceScrollTokenRef.current !== forceScrollToken) {
      handledForceScrollTokenRef.current = forceScrollToken;
      void scrollToBottom({ animation: "smooth", ignoreEscapes: true });
    } else if (streaming && isAtBottom && !escapedFromLock) {
      void scrollToBottom({ animation: "smooth", ignoreEscapes: false });
    }
  }, [escapedFromLock, forceScrollToken, isAtBottom, scrollKey, scrollToBottom, sessionId, streaming, suspended]);
}

/** 搜索期间解除底部滚动锁，并只在未发生跳转时恢复原底部位置。 */
function MobileSearchScrollBehavior({ active, targetMessageId }: { active: boolean; targetMessageId: string | null }) {
  const { escapedFromLock, isAtBottom, scrollRef, scrollToBottom, stopScroll } = useStickToBottomContext();
  const activeRef = useRef(false);
  const enteredAtBottomRef = useRef(false);
  const manuallyMovedRef = useRef(false);
  const movedRef = useRef(false);

  useLayoutEffect(() => {
    // 1. 进入搜索时先解除 ResizeObserver 的底部锁
    if (active && !activeRef.current) {
      enteredAtBottomRef.current = isAtBottom;
      manuallyMovedRef.current = false;
      movedRef.current = false;
      stopScroll();
    }
    if (active && activeRef.current && !escapedFromLock) stopScroll();

    // 2. 未跳转且未手动滚动的空搜索恢复原位置，否则保留阅读位置
    if (!active && activeRef.current && enteredAtBottomRef.current && !movedRef.current && !manuallyMovedRef.current) {
      void scrollToBottom({ animation: "smooth", ignoreEscapes: true });
    }
    activeRef.current = active;
  }, [active, escapedFromLock, isAtBottom, scrollToBottom, stopScroll]);

  useEffect(() => {
    if (!active) return;
    const scrollElement = scrollRef.current;
    if (!scrollElement) return;
    const markManualMove = () => { manuallyMovedRef.current = true; };
    scrollElement.addEventListener("touchmove", markManualMove, { passive: true });
    scrollElement.addEventListener("wheel", markManualMove, { passive: true });
    scrollElement.addEventListener("keydown", markManualMove);
    return () => {
      scrollElement.removeEventListener("touchmove", markManualMove);
      scrollElement.removeEventListener("wheel", markManualMove);
      scrollElement.removeEventListener("keydown", markManualMove);
    };
  }, [active, scrollRef]);

  useEffect(() => {
    if (active && targetMessageId !== null) movedRef.current = true;
  }, [active, targetMessageId]);
  return null;
}

function MobileScrollButton({
  unread,
  unreadAnchorVisited,
  onVisitUnread,
}: {
  unread: MobileUnreadState;
  unreadAnchorVisited: boolean;
  onVisitUnread: (messageId: string) => void;
}) {
  const { isAtBottom, scrollToBottom } = useStickToBottomContext();
  const visibleCount = unread.count > 99 ? "99+" : String(unread.count);
  return (
    <button
      className={`mobile-scroll-button ${isAtBottom ? "is-hidden" : ""}`}
      type="button"
      aria-hidden={isAtBottom}
      aria-label={unread.count > 0 ? `回到新消息，${unread.count} 条未读` : "回到对话底部"}
      tabIndex={isAtBottom ? -1 : 0}
      onClick={() => {
        if (unread.firstMessageId && !unreadAnchorVisited) {
          onVisitUnread(unread.firstMessageId);
          return;
        }
        void scrollToBottom({ animation: "smooth", ignoreEscapes: true });
      }}
    >
      {unread.count > 0 ? <span>{visibleCount}</span> : null}
      <ChevronDown size={21} />
    </button>
  );
}

/** 使用浏览器原生文本高亮注册表标记当前搜索目标中的命中词。 */
function MobileSearchTextHighlight({ query, messageId }: { query: string; messageId: string | null }) {
  useEffect(() => {
    // 1. 清理旧高亮并确认平台能力与当前目标
    const registry = (CSS as unknown as { highlights?: { set(name: string, value: unknown): void; delete(name: string): void } }).highlights;
    const HighlightConstructor = (window as Window & { Highlight?: new (...ranges: Range[]) => unknown }).Highlight;
    registry?.delete("akashic-search-match");
    if (!registry || !HighlightConstructor || !messageId || !query.trim()) return;
    const element = document.querySelector<HTMLElement>(`[data-message-id="${CSS.escape(messageId)}"]`);
    if (!element) return;
    const rawNeedle = query.trim();
    const needle = normalizeMobileSearchText(rawNeedle);
    if (needle.length !== rawNeedle.length) return;

    // 2. 收集目标消息内所有命中文本范围
    const ranges: Range[] = [];
    const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const text = node.nodeValue ?? "";
      const searchable = normalizeMobileSearchText(text);
      if (searchable.length !== text.length) continue;
      let offset = 0;
      while (offset < searchable.length) {
        const index = searchable.indexOf(needle, offset);
        if (index < 0) break;
        const range = document.createRange();
        range.setStart(node, index);
        range.setEnd(node, index + needle.length);
        ranges.push(range);
        offset = index + needle.length;
      }
    }

    // 3. 原子替换命中集合，并在目标变化或卸载时清理
    if (ranges.length > 0) registry.set("akashic-search-match", new HighlightConstructor(...ranges));
    return () => registry.delete("akashic-search-match");
  }, [messageId, query]);
  return null;
}

function toChatMessage(message: MobileMessage): ChatMessage {
  return {
    id: message.id,
    role: message.role,
    content: message.content,
    attachments: [],
    blocks: message.blocks.map(toAgentBlock),
    streaming: message.streaming,
    interrupted: message.interrupted,
    durationMs: message.durationSeconds !== undefined ? message.durationSeconds * 1000 : undefined,
  };
}

function isPluginTurnMessage(message: ChatMessage): boolean {
  return message.role === "assistant" && !message.id.startsWith("proactive:");
}

function pluginTurnId(message: ChatMessage): string | undefined {
  if (!message.streaming || !message.id.startsWith("assistant:")) return undefined;
  return message.id.slice("assistant:".length);
}

function toAgentBlock(block: MobileProcessBlock): AgentBlock {
  if (block.kind === "thinking") return { kind: "thinking", content: block.detail || block.title };
  const input = block.arguments === undefined
    ? (block.detail ? { description: block.detail } : {})
    : (block.detail && typeof block.arguments.description !== "string"
        ? { description: block.detail, ...block.arguments }
        : block.arguments);
  return {
    kind: "tool",
    callId: block.id,
    name: block.title,
    status: block.state === "running" ? "input-available" : block.state === "failed" ? "output-error" : "output-available",
    input,
    output: block.resultPreview ?? null,
    errorText: block.state === "failed" ? block.resultPreview ?? block.detail : undefined,
    durationMs: block.durationMillis,
  };
}

class MobileErrorBoundary extends React.Component<React.PropsWithChildren, { message: string | null }> {
  state: { message: string | null } = { message: null };

  static getDerivedStateFromError(error: unknown) {
    return { message: error instanceof Error ? error.message : "会话界面发生未知错误" };
  }

  componentDidCatch(error: unknown) {
    console.error("[mobile] render failed", error);
  }

  render() {
    if (!this.state.message) return this.props.children;
    return (
      <main className="mobile-fatal" role="alert">
        <AlertCircle className="mobile-fatal__mark" size={28} />
        <h1>会话界面没有正常载入</h1>
        <p>{this.state.message}</p>
        <button type="button" onClick={() => window.location.reload()}>
          <RefreshCw size={18} />
          重新载入
        </button>
      </main>
    );
  }
}

// Activity 的 adjustResize 已拥有 IME 高度；visualViewport 会在部分 Pixel WebView 上重复扣除键盘。
function syncMobileViewportHeight() {
  const viewportHeight = Math.max(1, Math.round(window.innerHeight));
  document.documentElement.style.setProperty("--mobile-viewport-height", `${viewportHeight}px`);
}

syncMobileViewportHeight();
window.addEventListener("resize", syncMobileViewportHeight);

const root = document.getElementById("root");
if (!root) throw new Error("Mobile Web root 不存在");
createRoot(root).render(
  <MobileErrorBoundary>
    <MobileNativeApp />
  </MobileErrorBoundary>,
);

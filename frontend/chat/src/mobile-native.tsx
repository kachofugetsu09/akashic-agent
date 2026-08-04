import "./mobile-polyfills";
import { installMobileBridge } from "./mobile-bridge";

import React, {
  lazy,
  startTransition,
  Suspense,
  type ReactNode,
  useCallback,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import { createRoot } from "react-dom/client";
import { useVirtualizer } from "@tanstack/react-virtual";
import {
  AlertCircle,
  ArchiveX,
  ArrowLeft,
  Check,
  ChevronRight,
  ChevronDown,
  ChevronUp,
  Copy,
  Download,
  FileText,
  LibraryBig,
  BookOpenText,
  Server,
  Timer,
  Menu,
  MessageSquarePlus,
  Paperclip,
  Palette,
  Puzzle,
  RefreshCw,
  Reply,
  RotateCcw,
  Search,
  SendHorizontal,
  Settings,
  Share2,
  Square,
  TimerReset,
  Wifi,
  WifiOff,
  X,
} from "lucide-react";
import { cycleTheme, initializeTheme, setTheme, useTheme } from "../../theme/src/theme-runtime";
import { ConversationNavigation } from "./conversation-navigation";
import {
  ComposerReply,
  MessageReplyReference,
  SharedMessageActions,
} from "./message-actions";
import { TooltipProvider } from "@/components/ui/tooltip";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  MobilePluginDashboard,
  MobilePluginSlot,
  receiveMobilePluginCatalog,
  receiveMobilePluginResult,
  type MobilePluginCatalog,
  type MobilePluginDashboardEntry,
  useMobilePluginDashboards,
} from "./mobile-plugin-runtime";
import {
  applyMobileStreamPatch,
  advanceMobileProjectionBaseline,
  advanceMobileUnreadTracking,
  allMobileAttachmentsReady,
  captureMobileComposerDraftWrite,
  formatMobileReplyNavigationAnnouncement,
  flushMobileComposerBeforePairing,
  formatMobileSelectionCopyText,
  isMobileImageViewerHistoryState,
  mobileMessageCanReply,
  mobileMessageHasCopyContent,
  mergeMobileComposerDraft,
  mobileSelectionActionAvailability,
  mobileComposerTextareaMetrics,
  mobileComposerDraftHydration,
  MOBILE_COMPOSER_DRAFT_MAX_LENGTH,
  nextMobileComposerDraftRevision,
  normalizeMobileComposerDraftText,
  normalizeMobileSearchText,
  reconcileMobileMessageSelection,
  reconcileMobileSnapshotMessages,
  reconcileMobileStreamItems,
  resolveMobileComposerDraft,
  resolveMobileReplyNavigationTarget,
  selectableMobileMessages,
  shouldClearAcceptedMobileComposerDraft,
  shouldClearMobileSelectionAfterShare,
  shouldSubmitMobileComposerKey,
  updateMobileSearchIndex,
  type MobileSearchIndexEntry,
  type MobileComposerDraft,
  type MobileComposerDraftWrite,
} from "./mobile-message-state";
import type { AgentBlock, ChatMessage } from "./main";
import { messageNeedsMarkdown } from "./message-rendering-policy";
import {
  advanceMobileStreamPresentation,
  MobileStreamProjectionStore,
} from "./mobile-stream-projection";
import {
  pushMobileSurface,
  readMobileSurfaceHistoryState,
  replaceMobileSurface,
  type MobileSurface,
} from "./mobile-surface-history";
import "./mobile-native.css";
import "./message-view.css";

const LazyChatMessageView = lazy(() =>
  import("./message-view").then(({ ChatMessageView }) => ({ default: ChatMessageView })),
);
const LazyMessageResponse = lazy(() =>
  import("@/components/ai-elements/message-response").then(({ MessageResponse }) => ({ default: MessageResponse })),
);

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

interface MobileConversationHandle {
  jumpToMessage(messageId: string, focus?: boolean): void;
}

interface MobileSession {
  id: string;
  title: string;
  lastMessagePreview?: string;
  lastMessageAt?: number;
  unreadCount: number;
  isRunning: boolean;
  isAvailable: boolean;
  canRemove: boolean;
}

interface MobileStreamPatch {
  protocolVersion: 3;
  projectionGeneration: number;
  selectedSessionId: string;
  messageIndex: number;
  messageId: string;
  searchRevision: number;
  durationSeconds?: number;
  contentAppend?: string;
  thinkingAppend?: {
    blockIndex: number;
    blockId: string;
    delta: string;
  };
  message?: MobileMessage;
  state?: MobileStatePatch;
}

type MobileStatePatch = Omit<MobileSnapshot, "protocolVersion" | "messages"> & {
  protocolVersion: 1;
};

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
  protocolVersion: 7;
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
  composer: {
    draft: MobileComposerDraft;
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
  runtimeInspection: MobileRuntimeInspection;
}

interface MobileRuntimeInspection {
  refreshing: boolean;
  detailLoading: boolean;
  snapshotId?: string;
  documents: MobileRuntimeDocument[];
  jobs: MobileRuntimeJob[];
  mcpServers: MobileRuntimeMcp[];
  pluginCount: number;
  skillCount: number;
  detail?: MobileRuntimeDetail;
  errorMessage?: string;
}

interface MobileRuntimeDocument {
  id: string;
  title: string;
  relativePath: string;
  description: string;
  available: boolean;
}

interface MobileRuntimeJob {
  id: string;
  name?: string;
  trigger: string;
  tier: string;
  fireAt: string;
  enabled: boolean;
}

interface MobileRuntimeMcp {
  ownerId: string;
  name: string;
  toolCount: number;
}

interface MobileRuntimeDetail {
  kind: "document" | "mcp" | "schedule";
  key: string;
  title: string;
  subtitle: string;
  markdown: string;
}

interface NativeBridge {
  reportHealthy(): void;
  requestSnapshot(): void;
  selectSession(sessionId: string): void;
  removeUnavailableSession(sessionId: string): void;
  createSession(): void;
  restartPairing(): void;
  reloadFromServer(): void;
  exportDiagnostics(): void;
  openSettings(): void;
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
  shareText(requestId: string, text: string): void;
  saveComposerDraft(sessionId: string, text: string, replyToMessageId: string, updatedAt: string): void;
  commitSharedText(
    draftId: string,
    sessionId: string,
    text: string,
    replyToMessageId: string,
  ): void;
  rejectSharedText(draftId: string, message: string): void;
  sendMessage(
    requestId: string,
    sessionId: string,
    text: string,
    replyToMessageId: string,
    attachmentIdsJson: string,
    sentDraftRevision: string,
  ): void;
  copyText(text: string): void;
  performActionHaptic(): void;
  sendCommand(command: string): void;
  refreshRuntimeInspection(): void;
  openRuntimeDocument(documentId: string): void;
  openRuntimeMcp(ownerId: string, serverName: string): void;
  openRuntimeJob(jobId: string): void;
  clearRuntimeInspectionDetail(): void;
  stopTurn(): void;
  queryPluginUi(
    requestId: string,
    ownerId: string,
    slot: string,
    sessionId: string | null,
    turnId: string | null,
    pluginId: string,
    method: string,
    payloadJson: string,
    cacheMode: string,
    transportMode: string,
  ): void;
  cancelPluginUiOwner(ownerId: string): void;
  setTheme(themeId: string): void;
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

/** 校验 native 在完整快照之后发送的单消息 streaming patch。 */
function parseMobileStreamPatch(value: unknown): MobileStreamPatch {
  const raw = requireRecord(value, "streamPatch");
  if (raw.protocolVersion !== 3) {
    throw new Error(`不支持的 stream patch 版本: ${String(raw.protocolVersion)}`);
  }
  const projectionGeneration = requireNonNegativeInteger(
    raw.projectionGeneration,
    "streamPatch.projectionGeneration",
  );
  const messageIndex = requireNonNegativeInteger(raw.messageIndex, "streamPatch.messageIndex");
  const messageId = requireString(raw.messageId, "streamPatch.messageId");
  const selectedSessionId = requireString(raw.selectedSessionId, "streamPatch.selectedSessionId");
  const message = raw.message === undefined ? undefined : parseMessage(raw.message, messageIndex);
  const state = raw.state === undefined ? undefined : parseMobileStatePatch(raw.state);
  if (message && (
    message.role !== "assistant"
    || message.sessionId !== selectedSessionId
  )) {
    throw new Error("stream patch 只能更新当前会话的助手消息");
  }
  if (message?.streaming && message.id !== messageId) {
    throw new Error("streaming patch 不得迁移消息 ID");
  }
  if (message && !message.streaming && (
    message.sessionId !== selectedSessionId
    || state === undefined
  )) {
    throw new Error("terminal patch 必须携带同会话控制状态");
  }
  if (state && (
    state.selectedSessionId !== selectedSessionId
    || state.projectionGeneration !== projectionGeneration
  )) {
    throw new Error("stream patch 控制状态与消息投影不一致");
  }
  if (state && message?.streaming !== false) {
    throw new Error("只有 terminal patch 可以携带控制状态");
  }
  const contentAppend = optionalString(raw.contentAppend, "streamPatch.contentAppend");
  const thinkingAppend = raw.thinkingAppend === undefined
    ? undefined
    : (() => {
        const append = requireRecord(raw.thinkingAppend, "streamPatch.thinkingAppend");
        return {
          blockIndex: requireNonNegativeInteger(append.blockIndex, "streamPatch.thinkingAppend.blockIndex"),
          blockId: requireString(append.blockId, "streamPatch.thinkingAppend.blockId"),
          delta: requireString(append.delta, "streamPatch.thinkingAppend.delta"),
        };
      })();
  if (message && (contentAppend !== undefined || thinkingAppend !== undefined)) {
    throw new Error("stream patch 不能同时携带完整消息和追加量");
  }
  if (!message && contentAppend === undefined && thinkingAppend === undefined) {
    throw new Error("stream patch 缺少消息变化");
  }
  return {
    protocolVersion: 3,
    projectionGeneration,
    selectedSessionId,
    messageIndex,
    messageId,
    searchRevision: requireNonNegativeInteger(raw.searchRevision, "streamPatch.searchRevision"),
    durationSeconds: raw.durationSeconds === undefined
      ? undefined
      : requireNonNegativeInteger(raw.durationSeconds, "streamPatch.durationSeconds"),
    contentAppend,
    thinkingAppend,
    message,
    state,
  };
}

/** 复用流式消息内语义未变化的块，避免已完成工具链随每个 token 重绘。 */
function reconcileMobileStreamMessage(previous: MobileMessage, next: MobileMessage): MobileMessage {
  const blocks = reconcileMobileStreamItems(previous.blocks, next.blocks, mobileProcessBlocksMatch);
  return blocks === next.blocks ? next : { ...next, blocks };
}

function mobileProcessBlocksMatch(previous: MobileProcessBlock, next: MobileProcessBlock) {
  return previous.id === next.id
    && previous.kind === next.kind
    && previous.title === next.title
    && previous.detail === next.detail
    && previous.state === next.state
    && previous.resultPreview === next.resultPreview
    && previous.durationMillis === next.durationMillis
    && JSON.stringify(previous.arguments) === JSON.stringify(next.arguments);
}

function mobileMessagePresentationMatches(previous: MobileMessage, next: MobileMessage) {
  if (previous.attachments.length !== next.attachments.length) return false;
  return previous.attachments.every((attachment, index) => {
    const candidate = next.attachments[index];
    return candidate !== undefined
      && attachment.id === candidate.id
      && attachment.filename === candidate.filename
      && attachment.contentType === candidate.contentType
      && attachment.sizeBytes === candidate.sizeBytes
      && attachment.transferredBytes === candidate.transferredBytes
      && attachment.state === candidate.state
      && attachment.canRemove === candidate.canRemove
      && attachment.contentUrl === candidate.contentUrl;
  });
}

function parseRuntimeInspection(value: unknown): MobileRuntimeInspection {
  const raw = requireRecord(value, "runtimeInspection");
  const detail = raw.detail === undefined || raw.detail === null
    ? undefined
    : (() => {
      const item = requireRecord(raw.detail, "runtimeInspection.detail");
      const kind = requireString(item.kind, "runtimeInspection.detail.kind");
      if (!["document", "mcp", "schedule"].includes(kind)) {
        throw new Error(`runtimeInspection.detail.kind 不受支持: ${kind}`);
      }
      return {
        kind: kind as MobileRuntimeDetail["kind"],
        key: requireString(item.key, "runtimeInspection.detail.key"),
        title: requireString(item.title, "runtimeInspection.detail.title"),
        subtitle: requireString(item.subtitle, "runtimeInspection.detail.subtitle"),
        markdown: requireString(item.markdown, "runtimeInspection.detail.markdown"),
      };
    })();
  return {
    refreshing: requireBoolean(raw.refreshing, "runtimeInspection.refreshing"),
    detailLoading: requireBoolean(raw.detailLoading, "runtimeInspection.detailLoading"),
    snapshotId: optionalString(raw.snapshotId, "runtimeInspection.snapshotId"),
    documents: requireArray(raw.documents, "runtimeInspection.documents", (value, index) => {
      const item = requireRecord(value, `runtimeInspection.documents[${index}]`);
      return {
        id: requireString(item.id, `runtimeInspection.documents[${index}].id`),
        title: requireString(item.title, `runtimeInspection.documents[${index}].title`),
        relativePath: requireString(item.relativePath, `runtimeInspection.documents[${index}].relativePath`),
        description: requireString(item.description, `runtimeInspection.documents[${index}].description`),
        available: requireBoolean(item.available, `runtimeInspection.documents[${index}].available`),
      };
    }),
    jobs: requireArray(raw.jobs, "runtimeInspection.jobs", (value, index) => {
      const item = requireRecord(value, `runtimeInspection.jobs[${index}]`);
      return {
        id: requireString(item.id, `runtimeInspection.jobs[${index}].id`),
        name: optionalString(item.name, `runtimeInspection.jobs[${index}].name`),
        trigger: requireString(item.trigger, `runtimeInspection.jobs[${index}].trigger`),
        tier: requireString(item.tier, `runtimeInspection.jobs[${index}].tier`),
        fireAt: requireString(item.fireAt, `runtimeInspection.jobs[${index}].fireAt`),
        enabled: requireBoolean(item.enabled, `runtimeInspection.jobs[${index}].enabled`),
      };
    }),
    mcpServers: requireArray(raw.mcpServers, "runtimeInspection.mcpServers", (value, index) => {
      const item = requireRecord(value, `runtimeInspection.mcpServers[${index}]`);
      return {
        ownerId: requireString(item.ownerId, `runtimeInspection.mcpServers[${index}].ownerId`),
        name: requireString(item.name, `runtimeInspection.mcpServers[${index}].name`),
        toolCount: requireNonNegativeInteger(item.toolCount, `runtimeInspection.mcpServers[${index}].toolCount`),
      };
    }),
    pluginCount: requireNonNegativeInteger(raw.pluginCount, "runtimeInspection.pluginCount"),
    skillCount: requireNonNegativeInteger(raw.skillCount, "runtimeInspection.skillCount"),
    detail,
    errorMessage: optionalString(raw.errorMessage, "runtimeInspection.errorMessage"),
  };
}

/** 在 native 协议边界校验完整快照，并只补齐 Kotlin 明确定义的默认字段。 */
function parseMobileSnapshot(value: unknown): MobileSnapshot {
  // 1. 校验协议版本与根对象
  const raw = requireRecord(value, "snapshot");
  if (raw.protocolVersion !== 7) throw new Error(`不支持的移动端协议版本: ${String(raw.protocolVersion)}`);
  const connection = requireRecord(raw.connection, "connection");
  const status = requireString(connection.status, "connection.status");
  if (!["connecting", "ready", "degraded", "reconnecting", "disconnected"].includes(status)) {
    throw new Error(`connection.status 不受支持: ${status}`);
  }

  // 2. 校验会话和消息
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
      isAvailable: requireBoolean(session.isAvailable, `sessions[${index}].isAvailable`),
      canRemove: requireBoolean(session.canRemove, `sessions[${index}].canRemove`),
    };
  });
  const messages = requireArray(raw.messages, "messages", parseMessage);

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
    protocolVersion: 7,
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
    composer: {
      draft: (() => {
        const draft = requireRecord(composer.draft, "composer.draft");
        const text = requireString(draft.text, "composer.draft.text");
        const replyToMessageId = optionalString(draft.replyToMessageId, "composer.draft.replyToMessageId");
        const updatedAt = draft.updatedAt === undefined || draft.updatedAt === null
          ? undefined
          : requireNonNegativeInteger(draft.updatedAt, "composer.draft.updatedAt");
        if ((text || replyToMessageId) && !updatedAt) {
          throw new Error("非空会话草稿缺少 revision");
        }
        return {
          text,
          replyToMessageId,
          updatedAt,
        };
      })(),
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
    runtimeInspection: parseRuntimeInspection(raw.runtimeInspection),
  };
}

/** Validate a native control-state patch while reusing the full snapshot boundary schema. */
function parseMobileStatePatch(value: unknown): MobileStatePatch {
  const raw = requireRecord(value, "statePatch");
  if (raw.protocolVersion !== 1) {
    throw new Error(`不支持的 state patch 版本: ${String(raw.protocolVersion)}`);
  }
  const parsed = parseMobileSnapshot({ ...raw, protocolVersion: 7, messages: [] });
  const { messages, protocolVersion, ...state } = parsed;
  void messages;
  void protocolVersion;
  return { protocolVersion: 1, ...state };
}

function parseMobilePluginCatalog(value: unknown): MobilePluginCatalog {
  const raw = requireRecord(value, "pluginCatalog");
  const revision = requireString(raw.catalogRevision, "pluginCatalog.catalogRevision");
  if (revision && !/^[0-9a-f]{64}$/.test(revision)) {
    throw new Error("pluginCatalog.catalogRevision 无效");
  }
  const plugins = requireArray(raw.plugins, "pluginCatalog.plugins", (item, index) => {
    const plugin = requireRecord(item, `pluginCatalog.plugins[${index}]`);
    const moduleUrl = requireString(plugin.moduleUrl, `pluginCatalog.plugins[${index}].moduleUrl`);
    const stylesheetUrl = optionalString(
      plugin.stylesheetUrl,
      `pluginCatalog.plugins[${index}].stylesheetUrl`,
    );
    for (const url of [moduleUrl, stylesheetUrl]) {
      if (url !== undefined && !url.startsWith("https://appassets.androidplatform.net/plugin-ui/")) {
        throw new Error(`pluginCatalog.plugins[${index}] 资源 URL 越界`);
      }
    }
    const navigation = optionalRecord(plugin.navigation, `pluginCatalog.plugins[${index}].navigation`);
    return {
      id: requireString(plugin.id, `pluginCatalog.plugins[${index}].id`),
      revision: requireString(plugin.revision, `pluginCatalog.plugins[${index}].revision`),
      moduleUrl,
      stylesheetUrl,
      navigation: navigation ? {
        label: requireString(navigation.label, `pluginCatalog.plugins[${index}].navigation.label`),
        description: requireString(
          navigation.description,
          `pluginCatalog.plugins[${index}].navigation.description`,
        ),
      } : undefined,
      slots: requireArray(plugin.slots, `pluginCatalog.plugins[${index}].slots`, (slot, slotIndex) => {
        const parsed = requireString(slot, `pluginCatalog.plugins[${index}].slots[${slotIndex}]`);
        if (!new Set(["turn.before_reasoning", "turn.before_tool", "turn.after_answer", "drawer.panel"]).has(parsed)) {
          throw new Error(`pluginCatalog.plugins[${index}] slot 无效`);
        }
        return parsed as "turn.before_reasoning" | "turn.before_tool" | "turn.after_answer" | "drawer.panel";
      }),
    };
  });
  if (new Set(plugins.map((plugin) => plugin.id)).size !== plugins.length) {
    throw new Error("pluginCatalog 插件 ID 重复");
  }
  return {
    catalogRevision: revision,
    updating: requireBoolean(raw.updating, "pluginCatalog.updating"),
    error: optionalString(raw.error, "pluginCatalog.error"),
    plugins,
  };
}

function parseMobilePluginResult(value: unknown) {
  const raw = requireRecord(value, "pluginResult");
  return {
    requestId: requireString(raw.requestId, "pluginResult.requestId"),
    resultJson: optionalString(raw.resultJson, "pluginResult.resultJson"),
    error: optionalString(raw.error, "pluginResult.error"),
  };
}

declare global {
  interface Window {
    AkashicNativeTransport?: {
      postMessage(message: string): void;
    };
    AkashicNative?: NativeBridge;
    AkashicMobile?: {
      receiveSnapshot(snapshot: unknown): void;
      receiveStreamPatch(patch: unknown): void;
      receiveStatePatch(patch: unknown): void;
      receivePluginCatalog(catalog: unknown): void;
      receivePluginUiResult(result: unknown): void;
      receiveSendResult(requestId: string, accepted: boolean): void;
      receiveShareResult(requestId: string, launched: boolean): void;
      receiveSharedText(draftId: string, sessionId: string, text: string): void;
      navigateBack(): boolean;
    };
  }
}

function MobileNativeApp() {
  const pluginDashboards = useMobilePluginDashboards();
  const [snapshot, setSnapshot] = useState<MobileSnapshot | null>(null);
  const [streamStore] = useState(() => new MobileStreamProjectionStore<MobileMessage>(
    {
      request: (callback) => window.requestAnimationFrame(callback),
      cancel: (handle) => window.cancelAnimationFrame(handle),
    },
    advanceMobileStreamPresentation,
  ));
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
  const [missingReplySourceId, setMissingReplySourceId] = useState<string | null>(null);
  const [replyNavigationAnnouncement, setReplyNavigationAnnouncement] = useState("");
  const [sharePending, setSharePending] = useState(false);
  const [shareStatus, setShareStatus] = useState<string | null>(null);
  const [selectedMessageIds, setSelectedMessageIds] = useState<Set<string>>(() => new Set());
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
  const conversationRef = useRef<MobileConversationHandle>(null);
  const copiedTimerRef = useRef<number | null>(null);
  const missingReplyTimerRef = useRef<number | null>(null);
  const replyAnnouncementTimerRef = useRef<number | null>(null);
  const searchHighlightTimerRef = useRef<number | null>(null);
  const previousSessionIdRef = useRef<string | undefined>(undefined);
  const handledNavigationTargetRef = useRef<string | undefined>(undefined);
  const pendingSendRequestRef = useRef<{
    requestId: string;
    draft: MobileComposerDraftWrite;
  } | null>(null);
  const pendingShareRequestRef = useRef<string | null>(null);
  const surfaceRef = useRef<MobileSurface>({ kind: "chat" });
  const selectionActiveRef = useRef(false);
  const appliedSharedTextIdsRef = useRef(new Set<string>());
  const pendingSharedTextRef = useRef<{ id: string; sessionId: string; text: string } | null>(null);
  const activeComposerDraftRef = useRef<MobileComposerDraftWrite | null>(null);
  const pendingComposerDraftRef = useRef<MobileComposerDraftWrite | null>(null);
  const optimisticComposerDraftsRef = useRef(new Map<string, MobileComposerDraftWrite>());
  const composerDraftTimerRef = useRef<number | null>(null);
  const snapshotMessagesRef = useRef<MobileMessage[]>([]);
  const streamSnapshotRef = useRef<MobileSnapshot | null>(null);
  const composerInputRef = useRef("");

  const saveComposerDraft = useCallback((draft: MobileComposerDraftWrite) => {
    optimisticComposerDraftsRef.current.set(draft.sessionId, draft);
    window.AkashicNative?.saveComposerDraft(
      draft.sessionId,
      draft.text,
      draft.replyToMessageId ?? "",
      String(draft.updatedAt),
    );
  }, []);

  const flushComposerDraft = useCallback(() => {
    if (composerDraftTimerRef.current !== null) {
      window.clearTimeout(composerDraftTimerRef.current);
      composerDraftTimerRef.current = null;
    }
    const pending = pendingComposerDraftRef.current;
    pendingComposerDraftRef.current = null;
    if (pending) saveComposerDraft(pending);
  }, [saveComposerDraft]);

  const scheduleComposerDraft = useCallback((draft: MobileComposerDraftWrite) => {
    pendingComposerDraftRef.current = draft;
    if (composerDraftTimerRef.current !== null) window.clearTimeout(composerDraftTimerRef.current);
    composerDraftTimerRef.current = window.setTimeout(() => {
      composerDraftTimerRef.current = null;
      const pending = pendingComposerDraftRef.current;
      pendingComposerDraftRef.current = null;
      if (pending) saveComposerDraft(pending);
    }, 250);
  }, [saveComposerDraft]);

  const updateComposerDraft = useCallback((text: string, target: MobileMessage | null) => {
    const current = activeComposerDraftRef.current;
    const normalizedText = normalizeMobileComposerDraftText(text);
    const draft = captureMobileComposerDraftWrite(
      current?.sessionId,
      normalizedText,
      target?.id,
      nextMobileComposerDraftRevision(current?.updatedAt, Date.now()),
    );
    setInput(normalizedText);
    setReplyTarget(target);
    if (!draft) return;
    activeComposerDraftRef.current = draft;
    optimisticComposerDraftsRef.current.delete(draft.sessionId);
    scheduleComposerDraft(draft);
  }, [scheduleComposerDraft]);

  const clearAcceptedComposerDraft = useCallback((sessionId: string) => {
    const current = activeComposerDraftRef.current;
    const cleared = {
      sessionId,
      text: "",
      updatedAt: nextMobileComposerDraftRevision(current?.updatedAt, Date.now()),
    };
    if (current?.sessionId !== sessionId) {
      saveComposerDraft(cleared);
      return;
    }
    if (composerDraftTimerRef.current !== null) {
      window.clearTimeout(composerDraftTimerRef.current);
      composerDraftTimerRef.current = null;
    }
    pendingComposerDraftRef.current = null;
    activeComposerDraftRef.current = cleared;
    setInput("");
    setReplyTarget(null);
    saveComposerDraft(cleared);
  }, [saveComposerDraft]);

  const applySharedText = useCallback((draftId: string, sessionId: string, text: string) => {
    const current = activeComposerDraftRef.current;
    if (!current || current.sessionId !== sessionId) return false;

    if (appliedSharedTextIdsRef.current.has(draftId)) {
      window.AkashicNative?.commitSharedText(
        draftId,
        sessionId,
        current.text,
        current.replyToMessageId ?? "",
      );
      return true;
    }

    // 1. 复用当前会话草稿 owner，并保留已有引用目标
    const merged = mergeMobileComposerDraft(current.text, text);
    if (merged === null) {
      window.AkashicNative?.rejectSharedText(draftId, "当前输入空间不足，请精简草稿后重试");
      return true;
    }
    const next = {
      ...current,
      text: merged,
      updatedAt: nextMobileComposerDraftRevision(current.updatedAt, Date.now()),
    };
    if (composerDraftTimerRef.current !== null) {
      window.clearTimeout(composerDraftTimerRef.current);
      composerDraftTimerRef.current = null;
    }
    pendingComposerDraftRef.current = null;
    activeComposerDraftRef.current = next;
    setInput(next.text);
    optimisticComposerDraftsRef.current.set(next.sessionId, next);

    // 2. 回到对话任务面并在持久化请求发出后确认消费
    if (surfaceRef.current.kind !== "chat") {
      replaceMobileSurface(window.history, { kind: "chat" });
      surfaceRef.current = { kind: "chat" };
      setSurface({ kind: "chat" });
    }
    setDrawerOpen(false);
    setSearchOpen(false);
    searchOpenRef.current = false;
    setSearchQuery("");
    normalizedSearchQueryRef.current = "";
    setSearchIndex(new Map());
    setSearchTargetId(null);
    setHighlightedMessageId(null);
    setCommandsOpen(false);
    setQueueOpen(false);
    selectionActiveRef.current = false;
    setSelectedMessageIds(new Set());
    appliedSharedTextIdsRef.current.add(draftId);
    window.AkashicNative?.commitSharedText(
      draftId,
      sessionId,
      next.text,
      next.replyToMessageId ?? "",
    );
    requestAnimationFrame(() => textareaRef.current?.focus());
    return true;
  }, []);

  useEffect(() => {
    let requestTimer: number | null = null;
    let snapshotAccepted = false;
    const requestSnapshot = () => {
      if (snapshotAccepted) return;
      window.AkashicNative?.requestSnapshot();
      requestTimer = window.setTimeout(requestSnapshot, 250);
    };
    const previousScrollRestoration = window.history.scrollRestoration;
    window.history.scrollRestoration = "manual";
    replaceMobileSurface(window.history, { kind: "chat" });
    const handlePopState = (event: PopStateEvent) => {
      const next = readMobileSurfaceHistoryState(event.state);
      surfaceRef.current = next;
      setSurface(next);
    };
    window.addEventListener("popstate", handlePopState);
    window.AkashicMobile = {
      receiveSnapshot(next) {
        let nextSnapshot: MobileSnapshot;
        try {
          nextSnapshot = parseMobileSnapshot(next);
          setSnapshotError(null);
        } catch (error) {
          console.error("[mobile] rejected native snapshot", error);
          setSnapshotError(error instanceof Error ? error.message : "原生快照无效");
          return;
        }
        const delivered = streamSnapshotRef.current;
        if (delivered && delivered.selectedSessionId === nextSnapshot.selectedSessionId) {
          nextSnapshot = {
            ...nextSnapshot,
            messages: reconcileMobileSnapshotMessages(
              delivered.messages,
              nextSnapshot.messages,
              mobileMessagePresentationMatches,
            ),
          };
        }
        streamSnapshotRef.current = nextSnapshot;
        if (searchOpenRef.current && normalizedSearchQueryRef.current) {
          setSearchIndex((current) => updateMobileSearchIndex(
            current,
            nextSnapshot.messages,
            normalizedSearchQueryRef.current,
            false,
          ));
        }
        setSnapshot(nextSnapshot);
        streamStore.clear();
        snapshotAccepted = true;
        if (requestTimer !== null) window.clearTimeout(requestTimer);
      },
      receiveStreamPatch(next) {
        let parsed: MobileStreamPatch;
        try {
          parsed = parseMobileStreamPatch(next);
          setSnapshotError(null);
        } catch (error) {
          console.error("[mobile] rejected native stream patch", error);
          setSnapshotError(error instanceof Error ? error.message : "原生流式 patch 无效");
          return;
        }
        const current = streamSnapshotRef.current;
        if (current === null) {
          window.AkashicNative?.requestSnapshot();
          return;
        }
        const previousMessage = current.messages[parsed.messageIndex];
        const reconciledPatch = previousMessage && parsed.message
          ? { ...parsed, message: reconcileMobileStreamMessage(previousMessage, parsed.message) }
          : parsed;
        const applied = applyMobileStreamPatch(current, reconciledPatch);
        if (applied === null || previousMessage === undefined) {
          window.AkashicNative?.requestSnapshot();
          return;
        }
        let nextSnapshot = applied;
        if (parsed.state) {
          const { protocolVersion, ...state } = parsed.state;
          void protocolVersion;
          nextSnapshot = { ...nextSnapshot, ...state };
        }
        const nextMessage = nextSnapshot.messages[parsed.messageIndex];
        if (nextMessage === undefined) throw new Error("stream patch 应产生目标消息");
        streamSnapshotRef.current = nextSnapshot;
        const immediate = !nextMessage.streaming
          || window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        streamStore.publish(parsed.messageId, previousMessage, nextMessage, immediate);
        if (searchOpenRef.current && normalizedSearchQueryRef.current) {
          setSearchIndex((index) => updateMobileSearchIndex(
            index,
            nextSnapshot.messages,
            normalizedSearchQueryRef.current,
            false,
          ));
        }
        if (parsed.state || !nextMessage.streaming) setSnapshot(nextSnapshot);
      },
      receiveStatePatch(next) {
        let patch: MobileStatePatch;
        try {
          patch = parseMobileStatePatch(next);
          setSnapshotError(null);
        } catch (error) {
          console.error("[mobile] rejected native state patch", error);
          setSnapshotError(error instanceof Error ? error.message : "原生状态 patch 无效");
          return;
        }
        const projected = streamSnapshotRef.current;
        if (
          projected === null
          || projected.selectedSessionId !== patch.selectedSessionId
          || projected.projectionGeneration !== patch.projectionGeneration
        ) {
          window.AkashicNative?.requestSnapshot();
          return;
        }
        const { protocolVersion, ...state } = patch;
        void protocolVersion;
        streamSnapshotRef.current = { ...projected, ...state };
        startTransition(() => {
          setSnapshot((current) => {
            if (
              current === null ||
              current.selectedSessionId !== patch.selectedSessionId ||
              current.projectionGeneration !== patch.projectionGeneration
            ) {
              window.AkashicNative?.requestSnapshot();
              return current;
            }
            return { ...current, ...state };
          });
        });
      },
      receivePluginCatalog(nextCatalog) {
        let parsed: MobilePluginCatalog;
        try {
          parsed = parseMobilePluginCatalog(nextCatalog);
        } catch (error) {
          console.error("[mobile] rejected plugin catalog", error);
          setPluginLoadError(error instanceof Error ? error.message : "插件目录无效");
          return;
        }
        void receiveMobilePluginCatalog(parsed).then(
          () => setPluginLoadError(null),
          (error: unknown) => {
            console.error("[mobile] failed to activate plugin catalog", error);
            setPluginLoadError(error instanceof Error ? error.message : "插件界面加载失败");
          },
        );
      },
      receivePluginUiResult(result) {
        try {
          receiveMobilePluginResult(parseMobilePluginResult(result));
        } catch (error) {
          console.error("[mobile] rejected plugin result", error);
        }
      },
      receiveSendResult(requestId, accepted) {
        const pendingSend = pendingSendRequestRef.current;
        if (pendingSend?.requestId !== requestId) return;
        pendingSendRequestRef.current = null;
        setSendPending(false);
        if (!accepted) return;
        setSendScrollRequest((current) => current + 1);
        if (shouldClearAcceptedMobileComposerDraft(activeComposerDraftRef.current, pendingSend.draft)) {
          clearAcceptedComposerDraft(pendingSend.draft.sessionId);
        } else {
          flushComposerDraft();
        }
        setCommandsOpen(false);
        textareaRef.current?.blur();
      },
      receiveShareResult(requestId, launched) {
        const pendingRequestId = pendingShareRequestRef.current;
        if (pendingRequestId !== requestId) return;
        pendingShareRequestRef.current = null;
        setSharePending(false);
        if (shouldClearMobileSelectionAfterShare(pendingRequestId, requestId, launched)) {
          selectionActiveRef.current = false;
          setSelectedMessageIds(new Set());
          setShareStatus(null);
          return;
        }
        setShareStatus("分享未打开，请重试");
      },
      receiveSharedText(draftId, sessionId, text) {
        if (!applySharedText(draftId, sessionId, text)) {
          pendingSharedTextRef.current = { id: draftId, sessionId, text };
        }
      },
      navigateBack() {
        if (selectionActiveRef.current) {
          if (pendingShareRequestRef.current !== null) return true;
          pendingShareRequestRef.current = null;
          setSharePending(false);
          setShareStatus(null);
          selectionActiveRef.current = false;
          setSelectedMessageIds(new Set());
          return true;
        }
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
    const receiveNativeMessage = (event: MessageEvent<unknown>) => {
      if (typeof event.data !== "string") return;
      try {
        const message = requireRecord(JSON.parse(event.data), "nativeMessage");
        const type = requireString(message.type, "nativeMessage.type");
        if (type === "mobile.snapshot") {
          window.AkashicMobile?.receiveSnapshot(message.payload);
          return;
        }
        if (type === "mobile.stream-patch") {
          window.AkashicMobile?.receiveStreamPatch(message.payload);
          return;
        }
        if (type === "mobile.state-patch") {
          window.AkashicMobile?.receiveStatePatch(message.payload);
          return;
        }
        if (type === "mobile.theme") {
          setTheme(requireString(message.payload, "nativeMessage.payload"), false);
          return;
        }
        if (type === "plugin.catalog") {
          window.AkashicMobile?.receivePluginCatalog(message.payload);
          return;
        }
        if (type === "plugin.result") {
          window.AkashicMobile?.receivePluginUiResult(message.payload);
          return;
        }
        throw new Error(`nativeMessage.type 无效: ${type}`);
      } catch (error) {
        console.error("[mobile] rejected native message", error);
      }
    };
    window.addEventListener("message", receiveNativeMessage);
    requestSnapshot();
    return () => {
      if (requestTimer !== null) window.clearTimeout(requestTimer);
      window.removeEventListener("popstate", handlePopState);
      window.removeEventListener("message", receiveNativeMessage);
      window.history.scrollRestoration = previousScrollRestoration;
      streamStore.clear();
      delete window.AkashicMobile;
    };
  }, [applySharedText, clearAcceptedComposerDraft, flushComposerDraft, streamStore]);

  useEffect(() => {
    if (!snapshot) return;
    const frame = window.requestAnimationFrame(() => {
      window.AkashicNative?.reportHealthy();
    });
    return () => window.cancelAnimationFrame(frame);
  }, [snapshot]);

  useEffect(() => {
    if (snapshot?.composer.isStopping || !snapshot?.composer.canStop || snapshot?.connection.error) {
      setStopRequested(false);
    }
  }, [snapshot?.composer.canStop, snapshot?.composer.isStopping, snapshot?.connection.error]);

  useLayoutEffect(() => {
    const sessionId = snapshot?.selectedSessionId;
    const previous = activeComposerDraftRef.current;
    if (!snapshot || !sessionId) {
      if (previous) flushComposerDraft();
      activeComposerDraftRef.current = null;
      setInput("");
      setReplyTarget(null);
      return;
    }

    // 1. 切换会话或原生确认 owner 写入时，成对恢复文字与引用
    const sessionChanged = previous?.sessionId !== sessionId;
    const optimistic = optimisticComposerDraftsRef.current.get(sessionId);
    const hydration = mobileComposerDraftHydration(snapshot.composer.draft, optimistic);
    const ownerAcknowledged = hydration.ownerAcknowledged;
    if (sessionChanged) flushComposerDraft();
    if (optimistic && !ownerAcknowledged && !sessionChanged) return;
    if (sessionChanged || ownerAcknowledged) {
      if (ownerAcknowledged) optimisticComposerDraftsRef.current.delete(sessionId);
      const resolved = resolveMobileComposerDraft(
        hydration.draft,
        snapshot.messages,
        sessionId,
      );
      const hydrated = captureMobileComposerDraftWrite(
        sessionId,
        resolved.text,
        resolved.replyTarget?.id,
        resolved.updatedAt ?? nextMobileComposerDraftRevision(undefined, Date.now()),
      );
      if (!hydrated) throw new Error("已选会话无法建立输入草稿");
      activeComposerDraftRef.current = hydrated;
      setInput(resolved.text);
      setReplyTarget(resolved.replyTarget);
      if (resolved.cleanedDraft) {
        const cleaned = {
          sessionId,
          ...resolved.cleanedDraft,
          updatedAt: nextMobileComposerDraftRevision(hydrated.updatedAt, Date.now()),
        };
        activeComposerDraftRef.current = cleaned;
        saveComposerDraft(cleaned);
      }
      return;
    }

    // 2. 当前引用目标消失时立即隐藏，并让原生 owner 清理悬空 ID
    if (!previous?.replyToMessageId) return;
    const resolved = resolveMobileComposerDraft(previous, snapshot.messages, sessionId);
    if (!resolved.cleanedDraft) return;
    const cleaned = {
      sessionId,
      ...resolved.cleanedDraft,
      updatedAt: nextMobileComposerDraftRevision(previous.updatedAt, Date.now()),
    };
    activeComposerDraftRef.current = cleaned;
    setReplyTarget(null);
    if (composerDraftTimerRef.current !== null) {
      window.clearTimeout(composerDraftTimerRef.current);
      composerDraftTimerRef.current = null;
    }
    pendingComposerDraftRef.current = null;
    saveComposerDraft(cleaned);
  }, [
    flushComposerDraft,
    saveComposerDraft,
    snapshot,
  ]);

  useLayoutEffect(() => {
    const pending = pendingSharedTextRef.current;
    if (pending && applySharedText(pending.id, pending.sessionId, pending.text)) {
      pendingSharedTextRef.current = null;
    }
  }, [applySharedText, snapshot?.composer.draft, snapshot?.selectedSessionId]);

  useEffect(() => {
    const flushWhenHidden = () => {
      if (document.visibilityState === "hidden") flushComposerDraft();
    };
    document.addEventListener("visibilitychange", flushWhenHidden);
    return () => {
      document.removeEventListener("visibilitychange", flushWhenHidden);
      flushComposerDraft();
    };
  }, [flushComposerDraft]);

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
    setSelectedMessageIds((current) => {
      if (current.size === 0) return current;
      const next = reconcileMobileMessageSelection(current, snapshot?.messages ?? []);
      selectionActiveRef.current = next.size > 0;
      if (next.size === current.size && [...next].every((messageId) => current.has(messageId))) return current;
      return next;
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
    if (missingReplyTimerRef.current !== null) window.clearTimeout(missingReplyTimerRef.current);
    if (replyAnnouncementTimerRef.current !== null) window.clearTimeout(replyAnnouncementTimerRef.current);
    if (searchHighlightTimerRef.current !== null) window.clearTimeout(searchHighlightTimerRef.current);
  }, []);

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
    // 1. 由虚拟列表先挂载目标行，再完成定位和焦点恢复
    conversationRef.current?.jumpToMessage(messageId, focus);

    // 2. 点亮目标状态层并恢复无障碍焦点
    setHighlightedMessageId(messageId);
    if (searchHighlightTimerRef.current !== null) window.clearTimeout(searchHighlightTimerRef.current);
    searchHighlightTimerRef.current = window.setTimeout(() => {
      setHighlightedMessageId((current) => current === messageId ? null : current);
      searchHighlightTimerRef.current = null;
    }, 1300);
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
    setSearchOpen(false);
    searchOpenRef.current = false;
    setSearchQuery("");
    normalizedSearchQueryRef.current = "";
    setSearchIndex(new Map());
    setSearchTargetId(null);
    setUnreadState({ count: 0 });
    setQueueOpen(false);
    pendingShareRequestRef.current = null;
    setSharePending(false);
    setShareStatus(null);
    selectionActiveRef.current = false;
    setSelectedMessageIds(new Set());
  }, [snapshot?.selectedSessionId]);

  useEffect(() => {
    setUnreadAnchorVisited(false);
  }, [unreadState.anchorKey]);

  useLayoutEffect(() => {
    snapshotMessagesRef.current = snapshot?.messages ?? [];
  }, [snapshot?.messages]);

  const baselineMessages = snapshot?.messages;
  useEffect(() => {
    if (baselineMessages) streamStore.reconcileBaseline(baselineMessages);
  }, [baselineMessages, streamStore]);

  useLayoutEffect(() => {
    composerInputRef.current = input;
  }, [input]);

  const copyMessage = useCallback((message: MobileMessage) => {
    window.AkashicNative?.copyText(message.content);
    setCopiedMessageId(message.id);
    if (copiedTimerRef.current !== null) window.clearTimeout(copiedTimerRef.current);
    copiedTimerRef.current = window.setTimeout(() => {
      setCopiedMessageId((current) => current === message.id ? null : current);
      copiedTimerRef.current = null;
    }, 1600);
  }, []);
  const enterSelection = useCallback((messageId: string) => {
    if (pendingShareRequestRef.current !== null) return;
    setDrawerOpen(false);
    setSearchOpen(false);
    searchOpenRef.current = false;
    setSearchQuery("");
    normalizedSearchQueryRef.current = "";
    setSearchIndex(new Map());
    setSearchTargetId(null);
    setHighlightedMessageId(null);
    setCommandsOpen(false);
    setQueueOpen(false);
    updateComposerDraft(composerInputRef.current, null);
    textareaRef.current?.blur();
    pendingShareRequestRef.current = null;
    setSharePending(false);
    setShareStatus(null);
    selectionActiveRef.current = true;
    setSelectedMessageIds(new Set([messageId]));
  }, [updateComposerDraft]);
  const toggleSelection = useCallback((messageId: string) => {
    if (pendingShareRequestRef.current !== null) return;
    setShareStatus(null);
    setSelectedMessageIds((current) => {
      const next = new Set(current);
      if (next.has(messageId)) next.delete(messageId);
      else next.add(messageId);
      selectionActiveRef.current = next.size > 0;
      return next;
    });
  }, []);
  const navigateToReply = useCallback((sourceMessageId: string, reply: MobileReply) => {
    const target = resolveMobileReplyNavigationTarget(reply.messageId, snapshotMessagesRef.current);
    if (target) {
      setMissingReplySourceId(null);
      const announcement = formatMobileReplyNavigationAnnouncement(target, formatMessageTime);
      setReplyNavigationAnnouncement("");
      if (replyAnnouncementTimerRef.current !== null) window.clearTimeout(replyAnnouncementTimerRef.current);
      replyAnnouncementTimerRef.current = window.setTimeout(() => {
        setReplyNavigationAnnouncement(announcement);
        replyAnnouncementTimerRef.current = null;
      }, 40);
      jumpToMessage(target.id, true);
      return;
    }
    setMissingReplySourceId(sourceMessageId);
    if (missingReplyTimerRef.current !== null) window.clearTimeout(missingReplyTimerRef.current);
    missingReplyTimerRef.current = window.setTimeout(() => {
      setMissingReplySourceId((current) => current === sourceMessageId ? null : current);
      missingReplyTimerRef.current = null;
    }, 1800);
  }, [jumpToMessage]);
  const retryMessageDelivery = useCallback((messageId: string) => {
    setRecoveringMessageIds((current) => new Set(current).add(messageId));
    window.AkashicNative?.performActionHaptic();
    window.AkashicNative?.retryFailedMessage(messageId);
  }, []);
  const replyToMessage = useCallback((message: MobileMessage) => {
    updateComposerDraft(composerInputRef.current, message);
  }, [updateComposerDraft]);

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
  const selectedSession = snapshot.sessions.find((session) => session.id === snapshot.selectedSessionId);
  const selectedSessionUnavailable = selectedSession?.isAvailable === false;

  const send = () => {
    const text = input.trim();
    const attachmentsReady = allMobileAttachmentsReady(snapshot.composer.attachments);
    if (sendPending || !snapshot.composer.canSend || !attachmentsReady) return;
    if (!text && snapshot.composer.attachments.length === 0) return;
    const native = window.AkashicNative;
    const sessionId = snapshot.selectedSessionId;
    if (!native) return;
    if (!sessionId) throw new Error("发送消息缺少当前会话 owner");
    const requestId = `send-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const sentDraft = captureMobileComposerDraftWrite(
      sessionId,
      input,
      replyTarget?.id,
      activeComposerDraftRef.current?.updatedAt,
    );
    if (!sentDraft) throw new Error("发送消息无法捕获当前会话草稿");
    flushComposerDraft();
    pendingSendRequestRef.current = { requestId, draft: sentDraft };
    setSendPending(true);
    native.sendMessage(
      requestId,
      sessionId,
      input,
      replyTarget?.id ?? "",
      JSON.stringify(snapshot.composer.attachments.map((attachment) => attachment.id)),
      String(sentDraft.updatedAt),
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
  const selectedMessages = selectableMobileMessages(snapshot.messages, selectedMessageIds);
  const selectionActive = selectedMessages.length > 0;
  const clearSelection = () => {
    pendingShareRequestRef.current = null;
    setSharePending(false);
    setShareStatus(null);
    selectionActiveRef.current = false;
    setSelectedMessageIds(new Set());
  };
  const copySelection = () => {
    const text = formatMobileSelectionCopyText(
      selectedMessages,
      (createdAt) => `${formatMessageDate(createdAt)} ${formatMessageTime(createdAt)}`,
    );
    window.AkashicNative?.copyText(text);
    window.AkashicNative?.performActionHaptic();
    clearSelection();
  };
  const shareSelection = () => {
    const text = formatMobileSelectionCopyText(
      selectedMessages,
      (createdAt) => `${formatMessageDate(createdAt)} ${formatMessageTime(createdAt)}`,
    );
    const requestId = crypto.randomUUID();
    pendingShareRequestRef.current = requestId;
    setSharePending(true);
    setShareStatus("正在打开系统分享");
    window.AkashicNative?.shareText(requestId, text);
  };
  const replyToSelection = () => {
    const target = selectedMessages.length === 1 ? selectedMessages[0] : undefined;
    if (!target || !mobileMessageCanReply(target, snapshot.selectedSessionId)) return;
    clearSelection();
    updateComposerDraft(input, target);
    requestAnimationFrame(() => textareaRef.current?.focus());
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
            selectionCount={selectedMessages.length}
            canCopySelection={selectedMessages.some(mobileMessageHasCopyContent)}
            canShareSelection={selectedMessages.some(mobileMessageHasCopyContent)}
            sharePending={sharePending}
            shareStatus={shareStatus}
            canReplyToSelection={selectedMessages.length === 1 && mobileMessageCanReply(selectedMessages[0], snapshot.selectedSessionId)}
            onToggleDrawer={toggleDrawer}
            onOpenSearch={openSearch}
            onCloseSearch={closeSearch}
            onSearchQuery={updateSearchQuery}
            onSearchSubmit={() => {
              if (searchTargetId !== null) jumpToMessage(searchTargetId, true);
            }}
            onCloseSelection={clearSelection}
            onCopySelection={copySelection}
            onShareSelection={shareSelection}
            onReplyToSelection={replyToSelection}
          />
        ) : (
          <MobilePluginTopBar
            title={surface.kind === "plugins"
              ? "插件"
              : surface.kind === "runtime"
                ? "知识与运行"
                : surface.kind === "runtime-detail"
                  ? snapshot.runtimeInspection.detail?.title ?? "详情"
                  : pluginDashboards.find((plugin) => plugin.id === surface.pluginId)?.label ?? "插件看板"}
            onBack={() => window.history.back()}
          />
        )}
        <MobileDrawer
          open={drawerOpen}
          snapshot={snapshot}
          pluginCount={pluginDashboards.length}
          onOpenRuntime={() => {
            window.AkashicNative?.refreshRuntimeInspection();
            navigateToSurface({ kind: "runtime" });
            closeDrawer();
          }}
          onOpenPlugins={() => {
            navigateToSurface({ kind: "plugins" });
            closeDrawer();
          }}
          onOpenSettings={() => {
            window.AkashicNative?.openSettings();
            closeDrawer();
          }}
          onRestartPairing={() => {
            flushMobileComposerBeforePairing(
              flushComposerDraft,
              () => window.AkashicNative?.restartPairing(),
            );
          }}
          onClose={closeDrawer}
        />
        <span className="mobile-a11y-announcement" aria-live="polite" aria-atomic="true">
          {replyNavigationAnnouncement}
        </span>
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

        {surface.kind === "runtime" ? (
          <RuntimeInspectionDirectory
            inspection={snapshot.runtimeInspection}
            onRefresh={() => window.AkashicNative?.refreshRuntimeInspection()}
            onOpenDocument={(documentId) => {
              window.AkashicNative?.openRuntimeDocument(documentId);
              navigateToSurface({ kind: "runtime-detail", detailKind: "document", key: documentId });
            }}
            onOpenMcp={(ownerId, name) => {
              window.AkashicNative?.openRuntimeMcp(ownerId, name);
              navigateToSurface({ kind: "runtime-detail", detailKind: "mcp", key: `${ownerId}/${name}` });
            }}
            onOpenJob={(jobId) => {
              window.AkashicNative?.openRuntimeJob(jobId);
              navigateToSurface({ kind: "runtime-detail", detailKind: "schedule", key: jobId });
            }}
          />
        ) : surface.kind === "runtime-detail" ? (
          <RuntimeInspectionDetail inspection={snapshot.runtimeInspection} />
        ) : surface.kind === "plugins" ? (
          <MobilePluginDirectory
            plugins={pluginDashboards}
            onOpen={(pluginId) => navigateToSurface({ kind: "dashboard", pluginId })}
          />
        ) : surface.kind === "dashboard" ? (
          <section className="mobile-plugin-dashboard" aria-label="插件看板">
            <MobilePluginDashboard pluginId={surface.pluginId} />
          </section>
        ) : null}
        <div
          className={`mobile-main-content ${surface.kind === "chat" ? "" : "surface-hidden"} ${replyTarget ? "replying" : ""} ${searchOpen ? "searching" : ""} ${selectionActive ? "selecting" : ""} ${queueOpen && snapshot.composer.pendingMessages.length > 1 ? "queueing" : ""} ${selectedSessionUnavailable ? "session-unavailable" : ""}`}
          aria-hidden={surface.kind === "chat" ? undefined : true}
          inert={drawerOpen || surface.kind !== "chat" ? true : undefined}
        >
          <MobileVirtualConversation
            ref={conversationRef}
            snapshot={snapshot}
            streamStore={streamStore}
            selectedMessageIds={selectedMessageIds}
            recoveringMessageIds={recoveringMessageIds}
            selectionActive={selectionActive}
            selectedSessionUnavailable={selectedSessionUnavailable}
            highlightedMessageId={highlightedMessageId}
            copiedMessageId={copiedMessageId}
            missingReplySourceId={missingReplySourceId}
            suspended={searchOpen || selectionActive}
            forceScrollToken={sendScrollRequest}
            unread={unreadState}
            unreadAnchorVisited={unreadAnchorVisited}
            messageElementsRef={messageElementsRef}
            onUnreadChange={setUnreadState}
            onVisitUnread={() => setUnreadAnchorVisited(true)}
            onEnterSelection={enterSelection}
            onToggleSelection={toggleSelection}
            onReplyToMessage={replyToMessage}
            onNavigateToReply={navigateToReply}
            onCopyMessage={copyMessage}
            onRetryMessageDelivery={retryMessageDelivery}
          />
          <MobileSearchTextHighlight query={searchQuery} messageId={searchTargetId} />

          {searchOpen ? (
            <MobileSearchNavigator
              query={searchQuery}
              count={searchResults.length}
              currentIndex={searchTargetIndex}
              onPrevious={() => moveSearch(-1)}
              onNext={() => moveSearch(1)}
            />
          ) : selectionActive ? null : selectedSessionUnavailable && selectedSession ? (
            <UnavailableSessionFooter key={selectedSession.id} session={selectedSession} />
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
              onInput={(value) => updateComposerDraft(value, replyTarget)}
              onToggleCommands={toggleCommands}
              onToggleQueue={() => setQueueOpen((current) => !current)}
              onCloseCommands={closeCommands}
              onSend={send}
              onStop={stop}
              onCancelReply={() => updateComposerDraft(input, null)}
            />
          )}
        </div>
      </main>
    </TooltipProvider>
  );
}

/** 让单消息 patch 只重渲染引用发生变化的消息行。 */
const MobileMessageRow = React.memo(function MobileMessageRow({
  source: baselineSource,
  streamStore,
  startsDay,
  followsSameRole,
  unreadCount,
  highlighted,
  selected,
  selectionActive,
  canReply,
  copied,
  deliveryActionBusy,
  replySourceUnavailable,
  selectedSessionUnavailable,
  messageElementsRef,
  onEnterSelection,
  onToggleSelection,
  onReplyToMessage,
  onNavigateToReply,
  onCopyMessage,
  onRetryMessageDelivery,
}: {
  source: MobileMessage;
  streamStore: MobileStreamProjectionStore<MobileMessage>;
  startsDay: boolean;
  followsSameRole: boolean;
  unreadCount: number;
  highlighted: boolean;
  selected: boolean;
  selectionActive: boolean;
  canReply: boolean;
  copied: boolean;
  deliveryActionBusy: boolean;
  replySourceUnavailable: boolean;
  selectedSessionUnavailable: boolean;
  messageElementsRef: React.RefObject<Map<string, HTMLDivElement>>;
  onEnterSelection: (messageId: string) => void;
  onToggleSelection: (messageId: string) => void;
  onReplyToMessage: (message: MobileMessage) => void;
  onNavigateToReply: (sourceMessageId: string, reply: MobileReply) => void;
  onCopyMessage: (message: MobileMessage) => void;
  onRetryMessageDelivery: (messageId: string) => void;
}) {
  const subscribe = useCallback(
    (listener: () => void) => streamStore.subscribe(baselineSource.id, listener),
    [baselineSource.id, streamStore],
  );
  const getSnapshot = useCallback(
    () => streamStore.read(baselineSource.id, baselineSource),
    [baselineSource, streamStore],
  );
  const source = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
  const message = toCachedChatMessage(source);
  const pluginTurn = !selectedSessionUnavailable && isPluginTurnMessage(message);
  const turnId = pluginTurnId(message);
  const leadingContent = useMemo(() => {
    const reply = source.reply;
    return reply ? (
      <MessageReplyReference
        role={reply.role}
        preview={reply.preview}
        unavailable={replySourceUnavailable}
        onNavigate={() => onNavigateToReply(source.id, reply)}
      />
    ) : undefined;
  }, [onNavigateToReply, replySourceUnavailable, source.id, source.reply]);
  const attachmentContent = useMemo(
    () => <MobileMessageAttachments attachments={source.attachments} />,
    [source.attachments],
  );
  const processStartContent = useMemo(() => pluginTurn ? (
    <MobilePluginSlot
      name="turn.before_reasoning"
      sessionId={source.sessionId}
      messageId={message.id}
      turnId={turnId}
    />
  ) : undefined, [message.id, pluginTurn, source.sessionId, turnId]);
  const beforeProcessBlock = useCallback((block: AgentBlock) => pluginTurn && block.kind === "tool" ? (
    <MobilePluginSlot
      name="turn.before_tool"
      sessionId={source.sessionId}
      messageId={message.id}
      turnId={turnId}
      block={block}
    />
  ) : null, [message.id, pluginTurn, source.sessionId, turnId]);
  const answerEndContent = useMemo(() => pluginTurn ? (
    <MobilePluginSlot
      name="turn.after_answer"
      sessionId={source.sessionId}
      messageId={message.id}
      turnId={turnId}
    />
  ) : undefined, [message.id, pluginTurn, source.sessionId, turnId]);
  const copyCurrentMessage = useCallback(
    () => onCopyMessage(getSnapshot()),
    [getSnapshot, onCopyMessage],
  );
  const replyToCurrentMessage = useCallback(
    () => onReplyToMessage(getSnapshot()),
    [getSnapshot, onReplyToMessage],
  );
  const retryCurrentMessage = useCallback(
    () => onRetryMessageDelivery(getSnapshot().id),
    [getSnapshot, onRetryMessageDelivery],
  );
  const requiresFullRenderer = source.reply !== undefined
    || source.blocks.length > 0
    || source.attachments.length > 0
    || messageNeedsMarkdown(source.content);
  return (
    <>
      {startsDay ? <MessageDateDivider createdAt={source.createdAt} /> : null}
      {unreadCount > 0 ? <MessageUnreadDivider count={unreadCount} /> : null}
      {!startsDay && followsSameRole ? (
        <div className={`mobile-role-divider ${source.role}`} />
      ) : null}
      <MessageSelectionTarget
        ref={(element) => {
          if (element) messageElementsRef.current.set(source.id, element);
          else messageElementsRef.current.delete(source.id);
        }}
        className={`mobile-message-anchor ${source.role} ${source.streaming ? "streaming" : ""} ${highlighted ? "search-target" : ""} ${selected ? "selected" : ""}`}
        data-message-id={source.id}
        tabIndex={-1}
        selectable={!source.streaming}
        selectionActive={selectionActive}
        selected={selected}
        onEnterSelection={() => onEnterSelection(source.id)}
        onToggleSelection={() => onToggleSelection(source.id)}
      >
        <div className="message-interaction-surface">
          {requiresFullRenderer ? (
            <Suspense fallback={<MobilePlainMessageView role={source.role} content={source.content} />}>
              <LazyChatMessageView
                message={message}
                onCopyToolDetail={copyToolDetail}
                leadingContent={leadingContent}
                attachmentContent={attachmentContent}
                processStartContent={processStartContent}
                beforeProcessBlock={beforeProcessBlock}
                answerEndContent={answerEndContent}
              />
            </Suspense>
          ) : (
            <>
              <MobilePlainMessageView role={source.role} content={source.content} />
              {pluginTurn ? (
                <MobilePluginSlot
                  name="turn.after_answer"
                  sessionId={source.sessionId}
                  messageId={message.id}
                  turnId={pluginTurnId(message)}
                />
              ) : null}
            </>
          )}
          <MessageMeta
            role={source.role}
            createdAt={source.createdAt}
            deliveryLabel={source.deliveryLabel}
            deliveryAction={source.deliveryAction}
            interrupted={source.interrupted}
            hasContent={Boolean(source.content)}
            copied={copied}
            canReply={canReply}
            onCopy={copyCurrentMessage}
            onReply={replyToCurrentMessage}
            deliveryActionBusy={deliveryActionBusy}
            onDeliveryAction={retryCurrentMessage}
          />
        </div>
      </MessageSelectionTarget>
    </>
  );
});

const MobilePlainMessageView = React.memo(function MobilePlainMessageView({ role, content }: { role: MobileMessage["role"]; content: string }) {
  return (
    <div className={`mobile-plain-message-view ${role}`}>
      <div className="mobile-plain-message-view__content">
        <p>{content}</p>
      </div>
    </div>
  );
});

function copyToolDetail(text: string) {
  window.AkashicNative?.copyText(text);
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
  selectionCount,
  canCopySelection,
  canShareSelection,
  sharePending,
  shareStatus,
  canReplyToSelection,
  onToggleDrawer,
  onOpenSearch,
  onCloseSearch,
  onSearchQuery,
  onSearchSubmit,
  onCloseSelection,
  onCopySelection,
  onShareSelection,
  onReplyToSelection,
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
  selectionCount: number;
  canCopySelection: boolean;
  canShareSelection: boolean;
  sharePending: boolean;
  shareStatus: string | null;
  canReplyToSelection: boolean;
  onToggleDrawer: () => void;
  onOpenSearch: () => void;
  onCloseSearch: () => void;
  onSearchQuery: (query: string) => void;
  onSearchSubmit: () => void;
  onCloseSelection: () => void;
  onCopySelection: () => void;
  onShareSelection: () => void;
  onReplyToSelection: () => void;
}) {
  const theme = useTheme();
  if (selectionCount > 0) {
    const actions = mobileSelectionActionAvailability(sharePending, {
      reply: canReplyToSelection,
      copy: canCopySelection,
      share: canShareSelection,
    });
    return (
      <header className="mobile-topbar selection-mode" aria-busy={sharePending}>
        <button className="mobile-icon-button" type="button" onClick={onCloseSelection} aria-label="退出消息选择" disabled={!actions.exit}>
          <X size={24} />
        </button>
        <strong aria-live="polite">{shareStatus ?? `已选择 ${selectionCount} 条`}</strong>
        {canReplyToSelection ? (
          <button className="mobile-icon-button" type="button" onClick={onReplyToSelection} aria-label="引用选中的消息" disabled={!actions.reply}>
            <Reply size={21} />
          </button>
        ) : null}
        {canCopySelection ? (
          <button className="mobile-icon-button" type="button" onClick={onCopySelection} aria-label="复制选中的消息" disabled={!actions.copy}>
            <Copy size={21} />
          </button>
        ) : null}
        {canShareSelection ? (
          <button className="mobile-icon-button" type="button" onClick={onShareSelection} aria-label="分享选中的消息" disabled={!actions.share}>
            <Share2 size={21} />
          </button>
        ) : null}
      </header>
    );
  }
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
      <button
        className="mobile-icon-button"
        type="button"
        onClick={() => {
          const next = cycleTheme();
          window.AkashicNative?.setTheme(next.requestedThemeId);
        }}
        aria-label={`切换主题，当前为${theme.label}`}
        title={`当前主题：${theme.label}`}
      >
        <Palette size={21} />
      </button>
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
  onOpenRuntime,
  onOpenPlugins,
  onOpenSettings,
  onRestartPairing,
  onClose,
}: {
  open: boolean;
  snapshot: MobileSnapshot;
  pluginCount: number;
  onOpenRuntime: () => void;
  onOpenPlugins: () => void;
  onOpenSettings: () => void;
  onRestartPairing: () => void;
  onClose: () => void;
}) {
  const drawerRef = useRef<HTMLElement>(null);
  useEffect(() => {
    if (open) requestAnimationFrame(() => drawerRef.current?.focus());
  }, [open]);
  return (
    <div className={`mobile-drawer-layer ${open ? "open" : ""}`} aria-hidden={!open}>
      <button className="mobile-drawer-scrim" type="button" onClick={onClose} aria-label="关闭会话抽屉" tabIndex={open ? 0 : -1} />
      <ConversationNavigation
        className="mobile-drawer"
        panelRef={drawerRef}
        dialog
        closeAction={(
          <button className="mobile-drawer__close" type="button" onClick={onClose} aria-label="关闭会话抽屉">
            <X size={24} />
          </button>
        )}
        destinations={[
          {
            id: "runtime",
            icon: <LibraryBig size={20} />,
            label: "知识与运行",
            description: "记忆 · MCP · 定时任务",
            featured: true,
            onActivate: onOpenRuntime,
          },
          {
            id: "plugins",
            icon: <Puzzle size={20} />,
            label: "插件",
            badge: pluginCount,
            onActivate: onOpenPlugins,
          },
        ]}
        sessions={snapshot.sessions.map((session) => ({
          id: session.id,
          title: session.title || "未命名会话",
          preview: session.isAvailable
            ? session.lastMessagePreview || "还没有消息"
            : "电脑端已不存在 · 本机保留历史",
          updatedLabel: session.lastMessageAt ? formatDrawerTime(session.lastMessageAt) : undefined,
          active: session.id === snapshot.selectedSessionId,
          unavailable: !session.isAvailable,
          state: !session.isAvailable ? (
            <ArchiveX size={19} aria-label="电脑端已不存在" />
          ) : session.isRunning ? <span className="session-running" aria-label="Agent 正在处理" />
            : session.unreadCount > 0 ? (
              <strong className="session-unread" aria-label={`${session.unreadCount} 条未读`}>
                {session.unreadCount > 99 ? "99+" : session.unreadCount}
              </strong>
            ) : session.id === snapshot.selectedSessionId ? <Check size={18} /> : null,
          onActivate: () => {
            window.AkashicNative?.selectSession(session.id);
            onClose();
          },
        }))}
        sessionAfterContent={<MobilePluginSlot name="drawer.panel" sessionId={snapshot.selectedSessionId} />}
        actions={[
          { id: "settings", icon: <Settings size={18} />, label: "设置", onActivate: onOpenSettings },
          {
            id: "diagnostics",
            icon: <FileText size={18} />,
            label: "导出诊断报告",
            onActivate: () => {
              window.AkashicNative?.exportDiagnostics();
              onClose();
            },
          },
          {
            id: "resync",
            icon: <RotateCcw size={18} />,
            label: snapshot.composer.isResyncing ? "正在重新同步" : "清理缓存并同步",
            disabled: !snapshot.composer.canResync,
            onActivate: () => {
              if (window.confirm("清除本机已同步消息和附件缓存，并从电脑重新拉取？连接状态会保留。")) {
                window.AkashicNative?.reloadFromServer();
                onClose();
              }
            },
          },
          { id: "pairing", icon: <RefreshCw size={18} />, label: "重新扫码", onActivate: onRestartPairing },
          {
            id: "new-chat",
            icon: <MessageSquarePlus size={18} />,
            label: "新聊天",
            primary: true,
            onActivate: () => {
              window.AkashicNative?.createSession();
              onClose();
            },
          },
        ]}
      />
    </div>
  );
}

function RuntimeInspectionDirectory({
  inspection,
  onRefresh,
  onOpenDocument,
  onOpenMcp,
  onOpenJob,
}: {
  inspection: MobileRuntimeInspection;
  onRefresh: () => void;
  onOpenDocument: (documentId: string) => void;
  onOpenMcp: (ownerId: string, name: string) => void;
  onOpenJob: (jobId: string) => void;
}) {
  return (
    <main className="runtime-library">
      <header className="runtime-library__intro">
        <span>当前电脑的只读投影</span>
        <h1>知识与运行</h1>
        <p>在一个地方查看人格、记忆、连接能力和正在等待的任务。</p>
        <button type="button" onClick={onRefresh} disabled={inspection.refreshing}>
          <RefreshCw size={17} className={inspection.refreshing ? "is-spinning" : ""} />
          {inspection.refreshing ? "正在刷新" : "刷新"}
        </button>
      </header>

      {inspection.errorMessage ? (
        <div className="runtime-library__error" role="alert">
          <AlertCircle size={18} />
          <span>{inspection.errorMessage}</span>
        </div>
      ) : null}

      <RuntimeSection
        className="documents"
        icon={<BookOpenText size={20} />}
        eyebrow="Knowledge"
        title="文档"
        count={inspection.documents.length}
      >
        {inspection.documents.map((document) => (
          <button
            className="runtime-item"
            type="button"
            key={document.id}
            disabled={!document.available}
            onClick={() => onOpenDocument(document.id)}
          >
            <span>
              <strong>{document.title}</strong>
              <small>{document.description}</small>
              <code>{document.relativePath}</code>
            </span>
            <ChevronRight size={18} />
          </button>
        ))}
      </RuntimeSection>

      <RuntimeSection
        className="mcp"
        icon={<Server size={20} />}
        eyebrow="Connections"
        title="MCP"
        count={inspection.mcpServers.length}
        meta={`${inspection.pluginCount} 插件 · ${inspection.skillCount} Skills`}
      >
        {inspection.mcpServers.length ? inspection.mcpServers.map((server) => (
          <button
            className="runtime-item"
            type="button"
            key={`${server.ownerId}/${server.name}`}
            onClick={() => onOpenMcp(server.ownerId, server.name)}
          >
            <span>
              <strong>{server.name}</strong>
              <small>{server.toolCount} 个工具 · {server.ownerId}</small>
            </span>
            <ChevronRight size={18} />
          </button>
        )) : <p className="runtime-empty">当前快照没有 MCP server。</p>}
      </RuntimeSection>

      <RuntimeSection
        className="schedules"
        icon={<Timer size={20} />}
        eyebrow="Automation"
        title="定时任务"
        count={inspection.jobs.length}
      >
        {inspection.jobs.length ? inspection.jobs.map((job) => (
          <button
            className="runtime-item"
            type="button"
            key={job.id}
            onClick={() => onOpenJob(job.id)}
          >
            <span>
              <strong>{job.name || "未命名定时任务"}</strong>
              <small>{formatRuntimeFireAt(job.fireAt)} · {job.trigger}/{job.tier}</small>
            </span>
            <span className={`runtime-status ${job.enabled ? "enabled" : ""}`}>
              {job.enabled ? "启用" : "停用"}
            </span>
            <ChevronRight size={18} />
          </button>
        )) : <p className="runtime-empty">当前没有定时任务。</p>}
      </RuntimeSection>
    </main>
  );
}

function RuntimeSection({
  className,
  icon,
  eyebrow,
  title,
  count,
  meta,
  children,
}: {
  className: string;
  icon: ReactNode;
  eyebrow: string;
  title: string;
  count: number;
  meta?: string;
  children: ReactNode;
}) {
  return (
    <section className={`runtime-section ${className}`}>
      <header>
        <span className="runtime-section__icon">{icon}</span>
        <span>
          <small>{eyebrow}</small>
          <strong>{title}</strong>
        </span>
        <span className="runtime-section__count">{count}</span>
      </header>
      {meta ? <p className="runtime-section__meta">{meta}</p> : null}
      <div className="runtime-section__items">{children}</div>
    </section>
  );
}

function RuntimeInspectionDetail({
  inspection,
}: {
  inspection: MobileRuntimeInspection;
}) {
  if (inspection.detailLoading) {
    return (
      <main className="runtime-detail runtime-detail--loading">
        <RefreshCw size={24} className="is-spinning" />
        <p>正在读取最新内容…</p>
      </main>
    );
  }
  if (!inspection.detail) {
    return (
      <main className="runtime-detail runtime-detail--loading">
        <AlertCircle size={24} />
        <p>{inspection.errorMessage || "详情暂时不可用"}</p>
      </main>
    );
  }
  return (
    <main className={`runtime-detail ${inspection.detail.kind}`}>
      <header>
        <span>{inspection.detail.kind === "document" ? "Markdown" : inspection.detail.kind === "mcp" ? "MCP Server" : "Schedule"}</span>
        <h1>{inspection.detail.title}</h1>
        <p>{inspection.detail.subtitle}</p>
      </header>
      <article className="runtime-markdown">
        <Suspense fallback={<div className="mobile-markdown-render-fallback">正在载入内容…</div>}>
          <LazyMessageResponse>{inspection.detail.markdown}</LazyMessageResponse>
        </Suspense>
      </article>
    </main>
  );
}

function formatRuntimeFireAt(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("zh-CN", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function UnavailableSessionFooter({ session }: { session: MobileSession }) {
  if (!session.canRemove) {
    return (
      <div className="mobile-unavailable-session" role="status">
        <span>
          <strong>电脑端已不存在</strong>
          <small>未发送的消息或附件仍保留在本机；已停止发送，避免重新创建会话。</small>
        </span>
        <button className="new-session" type="button" onClick={() => window.AkashicNative?.createSession()}>
          新聊天
        </button>
      </div>
    );
  }
  return (
    <div className="mobile-unavailable-session" role="status">
      <span>
        <strong>电脑端已不存在</strong>
        <small>历史仍保存在这台手机上，但不能继续发送。</small>
      </span>
      <Dialog>
        <DialogTrigger asChild>
          <button type="button">从本机移除</button>
        </DialogTrigger>
        <DialogContent
          className="mobile-session-remove-dialog"
          overlayClassName="mobile-session-remove-overlay"
          showCloseButton={false}
        >
          <DialogTitle>移除本机会话？</DialogTitle>
          <DialogDescription>
            “{session.title || "未命名会话"}”的本地历史和已缓存附件会被删除，电脑端数据不会受到影响。
          </DialogDescription>
          <DialogFooter className="mobile-session-remove-dialog__actions">
            <DialogClose asChild>
              <button type="button">取消</button>
            </DialogClose>
            <DialogClose asChild>
              <button
                className="destructive"
                type="button"
                onClick={() => {
                  window.AkashicNative?.performActionHaptic();
                  window.AkashicNative?.removeUnavailableSession(session.id);
                }}
              >
                移除
              </button>
            </DialogClose>
          </DialogFooter>
        </DialogContent>
      </Dialog>
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
  const hasOwner = snapshot.selectedSessionId !== undefined;
  const hasDraft = snapshot.composer.attachments.length > 0;
  const attachmentsReady = allMobileAttachmentsReady(snapshot.composer.attachments);
  const canSubmit = hasOwner && snapshot.composer.canSend && attachmentsReady && !sendPending && (!!input.trim() || hasDraft);
  const zoneRef = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) resizeMobileComposerTextarea(textarea);
  }, [input, textareaRef]);

  useEffect(() => {
    const textarea = textareaRef.current;
    const widthOwner = textarea?.parentElement;
    if (!textarea || !widthOwner) return;
    let lastWidth = widthOwner.clientWidth;
    const observer = new ResizeObserver(() => {
      const nextWidth = widthOwner.clientWidth;
      if (nextWidth === lastWidth) return;
      lastWidth = nextWidth;
      resizeMobileComposerTextarea(textarea);
    });
    observer.observe(widthOwner);
    return () => observer.disconnect();
  }, [textareaRef]);

  useLayoutEffect(() => {
    const zone = zoneRef.current;
    const content = zone?.closest<HTMLElement>(".mobile-main-content");
    if (!zone || !content) return;
    const updateInset = () => {
      content.style.setProperty("--mobile-composer-height", `${zone.offsetHeight}px`);
    };
    updateInset();
    const observer = new ResizeObserver(updateInset);
    observer.observe(zone);
    return () => {
      observer.disconnect();
      content.style.removeProperty("--mobile-composer-height");
    };
  }, []);

  return (
    <div ref={zoneRef} className="mobile-composer-zone">
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
        {replyTarget ? (
          <ComposerReply
            role={replyTarget.role}
            preview={replyPreview(replyTarget)}
            onCancel={onCancelReply}
          />
        ) : null}
        <div className="mobile-composer">
        <button className={`mobile-icon-button command-toggle ${commandsOpen ? "active" : ""}`} type="button" disabled={!hasOwner} onClick={onToggleCommands} aria-label={commandsOpen ? "关闭快捷命令" : "打开快捷命令"}>
          {commandsOpen ? <X size={22} /> : <Menu size={22} />}
        </button>
        <textarea
          ref={textareaRef}
          rows={1}
          maxLength={MOBILE_COMPOSER_DRAFT_MAX_LENGTH}
          value={input}
          disabled={!hasOwner}
          placeholder="输入消息"
          onChange={(event) => onInput(event.target.value)}
          onFocus={() => commandsOpen && onCloseCommands(false)}
          onKeyDown={(event) => {
            if (shouldSubmitMobileComposerKey({
              key: event.key,
              ctrlKey: event.ctrlKey,
              metaKey: event.metaKey,
              isComposing: event.nativeEvent.isComposing,
            })) {
              event.preventDefault();
              onSend();
            }
          }}
        />
        <button className="mobile-icon-button" type="button" disabled={sendPending || !hasOwner} onClick={() => window.AkashicNative?.chooseAttachments()} aria-label="添加附件">
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

function resizeMobileComposerTextarea(textarea: HTMLTextAreaElement) {
  textarea.style.height = "0px";
  const metrics = mobileComposerTextareaMetrics(textarea.scrollHeight);
  textarea.style.height = `${metrics.height}px`;
  textarea.style.overflowY = metrics.overflowY;
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
  const imageUrl = cached && /^image\//i.test(attachment.contentType.trim())
    ? attachment.contentUrl
    : undefined;
  const [imageRetry, setImageRetry] = useState(0);
  const [imageUnavailable, setImageUnavailable] = useState(false);
  const [viewerOpen, setViewerOpen] = useState(false);
  const viewerOpenRef = useRef(false);

  useEffect(() => {
    setImageRetry(0);
    setImageUnavailable(false);
  }, [attachment.contentUrl]);

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
      {imageUrl && !imageUnavailable ? (
        <Dialog open={viewerOpen} onOpenChange={changeViewer}>
          <DialogTrigger asChild>
            <button className="message-attachment-preview" type="button" aria-label={`预览 ${attachment.filename}`}>
              <img
                alt=""
                src={imageRetry === 0 ? imageUrl : `${imageUrl}${imageUrl.includes("?") ? "&" : "?"}retry=${imageRetry}`}
                onError={() => {
                  if (imageRetry === 0) setImageRetry(1);
                  else setImageUnavailable(true);
                }}
              />
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
      {imageUrl && !imageUnavailable ? (
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

const MessageMeta = React.memo(function MessageMeta({
  role,
  createdAt,
  deliveryLabel,
  deliveryAction,
  interrupted,
  hasContent,
  copied,
  canReply,
  deliveryActionBusy,
  onCopy,
  onReply,
  onDeliveryAction,
}: {
  role: MobileMessage["role"];
  createdAt: number;
  deliveryLabel?: string;
  deliveryAction?: MobileMessage["deliveryAction"];
  interrupted: boolean;
  hasContent: boolean;
  copied: boolean;
  canReply: boolean;
  deliveryActionBusy: boolean;
  onCopy: () => void;
  onReply: () => void;
  onDeliveryAction: () => void;
}) {
  const deliveryActionLabel = deliveryActionBusy
    ? "处理中"
    : deliveryAction === "retry" ? "重试" : "核对";
  const deliveryActionAria = deliveryAction === "retry"
    ? "发送失败，重试消息"
    : "结果待确认，核对消息状态";
  return (
    <div className={`mobile-message-meta ${role}`}>
      <div className="mobile-message-meta__text">
        <time dateTime={new Date(createdAt).toISOString()}>{formatMessageTime(createdAt)}</time>
        {role === "user" && deliveryLabel ? (
          <span className={deliveryAction ? `delivery-state ${deliveryAction}` : undefined}>
            {deliveryLabel}
          </span>
        ) : null}
        {role === "user" && deliveryAction ? (
          <button
            className={`mobile-delivery-action ${deliveryAction}`}
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
        {interrupted ? <span className="interrupted-label">本轮已中止</span> : null}
      </div>
      <SharedMessageActions
        canReply={canReply}
        canCopy={hasContent}
        copied={copied}
        onReply={onReply}
        onCopy={onCopy}
      />
    </div>
  );
});

const MessageSelectionTarget = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> & {
    selectable: boolean;
    selectionActive: boolean;
    selected: boolean;
    onEnterSelection: () => void;
    onToggleSelection: () => void;
  }
>(function MessageSelectionTarget({
  selectable,
  selectionActive,
  selected,
  onEnterSelection,
  onToggleSelection,
  children,
  ...props
}, ref) {
  const timerRef = useRef<number | null>(null);
  const suppressClickTimerRef = useRef<number | null>(null);
  const capturedPointerRef = useRef<number | null>(null);
  const originRef = useRef({ x: 0, y: 0 });
  const suppressClickRef = useRef(false);
  const cancelLongPress = () => {
    if (timerRef.current === null) return;
    window.clearTimeout(timerRef.current);
    timerRef.current = null;
  };
  const suppressNextClick = () => {
    suppressClickRef.current = true;
    if (suppressClickTimerRef.current !== null) window.clearTimeout(suppressClickTimerRef.current);
    suppressClickTimerRef.current = window.setTimeout(() => {
      suppressClickRef.current = false;
      suppressClickTimerRef.current = null;
    }, 700);
  };
  useEffect(() => () => {
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    if (suppressClickTimerRef.current !== null) window.clearTimeout(suppressClickTimerRef.current);
  }, []);
  const startLongPress = (clientX: number, clientY: number) => {
    suppressClickRef.current = false;
    originRef.current = { x: clientX, y: clientY };
    cancelLongPress();
    timerRef.current = window.setTimeout(() => {
      timerRef.current = null;
      suppressNextClick();
      window.AkashicNative?.performActionHaptic();
      onEnterSelection();
    }, 420);
  };
  const cancelMovedLongPress = (clientX: number, clientY: number) => {
    if (timerRef.current === null) return;
    if (Math.hypot(clientX - originRef.current.x, clientY - originRef.current.y) > 9) cancelLongPress();
  };

  return (
    <div
      {...props}
      ref={ref}
      role={selectionActive && selectable ? "checkbox" : undefined}
      aria-checked={selectionActive && selectable ? selected : undefined}
      aria-label={selectionActive && selectable ? `${selected ? "取消选择" : "选择"}此消息` : undefined}
      tabIndex={selectionActive && selectable ? 0 : props.tabIndex}
      onPointerDown={(event) => {
        if (!selectable || selectionActive || event.pointerType === "touch" || event.button !== 0 || isMessageInteractiveTarget(event.target)) return;
        event.currentTarget.setPointerCapture(event.pointerId);
        capturedPointerRef.current = event.pointerId;
        startLongPress(event.clientX, event.clientY);
      }}
      onPointerMove={(event) => {
        if (event.pointerType !== "touch") cancelMovedLongPress(event.clientX, event.clientY);
      }}
      onPointerUp={(event) => {
        cancelLongPress();
        if (capturedPointerRef.current === event.pointerId && event.currentTarget.hasPointerCapture(event.pointerId)) {
          event.currentTarget.releasePointerCapture(event.pointerId);
        }
        capturedPointerRef.current = null;
      }}
      onPointerCancel={(event) => {
        cancelLongPress();
        if (capturedPointerRef.current === event.pointerId && event.currentTarget.hasPointerCapture(event.pointerId)) {
          event.currentTarget.releasePointerCapture(event.pointerId);
        }
        capturedPointerRef.current = null;
      }}
      onTouchStart={(event) => {
        if (!selectable || selectionActive || event.touches.length !== 1 || isMessageInteractiveTarget(event.target)) return;
        const touch = event.touches[0];
        startLongPress(touch.clientX, touch.clientY);
      }}
      onTouchMove={(event) => {
        if (event.touches.length !== 1) {
          cancelLongPress();
          return;
        }
        const touch = event.touches[0];
        cancelMovedLongPress(touch.clientX, touch.clientY);
      }}
      onTouchEnd={cancelLongPress}
      onTouchCancel={cancelLongPress}
      onContextMenu={(event) => {
        if (!selectable || isMessageInteractiveTarget(event.target)) return;
        event.preventDefault();
        if (suppressClickRef.current) return;
        if (!selectionActive) {
          suppressNextClick();
          window.AkashicNative?.performActionHaptic();
          onEnterSelection();
        }
      }}
      onClickCapture={(event) => {
        if (suppressClickRef.current) {
          suppressClickRef.current = false;
          if (suppressClickTimerRef.current !== null) {
            window.clearTimeout(suppressClickTimerRef.current);
            suppressClickTimerRef.current = null;
          }
          event.preventDefault();
          event.stopPropagation();
          return;
        }
        if (!selectionActive || !selectable) return;
        event.preventDefault();
        event.stopPropagation();
        onToggleSelection();
      }}
      onKeyDown={(event) => {
        if (!selectionActive || !selectable || (event.key !== "Enter" && event.key !== " ")) return;
        event.preventDefault();
        onToggleSelection();
      }}
    >
      <div className="message-selection-content" inert={selectionActive ? true : undefined}>
        {children}
      </div>
    </div>
  );
});

function isMessageInteractiveTarget(target: EventTarget | null) {
  return target instanceof Element && target.closest("button, a, input, textarea, [role='button']") !== null;
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

interface MobileVirtualConversationProps {
  snapshot: MobileSnapshot;
  streamStore: MobileStreamProjectionStore<MobileMessage>;
  selectedMessageIds: Set<string>;
  recoveringMessageIds: Set<string>;
  selectionActive: boolean;
  selectedSessionUnavailable: boolean;
  highlightedMessageId: string | null;
  copiedMessageId: string | null;
  missingReplySourceId: string | null;
  suspended: boolean;
  forceScrollToken: number;
  unread: MobileUnreadState;
  unreadAnchorVisited: boolean;
  messageElementsRef: React.RefObject<Map<string, HTMLDivElement>>;
  onUnreadChange: (state: MobileUnreadState) => void;
  onVisitUnread: () => void;
  onEnterSelection: (messageId: string) => void;
  onToggleSelection: (messageId: string) => void;
  onReplyToMessage: (message: MobileMessage) => void;
  onNavigateToReply: (sourceMessageId: string, reply: MobileReply) => void;
  onCopyMessage: (message: MobileMessage) => void;
  onRetryMessageDelivery: (messageId: string) => void;
}

/** Render only the measured chat window and keep the latest edge stable during streaming. */
const MobileVirtualConversation = React.forwardRef<MobileConversationHandle, MobileVirtualConversationProps>(
  function MobileVirtualConversation({
    snapshot,
    streamStore,
    selectedMessageIds,
    recoveringMessageIds,
    selectionActive,
    selectedSessionUnavailable,
    highlightedMessageId,
    copiedMessageId,
    missingReplySourceId,
    suspended,
    forceScrollToken,
    unread,
    unreadAnchorVisited,
    messageElementsRef,
    onUnreadChange,
    onVisitUnread,
    onEnterSelection,
    onToggleSelection,
    onReplyToMessage,
    onNavigateToReply,
    onCopyMessage,
    onRetryMessageDelivery,
  }, ref) {
    const scrollRef = useRef<HTMLDivElement>(null);
    const sourceMessagesRef = useRef(snapshot.messages);
    sourceMessagesRef.current = snapshot.messages;
    const [isAtEnd, setIsAtEnd] = useState(true);
    const isAtEndRef = useRef(true);
    const restoredProjectionRef = useRef<string | undefined>(undefined);
    const restoreFrameRef = useRef<number | null>(null);
    const saveTimerRef = useRef<number | null>(null);
    const lastSavedRef = useRef("");
    const firstMessageId = snapshot.messages[0]?.id ?? "";
    const lastMessageId = snapshot.messages.at(-1)?.id ?? "";
    const messageIdentityKey = [
      snapshot.projectionGeneration,
      snapshot.messages.length,
      firstMessageId,
      lastMessageId,
    ].join("\u001f");
    const messageIndexById = useMemo(
      () => new Map(snapshot.messages.map((message, index) => [message.id, index])),
      // eslint-disable-next-line react-hooks/exhaustive-deps
      [messageIdentityKey],
    );
    const latestAssistantAt = useMemo(
      () => snapshot.messages.reduce(
        (latest, message) => message.role === "assistant" ? Math.max(latest, message.createdAt) : latest,
        0,
      ),
      // eslint-disable-next-line react-hooks/exhaustive-deps
      [messageIdentityKey],
    );

    const getScrollElement = useCallback(() => scrollRef.current, []);
    const estimateSize = useCallback(
      (index: number) => estimateMobileMessageHeight(sourceMessagesRef.current[index]),
      [],
    );
    const getItemKey = useCallback(
      (index: number) => sourceMessagesRef.current[index]?.id ?? index,
      [],
    );
    // TanStack Virtual 的有状态实例由组件持有，不能交给 React Compiler 自动记忆化。
    // eslint-disable-next-line react-hooks/incompatible-library
    const virtualizer = useVirtualizer({
      count: snapshot.messages.length,
      getScrollElement,
      estimateSize,
      getItemKey,
      anchorTo: "end",
      followOnAppend: suspended ? false : "auto",
      scrollEndThreshold: 48,
      overscan: 4,
      onChange(instance) {
        const next = instance.isAtEnd(2);
        if (next === isAtEndRef.current) return;
        isAtEndRef.current = next;
        setIsAtEnd(next);
      },
    });

    const jumpToMessage = useCallback((messageId: string, focus = false) => {
      const index = messageIndexById.get(messageId);
      if (index === undefined) return;
      virtualizer.scrollToIndex(index, { align: "center", behavior: "auto" });
      if (!focus) return;
      let attempts = 0;
      const focusWhenMounted = () => {
        const element = messageElementsRef.current.get(messageId);
        if (element) {
          element.focus({ preventScroll: true });
          return;
        }
        attempts += 1;
        if (attempts < 4) requestAnimationFrame(focusWhenMounted);
      };
      requestAnimationFrame(focusWhenMounted);
    }, [messageElementsRef, messageIndexById, virtualizer]);

    useImperativeHandle(ref, () => ({ jumpToMessage }), [jumpToMessage]);

    useLayoutEffect(() => {
      // 1. Restore an owned reading anchor or open at the latest message.
      const sessionId = snapshot.selectedSessionId;
      if (!sessionId || snapshot.messages.length === 0 || snapshot.composer.isResyncing) return;
      const projectionKey = `${sessionId}\u001f${snapshot.projectionGeneration}`;
      if (restoredProjectionRef.current === projectionKey) return;
      restoredProjectionRef.current = projectionKey;
      const target = snapshot.readingPosition;
      const targetIndex = target ? messageIndexById.get(target.messageId) : undefined;
      let attempts = 0;
      const restore = () => {
        if (target && targetIndex !== undefined) {
          virtualizer.scrollToIndex(targetIndex, { align: "start", behavior: "auto" });
          const item = virtualizer.getVirtualItems().find((candidate) => candidate.index === targetIndex);
          if (item) virtualizer.scrollToOffset(item.start - target.offsetPx, { behavior: "auto" });
        } else {
          virtualizer.scrollToEnd({ behavior: "auto" });
        }
        attempts += 1;
        if (attempts < 3) restoreFrameRef.current = requestAnimationFrame(restore);
        else restoreFrameRef.current = null;
      };
      restoreFrameRef.current = requestAnimationFrame(restore);
      return () => {
        if (restoreFrameRef.current !== null) cancelAnimationFrame(restoreFrameRef.current);
        restoreFrameRef.current = null;
      };
    }, [messageIndexById, snapshot.composer.isResyncing, snapshot.messages.length, snapshot.projectionGeneration, snapshot.readingPosition, snapshot.selectedSessionId, virtualizer]);

    useEffect(() => {
      // 2. Persist from virtual measurements, avoiding a full DOM layout walk.
      const scrollElement = scrollRef.current;
      const sessionId = snapshot.selectedSessionId;
      if (!scrollElement || !sessionId || snapshot.composer.isResyncing || suspended) return;
      const persist = () => {
        saveTimerRef.current = null;
        if (virtualizer.isAtEnd(2)) {
          const key = `${sessionId}\u001ftail\u001f${latestAssistantAt}`;
          if (key !== lastSavedRef.current) {
            lastSavedRef.current = key;
            window.AkashicNative?.markSessionReadThrough(sessionId, latestAssistantAt);
          }
          return;
        }
        const scrollOffset = virtualizer.scrollOffset ?? 0;
        const anchor = virtualizer.getVirtualItems().find((item) => item.end > scrollOffset + 1);
        const message = anchor ? sourceMessagesRef.current[anchor.index] : undefined;
        if (!anchor || !message) return;
        const offsetPx = Math.round(anchor.start - scrollOffset);
        const key = `${sessionId}\u001f${message.id}\u001f${offsetPx}`;
        if (key === lastSavedRef.current) return;
        lastSavedRef.current = key;
        window.AkashicNative?.saveReadingPosition(sessionId, message.id, offsetPx);
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
    }, [latestAssistantAt, snapshot.composer.isResyncing, snapshot.projectionGeneration, snapshot.selectedSessionId, suspended, virtualizer]);

    useEffect(() => {
      if (forceScrollToken === 0) return;
      virtualizer.scrollToEnd({ behavior: "auto" });
    }, [forceScrollToken, virtualizer]);

    useMobileUnreadTracking(
      snapshot.selectedSessionId,
      snapshot.messages,
      messageIdentityKey,
      snapshot.projectionGeneration,
      snapshot.composer.isResyncing,
      suspended,
      isAtEnd,
      !isAtEnd,
      onUnreadChange,
    );

    const virtualItems = virtualizer.getVirtualItems();
    return (
      <div className="mobile-conversation-frame">
        <div ref={scrollRef} className="mobile-conversation mobile-virtual-conversation" role="log">
          {snapshot.messages.length === 0 ? (
            <div className="mobile-empty">
              <h1>开始一段新对话</h1>
              <p>消息会通过电脑上的 Akashic 实时处理。</p>
            </div>
          ) : (
            <div
              className="mobile-virtual-conversation__content"
              style={{ height: virtualizer.getTotalSize() }}
            >
              {virtualItems.map((virtualItem) => {
                const index = virtualItem.index;
                const source = snapshot.messages[index];
                const previous = snapshot.messages[index - 1];
                return (
                  <div
                    className="mobile-virtual-row"
                    data-index={index}
                    key={virtualItem.key}
                    ref={virtualizer.measureElement}
                    style={{ transform: `translateY(${virtualItem.start}px)` }}
                  >
                    <MobileMessageRow
                      source={source}
                      streamStore={streamStore}
                      startsDay={!previous || !sameLocalDay(previous.createdAt, source.createdAt)}
                      followsSameRole={previous?.role === source.role}
                      unreadCount={source.id === unread.firstMessageId ? unread.count : 0}
                      highlighted={highlightedMessageId === source.id}
                      selected={selectedMessageIds.has(source.id)}
                      selectionActive={selectionActive}
                      canReply={mobileMessageCanReply(source, snapshot.selectedSessionId)}
                      copied={copiedMessageId === source.id}
                      deliveryActionBusy={recoveringMessageIds.has(source.id)}
                      replySourceUnavailable={missingReplySourceId === source.id}
                      selectedSessionUnavailable={selectedSessionUnavailable}
                      messageElementsRef={messageElementsRef}
                      onEnterSelection={onEnterSelection}
                      onToggleSelection={onToggleSelection}
                      onReplyToMessage={onReplyToMessage}
                      onNavigateToReply={onNavigateToReply}
                      onCopyMessage={onCopyMessage}
                      onRetryMessageDelivery={onRetryMessageDelivery}
                    />
                  </div>
                );
              })}
            </div>
          )}
        </div>
        <MobileScrollButton
          isAtBottom={isAtEnd}
          unread={unread}
          unreadAnchorVisited={unreadAnchorVisited}
          onVisitUnread={(messageId) => {
            onVisitUnread();
            jumpToMessage(messageId);
          }}
          onScrollToBottom={() => virtualizer.scrollToEnd({ behavior: "auto" })}
        />
      </div>
    );
  },
);

function estimateMobileMessageHeight(message: MobileMessage | undefined) {
  if (!message) return 180;
  const textLines = Math.ceil(Math.min(message.content.length, 2_400) / 34);
  const blockHeight = Math.min(message.blocks.length, 4) * 72;
  const attachmentHeight = Math.min(message.attachments.length, 2) * 112;
  return Math.min(720, 76 + textLines * 20 + blockHeight + attachmentHeight);
}

/** 按顶层助手消息追踪当前阅读位置之后的未读集合。 */
function useMobileUnreadTracking(
  sessionId: string | undefined,
  sourceMessages: MobileMessage[],
  messageIdentityKey: string,
  projectionGeneration: number,
  resyncing: boolean,
  suspended: boolean,
  isAtBottom: boolean,
  escapedFromLock: boolean,
  onUnreadChange: (state: MobileUnreadState) => void,
) {
  const sourceMessagesRef = useRef(sourceMessages);
  useEffect(() => {
    sourceMessagesRef.current = sourceMessages;
  }, [sourceMessages]);
  const trackedSessionRef = useRef<string | undefined>(undefined);
  const knownMessagesRef = useRef(new Map<string, MobileMessage>());
  const unseenMessageIdsRef = useRef<string[]>([]);
  const publishedUnreadKeyRef = useRef("");
  const projectionBaselineRef = useRef({ generation: projectionGeneration, rebuilding: false });
  const anchorMessageIdRef = useRef<string | undefined>(undefined);
  const anchorKeyRef = useRef<string | undefined>(undefined);
  const anchorOrdinalRef = useRef(0);

  useEffect(() => {
    const currentMessages = sourceMessagesRef.current;
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
      knownMessagesRef.current = new Map(currentMessages.map((message) => [message.id, message]));
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
      currentMessages,
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
    messageIdentityKey,
    suspended,
  ]);
}

function MobileScrollButton({
  isAtBottom,
  unread,
  unreadAnchorVisited,
  onVisitUnread,
  onScrollToBottom,
}: {
  isAtBottom: boolean;
  unread: MobileUnreadState;
  unreadAnchorVisited: boolean;
  onVisitUnread: (messageId: string) => void;
  onScrollToBottom: () => void;
}) {
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
        onScrollToBottom();
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
    const rawNeedle = query.trim();
    const needle = normalizeMobileSearchText(rawNeedle);
    if (needle.length !== rawNeedle.length) return;
    let retryFrame: number | null = null;
    let attempts = 0;

    const register = () => {
      retryFrame = null;
      const element = document.querySelector<HTMLElement>(`[data-message-id="${CSS.escape(messageId)}"]`);
      if (!element) {
        attempts += 1;
        if (attempts < 4) retryFrame = requestAnimationFrame(register);
        return;
      }

      // 2. 目标行由虚拟列表挂载后，收集其中所有命中文本范围
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
      if (ranges.length > 0) registry.set("akashic-search-match", new HighlightConstructor(...ranges));
    };

    // 3. 下一帧等待虚拟行落位，并在目标变化或卸载时原子清理
    retryFrame = requestAnimationFrame(register);
    return () => {
      if (retryFrame !== null) cancelAnimationFrame(retryFrame);
      registry.delete("akashic-search-match");
    };
  }, [messageId, query]);
  return null;
}

function toChatMessage(message: MobileMessage): ChatMessage {
  return {
    id: message.id,
    role: message.role,
    content: message.content,
    attachments: [],
    blocks: toCachedAgentBlocks(message.blocks),
    streaming: message.streaming,
    interrupted: message.interrupted,
    durationMs: message.durationSeconds !== undefined ? message.durationSeconds * 1000 : undefined,
  };
}

const chatMessageProjectionCache = new WeakMap<MobileMessage, ChatMessage>();
const agentBlockProjectionCache = new WeakMap<MobileProcessBlock, AgentBlock>();
const agentBlockListProjectionCache = new WeakMap<MobileProcessBlock[], AgentBlock[]>();

function toCachedChatMessage(message: MobileMessage): ChatMessage {
  const cached = chatMessageProjectionCache.get(message);
  if (cached !== undefined) return cached;
  const projected = toChatMessage(message);
  chatMessageProjectionCache.set(message, projected);
  return projected;
}

function toCachedAgentBlock(block: MobileProcessBlock): AgentBlock {
  const cached = agentBlockProjectionCache.get(block);
  if (cached !== undefined) return cached;
  const projected = toAgentBlock(block);
  agentBlockProjectionCache.set(block, projected);
  return projected;
}

function toCachedAgentBlocks(blocks: MobileProcessBlock[]): AgentBlock[] {
  const cached = agentBlockListProjectionCache.get(blocks);
  if (cached !== undefined) return cached;
  const projected = blocks.map(toCachedAgentBlock);
  agentBlockListProjectionCache.set(blocks, projected);
  return projected;
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
initializeTheme();
installMobileBridge();

const root = document.getElementById("root");
if (!root) throw new Error("Mobile Web root 不存在");
createRoot(root).render(
  <MobileErrorBoundary>
    <MobileNativeApp />
  </MobileErrorBoundary>,
);

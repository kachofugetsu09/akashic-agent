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
  ChevronDown,
  ChevronUp,
  Copy,
  FileText,
  Menu,
  MessageSquarePlus,
  Paperclip,
  RefreshCw,
  Reply,
  RotateCcw,
  Search,
  SendHorizontal,
  Share2,
  Square,
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
  MobilePluginSlot,
  receiveMobilePluginAssets,
  settleMobilePluginResponses,
  type MobilePluginAsset,
} from "./mobile-plugin-runtime";
import {
  advanceMobileProjectionBaseline,
  advanceMobileUnreadTracking,
  normalizeMobileSearchText,
  updateMobileSearchIndex,
  type MobileSearchIndexEntry,
} from "./mobile-message-state";
import type { AgentBlock, ChatMessage } from "./main";
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

export interface MobileSnapshot {
  protocolVersion: 2;
  connection: {
    label: string;
    status: ConnectionStatus;
    notice?: string;
    error?: string;
  };
  sessions: { id: string; title: string }[];
  selectedSessionId?: string;
  projectionGeneration: number;
  messages: MobileMessage[];
  pluginResponses: { requestId: string; resultJson?: string; error?: string }[];
  composer: {
    attachments: MobileAttachment[];
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
  retryDownloadedAttachment(attachmentId: string): void;
  touchDownloadedAttachment(attachmentId: string): void;
  openDownloadedAttachment(attachmentId: string): void;
  shareDownloadedAttachment(attachmentId: string): void;
  dismissError(): void;
  sendMessage(text: string, replyToMessageId: string): void;
  copyText(text: string): void;
  performReplyHaptic(): void;
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

/** 在 native 协议边界校验完整快照，并只补齐 Kotlin 明确定义的默认字段。 */
function parseMobileSnapshot(value: unknown): MobileSnapshot {
  // 1. 校验协议版本与根对象
  const raw = requireRecord(value, "snapshot");
  if (raw.protocolVersion !== 2) throw new Error(`不支持的移动端协议版本: ${String(raw.protocolVersion)}`);
  const connection = requireRecord(raw.connection, "connection");
  const status = requireString(connection.status, "connection.status");
  if (!["connecting", "ready", "degraded", "reconnecting", "disconnected"].includes(status)) {
    throw new Error(`connection.status 不受支持: ${status}`);
  }

  // 2. 校验会话、消息和插件响应
  const sessions = requireArray(raw.sessions, "sessions", (item, index) => {
    const session = requireRecord(item, `sessions[${index}]`);
    return { id: requireString(session.id, `sessions[${index}].id`), title: requireString(session.title, `sessions[${index}].title`) };
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
  return {
    protocolVersion: 2,
    connection: {
      label: requireString(connection.label, "connection.label"),
      status: status as ConnectionStatus,
      notice: optionalString(connection.notice, "connection.notice"),
      error: optionalString(connection.error, "connection.error"),
    },
    sessions,
    selectedSessionId: optionalString(raw.selectedSessionId, "selectedSessionId"),
    projectionGeneration,
    messages,
    pluginResponses,
    composer: {
      attachments: requireArray(composer.attachments, "composer.attachments", parseAttachment),
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
    };
  }
}

function MobileNativeApp() {
  const [snapshot, setSnapshot] = useState<MobileSnapshot | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [commandsOpen, setCommandsOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchTargetId, setSearchTargetId] = useState<string | null>(null);
  const [highlightedMessageId, setHighlightedMessageId] = useState<string | null>(null);
  const [unreadState, setUnreadState] = useState<MobileUnreadState>({ count: 0 });
  const [unreadAnchorVisited, setUnreadAnchorVisited] = useState(false);
  const [input, setInput] = useState("");
  const [replyTarget, setReplyTarget] = useState<MobileMessage | null>(null);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [stopRequested, setStopRequested] = useState(false);
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
    };
    window.AkashicNative?.reportReady();
    requestSnapshot();
    return () => {
      if (frame !== null) cancelAnimationFrame(frame);
      if (requestTimer !== null) window.clearTimeout(requestTimer);
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
    if (!text && snapshot.composer.attachments.length === 0) return;
    window.AkashicNative?.sendMessage(text, replyTarget?.id ?? "");
    setSendScrollRequest((current) => current + 1);
    setInput("");
    setReplyTarget(null);
    setCommandsOpen(false);
    textareaRef.current?.blur();
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

  return (
    <TooltipProvider>
      <main className="mobile-shell">
        <MobileTopBar
          status={snapshot.connection.status}
          label={snapshot.connection.label}
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
        <MobileDrawer
          open={drawerOpen}
          snapshot={snapshot}
          onClose={closeDrawer}
        />

        <div className={`mobile-main-content ${replyTarget ? "replying" : ""} ${searchOpen ? "searching" : ""}`} inert={drawerOpen ? true : undefined}>
          {pluginLoadError ? (
            <div className="mobile-plugin-load-error" role="status">
              插件界面暂不可用 · {pluginLoadError}
            </div>
          ) : null}
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
              stopRequested={stopRequested}
              replyTarget={replyTarget}
              onInput={setInput}
              onToggleCommands={toggleCommands}
              onCloseCommands={closeCommands}
              onSend={send}
              onStop={stop}
              onCancelReply={() => setReplyTarget(null)}
            />
          )}
        </div>
      </main>
    </TooltipProvider>
  );
}

function MobileTopBar({
  status,
  label,
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

function MobileDrawer({ open, snapshot, onClose }: { open: boolean; snapshot: MobileSnapshot; onClose: () => void }) {
  const drawerRef = useRef<HTMLElement>(null);
  useEffect(() => {
    if (open) requestAnimationFrame(() => drawerRef.current?.focus());
  }, [open]);
  return (
    <div className={`mobile-drawer-layer ${open ? "open" : ""}`} aria-hidden={!open}>
      <button className="mobile-drawer-scrim" type="button" onClick={onClose} aria-label="关闭会话抽屉" tabIndex={open ? 0 : -1} />
      <aside ref={drawerRef} className="mobile-drawer" role="dialog" aria-modal="true" aria-label="会话列表" tabIndex={-1}>
        <div className="mobile-drawer__heading">会话</div>
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
              <span>{session.title || "未命名会话"}</span>
              {session.id === snapshot.selectedSessionId ? <Check size={18} /> : null}
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
  stopRequested,
  replyTarget,
  onInput,
  onToggleCommands,
  onCloseCommands,
  onSend,
  onStop,
  onCancelReply,
}: {
  snapshot: MobileSnapshot;
  input: string;
  textareaRef: React.RefObject<HTMLTextAreaElement | null>;
  commandsOpen: boolean;
  stopRequested: boolean;
  replyTarget: MobileMessage | null;
  onInput: (value: string) => void;
  onToggleCommands: () => void;
  onCloseCommands: (restoreFocus?: boolean) => void;
  onSend: () => void;
  onStop: () => void;
  onCancelReply: () => void;
}) {
  const stopping = stopRequested || snapshot.composer.isStopping;
  const hasDraft = snapshot.composer.attachments.length > 0;
  const canSubmit = snapshot.composer.canSend && (!!input.trim() || hasDraft);

  return (
    <div className="mobile-composer-zone">
      <CommandSheet open={commandsOpen} commands={snapshot.composer.commands} onClose={onCloseCommands} />
      {snapshot.connection.notice ? <div className="connection-notice">{snapshot.connection.notice}</div> : null}
      {snapshot.connection.error ? (
        <button className="mobile-error" type="button" onClick={() => window.AkashicNative?.dismissError()}>
          <AlertCircle size={17} />
          <span>{snapshot.connection.error}</span>
          <X size={16} />
        </button>
      ) : null}
      {stopping ? <div className="stop-feedback" aria-live="polite">正在中止本轮处理…</div> : null}
      {hasDraft ? <DraftAttachments attachments={snapshot.composer.attachments} /> : null}
      <div className={`mobile-composer-frame ${replyTarget ? "has-reply" : ""}`}>
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
        <button className="mobile-icon-button" type="button" onClick={() => window.AkashicNative?.chooseAttachments()} aria-label="添加附件">
          <Paperclip size={22} />
        </button>
        {snapshot.composer.canStop || stopping ? (
          <button className={`mobile-send-button stop ${stopping ? "pending" : ""}`} type="button" onClick={onStop} aria-label={stopping ? "正在中止" : "中止回答"} disabled={stopping}>
            <Square size={17} fill="currentColor" />
          </button>
        ) : (
          <button className="mobile-send-button" type="button" onClick={onSend} aria-label="发送消息" disabled={!canSubmit}>
            <SendHorizontal size={22} />
          </button>
        )}
        </div>
      </div>
    </div>
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

function DraftAttachments({ attachments }: { attachments: MobileAttachment[] }) {
  return (
    <div className="draft-attachments">
      {attachments.map((attachment) => {
        const progress = attachment.sizeBytes > 0
          ? Math.min(100, Math.round(attachment.transferredBytes / attachment.sizeBytes * 100))
          : 0;
        return (
          <div className="draft-attachment" key={attachment.id}>
            <FileText size={19} />
            <div className="draft-attachment__body">
              <div className="draft-attachment__name">{attachment.filename}</div>
              <div className="draft-attachment__status">{attachment.state === "ready" ? "上传完成" : attachment.state === "failed" ? "上传失败" : `${progress}%`}</div>
              <div className="draft-attachment__track"><span style={{ inlineSize: `${progress}%` }} /></div>
            </div>
            {attachment.state === "failed" ? (
              <button type="button" onClick={() => window.AkashicNative?.retryAttachment(attachment.id)}>重试</button>
            ) : attachment.canRemove ? (
              <button type="button" onClick={() => window.AkashicNative?.removeAttachment(attachment.id)} aria-label={`移除 ${attachment.filename}`}><X size={18} /></button>
            ) : null}
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
  const status = attachment.state === "pending"
    ? "等待下载"
    : attachment.state === "downloading"
      ? `下载中 ${progress}%`
      : attachment.state === "failed"
        ? "下载失败"
        : attachment.state === "evicted"
          ? "缓存已清理"
          : "已下载";

  return (
    <div className={`mobile-message-attachment ${attachment.state}`}>
      {image ? (
        <Dialog onOpenChange={(open) => open && window.AkashicNative?.touchDownloadedAttachment(attachment.id)}>
          <DialogTrigger asChild>
            <button className="message-attachment-preview" type="button" aria-label={`预览 ${attachment.filename}`}>
              <img alt="" src={attachment.contentUrl} />
            </button>
          </DialogTrigger>
          <DialogContent className="image-preview-dialog">
            <DialogTitle className="sr-only">{attachment.filename}</DialogTitle>
            <img alt={attachment.filename} className="image-preview-full" src={attachment.contentUrl} />
          </DialogContent>
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
      {attachment.state === "failed" || attachment.state === "evicted" ? (
        <button className="message-attachment-action" type="button" onClick={() => window.AkashicNative?.retryDownloadedAttachment(attachment.id)}>重试</button>
      ) : cached ? (
        <button className="message-attachment-action icon" type="button" onClick={() => window.AkashicNative?.shareDownloadedAttachment(attachment.id)} aria-label={`分享 ${attachment.filename}`}><Share2 size={18} /></button>
      ) : null}
    </div>
  );
}

function MessageMeta({
  source,
  copied,
  canReply,
  onCopy,
  onReply,
}: {
  source: MobileMessage;
  copied: boolean;
  canReply: boolean;
  onCopy: () => void;
  onReply: () => void;
}) {
  return (
    <div className={`mobile-message-meta ${source.role}`}>
      <div className="mobile-message-meta__text">
        <time dateTime={new Date(source.createdAt).toISOString()}>{formatMessageTime(source.createdAt)}</time>
        {source.role === "user" && source.deliveryLabel ? <span>{source.deliveryLabel}</span> : null}
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
      window.AkashicNative?.performReplyHaptic();
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
  return null;
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

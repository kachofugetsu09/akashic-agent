export interface MobileMessageIdentity {
  id: string;
  sessionId: string;
  role: "user" | "assistant";
  createdAt: number;
}

export interface MobileSearchMessage {
  id: string;
  content: string;
  searchRevision: number;
  attachments: { filename: string }[];
}

export interface MobileSearchIndexEntry {
  revision: number;
  searchable: string;
  matches: boolean;
}

export interface MobileSelectableMessage {
  id: string;
  sessionId: string;
  role: "user" | "assistant";
  content: string;
  createdAt: number;
  streaming: boolean;
  replyable: boolean;
  attachments: { filename: string }[];
}

export interface MobileUnreadViewportState {
  isAtBottom: boolean;
  escapedFromLock: boolean;
  suspended: boolean;
}

export interface MobileUnreadTrackingResult<T extends MobileMessageIdentity> {
  knownMessages: Map<string, T>;
  unseenMessageIds: string[];
  messageIdMigrations: Map<string, string>;
}

export interface MobileProjectionBaselineState {
  generation: number;
  rebuilding: boolean;
}

export interface MobileReadingAnchor {
  messageId: string;
  offsetPx: number;
}

/** 同步原生阅读锚点，并把明确清除锚点解释为回到会话末尾。 */
export function updateMobileReadingRestoreTarget<T extends MobileReadingAnchor>(
  current: T | null | undefined,
  next: T | undefined,
) {
  const normalized = next ?? null;
  if (
    current?.messageId === normalized?.messageId
    && current?.offsetPx === normalized?.offsetPx
  ) return current ?? null;
  return normalized;
}

export interface MobileComposerDraft {
  text: string;
  replyToMessageId?: string;
}

export interface MobileComposerDraftWrite extends MobileComposerDraft {
  sessionId: string;
}

export interface MobileComposerDraftResolution<T extends MobileSelectableMessage> {
  text: string;
  replyTarget: T | null;
  cleanedDraft?: MobileComposerDraft;
}

export interface MobileComposerKeyboardEvent {
  key: string;
  ctrlKey: boolean;
  metaKey: boolean;
  isComposing: boolean;
}

export interface MobileComposerTextareaMetrics {
  height: number;
  overflowY: "auto" | "hidden";
}

export interface MobileReplyNavigationMessage {
  role: "user" | "assistant";
  createdAt: number;
}

export interface MobileSelectionActionAvailability {
  exit: boolean;
  reply: boolean;
  copy: boolean;
  share: boolean;
}

export const MOBILE_COMPOSER_DRAFT_MAX_LENGTH = 65_536;

export function normalizeMobileComposerDraftText(text: string) {
  return text.slice(0, MOBILE_COMPOSER_DRAFT_MAX_LENGTH);
}

export function flushMobileComposerBeforePairing(
  flush: () => void,
  restartPairing: () => void,
) {
  flush();
  restartPairing();
}

export function allMobileAttachmentsReady(attachments: readonly { state: string }[]) {
  return attachments.every((attachment) => attachment.state === "ready");
}

export function isMobileImageViewerHistoryState(state: unknown, attachmentId: string) {
  if (typeof state !== "object" || state === null) return false;
  return "akashicImageViewer" in state && state.akashicImageViewer === attachmentId;
}

export function normalizeMobileSearchText(value: string) {
  return value.toLocaleLowerCase("zh-CN");
}

export function selectableMobileMessages<T extends MobileSelectableMessage>(
  messages: readonly T[],
  selectedIds: ReadonlySet<string>,
) {
  return messages.filter((message) => selectedIds.has(message.id) && !message.streaming);
}

export function reconcileMobileMessageSelection<T extends MobileSelectableMessage>(
  selectedIds: ReadonlySet<string>,
  messages: readonly T[],
) {
  const visibleIds = new Set(
    messages.filter((message) => !message.streaming).map((message) => message.id),
  );
  return new Set([...selectedIds].filter((messageId) => visibleIds.has(messageId)));
}

export function mobileMessageHasCopyContent(message: MobileSelectableMessage) {
  return Boolean(message.content.trim()) || message.attachments.length > 0;
}

export function mobileMessageCanReply(
  message: MobileSelectableMessage,
  selectedSessionId: string | null | undefined,
) {
  return message.replyable && message.sessionId === selectedSessionId;
}

/** 只在当前投影中解析引用目标，避免跳到别的会话或过期历史。 */
export function resolveMobileReplyNavigationTarget<T extends { id: string }>(
  replyMessageId: string,
  messages: readonly T[],
) {
  return messages.find((message) => message.id === replyMessageId) ?? null;
}

export function formatMobileReplyNavigationAnnouncement(
  message: MobileReplyNavigationMessage,
  formatTime: (createdAt: number) => string,
) {
  return `已跳到${message.role === "assistant" ? "Akashic" : "你"} ${formatTime(message.createdAt)} 的消息`;
}

/** 把原生草稿解析为当前会话可展示的文字与引用。 */
export function resolveMobileComposerDraft<T extends MobileSelectableMessage>(
  draft: MobileComposerDraft,
  messages: readonly T[],
  selectedSessionId: string | null | undefined,
): MobileComposerDraftResolution<T> {
  // 1. 没有引用时直接使用原生 owner 的文字
  if (!draft.replyToMessageId) return { text: draft.text, replyTarget: null };

  // 2. 只恢复仍属于当前会话且允许引用的目标
  const target = messages.find((message) => message.id === draft.replyToMessageId);
  if (target && mobileMessageCanReply(target, selectedSessionId)) {
    return { text: draft.text, replyTarget: target };
  }
  return {
    text: draft.text,
    replyTarget: null,
    cleanedDraft: { text: draft.text },
  };
}

/** 在调度时捕获会话身份，避免延迟写入落到后续选中的会话。 */
export function captureMobileComposerDraftWrite(
  sessionId: string | null | undefined,
  text: string,
  replyToMessageId?: string,
): MobileComposerDraftWrite | null {
  if (!sessionId) return null;
  return {
    sessionId,
    text,
    ...(replyToMessageId ? { replyToMessageId } : {}),
  };
}

export function mobileComposerDraftMatches(
  draft: MobileComposerDraft,
  expected: MobileComposerDraft,
) {
  return draft.text === expected.text
    && (draft.replyToMessageId ?? undefined) === (expected.replyToMessageId ?? undefined);
}

/** 在原生确认写入前保持乐观草稿，避免旧快照复活已发送内容。 */
export function mobileComposerDraftHydration(
  owner: MobileComposerDraft,
  optimistic: MobileComposerDraftWrite | undefined,
) {
  const ownerAcknowledged = optimistic !== undefined
    && mobileComposerDraftMatches(owner, optimistic);
  return {
    draft: optimistic !== undefined && !ownerAcknowledged ? optimistic : owner,
    ownerAcknowledged,
  };
}

export function shouldClearAcceptedMobileComposerDraft(
  active: MobileComposerDraftWrite | null,
  sent: MobileComposerDraftWrite,
) {
  if (active?.sessionId !== sent.sessionId) return true;
  return mobileComposerDraftMatches(active, sent);
}

/** 只把显式桌面快捷键解释为发送，普通回车始终留给输入法换行。 */
export function shouldSubmitMobileComposerKey(event: MobileComposerKeyboardEvent) {
  return event.key === "Enter"
    && !event.isComposing
    && (event.ctrlKey || event.metaKey);
}

export function shouldClearMobileSelectionAfterShare(
  pendingRequestId: string | null,
  resultRequestId: string,
  launched: boolean,
) {
  return pendingRequestId === resultRequestId && launched;
}

export function mobileSelectionActionAvailability(
  sharePending: boolean,
  capabilities: Omit<MobileSelectionActionAvailability, "exit">,
): MobileSelectionActionAvailability {
  if (sharePending) return { exit: false, reply: false, copy: false, share: false };
  return { exit: true, ...capabilities };
}

/** 把输入内容高度限制在一至六行之间，超过后交给 textarea 内部滚动。 */
export function mobileComposerTextareaMetrics(scrollHeight: number): MobileComposerTextareaMetrics {
  const height = Math.min(164, Math.max(44, Math.ceil(scrollHeight)));
  return {
    height,
    overflowY: scrollHeight > 164 ? "auto" : "hidden",
  };
}

export function formatMobileSelectionCopyText(
  messages: readonly MobileSelectableMessage[],
  formatTimestamp: (createdAt: number) => string,
) {
  const copyable = messages.filter(mobileMessageHasCopyContent);
  if (copyable.length === 1) return mobileMessageCopyBody(copyable[0]);
  return copyable.map((message) => [
    `${message.role === "assistant" ? "Akashic" : "你"} · ${formatTimestamp(message.createdAt)}`,
    mobileMessageCopyBody(message),
  ].join("\n")).join("\n\n");
}

function mobileMessageCopyBody(message: MobileSelectableMessage) {
  return [
    message.content.trim(),
    ...message.attachments.map((attachment) => `[附件] ${attachment.filename}`),
  ].filter(Boolean).join("\n");
}

/** 只重算查询变化或原生修订号发生变化的消息搜索项。 */
export function updateMobileSearchIndex(
  current: ReadonlyMap<string, MobileSearchIndexEntry>,
  messages: readonly MobileSearchMessage[],
  query: string,
  queryChanged: boolean,
) {
  if (!query) return new Map<string, MobileSearchIndexEntry>();

  // 1. 复用稳定正文与匹配结果，只重算发生变化的消息
  const next = new Map<string, MobileSearchIndexEntry>();
  messages.forEach((message) => {
    const cached = current.get(message.id);
    const changed = cached?.revision !== message.searchRevision;
    if (!queryChanged && !changed && cached) {
      next.set(message.id, cached);
      return;
    }
    const searchable = changed || !cached
      ? normalizeMobileSearchText(
        [message.content, ...message.attachments.map((attachment) => attachment.filename)].join("\n"),
      )
      : cached.searchable;
    next.set(message.id, {
      revision: message.searchRevision,
      searchable,
      matches: searchable.includes(query),
    });
  });

  // 2. 以当前消息顺序重建 Map，canonical 迁移和切会话不会留下旧身份
  return next;
}

/** 把实时助手 turn 的临时 ID 对齐到最终 canonical ID。 */
export function reconcileAssistantMessageIds<T extends MobileMessageIdentity>(
  knownMessages: ReadonlyMap<string, T>,
  unseenMessageIds: readonly string[],
  currentMessages: readonly T[],
) {
  // 1. 将消失的实时 turn 与唯一 canonical 替代项配对
  const reconciledKnown = new Map(knownMessages);
  const messageIdMigrations = new Map<string, string>();
  const currentById = new Map(currentMessages.map((message) => [message.id, message]));
  const unknownAssistants = currentMessages.filter(
    (message) => message.role === "assistant" && !knownMessages.has(message.id),
  );
  let reconciledUnseen = [...unseenMessageIds];
  knownMessages.forEach((known, knownId) => {
    if (known.role !== "assistant" || currentById.has(knownId)) return;
    const replacements = unknownAssistants.filter(
      (message) => message.sessionId === known.sessionId && message.createdAt === known.createdAt,
    );
    if (replacements.length !== 1) return;
    const replacement = replacements[0];
    reconciledKnown.delete(knownId);
    reconciledKnown.set(replacement.id, replacement);
    messageIdMigrations.set(knownId, replacement.id);
    reconciledUnseen = reconciledUnseen.map(
      (messageId) => messageId === knownId ? replacement.id : messageId,
    );
  });

  // 2. 只报告真正新增的助手消息，并丢弃已消失的锚点
  const newlySeenAssistants = currentMessages.filter(
    (message) => message.role === "assistant" && !reconciledKnown.has(message.id),
  );
  return {
    knownMessages: new Map(currentMessages.map((message) => [message.id, message])),
    unseenMessageIds: reconciledUnseen.filter((messageId) => currentById.has(messageId)),
    newlySeenAssistants,
    messageIdMigrations,
  };
}

/** 以用户是否脱离滚动锁为准，更新当前会话的未读助手消息。 */
export function updateMobileUnreadMessageIds(
  current: readonly string[],
  newlySeen: readonly string[],
  viewport: MobileUnreadViewportState,
) {
  // 1. 只有仍受底部滚动锁控制时，到达底部才代表已经阅读
  if (!viewport.escapedFromLock && viewport.isAtBottom && !viewport.suspended) return [];

  // 2. 用户主动离开或搜索接管视口时，按首次出现的顶层消息去重累积
  if (newlySeen.length === 0 || (!viewport.escapedFromLock && !viewport.suspended)) return [...current];
  return [...new Set([...current, ...newlySeen])];
}

/** 在同步代际边界建立基线，否则推进当前会话的未读集合。 */
export function advanceMobileUnreadTracking<T extends MobileMessageIdentity>(
  knownMessages: ReadonlyMap<string, T>,
  unseenMessageIds: readonly string[],
  currentMessages: readonly T[],
  viewport: MobileUnreadViewportState,
  resetBaseline: boolean,
): MobileUnreadTrackingResult<T> {
  // 1. 全量同步中的空投影和首次回填都只建立已读基线
  if (resetBaseline) {
    return {
      knownMessages: new Map(currentMessages.map((message) => [message.id, message])),
      unseenMessageIds: [],
      messageIdMigrations: new Map(),
    };
  }

  // 2. 普通快照先对齐 canonical 身份，再按视口状态累计新 turn
  const reconciled = reconcileAssistantMessageIds(knownMessages, unseenMessageIds, currentMessages);
  return {
    knownMessages: reconciled.knownMessages,
    unseenMessageIds: updateMobileUnreadMessageIds(
      reconciled.unseenMessageIds,
      reconciled.newlySeenAssistants.map((message) => message.id),
      viewport,
    ),
    messageIdMigrations: reconciled.messageIdMigrations,
  };
}

/** 只在原生明确开启的破坏性投影代际内重建已读基线。 */
export function advanceMobileProjectionBaseline(
  current: MobileProjectionBaselineState,
  generation: number,
  resyncing: boolean,
) {
  // 1. 代际变化标记投影重建；普通断线的 SYNCING 不接管未读
  const generationChanged = generation !== current.generation;
  const rebuilding = current.rebuilding || generationChanged;

  // 2. 重建结束的首个完整快照仍作为基线，随后恢复正常追踪
  return {
    resetBaseline: rebuilding,
    state: {
      generation,
      rebuilding: rebuilding && resyncing,
    },
  };
}

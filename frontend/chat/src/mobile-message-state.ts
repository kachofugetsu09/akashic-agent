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

export function normalizeMobileSearchText(value: string) {
  return value.toLocaleLowerCase("zh-CN");
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

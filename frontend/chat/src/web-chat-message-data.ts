import type { AgentBlock, MessageAttachment, ThinkingBlock } from "./chat-message";
import type { SessionRow, UploadedFile } from "./web-chat-data";

export function uploadedFileToAttachment(file: UploadedFile): MessageAttachment {
  const filename = file.filename || "附件";
  return {
    id: file.artifact_id,
    type: "file",
    filename,
    mediaType: file.media_type || guessMediaType(filename),
    url: file.upload_url || artifactUrl(file.artifact_id),
  };
}

export function mergeAttachments(
  current: MessageAttachment[] | undefined,
  incoming: MessageAttachment[],
): MessageAttachment[] | undefined {
  if (!incoming.length) return current;
  const merged = [...(current ?? [])];
  const seen = new Set(merged.map((item) => item.path || item.id));
  incoming.forEach((item) => {
    const key = item.path || item.id;
    if (seen.has(key)) return;
    seen.add(key);
    merged.push(item);
  });
  return merged;
}

export function mediaToAttachments(media: unknown): MessageAttachment[] {
  if (!Array.isArray(media)) return [];
  return media.flatMap((item, index) => {
    if (typeof item === "string" && item.trim().length > 0) {
      // 历史 messages.extra.media 只读投影仍允许旧绝对路径。
      const path = item;
      const filename = filenameFromPath(path);
      return [{
        id: `${path}:${index}`,
        type: "file",
        filename,
        mediaType: guessMediaType(filename),
        url: mediaUrl(path),
        path,
      } satisfies MessageAttachment];
    }
    const descriptor = artifactDescriptor(item);
    if (!descriptor) return [];
    return [{
      id: descriptor.artifact_id,
      type: "file",
      filename: descriptor.filename || "附件",
      mediaType: descriptor.media_type || guessMediaType(descriptor.filename || ""),
      url: descriptor.url || artifactUrl(descriptor.artifact_id),
    } satisfies MessageAttachment];
  });
}

export function blocksWithFinalThinking(blocks: AgentBlock[], thinking: string | undefined): AgentBlock[] {
  const text = thinking?.trim();
  if (!text || blocks.some((block) => block.kind === "thinking")) return blocks;
  return [{ kind: "thinking", content: text } satisfies ThinkingBlock, ...blocks];
}

export function sessionLabel(session: SessionRow): string {
  const title = session.first_message_content?.trim() || "未命名对话";
  return title.length > 28 ? `${title.slice(0, 28)}...` : title;
}

const navigationTimeFormatter = new Intl.DateTimeFormat("zh-CN", {
  month: "numeric",
  day: "numeric",
});

export function formatNavigationTime(value: string | undefined): string | undefined {
  if (!value) return undefined;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? undefined : navigationTimeFormatter.format(date);
}

function mediaUrl(path: string): string {
  return `/api/chat/media?path=${encodeURIComponent(path)}`;
}

function artifactUrl(artifactId: string): string {
  return `/api/chat/artifacts/${encodeURIComponent(artifactId)}`;
}

type ArtifactDescriptor = {
  artifact_id: string;
  kind?: "file" | "image";
  filename?: string | null;
  media_type?: string | null;
  size_bytes?: number;
  sha256?: string;
  url?: string;
};

function artifactDescriptor(value: unknown): ArtifactDescriptor | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return null;
  const item = value as Record<string, unknown>;
  if (typeof item.artifact_id !== "string" || !item.artifact_id.trim()) return null;
  if (item.kind !== undefined && item.kind !== "file" && item.kind !== "image") return null;
  if (item.filename !== undefined && item.filename !== null && typeof item.filename !== "string") return null;
  if (item.media_type !== undefined && item.media_type !== null && typeof item.media_type !== "string") return null;
  if (item.url !== undefined
    && (typeof item.url !== "string" || !item.url.startsWith("/api/chat/artifacts/"))) return null;
  return item as ArtifactDescriptor;
}

function filenameFromPath(path: string): string {
  return path.split(/[\\/]/).pop() || "附件";
}

function guessMediaType(filename: string): string {
  const suffix = filename.split(".").pop()?.toLowerCase() || "";
  if (["apng", "avif", "gif", "jpg", "jpeg", "png", "svg", "webp"].includes(suffix)) {
    return `image/${suffix === "jpg" ? "jpeg" : suffix}`;
  }
  if (["mp4", "webm", "mov"].includes(suffix)) return `video/${suffix}`;
  if (["mp3", "ogg", "wav", "m4a"].includes(suffix)) return `audio/${suffix}`;
  if (suffix === "txt") return "text/plain";
  if (suffix === "pdf") return "application/pdf";
  return "application/octet-stream";
}

import type { MessageAttachment } from "./chat-message";
import type { SessionRow, UploadedFile } from "./web-chat-data";

export function uploadedFileToAttachment(file: UploadedFile): MessageAttachment {
  const filename = file.filename || "附件";
  return {
    id: file.artifact_id,
    type: "file",
    filename,
    mediaType: file.media_type || guessMediaType(filename),
    url: file.upload_url || artifactUrl(file.artifact_id),
    sizeBytes: file.size_bytes,
  };
}

// 完整标题交给 CSS 按实际宽度截断：按字数硬截会让宽屏也只剩 28 个字，侧栏搜不到后半段，
// 重命名框还会以"..."开头。首条消息可能是整段粘贴，只压成一行并限长防止撑大 DOM。
export function sessionLabel(session: SessionRow): string {
  const title = session.title?.trim() || session.first_message_content?.trim() || "未命名对话";
  return title.replace(/\s+/gu, " ").slice(0, 200);
}

function artifactUrl(artifactId: string): string {
  return `/api/chat/artifacts/${encodeURIComponent(artifactId)}`;
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

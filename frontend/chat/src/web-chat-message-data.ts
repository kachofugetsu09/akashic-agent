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
  };
}

export function sessionLabel(session: SessionRow): string {
  const title = session.title?.trim() || session.first_message_content?.trim() || "未命名对话";
  return title.length > 28 ? `${title.slice(0, 28)}...` : title;
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

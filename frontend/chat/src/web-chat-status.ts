export type ChatStatus = "idle" | "submitted" | "streaming" | "error";

export function isGeneratingChatStatus(status: ChatStatus): boolean {
  return status === "submitted" || status === "streaming";
}

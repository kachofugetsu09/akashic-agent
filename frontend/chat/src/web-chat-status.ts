export type ChatStatus = "idle" | "submitted" | "streaming" | "finalizing" | "error";

export function isGeneratingChatStatus(status: ChatStatus): boolean {
  return status === "submitted" || status === "streaming";
}

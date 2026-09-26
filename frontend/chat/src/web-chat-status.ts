import type { ReplyActivity, TimelineMessage } from "./message-timeline";

export type ChatStatus = "idle" | "submitted" | "streaming" | "finalizing" | "error";

export function isGeneratingChatStatus(status: ChatStatus): boolean {
  return status === "submitted" || status === "streaming";
}

/** 从消息事实衔接接纳与回复活动；保存输入不代表回复结束。 */
export function replyChatStatus(
  activities: ReplyActivity[], pending: number, messages: TimelineMessage[], available: boolean | null,
): ChatStatus {
  if (activities.some((item) => item.active)) return "streaming";
  if (pending) return "submitted";
  if (available === false) return "idle";

  // 1. 每个来源只跟踪最后一条用户输入及覆盖它的完成/控制记录。
  const inputs = new Map<string, { seq: number; boundary: number; paused: number }>();
  for (const message of messages) {
    const body = message.body;
    if (body.kind === "input" && message.author === "user") {
      inputs.set(message.source, { seq: message.seq, boundary: -1, paused: -1 });
      continue;
    }
    const input = inputs.get(message.source);
    if (!input) continue;
    if (body.kind === "output" && body.finish !== "continue") input.boundary = message.seq;
    if (body.kind === "control") {
      if (body.action === "abandon") input.boundary = Math.max(input.boundary, body.through_seq);
      else if (body.action === "resume") {
        if (body.through_seq >= input.paused) input.paused = -1;
      } else input.paused = Math.max(input.paused, body.through_seq);
    }
  }
  // 2. 撤权活动只等清理；未被终结的其他输入继续显示等待。
  for (const [source, input] of inputs) {
    if (input.seq > Math.max(input.boundary, input.paused)
      && !activities.some((item) => item.source === source)) return "submitted";
  }
  return activities.length ? "finalizing" : "idle";
}

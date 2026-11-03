// 标签模块与 iframe 里的编辑器之间的消息：标签只转交会话、可见性与主题，编辑器状态都在 iframe 里。
export const DESIGN_CHANNEL = "akashic.design.v1";

export type DesignToFrame =
  | { type: "init"; session: string | null; dark: boolean; active: boolean }
  | { type: "session"; session: string | null }
  | { type: "active"; active: boolean }
  | { type: "theme"; dark: boolean };

export type DesignFromFrame = { type: "ready" };

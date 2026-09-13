"use client";

import { configureKaomojiMarkdown } from "@/kaomoji-markdown";
import { cn } from "@/lib/utils";
import { canBatchStreamingMarkdown } from "@/message-rendering-policy";
import { memo, type ComponentProps, useEffect } from "react";
import MarkdownRender, {
  CodeBlockNode,
  MathBlockNode,
  MathInlineNode,
  setCustomComponents,
  type NodeComponentProps,
} from "markstream-react";
import "markstream-react/index.px.css";
import { useReducedMotion } from "motion/react";

export interface MessageResponseProps {
  children: string;
  className?: string;
  isAnimating?: boolean;
  streamBatchCharacters?: number;
}

function KaomojiLiteral({ node }: NodeComponentProps<{ content?: string }>) {
  return <span className="kaomoji-literal">{String(node.content ?? "")}</span>;
}

interface DeferredNode {
  code?: string;
  content?: string;
  raw?: string;
}

let mathStylesPromise: Promise<unknown> | undefined;

function useMathStyles() {
  useEffect(() => {
    mathStylesPromise ??= import("@/katex-styles");
  }, []);
}

/** 聊天代码块 chrome 对齐 dsh：只留语言标签与复制，去掉行号、折叠、字号等编辑器控件。 */
const CODE_BLOCK_PROPS = {
  showLineNumbers: false,
  showCollapseButton: false,
  showFontSizeButtons: false,
  showExpandButton: false,
  showPreviewButton: false,
  showTooltips: false,
  enableFontSizeControl: false,
  isShowPreview: false,
} as const;

function DeferredMathBlock({ node, ctx }: NodeComponentProps<DeferredNode>) {
  useMathStyles();
  if (!ctx?.final) return <pre className="markstream-deferred-source">{String(node.raw ?? node.content ?? "")}</pre>;
  return <MathBlockNode node={node as ComponentProps<typeof MathBlockNode>["node"]} />;
}

function DeferredMathInline({ node, ctx }: NodeComponentProps<DeferredNode>) {
  useMathStyles();
  if (!ctx?.final) return <code className="markstream-deferred-source">{String(node.raw ?? node.content ?? "")}</code>;
  return <MathInlineNode node={node as ComponentProps<typeof MathInlineNode>["node"]} />;
}

/** Mermaid 已下线；围栏改按普通代码块渲染源码。 */
function MermaidAsCode({ node, isDark }: NodeComponentProps<DeferredNode>) {
  return (
    <CodeBlockNode
      node={{ ...node, type: "code_block", language: "mermaid", code: String(node.code ?? node.content ?? "") } as ComponentProps<typeof CodeBlockNode>["node"]}
      isDark={isDark}
      {...CODE_BLOCK_PROPS}
    />
  );
}

setCustomComponents({
  kaomoji_literal: KaomojiLiteral,
  math_block: DeferredMathBlock,
  math_inline: DeferredMathInline,
  mermaid: MermaidAsCode,
});

/** Render complete or append-only Markdown with Markstream's incremental parser. */
export const MessageResponse = memo(function MessageResponse({
  children,
  className,
  isAnimating = false,
}: MessageResponseProps) {
  const reducedMotion = useReducedMotion();
  return (
    <div className={cn("message-response-markstream size-full", isAnimating && "is-streaming", className)}>
      <MarkdownRender
        content={children}
        final={!isAnimating}
        fade={false}
        typewriter={false}
        smoothStreaming={isAnimating && !reducedMotion}
        smoothStreamingOptions={{
          minCharsPerSecond: 24,
          maxCharsPerSecond: 140,
          targetLatencyMs: 260,
          catchUpLatencyMs: 120,
          catchUpThreshold: 64,
          maxCommitFps: 30,
          startDelayMs: 80,
          maxCharsPerCommit: 6,
        }}
        batchRendering={false}
        maxLiveNodes={0}
        viewportPriority
        codeBlockStream={isAnimating}
        renderCodeBlocksAsPre={isAnimating}
        codeBlockProps={CODE_BLOCK_PROPS}
        codeBlockLightTheme="vitesse-light"
        codeBlockDarkTheme="vitesse-dark"
        parseOptions={{ reuseStableTopLevelNodes: true }}
        customMarkdownIt={configureKaomojiMarkdown}
      />
    </div>
  );
}, (previous, next) => (
  previous.isAnimating === next.isAnimating
  && previous.className === next.className
  && previous.streamBatchCharacters === next.streamBatchCharacters
  && (
    previous.children === next.children
    || (
      next.isAnimating === true
      && canBatchStreamingMarkdown(
        previous.children,
        next.children,
        next.streamBatchCharacters ?? 1,
      )
    )
  )
));

MessageResponse.displayName = "MessageResponse";

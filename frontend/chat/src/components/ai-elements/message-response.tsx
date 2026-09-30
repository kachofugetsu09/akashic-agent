"use client";

import { configureReadingMarkdown } from "@/markdown-reading";
import { configureKaomojiMarkdown } from "@/kaomoji-markdown";
import { cn } from "@/lib/utils";
import { canBatchStreamingMarkdown } from "@/message-rendering-policy";
import { memo, type ComponentProps, useEffect } from "react";
import MarkdownRender, {
  MathBlockNode,
  MathInlineNode,
  setCustomComponents,
  type NodeComponentProps,
} from "markstream-react";
import "markstream-react/index.px.css";
import { useReducedMotion } from "motion/react";
import { CODE_BLOCK_PROPS, ReadingCode, ReadingList, ReadingParagraph, ReadingTable } from "@/markdown-reading-nodes";
import type { MarkdownIt } from "stream-markdown-parser";

function configureMessageMarkdown(markdown: MarkdownIt) {
  return configureReadingMarkdown(configureKaomojiMarkdown(markdown));
}

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

/** Mermaid 已下线，保留源码展示。 */
function MermaidAsCode(props: NodeComponentProps<DeferredNode>) {
  return <ReadingCode {...props} node={{ ...props.node, type: "code_block", language: "mermaid", code: String(props.node.code ?? props.node.content ?? ""), raw: String(props.node.raw ?? "") }} />;
}

setCustomComponents({
  kaomoji_literal: KaomojiLiteral,
  math_block: DeferredMathBlock,
  math_inline: DeferredMathInline,
  mermaid: MermaidAsCode,
  list: ReadingList,
  paragraph: ReadingParagraph,
  table: ReadingTable,
  code_block: ReadingCode,
});

/** Render complete or append-only Markdown with Markstream's incremental parser. */
export const MessageResponse = memo(function MessageResponse({
  children,
  className,
  isAnimating = false,
}: MessageResponseProps) {
  const reducedMotion = useReducedMotion();
  return (
    <div className={cn("message-response-markstream markdown-reading size-full", isAnimating && "is-streaming", className)}>
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
        customMarkdownIt={configureMessageMarkdown}
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

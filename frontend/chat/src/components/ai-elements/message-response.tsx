"use client";

import { configureKaomojiMarkdown } from "@/kaomoji-markdown";
import { cn } from "@/lib/utils";
import { canBatchStreamingMarkdown } from "@/message-rendering-policy";
import { memo, type ComponentProps, useEffect } from "react";
import MarkdownRender, {
  MathBlockNode,
  MathInlineNode,
  MermaidBlockNode,
  setCustomComponents,
  type NodeComponentProps,
} from "markstream-react";
import "markstream-react/index.px.css";

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

function DeferredMermaid({ node, ctx, isDark }: NodeComponentProps<DeferredNode>) {
  if (!ctx?.final) return <pre className="markstream-deferred-source">{String(node.raw ?? node.code ?? "")}</pre>;
  return (
    <MermaidBlockNode
      node={node as ComponentProps<typeof MermaidBlockNode>["node"]}
      loading={false}
      isDark={isDark}
    />
  );
}

setCustomComponents({
  kaomoji_literal: KaomojiLiteral,
  math_block: DeferredMathBlock,
  math_inline: DeferredMathInline,
  mermaid: DeferredMermaid,
});

/** Render complete or append-only Markdown with Markstream's incremental parser. */
export const MessageResponse = memo(function MessageResponse({
  children,
  className,
  isAnimating = false,
}: MessageResponseProps) {
  return (
    <div className={cn("message-response-markstream size-full", isAnimating && "is-streaming", className)}>
      <MarkdownRender
        content={children}
        final={!isAnimating}
        fade={false}
        typewriter={false}
        smoothStreaming={false}
        batchRendering={false}
        maxLiveNodes={0}
        viewportPriority
        codeBlockStream={isAnimating}
        renderCodeBlocksAsPre={isAnimating}
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

"use client";

import { configureReadingMarkdown, jumpToFootnote } from "@/markdown-reading";
import { configureKaomojiMarkdown } from "@/kaomoji-markdown";
import { cn } from "@/lib/utils";
import { canBatchStreamingMarkdown } from "@/message-rendering-policy";
import { memo, type ComponentProps, useEffect, useMemo } from "react";
import { useTheme } from "../../../../theme/src/theme-runtime";
import MarkdownRender, {
  MathBlockNode,
  MathInlineNode,
  setCustomComponents,
  setDefaultI18nMap,
  type NodeComponentProps,
} from "markstream-react";
import "markstream-react/index.px.css";
import { useReducedMotion } from "motion/react";
import { CODE_BLOCK_OPTIONS, CODE_BLOCK_PROPS, FootnoteSource, ReadingCode, ReadingFootnote, ReadingFootnoteAnchor, ReadingFootnoteReference, ReadingLink, ReadingList, ReadingParagraph, ReadingTable, footnoteLookup } from "@/markdown-reading-nodes";
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

// markstream 自带按钮的文案与提示默认是英文；高亮完成后代码块会从占位换成它的工具栏，
// 文案要与占位按钮（"复制代码"）和界面其余部分一致。
setDefaultI18nMap({
  "common.copy": "复制代码",
  "common.copied": "已复制",
  "common.decrease": "缩小字号",
  "common.reset": "重置字号",
  "common.increase": "放大字号",
  "common.expand": "展开",
  "common.collapse": "收起",
  "common.preview": "预览",
  "common.source": "源码",
  "common.export": "导出",
  "common.open": "打开",
  "common.close": "关闭",
  "common.zoomIn": "放大",
  "common.zoomOut": "缩小",
  "common.resetZoom": "重置缩放",
  "image.loadError": "图片加载失败",
  "image.loading": "正在加载图片…",
});

setCustomComponents({
  kaomoji_literal: KaomojiLiteral,
  math_block: DeferredMathBlock,
  math_inline: DeferredMathInline,
  mermaid: MermaidAsCode,
  footnote_anchor: ReadingFootnoteAnchor,
  footnote_reference: ReadingFootnoteReference,
  link: ReadingLink,
  list: ReadingList,
  paragraph: ReadingParagraph,
  table: ReadingTable,
  code_block: ReadingCode,
  footnote: ReadingFootnote,
});

/** Render complete or append-only Markdown with Markstream's incremental parser. */
export const MessageResponse = memo(function MessageResponse({
  children,
  className,
  isAnimating = false,
}: MessageResponseProps) {
  const reducedMotion = useReducedMotion();
  // markstream 的代码高亮按 isDark 选 light/dark 主题；不接入时暗色下代码块永远落在亮色分支。
  const theme = useTheme();
  // 脚注正文按同一原文的完整 GFM 解析取（见 ReadingFootnote）：生成中不提供，落定后惰性解析一次。
  const footnotes = useMemo(() => isAnimating ? null : footnoteLookup(children), [children, isAnimating]);
  return (
    <div className={cn("message-response-markstream markdown-reading size-full", isAnimating && "is-streaming", className)}
      onClickCapture={jumpToFootnote}>
      <FootnoteSource.Provider value={footnotes}>
      <MarkdownRender
        content={children}
        isDark={theme.colorScheme === "dark"}
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
        codeBlockOptions={CODE_BLOCK_OPTIONS}
        codeBlockLightTheme="vitesse-light"
        codeBlockDarkTheme="vitesse-dark"
        parseOptions={{ reuseStableTopLevelNodes: true }}
        customMarkdownIt={configureMessageMarkdown}
      />
      </FootnoteSource.Provider>
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

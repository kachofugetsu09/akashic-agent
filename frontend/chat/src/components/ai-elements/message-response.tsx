"use client";

import { configureReadingMarkdown, jumpToFootnote } from "@/markdown-reading";
import { configureKaomojiMarkdown } from "@/kaomoji-markdown";
import { cn } from "@/lib/utils";
import { canBatchStreamingMarkdown } from "@/message-rendering-policy";
import { memo, useEffect, useMemo } from "react";
import { useTheme } from "../../../../theme/src/theme-runtime";
import MarkdownRender, {
  setCustomComponents,
  setDefaultI18nMap,
  type NodeComponentProps,
} from "markstream-react";
import "markstream-react/index.px.css";
import { useReducedMotion } from "motion/react";
import { CODE_BLOCK_OPTIONS, CODE_BLOCK_PROPS, FootnoteSource, ReadingCode, ReadingFootnote, ReadingFootnoteAnchor, ReadingFootnoteReference, ReadingInlineCode, ReadingLink, ReadingList, ReadingParagraph, ReadingTable, footnoteLookup } from "@/markdown-reading-nodes";
import type { MarkdownIt } from "stream-markdown-parser";
import { mightContainMath, preloadMath, ReadingMath, type MathNode } from "@/reading-math";

function configureMessageMarkdown(markdown: MarkdownIt) {
  return configureReadingMarkdown(configureKaomojiMarkdown(markdown));
}

export interface MessageResponseProps {
  children: string;
  className?: string;
  isAnimating?: boolean;
  streamBatchCharacters?: number;
  /** 外层已按到达节奏放字（stream-pace），这里只保留贴身淡入和代码围栏的整体揭示。 */
  paced?: boolean;
}

function KaomojiLiteral({ node }: NodeComponentProps<{ content?: string }>) {
  return <span className="kaomoji-literal">{String(node.content ?? "")}</span>;
}

interface DeferredNode {
  code?: string;
  content?: string;
  raw?: string;
}

function MathBlock({ node }: NodeComponentProps<MathNode>) {
  return <ReadingMath node={node} block />;
}

function MathInline({ node }: NodeComponentProps<MathNode>) {
  return <ReadingMath node={node} block={false} />;
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
  math_block: MathBlock,
  math_inline: MathInline,
  mermaid: MermaidAsCode,
  footnote_anchor: ReadingFootnoteAnchor,
  footnote_reference: ReadingFootnoteReference,
  link: ReadingLink,
  inline_code: ReadingInlineCode,
  list: ReadingList,
  paragraph: ReadingParagraph,
  table: ReadingTable,
  code_block: ReadingCode,
  footnote: ReadingFootnote,
});

// 放字节奏跟着积压走（参考 Claude Desktop 的 paceReveal）：显示始终比到达晚约 0.3 秒，积压越多放得越快，
// 不设固定字速上限。旧参数把速度钳在 140 字/秒、每帧 6 字，快模型（≈300 字/秒）发完时只显示出三成，
// 剩下的在落定那一刻整段跳出。积压超过 400 字时按 120ms 追平。
// 上屏上限 60 次/秒：30 次时文字一格一格跳，高刷屏上尤其明显；每次上屏要重新解析整段，
// 再往上在低端机上会掉帧。两次上屏之间的平滑交给新文字的淡入（合成层动画，跟随屏幕刷新率）。
const STREAM_PACE = {
  minCharsPerSecond: 30,
  maxCharsPerSecond: 6000,
  targetLatencyMs: 300,
  catchUpLatencyMs: 120,
  catchUpThreshold: 400,
  maxCommitFps: 60,
  startDelayMs: 60,
  maxCharsPerCommit: 200,
};

// 外层放字后输入已经连续，渲染器只需紧跟：延迟压到一帧量级，避免两层平滑叠出 0.3 秒滞后。
const PACED_STREAM = { ...STREAM_PACE, targetLatencyMs: 50, catchUpLatencyMs: 50, startDelayMs: 0 };

/** Render complete or append-only Markdown with Markstream's incremental parser. */
export const MessageResponse = memo(function MessageResponse({
  children,
  className,
  isAnimating = false,
  paced = false,
}: MessageResponseProps) {
  const reducedMotion = useReducedMotion();
  // markstream 的代码高亮按 isDark 选 light/dark 主题；不接入时暗色下代码块永远落在亮色分支。
  const theme = useTheme();
  // 脚注正文按同一原文的完整 GFM 解析取（见 ReadingFootnote）：生成中不提供，落定后惰性解析一次。
  const footnotes = useMemo(() => isAnimating ? null : footnoteLookup(children), [children, isAnimating]);
  // 第一个 $ 出现就开始加载 KaTeX 与字体，等公式闭合时已经能直接排版。
  const hasMath = mightContainMath(children);
  useEffect(() => { if (hasMath) void preloadMath(); }, [hasMath]);
  return (
    <div className={cn("message-response-markstream markdown-reading size-full", isAnimating && "is-streaming", className)}
      onClickCapture={jumpToFootnote}>
      <FootnoteSource.Provider value={footnotes}>
      <MarkdownRender
        content={children}
        isDark={theme.colorScheme === "dark"}
        final={!isAnimating}
        fade={isAnimating && !reducedMotion}
        typewriter={false}
        smoothStreaming={isAnimating && !reducedMotion}
        smoothStreamingOptions={paced ? PACED_STREAM : STREAM_PACE}
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
  && previous.paced === next.paced
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

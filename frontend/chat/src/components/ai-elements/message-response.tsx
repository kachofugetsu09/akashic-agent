"use client";

import { cn } from "@/lib/utils";
import { IncrementalMarkdownBlocks } from "@/incremental-markdown-blocks";
import { detectMessageRenderingFeatures } from "@/message-rendering-policy";
import { cjk } from "@streamdown/cjk";
import type { Element, Parent, Root, RootContent } from "hast";
import { memo, useEffect, useMemo, useRef, useState } from "react";
import { parseMarkdownIntoBlocks, Streamdown } from "streamdown";
import type { PluginConfig } from "streamdown";
import type { PluggableList } from "unified";

export type MessageResponseProps = React.ComponentProps<typeof Streamdown>;

const baseStreamdownPlugins: PluginConfig = { cjk };
let codePluginPromise: Promise<NonNullable<PluginConfig["code"]>> | undefined;
let mathPluginPromise: Promise<NonNullable<PluginConfig["math"]>> | undefined;
let mermaidPluginPromise: Promise<NonNullable<PluginConfig["mermaid"]>> | undefined;
const kaomojiPlaceholder = "\uE000AKASHIC_KAOMOJI_";
const kaomojiPattern = /^([（(])([^()\n（）]{0,24}[・ω｀´＾＿ー∀▽дД﹏꒳][^()\n（）]{0,24})([）)])/;

export const MessageResponse = memo(
  ({ className, children, rehypePlugins, isAnimating = false, ...props }: MessageResponseProps) => {
    const incrementalBlocksRef = useRef<IncrementalMarkdownBlocks | null>(null);
    incrementalBlocksRef.current ??= new IncrementalMarkdownBlocks(parseMarkdownIntoBlocks);
    const parseMarkdownIntoBlocksFn = useMemo(
      () => (markdown: string) => incrementalBlocksRef.current!.parse(markdown, isAnimating),
      [isAnimating],
    );
    const prepared = useMemo(() => prepareKaomojiMarkdown(children), [children]);
    const mergedRehypePlugins = useMemo<PluggableList>(
      () => [...(rehypePlugins ?? []), [restoreKaomojiPlugin, prepared.kaomoji]],
      [prepared.kaomoji, rehypePlugins],
    );
    const requestedFeatures = useMemo(
      () => isAnimating
        ? { code: false, math: false, mermaid: false }
        : detectMessageRenderingFeatures(prepared.markdown),
      [isAnimating, prepared.markdown],
    );
    const [richPlugins, setRichPlugins] = useState<PluginConfig>({});

    useEffect(() => {
      const pending: Promise<Partial<PluginConfig>>[] = [];
      if (requestedFeatures.code && !richPlugins.code) {
        codePluginPromise ??= import("@streamdown/code").then((module) => module.code);
        pending.push(codePluginPromise.then((code) => ({ code })));
      }
      if (requestedFeatures.math && !richPlugins.math) {
        mathPluginPromise ??= import("@streamdown/math").then((module) => module.math);
        pending.push(mathPluginPromise.then((math) => ({ math })));
      }
      if (requestedFeatures.mermaid && !richPlugins.mermaid) {
        mermaidPluginPromise ??= import("@streamdown/mermaid").then((module) => module.mermaid);
        pending.push(mermaidPluginPromise.then((mermaid) => ({ mermaid })));
      }
      if (pending.length === 0) return;

      let cancelled = false;
      const load = () => {
        void Promise.all(pending).then((loaded) => {
          if (!cancelled) setRichPlugins((current) => Object.assign({}, current, ...loaded));
        });
      };
      const idleId = typeof window.requestIdleCallback === "function"
        ? window.requestIdleCallback(load, { timeout: 1_500 })
        : window.setTimeout(load, 32);
      return () => {
        cancelled = true;
        if (typeof window.cancelIdleCallback === "function") window.cancelIdleCallback(idleId);
        else window.clearTimeout(idleId);
      };
    }, [requestedFeatures.code, requestedFeatures.math, requestedFeatures.mermaid, richPlugins]);

    const plugins = useMemo(
      () => isAnimating ? baseStreamdownPlugins : { ...baseStreamdownPlugins, ...richPlugins },
      [isAnimating, richPlugins],
    );
    return (
      <Streamdown
        {...props}
        className={cn("size-full [&>*:first-child]:mt-0 [&>*:last-child]:mb-0", className)}
        isAnimating={isAnimating}
        parseIncompleteMarkdown={!isAnimating}
        parseMarkdownIntoBlocksFn={parseMarkdownIntoBlocksFn}
        plugins={plugins}
        rehypePlugins={mergedRehypePlugins}
      >
        {prepared.markdown}
      </Streamdown>
    );
  },
  (previous, next) => previous.children === next.children && previous.isAnimating === next.isAnimating,
);

MessageResponse.displayName = "MessageResponse";

function prepareKaomojiMarkdown(children: MessageResponseProps["children"]) {
  const markdown = typeof children === "string"
    ? children.replace(/\uE000AKASHIC_KAOMOJI_\d+\uE000/g, "")
    : "";
  const kaomoji: string[] = [];
  let fenced = false;
  const lines = markdown.split(/(\n)/);
  const masked = lines.map((line) => {
    if (line === "\n") return line;
    if (/^\s*(```|~~~)/.test(line)) {
      fenced = !fenced;
      return line;
    }
    if (fenced) return line;
    return maskKaomojiInLine(line, kaomoji);
  }).join("");
  return { markdown: masked, kaomoji };
}

function maskKaomojiInLine(line: string, kaomoji: string[]) {
  let result = "";
  let index = 0;
  while (index < line.length) {
    if (line[index] === "`") {
      const next = line.indexOf("`", index + 1);
      if (next === -1) {
        result += line.slice(index);
        break;
      }
      result += line.slice(index, next + 1);
      index = next + 1;
      continue;
    }
    const match = kaomojiPattern.exec(line.slice(index));
    if (match?.index === 0) {
      const value = match[0];
      kaomoji.push(value);
      result += `${kaomojiPlaceholder}${kaomoji.length - 1}\uE000`;
      index += value.length;
      continue;
    }
    result += line[index];
    index += 1;
  }
  return result;
}

function restoreKaomojiPlugin(kaomoji: string[]) {
  return (tree: Root) => {
    if (kaomoji.length > 0) restoreKaomojiNodes(tree, kaomoji);
  };
}

function restoreKaomojiNodes(parent: Parent, kaomoji: string[]) {
  parent.children = parent.children.flatMap((child) => {
    if (child.type === "text") return restoreKaomojiText(child.value, kaomoji);
    if (child.type === "element" && !isLiteralElement(child)) restoreKaomojiNodes(child, kaomoji);
    return [child];
  }) as Parent["children"];
}

function restoreKaomojiText(value: string, kaomoji: string[]): RootContent[] {
  const nodes: RootContent[] = [];
  const pattern = new RegExp(`${kaomojiPlaceholder}(\\d+)\\uE000`, "g");
  let offset = 0;
  for (const match of value.matchAll(pattern)) {
    const index = match.index ?? 0;
    if (index > offset) nodes.push({ type: "text", value: value.slice(offset, index) });
    const text = kaomoji[Number(match[1])] ?? match[0];
    nodes.push({
      type: "element",
      tagName: "span",
      properties: { className: ["kaomoji-literal"] },
      children: [{ type: "text", value: text }],
    });
    offset = index + match[0].length;
  }
  if (offset < value.length) nodes.push({ type: "text", value: value.slice(offset) });
  return nodes.length ? nodes : [{ type: "text", value }];
}

function isLiteralElement(element: Element) {
  return ["code", "pre", "kbd", "samp"].includes(element.tagName);
}

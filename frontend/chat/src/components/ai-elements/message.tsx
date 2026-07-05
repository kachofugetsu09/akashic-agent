"use client";

import { Button } from "@/components/ui/button";
import {
  ButtonGroup,
  ButtonGroupText,
} from "@/components/ui/button-group";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { cjk } from "@streamdown/cjk";
import { code } from "@streamdown/code";
import { math } from "@streamdown/math";
import { mermaid } from "@streamdown/mermaid";
import type { UIMessage } from "ai";
import type { Element, Parent, Root, RootContent } from "hast";
import { ChevronLeftIcon, ChevronRightIcon } from "lucide-react";
import type { ComponentProps, HTMLAttributes, ReactElement } from "react";
import {
  createContext,
  memo,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { Streamdown } from "streamdown";
import type { PluggableList } from "unified";

export type MessageProps = HTMLAttributes<HTMLDivElement> & {
  from: UIMessage["role"];
};

export const Message = ({ className, from, ...props }: MessageProps) => (
  <div
    className={cn(
      "group flex w-full max-w-[95%] flex-col gap-2",
      from === "user" ? "is-user ml-auto justify-end" : "is-assistant",
      className
    )}
    {...props}
  />
);

export type MessageContentProps = HTMLAttributes<HTMLDivElement>;

export const MessageContent = ({
  children,
  className,
  ...props
}: MessageContentProps) => (
  <div
    className={cn(
      "is-user:dark flex w-fit min-w-0 max-w-full flex-col gap-2 overflow-hidden text-sm",
      "group-[.is-user]:ml-auto group-[.is-user]:rounded-lg group-[.is-user]:bg-secondary group-[.is-user]:px-4 group-[.is-user]:py-3 group-[.is-user]:text-foreground",
      "group-[.is-assistant]:text-foreground",
      className
    )}
    {...props}
  >
    {children}
  </div>
);

export type MessageActionsProps = ComponentProps<"div">;

export const MessageActions = ({
  className,
  children,
  ...props
}: MessageActionsProps) => (
  <div className={cn("flex items-center gap-1", className)} {...props}>
    {children}
  </div>
);

export type MessageActionProps = ComponentProps<typeof Button> & {
  tooltip?: string;
  label?: string;
};

export const MessageAction = ({
  tooltip,
  children,
  label,
  variant = "ghost",
  size = "icon-sm",
  ...props
}: MessageActionProps) => {
  const button = (
    <Button size={size} type="button" variant={variant} {...props}>
      {children}
      <span className="sr-only">{label || tooltip}</span>
    </Button>
  );

  if (tooltip) {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>{button}</TooltipTrigger>
          <TooltipContent>
            <p>{tooltip}</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }

  return button;
};

interface MessageBranchContextType {
  currentBranch: number;
  totalBranches: number;
  goToPrevious: () => void;
  goToNext: () => void;
  branches: ReactElement[];
  setBranches: (branches: ReactElement[]) => void;
}

const MessageBranchContext = createContext<MessageBranchContextType | null>(
  null
);

const useMessageBranch = () => {
  const context = useContext(MessageBranchContext);

  if (!context) {
    throw new Error(
      "MessageBranch components must be used within MessageBranch"
    );
  }

  return context;
};

export type MessageBranchProps = HTMLAttributes<HTMLDivElement> & {
  defaultBranch?: number;
  onBranchChange?: (branchIndex: number) => void;
};

export const MessageBranch = ({
  defaultBranch = 0,
  onBranchChange,
  className,
  ...props
}: MessageBranchProps) => {
  const [currentBranch, setCurrentBranch] = useState(defaultBranch);
  const [branches, setBranches] = useState<ReactElement[]>([]);

  const handleBranchChange = useCallback(
    (newBranch: number) => {
      setCurrentBranch(newBranch);
      onBranchChange?.(newBranch);
    },
    [onBranchChange]
  );

  const goToPrevious = useCallback(() => {
    const newBranch =
      currentBranch > 0 ? currentBranch - 1 : branches.length - 1;
    handleBranchChange(newBranch);
  }, [currentBranch, branches.length, handleBranchChange]);

  const goToNext = useCallback(() => {
    const newBranch =
      currentBranch < branches.length - 1 ? currentBranch + 1 : 0;
    handleBranchChange(newBranch);
  }, [currentBranch, branches.length, handleBranchChange]);

  const contextValue = useMemo<MessageBranchContextType>(
    () => ({
      branches,
      currentBranch,
      goToNext,
      goToPrevious,
      setBranches,
      totalBranches: branches.length,
    }),
    [branches, currentBranch, goToNext, goToPrevious]
  );

  return (
    <MessageBranchContext.Provider value={contextValue}>
      <div
        className={cn("grid w-full gap-2 [&>div]:pb-0", className)}
        {...props}
      />
    </MessageBranchContext.Provider>
  );
};

export type MessageBranchContentProps = HTMLAttributes<HTMLDivElement>;

export const MessageBranchContent = ({
  children,
  ...props
}: MessageBranchContentProps) => {
  const { currentBranch, setBranches, branches } = useMessageBranch();
  const childrenArray = useMemo(
    () => (Array.isArray(children) ? children : [children]),
    [children]
  );

  // Use useEffect to update branches when they change
  useEffect(() => {
    if (branches.length !== childrenArray.length) {
      setBranches(childrenArray);
    }
  }, [childrenArray, branches, setBranches]);

  return childrenArray.map((branch, index) => (
    <div
      className={cn(
        "grid gap-2 overflow-hidden [&>div]:pb-0",
        index === currentBranch ? "block" : "hidden"
      )}
      key={branch.key}
      {...props}
    >
      {branch}
    </div>
  ));
};

export type MessageBranchSelectorProps = ComponentProps<typeof ButtonGroup>;

export const MessageBranchSelector = ({
  className,
  ...props
}: MessageBranchSelectorProps) => {
  const { totalBranches } = useMessageBranch();

  // Don't render if there's only one branch
  if (totalBranches <= 1) {
    return null;
  }

  return (
    <ButtonGroup
      className={cn(
        "[&>*:not(:first-child)]:rounded-l-md [&>*:not(:last-child)]:rounded-r-md",
        className
      )}
      orientation="horizontal"
      {...props}
    />
  );
};

export type MessageBranchPreviousProps = ComponentProps<typeof Button>;

export const MessageBranchPrevious = ({
  children,
  ...props
}: MessageBranchPreviousProps) => {
  const { goToPrevious, totalBranches } = useMessageBranch();

  return (
    <Button
      aria-label="Previous branch"
      disabled={totalBranches <= 1}
      onClick={goToPrevious}
      size="icon-sm"
      type="button"
      variant="ghost"
      {...props}
    >
      {children ?? <ChevronLeftIcon size={14} />}
    </Button>
  );
};

export type MessageBranchNextProps = ComponentProps<typeof Button>;

export const MessageBranchNext = ({
  children,
  ...props
}: MessageBranchNextProps) => {
  const { goToNext, totalBranches } = useMessageBranch();

  return (
    <Button
      aria-label="Next branch"
      disabled={totalBranches <= 1}
      onClick={goToNext}
      size="icon-sm"
      type="button"
      variant="ghost"
      {...props}
    >
      {children ?? <ChevronRightIcon size={14} />}
    </Button>
  );
};

export type MessageBranchPageProps = HTMLAttributes<HTMLSpanElement>;

export const MessageBranchPage = ({
  className,
  ...props
}: MessageBranchPageProps) => {
  const { currentBranch, totalBranches } = useMessageBranch();

  return (
    <ButtonGroupText
      className={cn(
        "border-none bg-transparent text-muted-foreground shadow-none",
        className
      )}
      {...props}
    >
      {currentBranch + 1} of {totalBranches}
    </ButtonGroupText>
  );
};

export type MessageResponseProps = ComponentProps<typeof Streamdown>;

const streamdownPlugins = { cjk, code, math, mermaid };
const kaomojiPlaceholder = "\uE000AKASHIC_KAOMOJI_";
const kaomojiPattern = /^([（(])([^()\n（）]{0,24}[・ω｀´＾＿ー∀▽дД﹏꒳][^()\n（）]{0,24})([）)])/;

export const MessageResponse = memo(
  ({ className, children, rehypePlugins, ...props }: MessageResponseProps) => {
    const prepared = useMemo(
      () => prepareKaomojiMarkdown(children),
      [children]
    );
    const mergedRehypePlugins = useMemo<PluggableList>(
      () => [
        ...(rehypePlugins ?? []),
        [restoreKaomojiPlugin, prepared.kaomoji],
      ],
      [prepared.kaomoji, rehypePlugins]
    );

    return (
      <Streamdown
        className={cn(
          "size-full [&>*:first-child]:mt-0 [&>*:last-child]:mb-0",
          className
        )}
        plugins={streamdownPlugins}
        rehypePlugins={mergedRehypePlugins}
        {...props}
      >
        {prepared.markdown}
      </Streamdown>
    );
  },
  (prevProps, nextProps) =>
    prevProps.children === nextProps.children &&
    nextProps.isAnimating === prevProps.isAnimating
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
      const placeholder = `${kaomojiPlaceholder}${kaomoji.length}\uE000`;
      kaomoji.push(value);
      result += placeholder;
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
    if (kaomoji.length === 0) return;
    restoreKaomojiNodes(tree, kaomoji);
  };
}

function restoreKaomojiNodes(parent: Parent, kaomoji: string[]) {
  parent.children = parent.children.flatMap((child) => {
    if (child.type === "text") {
      return restoreKaomojiText(child.value, kaomoji);
    }
    if (child.type === "element" && !isLiteralElement(child)) {
      restoreKaomojiNodes(child, kaomoji);
    }
    return [child];
  }) as Parent["children"];
}

function restoreKaomojiText(value: string, kaomoji: string[]): RootContent[] {
  const nodes: RootContent[] = [];
  const pattern = new RegExp(`${kaomojiPlaceholder}(\\d+)\\uE000`, "g");
  let offset = 0;
  for (const match of value.matchAll(pattern)) {
    const index = match.index ?? 0;
    if (index > offset) {
      nodes.push({ type: "text", value: value.slice(offset, index) });
    }
    const text = kaomoji[Number(match[1])] ?? match[0];
    nodes.push({
      type: "element",
      tagName: "span",
      properties: { className: ["kaomoji-literal"] },
      children: [{ type: "text", value: text }],
    });
    offset = index + match[0].length;
  }
  if (offset < value.length) {
    nodes.push({ type: "text", value: value.slice(offset) });
  }
  return nodes.length ? nodes : [{ type: "text", value }];
}

function isLiteralElement(element: Element) {
  return ["code", "pre", "kbd", "samp"].includes(element.tagName);
}

export type MessageToolbarProps = ComponentProps<"div">;

export const MessageToolbar = ({
  className,
  children,
  ...props
}: MessageToolbarProps) => (
  <div
    className={cn(
      "mt-4 flex w-full items-center justify-between gap-4",
      className
    )}
    {...props}
  >
    {children}
  </div>
);

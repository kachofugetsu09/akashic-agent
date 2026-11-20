import { createElement, type ReactElement } from "react";
import type { ToolIconSpec } from "@akashic/chat-ui-v1";
import {
  Activity,
  BarChart2,
  Blocks,
  BookOpen,
  BookmarkPlus,
  BookmarkX,
  Bot,
  Calendar,
  Clock,
  Code2,
  CornerDownLeft,
  Eye,
  FilePlus,
  FileSearch,
  FileText,
  FolderOpen,
  FolderSymlink,
  Gamepad2,
  GitFork,
  GitPullRequest,
  Globe,
  HeartPulse,
  History,
  LayoutTemplate,
  type LucideIcon,
  MapPin,
  MessageSquareText,
  MessagesSquare,
  Monitor,
  Moon,
  Network,
  PackagePlus,
  PenTool,
  Pencil,
  Plug,
  RotateCw,
  Rss,
  Search,
  Send,
  Smartphone,
  Square,
  Target,
  Terminal,
  UserRoundSearch,
  Wrench,
} from "lucide-react";
import type { ToolBlock } from "./chat-message";
import { readToolIcon } from "./message-timeline";
import { pluginToolIcon, usePluginUiCatalogVersion } from "./plugin-ui-runtime";

const LUCIDE_TOOL_ICONS: Record<string, LucideIcon> = {
  "activity": Activity,
  "bar-chart-2": BarChart2,
  "blocks": Blocks,
  "book-open": BookOpen,
  "bookmark-plus": BookmarkPlus,
  "bookmark-x": BookmarkX,
  "bot": Bot,
  "calendar": Calendar,
  "clock": Clock,
  "code-2": Code2,
  "corner-down-left": CornerDownLeft,
  "eye": Eye,
  "file-plus": FilePlus,
  "file-search": FileSearch,
  "file-text": FileText,
  "folder-open": FolderOpen,
  "folder-symlink": FolderSymlink,
  "gamepad-2": Gamepad2,
  "git-fork": GitFork,
  "git-pull-request": GitPullRequest,
  "globe": Globe,
  "heart-pulse": HeartPulse,
  "history": History,
  "layout-template": LayoutTemplate,
  "map-pin": MapPin,
  "message-square-text": MessageSquareText,
  "messages-square": MessagesSquare,
  "monitor": Monitor,
  "moon": Moon,
  "network": Network,
  "package-plus": PackagePlus,
  "pen-tool": PenTool,
  "pencil": Pencil,
  "plug": Plug,
  "rotate-cw": RotateCw,
  "rss": Rss,
  "search": Search,
  "send": Send,
  "smartphone": Smartphone,
  "square": Square,
  "target": Target,
  "terminal": Terminal,
  "user-round-search": UserRoundSearch,
  "wrench": Wrench,
};

function renderSpec(spec: ToolIconSpec | undefined, className: string, size: number): ReactElement | null {
  if (!spec) return null;
  if (spec.kind === "svg") {
    if (!spec.nodes.length) return null;
    return (
      <svg
        className={className}
        width={size}
        height={size}
        viewBox={spec.viewBox}
        fill="none"
        stroke="currentColor"
        strokeWidth={1.75}
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        {spec.nodes.map((node, index) => createElement(node.tag, { key: index, ...node.attrs }))}
      </svg>
    );
  }
  if (!Object.hasOwn(LUCIDE_TOOL_ICONS, spec.name)) return null;
  const Icon = LUCIDE_TOOL_ICONS[spec.name];
  return <Icon className={className} size={size} aria-hidden="true" />;
}

export function ToolStepIcon({
  block,
  className = "tool-step-icon",
  size = 14,
}: {
  block: ToolBlock;
  className?: string;
  size?: number;
}) {
  usePluginUiCatalogVersion();
  const dynamic = pluginToolIcon({ name: block.name, arguments: block.input }, block.owner);
  const resolvedDynamic = dynamic ? renderSpec(dynamic, className, size) : null;
  if (resolvedDynamic) return resolvedDynamic;
  const resolvedStatic = renderSpec(readToolIcon(block.icon), className, size);
  if (resolvedStatic) return resolvedStatic;
  return <Wrench className={className} size={size} aria-hidden="true" />;
}

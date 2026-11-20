import { createElement } from "react";
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

const ALLOWED_SVG_TAGS = new Set(["path", "circle", "rect", "line", "polyline", "polygon", "ellipse"]);
const ALLOWED_SVG_ATTRS = new Set([
  "d", "cx", "cy", "r", "rx", "ry", "x", "y", "x1", "y1", "x2", "y2",
  "width", "height", "points", "fill", "stroke",
  "fillRule", "clipRule", "strokeLinecap", "strokeLinejoin", "strokeWidth",
]);

function normalizeSpec(raw: string | ToolIconSpec | null | undefined): ToolIconSpec | null {
  if (!raw) return null;
  if (typeof raw === "string") return { kind: "lucide", name: raw };
  return raw;
}

function safeSvgAttrs(attrs: Record<string, string>): Record<string, string> {
  const safe: Record<string, string> = {};
  for (const [key, value] of Object.entries(attrs)) {
    if (!ALLOWED_SVG_ATTRS.has(key) || typeof value !== "string") continue;
    if (key === "fill" || key === "stroke") {
      const lower = value.trim().toLowerCase();
      if (lower !== "none" && lower !== "currentcolor") continue;
      safe[key] = lower === "none" ? "none" : "currentColor";
    } else {
      safe[key] = value;
    }
  }
  return safe;
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
  const spec = normalizeSpec(dynamic) ?? block.icon ?? null;

  if (spec?.kind === "svg" && spec.nodes.length > 0) {
    const validNodes = spec.nodes.filter((node) => ALLOWED_SVG_TAGS.has(node.tag));
    if (validNodes.length > 0) {
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
          {validNodes.map((node, index) =>
            createElement(node.tag, { key: index, ...safeSvgAttrs(node.attrs) }),
          )}
        </svg>
      );
    }
  }

  const Icon = spec?.kind === "lucide" ? (LUCIDE_TOOL_ICONS[spec.name] ?? Wrench) : Wrench;
  return <Icon className={className} size={size} aria-hidden="true" />;
}

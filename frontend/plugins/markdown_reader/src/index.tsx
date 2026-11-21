import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  type MouseEvent as ReactMouseEvent,
} from "react";
import { createRoot } from "react-dom/client";
import {
  AnimatePresence,
  LayoutGroup,
  MotionConfig,
  motion,
  useIsPresent,
} from "motion/react";
import {
  ArrowLeft,
  ArrowUp,
  ChevronDown,
  ChevronRight,
  Copy,
  Check,
  Download,
  Eye,
  EyeOff,
  FileCode,
  FileImage,
  FileText,
  Folder,
  FolderOpen,
  HardDrive,
  Home,
  LayoutGrid,
  List,
  ListTree,
  PanelLeft,
  Pencil,
  Quote,
  RefreshCw,
  Search,
  Star,
  Terminal,
  X,
} from "lucide-react";
import type {
  ConversationIntent,
  ConversationIntentStatus,
  ConversationTabView,
} from "@akashic/conversation-tools-v2";
import type { WebHostContextV1, WebUiDisposer } from "@akashic/web-ui-v1";
import { renderMarkdownProse } from "../../../../packages/akashic-plugin-kit/src/markdown-prose";
import { FileSyntax } from "./syntax";
import "./style.css";

type Entry = { name: string; path: string; kind?: "directory" | "file" | "other" };
type RootPreset = { id: string; label: string; path: string };
type Result = {
  status: string;
  path?: string;
  text?: string;
  bytes?: number;
  mime_type?: string;
  mtime_ms?: number;
  truncated?: boolean;
  tail?: boolean;
  message?: string;
  error?: string;
  root?: string | null;
  roots?: RootPreset[];
  open_file?: string;
  parent?: string | null;
  after?: string | null;
  items?: Entry[];
};

type OutlineItem = {
  id: string;
  text: string;
  level: number;
  el: HTMLElement;
};

const API = "/api/dashboard/markdown_reader";
const PINS_KEY = "akashic.files.pins.v1";
const VIEW_MODE_KEY = "akashic.files.viewMode.v1";
const WIDE_BREAKPOINT_PX = 720;
const NARROW_BREAKPOINT_PX = 460;
const MAX_PINS = 8;
const IMPORT_LIMIT_BYTES = 50 * 1024 * 1024;

const SPRING = {
  type: "spring" as const,
  stiffness: 460,
  damping: 38,
  mass: 0.8,
};

const FAST_SPRING = {
  type: "spring" as const,
  stiffness: 540,
  damping: 40,
  mass: 0.72,
};

// Codex 16×16 软角文件夹图标（evenodd 路径，data-icon="fill" 免双重描边）。
const FOLDER_ICON =
  '<svg viewBox="0 0 16 16" fill="none" data-icon="fill" aria-hidden="true"><path fill-rule="evenodd" clip-rule="evenodd" d="M5.559 2.141C6.065 2.141 6.558 2.302 6.967 2.599L7.818 3.219C8.047 3.385 8.324 3.475 8.607 3.475H11.967C13.287 3.475 14.358 4.546 14.358 5.867V11.467C14.358 12.787 13.287 13.858 11.967 13.858H4.033C2.712 13.858 1.641 12.787 1.641 11.467V4.533C1.642 3.212 2.712 2.142 4.033 2.141H5.559ZM2.692 7.859V11.467C2.692 12.207 3.292 12.808 4.033 12.808H11.967C12.708 12.808 13.308 12.208 13.308 11.467V7.859H2.692ZM4.033 3.192C3.292 3.192 2.692 3.792 2.692 4.533V6.808H13.308V5.867C13.308 5.126 12.708 4.525 11.967 4.525H8.607C8.102 4.525 7.609 4.365 7.2 4.068L6.348 3.448C6.119 3.281 5.843 3.192 5.559 3.192H4.033Z" fill="currentColor"/></svg>';

const STATUS_MESSAGES: Record<string, string> = {
  not_found: "服务端路径不存在",
  not_directory: "该路径不是目录",
  permission_denied: "没有读取权限",
  not_file: "只能预览普通文件",
  too_large: "文件过大，请下载后查看",
  invalid_text: "文件不是 UTF-8 文本",
  invalid_path: "路径格式无效",
  offline: "执行主机离线",
  io_error: "磁盘读取失败",
};

function basename(path: string): string {
  if (!path) return "";
  if (path === "/") return "/";
  const trimmed = path.length > 1 ? path.replace(/\/+$/, "") : path;
  return trimmed.slice(trimmed.lastIndexOf("/") + 1) || trimmed;
}

function dirnameOf(path: string): string {
  const idx = path.lastIndexOf("/");
  if (idx <= 0) return "/";
  return path.slice(0, idx);
}

function normalizePosix(path: string): string {
  const absolute = path.startsWith("/");
  const stack: string[] = [];
  for (const part of path.split("/")) {
    if (!part || part === ".") continue;
    if (part === "..") stack.pop();
    else stack.push(part);
  }
  return (absolute ? "/" : "") + stack.join("/") || (absolute ? "/" : ".");
}

function isMarkdown(path: string): boolean {
  return /\.(md|markdown|mdown)$/i.test(path);
}

function fileCategory(name: string, kind: "directory" | "file" | "other"): "folder" | "markdown" | "code" | "image" | "text" | "other" {
  if (kind === "directory") return "folder";
  if (kind === "other") return "other";
  if (isMarkdown(name)) return "markdown";
  if (/\.(png|jpe?g|webp|gif|svg|ico|bmp|avif)$/i.test(name)) return "image";
  if (/\.(tsx?|jsx?|mjs|cjs|py|pyi|json|jsonl|ya?ml|toml|ini|conf|sh|bash|zsh|css|scss|html?|xml|go|rs|sql|diff|patch)$/i.test(name)) {
    return "code";
  }
  return "text";
}

function fileTypeLabel(name: string, kind: "directory" | "file" | "other"): string {
  if (kind === "directory") return "文件夹";
  if (kind === "other") return "特殊";
  const cat = fileCategory(name, kind);
  if (cat === "markdown") return "Markdown";
  if (cat === "image") return "图像";
  const dot = name.lastIndexOf(".");
  if (dot > 0 && dot < name.length - 1) {
    return name.slice(dot + 1).toUpperCase();
  }
  return "文件";
}

function formatBytes(bytes: number | undefined): string {
  if (bytes === undefined || !Number.isFinite(bytes)) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function readPins(): string[] {
  try {
    const raw: unknown = JSON.parse(window.localStorage.getItem(PINS_KEY) ?? "[]");
    return Array.isArray(raw)
      ? raw.filter((item): item is string => typeof item === "string" && item.startsWith("/")).slice(0, MAX_PINS)
      : [];
  } catch {
    return [];
  }
}

function writePins(pins: readonly string[]): void {
  try {
    window.localStorage.setItem(PINS_KEY, JSON.stringify(pins.slice(0, MAX_PINS)));
  } catch {
    // 无痕或存储满时只在内存生效。
  }
}

function readViewMode(): "list" | "grid" {
  try {
    return window.localStorage.getItem(VIEW_MODE_KEY) === "grid" ? "grid" : "list";
  } catch {
    return "list";
  }
}

async function copyString(value: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(value);
      return true;
    }
  } catch {
    // 非 HTTPS 下回退临时选区复制。
  }
  try {
    const area = document.createElement("textarea");
    area.value = value;
    area.setAttribute("readonly", "true");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.append(area);
    area.select();
    const ok = document.execCommand("copy");
    area.remove();
    return ok;
  } catch {
    return false;
  }
}

function EntryIcon({ name, kind, size = 16 }: { name: string; kind: "directory" | "file" | "other"; size?: number }) {
  const cat = fileCategory(name, kind);
  if (cat === "folder") {
    return <Folder size={size} strokeWidth={1.75} className="md-icon-folder" aria-hidden="true" />;
  }
  if (cat === "markdown") {
    return <FileText size={size} strokeWidth={1.75} className="md-icon-md" aria-hidden="true" />;
  }
  if (cat === "code") {
    return <FileCode size={size} strokeWidth={1.75} className="md-icon-code" aria-hidden="true" />;
  }
  if (cat === "image") {
    return <FileImage size={size} strokeWidth={1.75} className="md-icon-image" aria-hidden="true" />;
  }
  return <FileText size={size} strokeWidth={1.75} className="md-icon-plain" aria-hidden="true" />;
}

function RootIcon({ id, size = 14 }: { id: string; size?: number }) {
  if (id === "memory") return <FolderOpen size={size} strokeWidth={1.75} aria-hidden="true" />;
  if (id === "home") return <Home size={size} strokeWidth={1.75} aria-hidden="true" />;
  if (id === "workdir") return <Terminal size={size} strokeWidth={1.75} aria-hidden="true" />;
  return <HardDrive size={size} strokeWidth={1.75} aria-hidden="true" />;
}

interface EntryRowProps {
  item: Entry;
  viewMode: "list" | "grid";
  wide: boolean;
  activePath: string;
  leavingTargetPath: string | null;
  onOpen: (item: Entry, kind: "directory" | "file" | "other") => void;
}

function EntryRow({
  item,
  viewMode,
  wide,
  activePath,
  leavingTargetPath,
  onOpen,
}: EntryRowProps) {
  const isPresent = useIsPresent();
  const kind = item.kind ?? (item.name.endsWith("/") ? "directory" : "file");
  const displayName = item.name.endsWith("/") && item.name.length > 1 ? item.name.slice(0, -1) : item.name;
  const shouldBindMorph = isPresent || leavingTargetPath !== item.path;
  const isMorphingOut = !isPresent && leavingTargetPath === item.path;
  const isCurrent = Boolean(activePath && item.path === activePath);

  if (viewMode === "grid") {
    return (
      <button
        type="button"
        className="md-grid-tile md-entry"
        title={item.path}
        data-kind={kind}
        disabled={kind === "other"}
        aria-current={isCurrent ? "true" : undefined}
        onClick={() => onOpen(item, kind)}
      >
        <span className="md-grid-icon">
          <EntryIcon name={displayName} kind={kind} size={28} />
        </span>
        <span className="md-morph-anchor">
          {shouldBindMorph ? (
            <motion.span
              layoutId={`morph-grid-${item.path}`}
              layout="position"
              transition={SPRING}
              className="md-grid-label"
              style={{ opacity: isMorphingOut ? 0 : 1 }}
            >
              {displayName}
            </motion.span>
          ) : (
            <span className="md-grid-label" style={{ opacity: 0 }}>
              {displayName}
            </span>
          )}
        </span>
        {kind === "other" ? (
          <span className="md-entry-badge">特殊</span>
        ) : (
          <span className="md-grid-sub">{fileTypeLabel(displayName, kind)}</span>
        )}
      </button>
    );
  }

  return (
    <button
      type="button"
      className="md-entry"
      title={item.path}
      data-kind={kind}
      disabled={kind === "other"}
      aria-current={isCurrent ? "true" : undefined}
      onClick={() => onOpen(item, kind)}
    >
      <span className="md-entry-left">
        <EntryIcon name={displayName} kind={kind} size={16} />
        <span className="md-morph-anchor">
          {shouldBindMorph ? (
            <motion.span
              layoutId={`morph-list-${item.path}`}
              layout="position"
              transition={SPRING}
              className="md-entry-label"
              style={{ opacity: isMorphingOut ? 0 : 1 }}
            >
              {displayName}
            </motion.span>
          ) : (
            <span className="md-entry-label" style={{ opacity: 0 }}>
              {displayName}
            </span>
          )}
        </span>
      </span>
      <span className="md-entry-right">
        {kind === "other" ? (
          <span className="md-entry-badge">特殊</span>
        ) : (
          <>
            {wide && <span className="md-entry-type">{fileTypeLabel(displayName, kind)}</span>}
            {kind === "directory" ? (
              <ChevronRight size={14} className="md-entry-chevron" aria-hidden="true" />
            ) : (
              <span className="md-entry-chevron-spacer" aria-hidden="true" />
            )}
          </>
        )}
      </span>
    </button>
  );
}

interface CrumbSegment {
  path: string;
  label: string;
}

function FileManagerApp({
  ctx,
  view,
}: {
  ctx: WebHostContextV1;
  view: ConversationTabView;
}) {
  const rootRef = useRef<HTMLElement | null>(null);
  const articleRef = useRef<HTMLElement | null>(null);
  const proseHostRef = useRef<HTMLDivElement | null>(null);
  const filterInputRef = useRef<HTMLInputElement | null>(null);
  const pathInputRef = useRef<HTMLInputElement | null>(null);

  const [panelWidth, setPanelWidth] = useState<number>(540);
  const [session, setSession] = useState<string | null>(view.session);
  const [mode, setMode] = useState<"browse" | "read">("browse");
  const [directory, setDirectory] = useState<string>("");
  const [path, setPath] = useState<string>("");
  const [workdir, setWorkdir] = useState<string | null | undefined>(undefined);
  const [roots, setRoots] = useState<RootPreset[]>([]);
  const [pins, setPins] = useState<string[]>(() => readPins());
  const [items, setItems] = useState<Entry[]>([]);
  const [afterCursor, setAfterCursor] = useState<string | null>(null);
  const [parentDir, setParentDir] = useState<string | null>(null);
  const [showHidden, setShowHidden] = useState(false);
  const [viewMode, setViewMode] = useState<"list" | "grid">(() => readViewMode());
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [placesOpen, setPlacesOpen] = useState(false);
  const [ellipsisOpen, setEllipsisOpen] = useState(false);
  const [pathEditing, setPathEditing] = useState(false);
  const [pathInput, setPathInput] = useState("");
  const [filterOpen, setFilterOpen] = useState(false);
  const [filterText, setFilterText] = useState("");
  const [outlineOpen, setOutlineOpen] = useState(false);
  const [outlineItems, setOutlineItems] = useState<OutlineItem[]>([]);
  const [source, setSource] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [docResult, setDocResult] = useState<Result | null>(null);
  const [selectedLines, setSelectedLines] = useState<{ start: number; end: number } | null>(null);
  const [copyFeedback, setCopyFeedback] = useState<"" | "copied" | "failed">("");
  const [quoteFeedback, setQuoteFeedback] = useState(false);
  const [navNote, setNavNote] = useState<{ text: string; isError: boolean; retry: (() => void) | null }>({
    text: "",
    isError: false,
    retry: null,
  });
  const [stageNote, setStageNote] = useState<{ text: string; isError: boolean; retry: (() => void) | null }>({
    text: "",
    isError: false,
    retry: null,
  });
  const [direction, setDirection] = useState<1 | -1>(1);
  const [leavingTargetPath, setLeavingTargetPath] = useState<string | null>(null);
  const [imageBlobUrl, setImageBlobUrl] = useState<string>("");
  const [imageError, setImageError] = useState<string>("");

  const syntaxRef = useRef<FileSyntax | null>(null);
  if (!syntaxRef.current) syntaxRef.current = new FileSyntax();
  const filesAbortRef = useRef<AbortController | null>(null);
  const docAbortRef = useRef<AbortController | null>(null);
  const navigationRef = useRef(0);
  const generationRef = useRef(0);
  const pendingLineRef = useRef<number | null>(null);
  const pendingFragmentRef = useRef<string>("");
  const blobUrlsRef = useRef<string[]>([]);
  const copyTimerRef = useRef(0);
  const quoteTimerRef = useRef(0);
  const targetTimerRef = useRef(0);
  const disposedRef = useRef(false);

  // 最新可变快照给异步回调使用
  const stateRef = useRef({
    session,
    directory,
    path,
    workdir,
    roots,
    showHidden,
    source,
    outlineOpen,
    docResult,
    selectedLines,
    downloading,
    mode,
  });
  stateRef.current = {
    session,
    directory,
    path,
    workdir,
    roots,
    showHidden,
    source,
    outlineOpen,
    docResult,
    selectedLines,
    downloading,
    mode,
  };

  const clearBlobs = useCallback(() => {
    for (const url of blobUrlsRef.current) URL.revokeObjectURL(url);
    blobUrlsRef.current = [];
  }, []);

  const rawUrl = useCallback((targetPath: string, download: boolean) => {
    const curSession = stateRef.current.session;
    const query = new URLSearchParams({
      path: targetPath,
      ...(curSession ? { session: curSession } : {}),
      ...(download ? { download: "1" } : {}),
    });
    return `${API}/raw?${query}`;
  }, []);

  const requestJson = useCallback(
    async (route: string, params: Record<string, string>, signal?: AbortSignal): Promise<Result> => {
      const curSession = stateRef.current.session;
      const query = new URLSearchParams({
        ...(curSession ? { session: curSession } : {}),
        ...params,
      });
      const response = await ctx.http.request(`${API}/${route}?${query}`, signal ? { signal } : undefined);
      let data: Result | null = null;
      try {
        data = (await response.json()) as Result;
      } catch {
        if (!response.ok) throw new Error(`服务端响应异常（${response.status}）`);
        throw new Error("服务端返回格式无效");
      }
      if (!response.ok) {
        throw new Error(data?.message || data?.error || `请求失败（${response.status}）`);
      }
      return data;
    },
    [ctx]
  );

  const normalizeRoots = useCallback((result: Result): RootPreset[] => {
    if (Array.isArray(result.roots) && result.roots.length) return result.roots;
    if (typeof result.root === "string" && result.root) {
      return [{ id: "workdir", label: "会话目录", path: result.root }];
    }
    return stateRef.current.roots;
  }, []);

  const attachImageElement = useCallback(
    (img: HTMLImageElement, targetPath: string, gen: number, signal?: AbortSignal) => {
      img.alt = basename(targetPath);
      void ctx.http
        .request(rawUrl(targetPath, false), signal ? { signal } : undefined)
        .then(async (res) => {
          if (!res.ok) throw new Error(`图片读取失败（${res.status}）`);
          return res.blob();
        })
        .then((blob) => {
          if (disposedRef.current || gen !== generationRef.current || !img.isConnected) return;
          const url = URL.createObjectURL(blob);
          blobUrlsRef.current.push(url);
          img.src = url;
        })
        .catch(() => {
          if (disposedRef.current || gen !== generationRef.current || !img.isConnected) return;
          const note = document.createElement("span");
          note.className = "md-image-note";
          note.textContent = `[图片未能读取：${basename(targetPath)}]`;
          img.replaceWith(note);
        });
    },
    [ctx, rawUrl]
  );

  const openFileRef = useRef<
    (
      targetPath: string,
      options?: { tail?: boolean; line?: number | null; fragment?: string; allowDirectoryFallback?: boolean; dir?: 1 | -1 }
    ) => Promise<"opened" | "directory" | "failed">
  >(async () => "failed");

  const filesRef = useRef<
    (targetPath?: string, after?: string, opts?: { dir?: 1 | -1; morphOutPath?: string | null; hiddenOverride?: boolean }) => Promise<void>
  >(async () => {});

  const syncParentDirectory = useCallback(
    async (parent: string) => {
      const navId = ++navigationRef.current;
      try {
        const result = await requestJson("files", {
          path: parent,
          ...(stateRef.current.showHidden ? { hidden: "1" } : {}),
        });
        if (disposedRef.current || navId !== navigationRef.current || result.status !== "available") return;
        setWorkdir(result.root ?? null);
        setRoots(normalizeRoots(result));
        const resolved = result.path || parent;
        setDirectory(resolved);
        setParentDir(result.parent ?? (resolved !== "/" ? dirnameOf(resolved) : null));
        setAfterCursor(result.after ?? null);
        setItems(result.items ?? []);
        setPathInput(resolved);
      } catch {
        // 附带目录同步失败不影响当前已打开的文件。
      }
    },
    [normalizeRoots, requestJson]
  );

  const loadFiles = useCallback(
    async (
      targetPath = "",
      after = "",
      opts?: { dir?: 1 | -1; morphOutPath?: string | null; hiddenOverride?: boolean }
    ) => {
      const navId = ++navigationRef.current;
      filesAbortRef.current?.abort();
      const abort = (filesAbortRef.current = new AbortController());
      const useHidden = opts?.hiddenOverride ?? stateRef.current.showHidden;

      if (!after) {
        setDirection(opts?.dir ?? (targetPath && stateRef.current.directory && targetPath.length < stateRef.current.directory.length ? -1 : 1));
        setLeavingTargetPath(opts?.morphOutPath ?? null);
        setMode("browse");
      }

      view.setIndicator?.({ busy: true, detail: "读取目录" });
      setNavNote({
        text: `正在读取服务端目录${targetPath ? ` ${basename(targetPath)}` : ""}…`,
        isError: false,
        retry: null,
      });

      try {
        const result = await requestJson(
          "files",
          {
            path: targetPath,
            ...(after ? { after } : {}),
            ...(useHidden ? { hidden: "1" } : {}),
          },
          abort.signal
        );
        if (disposedRef.current || navId !== navigationRef.current) return;
        setWorkdir(result.root ?? null);
        setRoots(normalizeRoots(result));

        if (result.status !== "available") {
          const msg =
            result.message ||
            `${STATUS_MESSAGES[result.status] || "目录读取失败"}${result.path ? `：${result.path}` : ""}`;
          setPathEditing(true);
          setNavNote({
            text: msg,
            isError: true,
            retry: () => void filesRef.current(targetPath, after, opts),
          });
          return;
        }

        const resolvedDir = result.path || targetPath;
        const dirChanged = resolvedDir !== stateRef.current.directory;
        setDirectory(resolvedDir);
        setParentDir(result.parent ?? (resolvedDir !== "/" ? dirnameOf(resolvedDir) : null));
        setAfterCursor(result.after ?? null);
        if (after && !dirChanged) {
          setItems((prev) => [...prev, ...(result.items ?? [])]);
        } else {
          setItems(result.items ?? []);
          if (dirChanged) {
            setFilterText("");
            setFilterOpen(false);
          }
        }
        setPathInput(resolvedDir);
        setNavNote({ text: "", isError: false, retry: null });

        if (!result.open_file) {
          view.setTitle?.(`文件 · ${basename(resolvedDir)}`.slice(0, 80));
        }

        if (result.open_file) {
          setPathEditing(false);
          await openFileRef.current(result.open_file, { dir: 1 });
        }
      } catch (error) {
        if (abort.signal.aborted || disposedRef.current || navId !== navigationRef.current) return;
        setPathEditing(true);
        setNavNote({
          text: error instanceof Error ? error.message : String(error),
          isError: true,
          retry: () => void filesRef.current(targetPath, after, opts),
        });
      } finally {
        if (filesAbortRef.current === abort) {
          filesAbortRef.current = null;
          if (!docAbortRef.current) view.setIndicator?.(null);
        }
      }
    },
    [normalizeRoots, requestJson, view]
  );

  filesRef.current = loadFiles;

  const openFile = useCallback(
    async (
      targetPath: string,
      options: {
        tail?: boolean;
        line?: number | null;
        fragment?: string;
        allowDirectoryFallback?: boolean;
        dir?: 1 | -1;
      } = {}
    ): Promise<"opened" | "directory" | "failed"> => {
      if (!targetPath) return "failed";
      const gen = ++generationRef.current;
      docAbortRef.current?.abort();
      const abort = (docAbortRef.current = new AbortController());
      syntaxRef.current?.cancel();

      view.setIndicator?.({ busy: true, detail: basename(targetPath) });
      setStageNote({ text: `正在读取 ${basename(targetPath)}…`, isError: false, retry: null });

      try {
        const result = await requestJson(
          "document",
          {
            path: targetPath,
            ...(options.tail ? { tail: "1" } : {}),
          },
          abort.signal
        );
        if (disposedRef.current || gen !== generationRef.current) return "failed";
        if (result.status === "not_file" && options.allowDirectoryFallback) {
          setStageNote({ text: "", isError: false, retry: null });
          return "directory";
        }
        if (result.status !== "available" && result.status !== "binary") {
          const msg = result.message || `${STATUS_MESSAGES[result.status] || "读取失败"}：${targetPath}`;
          setStageNote({
            text: msg,
            isError: true,
            retry: () => void openFileRef.current(targetPath, options),
          });
          setMode("read");
          return "failed";
        }

        const resolvedPath = result.path || targetPath;
        const fileChanged = resolvedPath !== stateRef.current.path;
        pendingLineRef.current = options.line ?? null;
        pendingFragmentRef.current = options.fragment ?? "";

        setDirection(options.dir ?? 1);
        setLeavingTargetPath(resolvedPath);
        setPath(resolvedPath);
        setDocResult(result);
        setSelectedLines(null);
        if (fileChanged) {
          setSource(Boolean(options.line && isMarkdown(resolvedPath)));
        }
        setMode("read");
        setPathEditing(false);
        setPlacesOpen(false);
        setEllipsisOpen(false);
        setStageNote({ text: "", isError: false, retry: null });
        view.setTitle?.(`文件 · ${basename(resolvedPath)}`.slice(0, 80));

        const parent = dirnameOf(resolvedPath);
        if (resolvedPath.startsWith("/") && parent !== stateRef.current.directory && !filesAbortRef.current) {
          void syncParentDirectory(parent);
        }
        return "opened";
      } catch (error) {
        if (abort.signal.aborted || disposedRef.current || gen !== generationRef.current) return "failed";
        setStageNote({
          text: error instanceof Error ? error.message : String(error),
          isError: true,
          retry: () => void openFileRef.current(targetPath, options),
        });
        setMode("read");
        return "failed";
      } finally {
        if (docAbortRef.current === abort) {
          docAbortRef.current = null;
          if (!filesAbortRef.current) view.setIndicator?.(null);
        }
      }
    },
    [requestJson, syncParentDirectory, view]
  );

  openFileRef.current = openFile;

  const refreshSessionRoots = useCallback(async () => {
    try {
      const result = await requestJson("files", {
        path: stateRef.current.directory,
        ...(stateRef.current.showHidden ? { hidden: "1" } : {}),
      });
      if (disposedRef.current) return;
      const nextWorkdir = result.root ?? null;
      const prevWorkdir = stateRef.current.workdir;
      setWorkdir(nextWorkdir);
      setRoots(normalizeRoots(result));
      if (nextWorkdir && nextWorkdir !== prevWorkdir && nextWorkdir !== stateRef.current.directory) {
        void loadFiles(nextWorkdir);
      }
    } catch {
      // 后台刷新根目录列表失败不打断当前阅读。
    }
  }, [loadFiles, normalizeRoots, requestJson]);

  const openOrBrowse = useCallback(
    async (targetPath: string, line: number | null) => {
      if (targetPath.endsWith("/") && targetPath.length > 1) {
        await loadFiles(targetPath);
        return;
      }
      const opened = await openFile(targetPath, { line, allowDirectoryFallback: true });
      if (opened === "directory") {
        await loadFiles(targetPath);
      }
    },
    [loadFiles, openFile]
  );

  const handleIntent = useCallback(
    async (intent: ConversationIntent): Promise<ConversationIntentStatus> => {
      if (intent.name !== "workspace.document.open.v1" && intent.name !== "workspace.file.open.v1") {
        return "invalid";
      }
      const payload = intent.payload as { path?: unknown; line?: unknown } | null;
      if (
        !payload ||
        typeof payload.path !== "string" ||
        !payload.path ||
        payload.path.length > 4096 ||
        /[\x00-\x1f]/.test(payload.path) ||
        /^[a-z][a-z0-9+.-]*:/i.test(payload.path) ||
        payload.path.startsWith("//")
      ) {
        return "invalid";
      }
      const line =
        typeof payload.line === "number" && Number.isInteger(payload.line) && payload.line > 0
          ? payload.line
          : null;
      await openOrBrowse(payload.path, line);
      return "handled";
    },
    [openOrBrowse]
  );

  // 宿主订阅与初次目录加载
  useEffect(() => {
    disposedRef.current = false;
    let subscribing = true;
    const stops: WebUiDisposer[] = [
      view.onSessionChange((nextSession) => {
        if (nextSession === stateRef.current.session && stateRef.current.directory) return;
        setSession(nextSession);
        stateRef.current.session = nextSession;
        setWorkdir(undefined);
        if (!stateRef.current.directory) void filesRef.current("");
        else void refreshSessionRoots();
      }),
      view.onIntent(async (intent) => handleIntent(intent)),
      view.onActiveChange((active) => {
        if (active && !subscribing && !stateRef.current.directory && !filesAbortRef.current) {
          void filesRef.current("");
        }
      }),
    ];
    subscribing = false;
    if (!stateRef.current.directory && !filesAbortRef.current) {
      void loadFiles("");
    }
    return () => {
      disposedRef.current = true;
      generationRef.current += 1;
      navigationRef.current += 1;
      filesAbortRef.current?.abort();
      docAbortRef.current?.abort();
      syntaxRef.current?.cancel();
      clearBlobs();
      window.clearTimeout(copyTimerRef.current);
      window.clearTimeout(quoteTimerRef.current);
      window.clearTimeout(targetTimerRef.current);
      stops.forEach((stop) => stop());
    };
  }, [clearBlobs, handleIntent, loadFiles, refreshSessionRoots, view]);

  // 监测面板容器宽度以切换停靠模式 / 宽屏模式
  useEffect(() => {
    const el = rootRef.current;
    if (!el) return;
    const measure = () => {
      if (el.clientWidth > 0) setPanelWidth(el.clientWidth);
    };
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const onDocClick = () => {
      setPlacesOpen(false);
      setEllipsisOpen(false);
    };
    document.addEventListener("click", onDocClick);
    return () => document.removeEventListener("click", onDocClick);
  }, []);

  // 点击 Backspace 返回上级目录或退出文件阅读
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) {
        return;
      }
      if (!rootRef.current?.contains(document.activeElement) && document.activeElement !== document.body) {
        return;
      }
      if (event.key === "Backspace") {
        if (stateRef.current.mode === "read") {
          event.preventDefault();
          const curPath = stateRef.current.path;
          setDirection(-1);
          setLeavingTargetPath(curPath);
          setMode("browse");
          if (stateRef.current.directory) {
            view.setTitle?.(`文件 · ${basename(stateRef.current.directory)}`.slice(0, 80));
          }
        } else if (parentDir && parentDir !== stateRef.current.directory) {
          event.preventDefault();
          void loadFiles(parentDir, "", { dir: -1, morphOutPath: stateRef.current.directory });
        }
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [loadFiles, parentDir, view]);

  const jumpFragment = useCallback((fragment: string) => {
    const article = articleRef.current;
    if (!article || !fragment) return;
    let id: string;
    try {
      id = decodeURIComponent(fragment);
    } catch {
      return;
    }
    const nodes = [...article.querySelectorAll<HTMLElement>("[id]")];
    const heading = nodes.find((node) => node.id === `md-${id}`) ?? nodes.find((node) => node.id === id);
    heading?.scrollIntoView({ block: "start" });
  }, []);

  const revealLine = useCallback((lineNo: number) => {
    const article = articleRef.current;
    if (!article) return;
    const row = article.querySelector<HTMLElement>(`.md-code-row[data-line="${lineNo}"]`);
    if (!row) return;
    row.scrollIntoView({ block: "center" });
    row.classList.add("is-target");
    window.clearTimeout(targetTimerRef.current);
    targetTimerRef.current = window.setTimeout(() => row.classList.remove("is-target"), 1600);
  }, []);

  // 渲染 Markdown / 代码 / 图片正文的命令式 DOM 桥接（保留共用 markdown-prose 与 syntax worker）
  useEffect(() => {
    const article = articleRef.current;
    const proseHost = proseHostRef.current;
    if (mode !== "read" || !article || !docResult || !path) {
      syntaxRef.current?.cancel();
      clearBlobs();
      setImageBlobUrl("");
      setImageError("");
      setOutlineItems([]);
      return undefined;
    }

    const gen = generationRef.current;
    syntaxRef.current?.cancel();
    clearBlobs();
    setImageBlobUrl("");
    setImageError("");
    proseHost?.replaceChildren();

    if (docResult.mime_type) {
      const controller = new AbortController();
      void ctx.http
        .request(rawUrl(path, false), { signal: controller.signal })
        .then(async (res) => {
          if (!res.ok) throw new Error(`图片读取失败（${res.status}）`);
          return res.blob();
        })
        .then((blob) => {
          if (disposedRef.current || gen !== generationRef.current) return;
          const url = URL.createObjectURL(blob);
          blobUrlsRef.current.push(url);
          setImageBlobUrl(url);
        })
        .catch((err) => {
          if (controller.signal.aborted || disposedRef.current || gen !== generationRef.current) return;
          setImageError(err instanceof Error ? err.message : `图片未能读取：${basename(path)}`);
        });
      return () => controller.abort();
    }

    if (docResult.status === "binary") {
      setOutlineItems([]);
      return undefined;
    }

    const text = docResult.text ?? "";
    if (!text || !proseHost) {
      setOutlineItems([]);
      return undefined;
    }

    if (isMarkdown(path) && !source) {
      const dir = dirnameOf(path);
      const pendingImages: Array<{ img: HTMLImageElement; target: string }> = [];
      const prose = renderMarkdownProse(text, {
        onImage: (image, src) => {
          if (
            !src ||
            /^[a-z][a-z0-9+.-]*:/i.test(src) ||
            src.startsWith("//") ||
            !/\.(png|jpe?g|webp|gif)$/i.test(src.split("?")[0])
          ) {
            return false;
          }
          const joined = src.startsWith("/") ? src : `${dir === "/" ? "" : dir}/${src}`;
          pendingImages.push({ img: image, target: normalizePosix(joined) });
          return true;
        },
      });
      proseHost.append(prose);
      const controller = new AbortController();
      for (const item of pendingImages) {
        attachImageElement(item.img, item.target, gen, controller.signal);
      }

      // 构建大纲
      const headings = [...proseHost.querySelectorAll<HTMLElement>(".md-prose :is(h1,h2,h3,h4,h5,h6)")].map(
        (h, i) => ({
          id: h.id || `h-${i}`,
          text: h.textContent || "标题",
          level: Number(h.tagName.slice(1)),
          el: h,
        })
      );
      setOutlineItems(headings);

      if (pendingFragmentRef.current) {
        const frag = pendingFragmentRef.current;
        pendingFragmentRef.current = "";
        jumpFragment(frag);
      }
      return () => controller.abort();
    }

    // 源码 / 代码行号视图
    setOutlineItems([]);
    const lines = text.replace(/\r\n/g, "\n").split("\n");
    if (lines.length > 1 && lines[lines.length - 1] === "") lines.pop();
    const pre = document.createElement("div");
    pre.className = "md-source md-code-view";
    pre.setAttribute("role", "region");
    pre.setAttribute("aria-label", `${basename(path)} 代码内容`);
    lines.forEach((lineText, idx) => {
      const lineNo = idx + 1;
      const row = document.createElement("div");
      row.className = "md-code-row";
      row.dataset.line = String(lineNo);
      const gutter = document.createElement("button");
      gutter.type = "button";
      gutter.className = "md-code-ln";
      gutter.textContent = String(lineNo);
      gutter.title = "点击选择行，Shift+点击选择范围后可引用到对话";
      gutter.onclick = (event) => {
        setSelectedLines((prev) => {
          let next: { start: number; end: number } | null;
          if (event.shiftKey && prev) {
            next = { start: Math.min(prev.start, lineNo), end: Math.max(prev.start, lineNo) };
          } else if (prev && prev.start === lineNo && prev.end === lineNo) {
            next = null;
          } else {
            next = { start: lineNo, end: lineNo };
          }
          pre.querySelectorAll<HTMLElement>(".md-code-row").forEach((r) => {
            const n = Number(r.dataset.line);
            r.classList.toggle("is-selected", Boolean(next && n >= next.start && n <= next.end));
          });
          return next;
        });
      };
      const code = document.createElement("span");
      code.className = "md-code-text";
      code.textContent = lineText || " ";
      row.append(gutter, code);
      pre.append(row);
    });
    proseHost.append(pre);
    article.scrollTop = 0;
    syntaxRef.current?.highlight(path, lines, pre);
    if (pendingLineRef.current !== null) {
      const targetLine = pendingLineRef.current;
      pendingLineRef.current = null;
      revealLine(targetLine);
    }
    return undefined;
  }, [attachImageElement, clearBlobs, ctx, docResult, jumpFragment, mode, path, rawUrl, revealLine, source]);

  const handleArticleClick = (event: ReactMouseEvent<HTMLElement>) => {
    const link = (event.target as Element).closest("a");
    if (!link) return;
    const href = link.getAttribute("href") || "";
    if (href.startsWith("#")) {
      event.preventDefault();
      jumpFragment(href.slice(1));
      return;
    }
    if (/^(https?|mailto):/i.test(href)) return;
    event.preventDefault();
    if (/^[a-z][a-z0-9+.-]*:/i.test(href)) return;
    const [raw, fragment] = href.split("#");
    let relative: string;
    try {
      relative = decodeURIComponent(raw);
    } catch {
      setStageNote({ text: "链接路径编码无效。", isError: true, retry: null });
      return;
    }
    const base = dirnameOf(path);
    const nextPath = relative.startsWith("/")
      ? relative
      : normalizePosix(`${base === "/" ? "" : base}/${relative}`);
    void openFile(nextPath, { fragment, allowDirectoryFallback: true }).then((res) => {
      if (res === "directory") {
        void loadFiles(nextPath);
      }
    });
  };

  const togglePin = () => {
    if (!directory) return;
    setPins((prev) => {
      const next = prev.includes(directory)
        ? prev.filter((item) => item !== directory)
        : [directory, ...prev.filter((item) => item !== directory)].slice(0, MAX_PINS);
      writePins(next);
      return next;
    });
  };

  const toggleHidden = () => {
    const next = !showHidden;
    setShowHidden(next);
    void loadFiles(directory, "", { hiddenOverride: next });
  };

  const handleCopyPath = async () => {
    const target = mode === "read" && path ? path : directory;
    if (!target) return;
    const ok = await copyString(target);
    setCopyFeedback(ok ? "copied" : "failed");
    window.clearTimeout(copyTimerRef.current);
    copyTimerRef.current = window.setTimeout(() => setCopyFeedback(""), 1500);
  };

  const handleDownload = async () => {
    if (!path || downloading) return;
    const targetPath = path;
    setDownloading(true);
    try {
      const response = await ctx.http.request(rawUrl(targetPath, true));
      if (!response.ok) throw new Error(`下载失败（${response.status}）`);
      const blob = await response.blob();
      if (disposedRef.current) return;
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = basename(targetPath);
      document.body.append(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 2000);
    } catch (error) {
      if (!disposedRef.current) {
        setStageNote({
          text: error instanceof Error ? error.message : "下载失败",
          isError: true,
          retry: () => void handleDownload(),
        });
      }
    } finally {
      if (!disposedRef.current) setDownloading(false);
    }
  };

  const handleQuote = () => {
    if (!session || !path || typeof docResult?.text !== "string") return;
    const browserSelection = window.getSelection()?.toString().trim() ?? "";
    const allLines = docResult.text.replace(/\r\n/g, "\n").split("\n");
    let snippet = "";
    let sourceLabel = `文件 · ${path}`;
    if (selectedLines) {
      const { start, end } = selectedLines;
      snippet = allLines.slice(start - 1, Math.min(end, start + 120)).join("\n");
      sourceLabel = `文件 · ${path}:${start}${end > start ? `-${end}` : ""}`;
    } else if (browserSelection) {
      snippet = browserSelection.slice(0, 4000);
    } else {
      snippet = allLines.slice(0, 40).join("\n");
      if (allLines.length > 40) sourceLabel = `文件 · ${path}:1-40`;
    }
    if (!snippet) return;
    const status = view.quote?.({
      session,
      source: sourceLabel.slice(0, 200),
      text: `\`\`\`\n${snippet.slice(0, 7800)}\n\`\`\``,
    });
    if (status === "inserted") {
      setQuoteFeedback(true);
      window.clearTimeout(quoteTimerRef.current);
      quoteTimerRef.current = window.setTimeout(() => setQuoteFeedback(false), 1500);
    }
  };

  const handleRefresh = async () => {
    if (mode === "browse" && directory) {
      await loadFiles(directory);
    } else if (mode === "read" && path) {
      await openFile(path, { tail: Boolean(docResult?.tail) });
    } else {
      await loadFiles("");
    }
  };

  const handleOpenEntry = (item: Entry, kind: "directory" | "file" | "other") => {
    if (kind === "directory") {
      void loadFiles(item.path, "", { dir: 1, morphOutPath: item.path });
      return;
    }
    if (kind === "other") return;
    void openFile(item.path, { dir: 1 });
  };

  const handleReturnToBrowse = () => {
    if (mode === "read") {
      setDirection(-1);
      setLeavingTargetPath(path);
      setMode("browse");
      if (directory) {
        view.setTitle?.(`文件 · ${basename(directory)}`.slice(0, 80));
      }
      return;
    }
    if (parentDir && parentDir !== directory) {
      void loadFiles(parentDir, "", { dir: -1, morphOutPath: directory });
    }
  };

  const isWide = panelWidth >= WIDE_BREAKPOINT_PX;
  const isNarrow = panelWidth > 0 && panelWidth < NARROW_BREAKPOINT_PX;
  const activeFullPath = mode === "read" && path ? path : directory;

  // 构建面包屑段列表（browse 时为 directory，read 时为 directory + file；保留相对/绝对前缀给 title 契约）
  const segments = useMemo<CrumbSegment[]>(() => {
    const baseDir = activeFullPath || "/";
    if (baseDir === "/") return [];
    const absolute = baseDir.startsWith("/");
    const parts = baseDir.split("/").filter(Boolean);
    const out: CrumbSegment[] = [];
    let acc = "";
    for (const part of parts) {
      acc = acc ? `${acc}/${part}` : absolute ? `/${part}` : part;
      out.push({ path: acc, label: part });
    }
    if (out.length > 0 && activeFullPath) {
      out[out.length - 1] = { ...out[out.length - 1], path: activeFullPath };
    }
    return out;
  }, [activeFullPath]);

  // 依据面板宽度与尾部文字长度智能折叠中间层级为 ···，保证当前文件名/目录名永远完整可读且不撞编辑按钮
  const tailTwoChars = segments.slice(-2).reduce((acc, s) => acc + s.label.length, 0);
  const maxTail = isWide
    ? (tailTwoChars > 34 ? 2 : 3)
    : panelWidth < 560
    ? (tailTwoChars > 12 ? 1 : 2)
    : (tailTwoChars > 20 ? 1 : 2);
  const keepTail = segments.length > maxTail ? maxTail : segments.length;
  const collapsedSegments = segments.length > keepTail ? segments.slice(0, segments.length - keepTail) : [];
  const visibleSegments = keepTail > 0 ? segments.slice(segments.length - keepTail) : [];
  const hideInlineRoot = panelWidth > 0 && panelWidth < 460 && collapsedSegments.length > 0;

  const filteredItems = useMemo(() => {
    const needle = filterText.trim().toLowerCase();
    if (!needle) return items;
    return items.filter((item) => item.name.toLowerCase().includes(needle));
  }, [filterText, items]);

  const activePlace = useMemo(() => {
    const found = roots.find((r) => directory === r.path);
    if (found) return found;
    const prefixMatch = roots
      .filter((r) => directory && (directory === r.path || directory.startsWith(r.path === "/" ? "/" : `${r.path}/`)))
      .sort((a, b) => b.path.length - a.path.length)[0];
    return prefixMatch ?? roots[0] ?? null;
  }, [directory, roots]);

  const hasFile = Boolean(path && docResult);
  const isMd = hasFile && isMarkdown(path) && docResult?.status === "available" && typeof docResult.text === "string";
  const hasText = hasFile && docResult?.status === "available" && typeof docResult.text === "string";
  const canDownload =
    hasFile &&
    (docResult?.bytes ?? 0) <= IMPORT_LIMIT_BYTES &&
    (docResult?.status === "available" || docResult?.status === "binary");
  const isPinned = Boolean(directory && pins.includes(directory));
  const needle = filterText.trim();
  const canGoBack = mode === "read" || Boolean(parentDir && parentDir !== directory);

  const quoteLabel = selectedLines
    ? selectedLines.start === selectedLines.end
      ? `引用 L${selectedLines.start}`
      : `引用 L${selectedLines.start}-${selectedLines.end}`
    : quoteFeedback
    ? "已引用"
    : "引用";

  const viewKey = mode === "read" ? `read:${path || "empty"}` : `browse:${directory || "init"}`;

  return (
    <MotionConfig reducedMotion="user">
      <section
        ref={rootRef}
        className="md-reader"
        data-mode={mode}
        data-wide={String(isWide)}
        data-narrow={String(isNarrow)}
        data-nav-open={String(isWide && sidebarOpen)}
        data-has-file={String(hasFile)}
      >
        {/* 1. 统一单行导航工具栏（45px 含 1px 底边 = 44px 偶数内高） */}
        <LayoutGroup id="md-reader-nav">
          <div className="md-bar">
            <div className="md-bar-left">
              {isWide ? (
                <button
                  type="button"
                  className="md-icon-btn"
                  aria-pressed={sidebarOpen}
                  aria-label="切换位置边栏"
                  title="显示或隐藏位置边栏"
                  onClick={() => setSidebarOpen((v) => !v)}
                >
                  <PanelLeft size={14} strokeWidth={1.75} aria-hidden="true" />
                </button>
              ) : (
                <div className="md-places-wrap" onClick={(e) => e.stopPropagation()}>
                  <button
                    type="button"
                    className="md-places-trigger"
                    aria-expanded={placesOpen}
                    aria-label="切换位置或收藏目录"
                    onClick={() => {
                      setEllipsisOpen(false);
                      setPlacesOpen((v) => !v);
                    }}
                  >
                    <FolderOpen size={14} strokeWidth={1.75} className="md-places-icon" aria-hidden="true" />
                    <span className="md-places-label">{activePlace?.label ?? "位置"}</span>
                    <ChevronDown size={12} strokeWidth={1.75} aria-hidden="true" />
                  </button>

                  <AnimatePresence>
                    {placesOpen && (
                      <motion.div
                        initial={{ opacity: 0, y: -4, scale: 0.96 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: -4, scale: 0.96 }}
                        transition={FAST_SPRING}
                        className="md-floating-menu md-places-menu"
                        role="menu"
                      >
                        <div className="md-menu-label">位置</div>
                        {roots.map((preset) => (
                          <button
                            key={preset.id}
                            type="button"
                            className="md-menu-row"
                            aria-current={directory === preset.path ? "true" : undefined}
                            title={preset.path}
                            onClick={() => {
                              setPlacesOpen(false);
                              void loadFiles(preset.path);
                            }}
                          >
                            <RootIcon id={preset.id} size={14} />
                            <span>{preset.label}</span>
                          </button>
                        ))}
                        {pins.length > 0 && (
                          <>
                            <div className="md-menu-rule" />
                            <div className="md-menu-label">收藏</div>
                            {pins.map((pin) => (
                              <button
                                key={pin}
                                type="button"
                                className="md-menu-row"
                                aria-current={directory === pin ? "true" : undefined}
                                title={pin}
                                onClick={() => {
                                  setPlacesOpen(false);
                                  void loadFiles(pin);
                                }}
                              >
                                <Star size={14} strokeWidth={1.75} className="md-icon-star" aria-hidden="true" />
                                <span>{basename(pin)}</span>
                              </button>
                            ))}
                          </>
                        )}
                        <div className="md-menu-rule" />
                        {directory && (
                          <button
                            type="button"
                            className="md-menu-row"
                            onClick={() => {
                              setPlacesOpen(false);
                              togglePin();
                            }}
                          >
                            <Star
                              size={14}
                              strokeWidth={1.75}
                              className={isPinned ? "md-icon-star-filled" : "md-icon-star"}
                              aria-hidden="true"
                            />
                            <span>{isPinned ? "取消收藏当前目录" : "收藏当前目录"}</span>
                          </button>
                        )}
                        <button
                          type="button"
                          className="md-menu-row"
                          onClick={() => {
                            setPlacesOpen(false);
                            setPathInput(activeFullPath);
                            setPathEditing(true);
                          }}
                        >
                          <Terminal size={14} strokeWidth={1.75} aria-hidden="true" />
                          <span>前往任意路径…</span>
                        </button>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              )}

              <button
                type="button"
                className="md-icon-btn"
                disabled={!canGoBack}
                aria-label={mode === "read" ? "返回目录" : "返回上一级"}
                title={mode === "read" ? "返回目录 (Backspace)" : "返回上一级 (Backspace)"}
                onClick={handleReturnToBrowse}
              >
                <ArrowLeft size={14} strokeWidth={1.75} aria-hidden="true" />
              </button>
            </div>

            {/* 中央地址栏：智能折叠面包屑 + 行内路径编辑 */}
            <div className="md-location">
              {pathEditing ? (
                <form
                  className="md-open"
                  onSubmit={(e: FormEvent) => {
                    e.preventDefault();
                    const target = pathInput.trim();
                    if (target) {
                      setPathEditing(false);
                      void loadFiles(target);
                    }
                  }}
                >
                  <input
                    ref={pathInputRef}
                    autoFocus
                    value={pathInput}
                    onChange={(e) => setPathInput(e.target.value)}
                    onKeyDown={(e: ReactKeyboardEvent<HTMLInputElement>) => {
                      if (e.key === "Escape") {
                        setPathEditing(false);
                        setPathInput(activeFullPath);
                      }
                    }}
                    placeholder="服务端路径（如 /srv/... 或 ~/...）"
                    aria-label="服务端目录或文件路径"
                    autoComplete="off"
                    spellCheck={false}
                  />
                  <button type="submit" className="md-open-submit">
                    前往
                  </button>
                  <button
                    type="button"
                    className="md-open-cancel"
                    aria-label="取消输入路径"
                    onClick={() => setPathEditing(false)}
                  >
                    <X size={12} strokeWidth={1.75} aria-hidden="true" />
                  </button>
                </form>
              ) : (
                <>
                  <nav className="md-crumbs" aria-label="路径导航">
                    {!hideInlineRoot && (
                      <button
                        type="button"
                        className={`md-crumb md-crumb-root ${segments.length === 0 ? "is-current" : ""}`}
                        title="/"
                        aria-current={segments.length === 0 ? "location" : undefined}
                        onClick={() => void loadFiles("/", "", { dir: -1 })}
                      >
                        <Home size={12} strokeWidth={1.75} aria-hidden="true" />
                        {segments.length === 0 && (
                          <span className="md-name md-sr-name" title="/">
                            /
                          </span>
                        )}
                      </button>
                    )}

                    {collapsedSegments.length > 0 && (
                      <span
                        className="md-crumb-ellipsis-wrap"
                        onClick={(e) => e.stopPropagation()}
                      >
                        {!hideInlineRoot && (
                          <ChevronRight size={12} className="md-crumb-sep" aria-hidden="true" />
                        )}
                        <button
                          type="button"
                          className="md-crumb-ellipsis"
                          aria-expanded={ellipsisOpen}
                          aria-label="展开折叠的祖先目录"
                          onClick={() => {
                            setPlacesOpen(false);
                            setEllipsisOpen((v) => !v);
                          }}
                        >
                          ···
                        </button>

                        <AnimatePresence>
                          {ellipsisOpen && (
                            <motion.div
                              initial={{ opacity: 0, y: -4, scale: 0.96 }}
                              animate={{ opacity: 1, y: 0, scale: 1 }}
                              exit={{ opacity: 0, y: -4, scale: 0.96 }}
                              transition={FAST_SPRING}
                              className="md-floating-menu md-ellipsis-menu"
                              role="menu"
                            >
                              {hideInlineRoot && (
                                <button
                                  type="button"
                                  className="md-menu-row"
                                  title="/"
                                  onClick={() => {
                                    setEllipsisOpen(false);
                                    void loadFiles("/", "", { dir: -1 });
                                  }}
                                >
                                  <Home size={14} strokeWidth={1.75} className="md-icon-plain" aria-hidden="true" />
                                  <span>/</span>
                                </button>
                              )}
                              {collapsedSegments.map((seg) => (
                                <button
                                  key={seg.path}
                                  type="button"
                                  className="md-menu-row"
                                  title={seg.path}
                                  onClick={() => {
                                    setEllipsisOpen(false);
                                    void loadFiles(seg.path, "", { dir: -1 });
                                  }}
                                >
                                  <Folder size={14} strokeWidth={1.75} className="md-icon-folder" aria-hidden="true" />
                                  <span>{seg.label}</span>
                                </button>
                              ))}
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </span>
                    )}

                    {visibleSegments.map((seg, idx) => {
                      const isLeaf = idx === visibleSegments.length - 1;
                      return (
                        <span key={seg.path} className="md-crumb-segment">
                          <ChevronRight size={12} className="md-crumb-sep" aria-hidden="true" />
                          <button
                            type="button"
                            className={`md-crumb ${isLeaf ? "is-current" : ""}`}
                            title={seg.path}
                            aria-current={isLeaf ? "location" : undefined}
                            onClick={() => {
                              if (!isLeaf) {
                                void loadFiles(seg.path, "", {
                                  dir: -1,
                                  morphOutPath: mode === "read" ? path : directory,
                                });
                              }
                            }}
                          >
                            {isLeaf ? (
                              <motion.span
                                layoutId={`morph-${viewMode}-${seg.path}`}
                                layout="position"
                                transition={SPRING}
                                className="md-crumb-label md-name"
                                title={seg.path}
                              >
                                {seg.label}
                              </motion.span>
                            ) : (
                              <span className="md-crumb-label" title={seg.path}>
                                {seg.label}
                              </span>
                            )}
                          </button>
                        </span>
                      );
                    })}

                    {/* 初次尚未拿到目录前保证 .md-name 占位不缺失 */}
                    {segments.length === 0 && !activeFullPath && (
                      <span className="md-name md-crumb-placeholder" title="" />
                    )}
                  </nav>

                  <button
                    type="button"
                    className="md-location-edit"
                    aria-label="输入任意服务端路径"
                    title="输入服务端目录或文件路径"
                    onClick={() => {
                      setPathInput(activeFullPath);
                      setPathEditing(true);
                    }}
                  >
                    <Pencil size={12} strokeWidth={1.75} aria-hidden="true" />
                  </button>
                </>
              )}
            </div>

            {/* 右侧控制组 */}
            <div className="md-bar-right">
              {mode === "browse" && (
                <>
                  <button
                    type="button"
                    className="md-icon-btn md-icon-search"
                    aria-pressed={filterOpen}
                    aria-label="筛选当前目录"
                    title="筛选当前目录"
                    onClick={() => {
                      setFilterOpen((v) => {
                        const next = !v;
                        if (!next) setFilterText("");
                        else window.setTimeout(() => filterInputRef.current?.focus(), 40);
                        return next;
                      });
                    }}
                  >
                    <Search size={14} strokeWidth={1.75} aria-hidden="true" />
                  </button>

                  <div className="md-view-seg" role="group" aria-label="视图模式">
                    <button
                      type="button"
                      className="md-seg-btn"
                      aria-pressed={viewMode === "list"}
                      aria-label="列表视图"
                      title="列表视图"
                      onClick={() => {
                        setViewMode("list");
                        try { window.localStorage.setItem(VIEW_MODE_KEY, "list"); } catch {}
                      }}
                    >
                      <List size={14} strokeWidth={1.75} aria-hidden="true" />
                    </button>
                    <button
                      type="button"
                      className="md-seg-btn"
                      aria-pressed={viewMode === "grid"}
                      aria-label="网格视图"
                      title="网格视图"
                      onClick={() => {
                        setViewMode("grid");
                        try { window.localStorage.setItem(VIEW_MODE_KEY, "grid"); } catch {}
                      }}
                    >
                      <LayoutGrid size={14} strokeWidth={1.75} aria-hidden="true" />
                    </button>
                  </div>
                </>
              )}

              {mode === "read" && (
                <button
                  type="button"
                  className="md-icon-btn"
                  aria-label={copyFeedback === "copied" ? "已复制路径" : "复制路径"}
                  title="复制当前文件路径"
                  onClick={() => void handleCopyPath()}
                >
                  {copyFeedback === "copied" ? (
                    <Check size={14} strokeWidth={1.75} aria-hidden="true" />
                  ) : (
                    <Copy size={14} strokeWidth={1.75} aria-hidden="true" />
                  )}
                </button>
              )}

              <button
                type="button"
                className="md-icon-btn"
                aria-label="刷新"
                title="刷新当前视图"
                onClick={() => void handleRefresh()}
              >
                <RefreshCw size={14} strokeWidth={1.75} aria-hidden="true" />
              </button>
            </div>
          </div>

          {/* 可展开筛选栏 */}
          <AnimatePresence initial={false}>
            {filterOpen && mode === "browse" && (
              <motion.div
                // motion-exempt: 阅读器筛选栏、大纲面板展开时由 motion 库逐帧驱动高度，低频，ADR 0166 已知限制
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 41, opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={FAST_SPRING}
                className="md-filter-drawer"
              >
                <div className="md-filter-inner">
                  <Search size={14} strokeWidth={1.75} className="md-filter-icon" aria-hidden="true" />
                  <input
                    ref={filterInputRef}
                    type="search"
                    value={filterText}
                    onChange={(e) => setFilterText(e.target.value)}
                    placeholder="筛选当前目录"
                    aria-label="筛选当前目录"
                    autoComplete="off"
                    spellCheck={false}
                  />
                  {filterText && (
                    <button
                      type="button"
                      className="md-icon-btn md-filter-clear"
                      aria-label="清空筛选"
                      onClick={() => setFilterText("")}
                    >
                      <X size={12} strokeWidth={1.75} aria-hidden="true" />
                    </button>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* 2. 主工作区：宽屏下可展开 Thunar Places 边栏；停靠与窄屏下 100% 留给列表或阅读 */}
          <div className="md-body">
            {isWide && sidebarOpen && (
              <aside className="md-nav" aria-label="位置与收藏">
                <div className="md-nav-inner">
                    <div className="md-sidebar-group">
                      <div className="md-sidebar-heading">位置</div>
                      {roots.map((preset) => (
                        <button
                          key={preset.id}
                          type="button"
                          className="md-sidebar-row"
                          aria-current={directory === preset.path ? "true" : undefined}
                          title={preset.path}
                          onClick={() => void loadFiles(preset.path)}
                        >
                          <RootIcon id={preset.id} size={14} />
                          <span>{preset.label}</span>
                        </button>
                      ))}
                    </div>

                    <div className="md-sidebar-rule" />

                    <div className="md-sidebar-group">
                      <div className="md-sidebar-heading-row">
                        <span className="md-sidebar-heading">收藏</span>
                        {directory && (
                          <button
                            type="button"
                            className="md-sidebar-pin-btn"
                            aria-pressed={isPinned}
                            aria-label={isPinned ? "取消收藏当前目录" : "收藏当前目录"}
                            title={isPinned ? "取消收藏当前目录" : "收藏当前目录"}
                            onClick={togglePin}
                          >
                            <Star
                              size={12}
                              strokeWidth={1.75}
                              className={isPinned ? "md-icon-star-filled" : "md-icon-star"}
                              aria-hidden="true"
                            />
                          </button>
                        )}
                      </div>
                      {pins.length > 0 ? (
                        pins.map((pin) => (
                          <button
                            key={pin}
                            type="button"
                            className="md-sidebar-row"
                            aria-current={directory === pin ? "true" : undefined}
                            title={pin}
                            onClick={() => void loadFiles(pin)}
                          >
                            <Star size={14} strokeWidth={1.75} className="md-icon-star" aria-hidden="true" />
                            <span>{basename(pin)}</span>
                          </button>
                        ))
                      ) : (
                        <p className="md-sidebar-empty">暂无收藏目录</p>
                      )}
                    </div>

                    <div className="md-sidebar-footer">
                      <button
                        type="button"
                        className="md-sidebar-row"
                        onClick={() => {
                          setPathInput(activeFullPath);
                          setPathEditing(true);
                        }}
                      >
                        <Terminal size={14} strokeWidth={1.75} aria-hidden="true" />
                        <span>前往路径…</span>
                      </button>
                    </div>
                  </div>
                </aside>
              )}

            {/* 中央舞台 */}
            <div className="md-stage">
              <AnimatePresence mode="popLayout" initial={false}>
                <motion.div
                  key={viewKey}
                  initial={{
                    opacity: 0,
                    scale: direction > 0 ? 1.018 : 0.982,
                  }}
                  animate={{
                    opacity: 1,
                    scale: 1,
                  }}
                  exit={{
                    opacity: 0,
                    scale: direction > 0 ? 0.982 : 1.018,
                  }}
                  transition={{
                    opacity: { duration: 0.18, ease: [0.22, 1, 0.36, 1] },
                    scale: { duration: 0.24, ease: [0.22, 1, 0.36, 1] },
                  }}
                  className="md-stage-panel"
                >
                  {mode === "browse" ? (
                    <div className="md-browse-view">
                      {navNote.text && (
                        <div
                          className="md-note-row"
                          data-tone={navNote.isError ? "error" : undefined}
                          role="status"
                          aria-live="polite"
                        >
                          <p className="md-note">{navNote.text}</p>
                          {navNote.retry && (
                            <button type="button" className="md-inline-btn" onClick={navNote.retry}>
                              重试
                            </button>
                          )}
                        </div>
                      )}

                      {isWide && viewMode === "list" && (
                        <div className="md-table-header">
                          <span>名称</span>
                          <span className="md-table-col-type">类型</span>
                        </div>
                      )}

                      <div className={viewMode === "grid" ? "md-list md-grid" : "md-list"}>
                        {!needle && parentDir && parentDir !== directory && viewMode === "list" && (
                          <button
                            type="button"
                            className="md-entry"
                            data-kind="parent"
                            title={parentDir}
                            onClick={() => void loadFiles(parentDir, "", { dir: -1, morphOutPath: directory })}
                          >
                            <span className="md-entry-left">
                              <ArrowUp size={16} strokeWidth={1.75} className="md-icon-plain" aria-hidden="true" />
                              <span className="md-entry-label">上一级</span>
                            </span>
                            <span className="md-entry-right">
                              {isWide && <span className="md-entry-type">{basename(parentDir)}</span>}
                              <span className="md-entry-chevron-spacer" aria-hidden="true" />
                            </span>
                          </button>
                        )}

                        {filteredItems.map((item) => (
                          <EntryRow
                            key={item.path}
                            item={item}
                            viewMode={viewMode}
                            wide={isWide}
                            activePath={path}
                            leavingTargetPath={leavingTargetPath}
                            onOpen={handleOpenEntry}
                          />
                        ))}

                        {filteredItems.length === 0 && !navNote.text && (
                          <p className="md-empty-list">
                            {needle ? "没有匹配的条目。" : "此目录为空。"}
                          </p>
                        )}

                        {!needle && afterCursor && (
                          <button
                            type="button"
                            className="md-entry md-load-more"
                            data-more="true"
                            onClick={() => void loadFiles(directory, afterCursor)}
                          >
                            <span className="md-entry-label">加载更多</span>
                          </button>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="md-read-view">
                      {/* 紧凑阅读工具条（39px 含 1px 底边 = 38px 偶数内高），停靠宽度下也不挤 */}
                      <div className="md-reader-bar">
                        <div className="md-reader-meta">
                          <span className="md-meta">
                            {formatBytes(docResult?.bytes) || fileTypeLabel(basename(path), "file")}
                          </span>
                          {docResult?.tail && <span className="md-meta-tag">末尾片段</span>}
                        </div>

                        <div className="md-tools">
                          {isMd && (
                            <>
                              <button
                                type="button"
                                className="md-tool-btn"
                                aria-pressed={source}
                                onClick={() => setSource((v) => !v)}
                              >
                                <FileCode size={14} strokeWidth={1.75} aria-hidden="true" />
                                <span>源码</span>
                              </button>
                              <button
                                type="button"
                                className="md-tool-btn"
                                aria-pressed={outlineOpen}
                                onClick={() => setOutlineOpen((v) => !v)}
                              >
                                <ListTree size={14} strokeWidth={1.75} aria-hidden="true" />
                                <span>大纲</span>
                              </button>
                            </>
                          )}
                          {hasText && session && (
                            <button
                              type="button"
                              className="md-tool-btn"
                              onClick={handleQuote}
                            >
                              <Quote size={14} strokeWidth={1.75} aria-hidden="true" />
                              <span>{quoteLabel}</span>
                            </button>
                          )}
                          {canDownload && (
                            <button
                              type="button"
                              className="md-tool-btn md-download"
                              disabled={downloading}
                              onClick={() => void handleDownload()}
                            >
                              <Download size={14} strokeWidth={1.75} aria-hidden="true" />
                              <span>{downloading ? "下载中…" : "下载"}</span>
                            </button>
                          )}
                        </div>
                      </div>

                      {/* 可折叠大纲面板 */}
                      <AnimatePresence initial={false}>
                        {outlineOpen && isMd && (
                          <motion.nav
                            // motion-exempt: 阅读器筛选栏、大纲面板展开时由 motion 库逐帧驱动高度，低频，ADR 0166 已知限制
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: "auto", opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            transition={FAST_SPRING}
                            className="md-outline"
                            aria-label="文档大纲"
                          >
                            {source ? (
                              <p>切回阅读模式查看文档大纲。</p>
                            ) : outlineItems.length > 0 ? (
                              outlineItems.map((h) => (
                                <button
                                  key={h.id}
                                  type="button"
                                  style={{ paddingInlineStart: `${8 + (h.level - 1) * 12}px` }}
                                  onClick={() => {
                                    h.el.scrollIntoView({ block: "start" });
                                    setOutlineOpen(false);
                                    h.el.tabIndex = -1;
                                    h.el.focus({ preventScroll: true });
                                  }}
                                >
                                  {h.text}
                                </button>
                              ))
                            ) : (
                              <p>文档没有标题。</p>
                            )}
                          </motion.nav>
                        )}
                      </AnimatePresence>

                      {stageNote.text && (
                        <div
                          className="md-note-row"
                          data-tone={stageNote.isError ? "error" : undefined}
                          role="status"
                          aria-live="polite"
                        >
                          <p className="md-note">{stageNote.text}</p>
                          {stageNote.retry && (
                            <button type="button" className="md-inline-btn" onClick={stageNote.retry}>
                              重试
                            </button>
                          )}
                        </div>
                      )}

                      {docResult?.truncated && path && (
                        <div className="md-banner">
                          <span>
                            {docResult.tail
                              ? `文件较大${docResult.bytes ? `（共 ${formatBytes(docResult.bytes)}）` : ""}，当前显示末尾片段`
                              : `文件较大${docResult.bytes ? `（共 ${formatBytes(docResult.bytes)}）` : ""}，当前仅显示前 256 KB`}
                          </span>
                          <button
                            type="button"
                            className="md-inline-btn"
                            onClick={() => void openFile(path, { tail: !docResult.tail })}
                          >
                            {docResult.tail ? "查看开头" : "查看末尾"}
                          </button>
                        </div>
                      )}

                      <article
                        ref={articleRef}
                        className="md-article"
                        tabIndex={0}
                        aria-label="文件内容预览"
                        onClick={handleArticleClick}
                      >
                        {docResult?.mime_type && (
                          <div className="md-media-preview">
                            {imageBlobUrl && <img src={imageBlobUrl} alt={basename(path)} />}
                            {imageError && <span className="md-image-note">{imageError}</span>}
                            <p className="md-media-caption">
                              {path}
                              {docResult.bytes !== undefined ? ` · ${formatBytes(docResult.bytes)}` : ""}
                            </p>
                          </div>
                        )}

                        {docResult?.status === "binary" && (
                          <div className="md-binary-card">
                            <strong>{basename(path)} 是二进制文件</strong>
                            <p>路径：{path}</p>
                            <p>
                              大小：{formatBytes(docResult.bytes) || "未知"} · 为避免阻塞 IO，不直接加载二进制正文。
                            </p>
                            <div className="md-binary-actions">
                              <button
                                type="button"
                                className="md-inline-btn"
                                onClick={() => void handleCopyPath()}
                              >
                                复制服务端路径
                              </button>
                              {canDownload && (
                                <button
                                  type="button"
                                  className="md-inline-btn"
                                  onClick={() => void handleDownload()}
                                >
                                  下载到本地
                                </button>
                              )}
                            </div>
                          </div>
                        )}

                        {docResult?.status === "available" && !docResult.mime_type && !docResult.text && (
                          <p className="md-empty-file">这是一个空文件。</p>
                        )}
                        <div ref={proseHostRef} className="md-prose-host" />
                      </article>
                    </div>
                  )}
                </motion.div>
              </AnimatePresence>
            </div>
          </div>
        </LayoutGroup>

        {/* 3. 底部状态栏（29px 含 1px 顶边 = 28px 偶数内高） */}
        <div className="md-status-bar">
          <span className="md-status-left">
            {mode === "browse"
              ? needle
                ? `筛选 ${filteredItems.length} / ${items.length} 项`
                : `${items.length} 项${afterCursor ? "+" : ""}`
              : `${fileTypeLabel(basename(path), "file")}${docResult?.bytes !== undefined ? ` · ${formatBytes(docResult.bytes)}` : ""}`}
          </span>

          <div className="md-status-right">
            {mode === "browse" ? (
              <button
                type="button"
                className="md-status-btn"
                aria-pressed={showHidden}
                onClick={toggleHidden}
              >
                {showHidden ? <Eye size={12} strokeWidth={1.75} aria-hidden="true" /> : <EyeOff size={12} strokeWidth={1.75} aria-hidden="true" />}
                <span>隐藏项：{showHidden ? "开" : "关"}</span>
              </button>
            ) : (
              <button
                type="button"
                className="md-status-btn"
                onClick={handleReturnToBrowse}
              >
                <ArrowLeft size={12} strokeWidth={1.75} aria-hidden="true" />
                <span>返回目录</span>
              </button>
            )}
          </div>
        </div>
      </section>
    </MotionConfig>
  );
}

export function activate(ctx: WebHostContextV1): WebUiDisposer {
  return ctx.ui.inject("conversation.tools.v2", (mount) =>
    mount.register({
      id: "markdown-reader",
      accepts: ["workspace.document.open.v1", "workspace.file.open.v1"],
      label: "文件",
      order: 25,
      icon: FOLDER_ICON,
      shortcut: "Mod+Shift+E",
      hint: "浏览服务端工作区、会话目录或指定路径文件",
      render(host, _entry, raw) {
        const root = createRoot(host);
        root.render(<FileManagerApp ctx={ctx} view={raw as ConversationTabView} />);
        return () => root.unmount();
      },
    })
  );
}

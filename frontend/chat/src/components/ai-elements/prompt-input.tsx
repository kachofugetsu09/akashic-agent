"use client";

import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupTextarea,
} from "@/components/ui/input-group";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import type { FileUIPart } from "ai";
import { nanoid } from "nanoid";
import type {
  ChangeEventHandler,
  ClipboardEventHandler,
  ComponentProps,
  FormEvent,
  FormEventHandler,
  HTMLAttributes,
  KeyboardEventHandler,
  ReactNode,
} from "react";
import {
  Children,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

/** 输入卡里的附件：本地 Blob URL 加选择时就知道的字节数（卡片上显示大小）。 */
export type PromptInputFile = FileUIPart & { id: string; sizeBytes?: number };

/** 拒收原因：单个超过大小上限，或这条消息的附件数已满。 */
export type RejectReason = "size" | "count";

export interface AttachmentsContext {
  files: PromptInputFile[];
  /** 返回实际入列的个数：满额或超限时可能为 0。 */
  add: (files: File[] | FileList) => number;
  remove: (id: string) => void;
  openFileDialog: () => void;
}

const LocalAttachmentsContext = createContext<AttachmentsContext | null>(null);

export const usePromptInputAttachments = () => {
  const context = useContext(LocalAttachmentsContext);
  if (!context) {
    throw new Error(
      "usePromptInputAttachments must be used within a PromptInput"
    );
  }
  return context;
};

export interface PromptInputMessage {
  text: string;
  files: Omit<PromptInputFile, "id">[];
}

export type PromptInputProps = Omit<
  HTMLAttributes<HTMLFormElement>,
  "onSubmit" | "onError"
> & {
  multiple?: boolean;
  /** 附件归属（会话草稿键）；变化时当前附件存到旧键名下，换出新键的那一份。 */
  attachmentKey?: string;
  /** 单个附件上限（字节）；超出的文件不入列，交给 onRejectFiles 提示。 */
  maxFileSize?: number;
  /** 一条消息最多带几个附件；多出的不入列。 */
  maxFiles?: number;
  onRejectFiles?: (files: File[], reason: RejectReason) => void;
  /** 可返回剩余草稿此后归属的键（新会话首次被接纳时拿到的会话 ID），未发出的附件随之迁过去。 */
  onSubmit: (
    message: PromptInputMessage,
    event: FormEvent<HTMLFormElement>
  ) => void | Promise<void | string | undefined>;
};

/** 管理当前表单的附件，发送成功后才释放对应的 Blob URL。 */
export const PromptInput = ({
  className,
  multiple,
  attachmentKey = "",
  maxFileSize,
  maxFiles = Number.POSITIVE_INFINITY,
  onRejectFiles,
  onSubmit,
  children,
  ...props
}: PromptInputProps) => {
  // 1. 当前表单拥有附件；提交中只隐藏本次发送持有的文件。
  const inputRef = useRef<HTMLInputElement | null>(null);
  const formRef = useRef<HTMLFormElement | null>(null);
  const submittingRef = useRef(false);
  const [submittedFileIds, setSubmittedFileIds] = useState<string[]>([]);
  // 提交中的附件属于上一条消息，不占下一份草稿的名额。
  const submittedRef = useRef<string[]>([]);

  const [files, setFiles] = useState<PromptInputFile[]>([]);
  // 附件与草稿一样按会话保留：切走时存到旧会话名下，切回时取出，不跟到别的会话里被发出去。
  // carry：new: 草稿首次被接纳后，它的附件归属换成真实会话，键随后变过去时不算切会话。
  const [parked, setParked] = useState<{ key: string; others: ReadonlyMap<string, PromptInputFile[]>; carry: { from: string; to: string } | null }>(
    () => ({ key: attachmentKey, others: new Map(), carry: null }));
  if (parked.key !== attachmentKey) {
    const owner = parked.carry?.from === parked.key ? parked.carry.to : parked.key;
    const others = new Map(parked.others);
    if (files.length) others.set(owner, [...(others.get(owner) ?? []), ...files]);
    const restored = others.get(attachmentKey) ?? [];
    others.delete(attachmentKey);
    setParked({ key: attachmentKey, others, carry: null });
    setFiles(restored);
  } else if (parked.others.has(attachmentKey)) {
    // 接纳晚于切换时，迁到当前键名下的附件直接接回输入卡。
    const others = new Map(parked.others);
    const waiting = others.get(attachmentKey) ?? [];
    others.delete(attachmentKey);
    setParked({ ...parked, others });
    setFiles([...waiting, ...files]);
  }
  // 卸载时读取最新附件（含存起来的），避免闭包持有旧列表而漏掉 Blob URL。
  const filesRef = useRef(files);
  const parkedRef = useRef(parked.others);

  useEffect(() => {
    filesRef.current = files;
    parkedRef.current = parked.others;
  }, [files, parked]);

  const openFileDialog = useCallback(() => {
    inputRef.current?.click();
  }, []);

  // 2. 选择、粘贴和拖拽共用同一条附件入列路径。
  const add = useCallback(
    (fileList: File[] | FileList) => {
      // 超限文件在选择时就拒掉：等到发送才被服务端 413，用户已经写完话、还会被引导去"重试"一个必然失败的上传。
      // 只剔除超限的那几个，其余照常入列（Desktop 同样部分接收）。
      const all = Array.from(fileList);
      const sized = maxFileSize === undefined ? all : all.filter((file) => file.size <= maxFileSize);
      if (sized.length < all.length) onRejectFiles?.(all.filter((file) => !sized.includes(file)), "size");
      const room = Math.max(0, maxFiles - filesRef.current.filter((file) => !submittedRef.current.includes(file.id)).length);
      const incoming = sized.slice(0, room);
      if (incoming.length < sized.length) onRejectFiles?.(sized.slice(room), "count");
      if (!incoming.length) return 0;
      // Blob URL 在 updater 外创建：updater 可能被 React 重复调用，里面建的 URL 会有一份无人释放。
      const next: PromptInputFile[] = incoming.map((file) => ({
        filename: file.name,
        id: nanoid(),
        mediaType: file.type,
        sizeBytes: file.size,
        type: "file",
        url: URL.createObjectURL(file),
      }));
      setFiles((prev) => [...prev, ...next]);
      return next.length;
    },
    [maxFileSize, maxFiles, onRejectFiles]
  );

  // 发送成功时本次附件可能已随切换会话存到别处，两边一起移除。
  const remove = useCallback(
    (id: string) => {
      setFiles((prev) => {
        const found = prev.find((file) => file.id === id);
        if (found?.url) {
          URL.revokeObjectURL(found.url);
        }
        return prev.filter((file) => file.id !== id);
      });
      setParked((prev) => {
        if (![...prev.others.values()].some((list) => list.some((file) => file.id === id))) return prev;
        const others = new Map<string, PromptInputFile[]>();
        for (const [key, list] of prev.others) {
          const found = list.find((file) => file.id === id);
          if (found?.url) URL.revokeObjectURL(found.url);
          const rest = list.filter((file) => file.id !== id);
          if (rest.length) others.set(key, rest);
        }
        return { ...prev, others };
      });
    },
    []
  );

  // 把 from 名下尚未发出的附件交给 to：键还没变就记下 carry，已经变了就直接迁移停放的那份。
  const adopt = useCallback((from: string, to: string) => {
    setParked((prev) => {
      const others = new Map(prev.others);
      const moving = others.get(from);
      if (moving) {
        others.delete(from);
        others.set(to, [...(others.get(to) ?? []), ...moving]);
      }
      return { key: prev.key, others, carry: prev.key === from ? { from, to } : prev.carry };
    });
  }, []);

  useEffect(() => {
    const target = formRef.current;
    if (!target) return;

    const onDragOver = (event: DragEvent) => {
      if (event.dataTransfer?.types?.includes("Files")) event.preventDefault();
    };
    const onDrop = (event: DragEvent) => {
      if (event.dataTransfer?.types?.includes("Files")) event.preventDefault();
      if (event.dataTransfer?.files && event.dataTransfer.files.length > 0) {
        add(event.dataTransfer.files);
      }
    };
    target.addEventListener("dragover", onDragOver);
    target.addEventListener("drop", onDrop);
    return () => {
      target.removeEventListener("dragover", onDragOver);
      target.removeEventListener("drop", onDrop);
    };
  }, [add]);

  useEffect(
    () => () => {
      for (const file of [...filesRef.current, ...[...parkedRef.current.values()].flat()]) {
        if (file.url) URL.revokeObjectURL(file.url);
      }
    },
    []
  );

  const handleChange: ChangeEventHandler<HTMLInputElement> = useCallback(
    (event) => {
      if (event.currentTarget.files) {
        add(event.currentTarget.files);
      }
      // 清空选择器，允许再次选择刚移除的同名文件。
      event.currentTarget.value = "";
    },
    [add]
  );

  const attachmentsCtx = useMemo<AttachmentsContext>(
    () => ({
      add,
      files: files.filter((item) => !submittedFileIds.includes(item.id)),
      openFileDialog,
      remove,
    }),
    [files, submittedFileIds, add, remove, openFileDialog]
  );

  // 3. 本次发送与下一份草稿分开；失败保留附件，由调用者显示原因。
  const handleSubmit: FormEventHandler<HTMLFormElement> = useCallback(
    async (event) => {
      event.preventDefault();
      if (submittingRef.current) return;
      submittingRef.current = true;

      const form = event.currentTarget;
      const formData = new FormData(form);
      const text = (formData.get("message") as string) || "";

      // 提交中的附件由本次发送持有；编辑器仍可接收下一份草稿。
      const submitted = files.map(({ id, ...file }) => ({ id, file }));
      const submittedKey = attachmentKey;
      submittedRef.current = submitted.map((item) => item.id);
      setSubmittedFileIds(submittedRef.current);
      form.reset();

      try {
        // 原 Blob URL 直接交给上传 owner，成功后才释放，不做 Base64 往返转换。
        const owner = await onSubmit({ files: submitted.map((item) => item.file), text }, event);
        for (const item of submitted) remove(item.id);
        if (typeof owner === "string" && owner !== submittedKey) adopt(submittedKey, owner);
      } catch {
        // 发送者显示失败原因；原附件保留供重试，不清除后来输入的内容。
      } finally {
        submittingRef.current = false;
        submittedRef.current = [];
        setSubmittedFileIds([]);
      }
    },
    [files, onSubmit, remove, adopt, attachmentKey]
  );

  return (
    <LocalAttachmentsContext.Provider value={attachmentsCtx}>
      <input
        aria-label="选择要上传的文件"
        className="hidden"
        multiple={multiple}
        onChange={handleChange}
        ref={inputRef}
        title="选择要上传的文件"
        type="file"
      />
      <form
        className={cn("w-full", className)}
        onSubmit={handleSubmit}
        ref={formRef}
        {...props}
      >
        <InputGroup className="overflow-hidden">{children}</InputGroup>
      </form>
    </LocalAttachmentsContext.Provider>
  );
};

export type PromptInputBodyProps = HTMLAttributes<HTMLDivElement>;

export const PromptInputBody = ({
  className,
  ...props
}: PromptInputBodyProps) => (
  <div className={cn("contents", className)} {...props} />
);

export type PromptInputTextareaProps = ComponentProps<
  typeof InputGroupTextarea
> & {
  /** 粘贴的纯文本达到这个 UTF-8 字节数就转成附件（按住 Shift 粘贴照常插入）。 */
  pasteAsFile?: { bytes: number; filename: string; onConverted?: (lines: number) => void };
};

// 外接键盘的平板会报告精细指针或悬停能力，此时仍按回车发送。
export const touchOnly = () => window.matchMedia("(hover: none) and (pointer: coarse)").matches;

export const PromptInputTextarea = ({
  onKeyDown,
  onKeyUp,
  onBlur,
  className,
  placeholder = "What would you like to know?",
  pasteAsFile,
  ...props
}: PromptInputTextareaProps) => {
  // 粘贴事件不带修饰键：只在按下 Shift+V 时记下，紧随其后的粘贴照常插入；松键、失焦即清掉，菜单粘贴不受影响。
  const shiftHeld = useRef(false);
  const attachments = usePromptInputAttachments();
  const [isComposing, setIsComposing] = useState(false);

  const handleKeyDown: KeyboardEventHandler<HTMLTextAreaElement> = useCallback(
    (e) => {
      // Call the external onKeyDown handler first
      shiftHeld.current = e.shiftKey && e.code === "KeyV";
      onKeyDown?.(e);

      // If the external handler prevented default, don't run internal logic
      if (e.defaultPrevented) {
        return;
      }

      if (e.key === "Enter") {
        // Safari 在 compositionend 之后才派发确认候选词的回车，isComposing 已为 false，只剩 keyCode 229 可辨认。
        if (isComposing || e.nativeEvent.isComposing || e.nativeEvent.keyCode === 229) {
          return;
        }
        // 纯触屏（无悬停、粗指针）上回车换行，发送交给按钮；与移动端聊天应用的手感一致。
        if (e.shiftKey || touchOnly()) {
          return;
        }
        e.preventDefault();

        // Check if the submit button is disabled before submitting
        const { form } = e.currentTarget;
        const submitButton = form?.querySelector(
          'button[type="submit"]'
        ) as HTMLButtonElement | null;
        if (!submitButton || submitButton.disabled) {
          return;
        }

        form?.requestSubmit();
      }

      // Remove last attachment when Backspace is pressed and textarea is empty
      if (
        e.key === "Backspace" &&
        e.currentTarget.value === "" &&
        attachments.files.length > 0
      ) {
        e.preventDefault();
        const lastAttachment = attachments.files[attachments.files.length - 1];
        if (lastAttachment) {
          attachments.remove(lastAttachment.id);
        }
      }
    },
    [onKeyDown, isComposing, attachments]
  );

  const handlePaste: ClipboardEventHandler<HTMLTextAreaElement> = useCallback(
    (event) => {
      const items = event.clipboardData?.items;

      if (!items) {
        return;
      }
      // 1. 先分出真正的文件：Office 类来源同时带文本、HTML 与一张渲染图，那张图只是文字的替代渲染。
      const text = event.clipboardData.getData("text/plain");
      const pasted = Array.from(items, (item) => item.kind === "file" ? item.getAsFile() : null).filter((file): file is File => file !== null);
      const rendering = Boolean(text) && event.clipboardData.types.includes("text/html")
        && pasted.length === 1 && pasted[0].type.startsWith("image/");
      const files = rendering ? [] : pasted;
      // 2. 长文本转附件（Desktop 同样在 4KB 处转换），与同时粘贴的文件一起入列；一个都没接纳就照常插入文字。
      if (pasteAsFile && text && !shiftHeld.current && new Blob([text]).size >= pasteAsFile.bytes) {
        const accepted = attachments.add([new File([text], pasteAsFile.filename, { type: "text/plain" }), ...files]);
        if (!accepted) return;
        event.preventDefault();
        pasteAsFile.onConverted?.(text.split("\n").length);
        return;
      }
      // 3. 普通粘贴里的文件入列；没有接纳时不拦文字。
      if (files.length > 0 && attachments.add(files) > 0) {
        event.preventDefault();
      }
    },
    [attachments, pasteAsFile]
  );

  const handleCompositionEnd = useCallback(() => setIsComposing(false), []);
  const handleCompositionStart = useCallback(() => setIsComposing(true), []);

  return (
    <InputGroupTextarea
      className={cn("field-sizing-content max-h-48 min-h-16", className)}
      name="message"
      onCompositionEnd={handleCompositionEnd}
      onCompositionStart={handleCompositionStart}
      onKeyDown={handleKeyDown}
      onKeyUp={(event) => {
        shiftHeld.current = false;
        onKeyUp?.(event);
      }}
      onBlur={(event) => {
        shiftHeld.current = false;
        onBlur?.(event);
      }}
      onPaste={handlePaste}
      placeholder={placeholder}
      enterKeyHint={touchOnly() ? "enter" : "send"}
      {...props}
    />
  );
};

export type PromptInputFooterProps = Omit<
  ComponentProps<typeof InputGroupAddon>,
  "align"
>;

export const PromptInputFooter = ({
  className,
  ...props
}: PromptInputFooterProps) => (
  <InputGroupAddon
    align="block-end"
    className={cn("justify-between gap-1", className)}
    {...props}
  />
);

export type PromptInputToolsProps = HTMLAttributes<HTMLDivElement>;

export const PromptInputTools = ({
  className,
  ...props
}: PromptInputToolsProps) => (
  <div
    className={cn("flex min-w-0 items-center gap-1", className)}
    {...props}
  />
);

export type PromptInputButtonTooltip =
  | string
  | {
      content: ReactNode;
      shortcut?: string;
      side?: ComponentProps<typeof TooltipContent>["side"];
    };

export type PromptInputButtonProps = ComponentProps<typeof InputGroupButton> & {
  tooltip?: PromptInputButtonTooltip;
};

export const PromptInputButton = ({
  variant = "ghost",
  className,
  size,
  tooltip,
  ...props
}: PromptInputButtonProps) => {
  const [tooltipOpen, setTooltipOpen] = useState(false);
  const newSize =
    size ?? (Children.count(props.children) > 1 ? "sm" : "icon-sm");

  const button = (
    <InputGroupButton
      className={cn(className)}
      size={newSize}
      type="button"
      variant={variant}
      {...props}
    />
  );

  if (!tooltip) {
    return button;
  }

  const tooltipContent =
    typeof tooltip === "string" ? tooltip : tooltip.content;
  const shortcut = typeof tooltip === "string" ? undefined : tooltip.shortcut;
  const side = typeof tooltip === "string" ? "top" : (tooltip.side ?? "top");

  return (
    <Tooltip
      open={tooltipOpen}
      onOpenChange={(next) => {
        // Hover-only: ignore focus-driven opens (e.g. after file dialog returns focus).
        if (!next) setTooltipOpen(false);
      }}
    >
      <TooltipTrigger
        asChild
        onPointerEnter={() => setTooltipOpen(true)}
        onPointerLeave={() => setTooltipOpen(false)}
        onPointerDown={() => setTooltipOpen(false)}
      >
        {button}
      </TooltipTrigger>
      <TooltipContent side={side}>
        {tooltipContent}
        {shortcut && (
          <span className="ml-2 text-muted-foreground">{shortcut}</span>
        )}
      </TooltipContent>
    </Tooltip>
  );
};

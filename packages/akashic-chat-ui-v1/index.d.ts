import type { ComponentType, ReactNode } from "react";

/** 聊天宿主提供的插槽；插件只在目录声明过的插槽里挂载。 */
export type PluginUiSlotName =
  | "turn.before_reasoning"
  | "turn.after_answer"
  | "drawer.panel"
  | "tool.result"
  | "message.attachment"
  | "chat.notice"
  | "header.end"
  | "composer.toolbar"
  | "composer.status"
  | "composer.suggest"
  | "scope.fields"
  | "scope.badge"
  | "navigation.groups";

/** 宿主挂载插件元素的插槽；composer.suggest 与 navigation.groups 是数据源，不挂载。 */
export type PluginUiMountSlotName = Exclude<PluginUiSlotName, "composer.suggest" | "navigation.groups">;

export type WorkbenchIntentStatus = "handled" | "unhandled" | "invalid" | "failed";

export interface PluginUiQueryOptions {
  cache?: "none" | "memory" | "immutable";
}

/** 每次挂载都有的宿主能力，与插槽无关。 */
export interface PluginUiHostContext {
  /** 挂载内的事实失效提示；不改变挂载身份。 */
  onInvalidate(callback: () => void): () => void;
  /** 异步展示失败时交还宿主降级内容；不改变消息。 */
  reportError(message: string): void;
  /**
   * 页面重新可见时回调：切回标签页、窗口重新获得焦点、外层页面重新显示聊天。
   * 插件据此重读可能在别处改过的数据，不必自己监听窗口事件。返回取消订阅。
   */
  onShown(callback: () => void): () => void;
  capabilities: {
    queryCacheModes?: readonly ("none" | "memory" | "immutable")[];
  };
  /**
   * 本插件后端 follow 推送的当前会话状态：先回放最近一帧，之后每帧回调；
   * null 表示旧状态已失效（断线、插件换代），插件应丢弃它。返回取消订阅。
   */
  subscribe(callback: (value: Record<string, unknown> | null) => void): () => void;
  /** 只查询本插件注册的方法；宿主不替插件选择其他插件。 */
  query(method: string, payload?: Record<string, unknown>, options?: PluginUiQueryOptions): Promise<Record<string, unknown>>;
  /**
   * 在右侧工具区打开：只在点击或键盘激活的处理函数里调用；没有接受者时得到 unhandled，插件回到自己的原有行为。
   * accepts 的结果会随宿主广播变化，变化时经 onInvalidate 通知，插件在回调里重新判断是否显示入口。
   */
  workbench: {
    accepts(name: string): boolean;
    open(name: string, payload: unknown): Promise<WorkbenchIntentStatus>;
  };
}

/**
 * 每个挂载插槽里宿主保证提供的事实：字段在这里写成必需的，插件就不必再自己检查。
 * 新增插槽时在这里登记它的事实，宿主的挂载点按同一份类型传值。
 */
export interface PluginUiSlotFacts {
  /** 一轮回答开始前；source 是这轮输入所在的会话来源。 */
  "turn.before_reasoning": { sessionId: string; messageId: string; block: { source: string } };
  /** 完整回答之后。 */
  "turn.after_answer": { sessionId: string; messageId: string };
  /** 本插件注册的工具的一次结果；只交给工具 owner。 */
  "tool.result": { block: ToolResultFacts };
  /** 消息里的一个附件；插件按文件名、媒体类型在 accepts 里认领。 */
  "message.attachment": { sessionId: string; messageId: string; block: MessageAttachmentFacts };
  /** 输入框上方的通知；还没有会话时 sessionId 为空。 */
  "chat.notice": { sessionId?: string };
  "header.end": { sessionId: string };
  /** 输入框工具位；新会话还没有 sessionId。 */
  "composer.toolbar": { sessionId?: string; composer: ComposerControls };
  "composer.status": { sessionId: string };
  /** 会话导航下的插件面板；没有会话时不挂载。 */
  "drawer.panel": { sessionId: string };
  /** 新建某个维度取值的表单里的附加设置；取值尚未产生。 */
  "scope.fields": { scope: { dimension: string }; fields: ScopeFieldControls };
  /** 跟在某个取值名字后面，位于可点击的行里，只放不可交互的短文字。 */
  "scope.badge": { scope: { dimension: string; value: string } };
}

/** 某个挂载插槽的完整上下文；不带参数时是所有插槽上下文的联合，按 slot 收窄。 */
export type PluginUiContext<S extends PluginUiMountSlotName = PluginUiMountSlotName> = {
  [K in S]: PluginUiHostContext & { slot: K } & PluginUiSlotFacts[K];
}[S];

/** 插槽交给 accepts 判断的数据；没有 block 的插槽为 undefined。 */
export type PluginUiSlotBlock<S extends PluginUiMountSlotName> = PluginUiSlotFacts[S] extends { block: infer B } ? B : undefined;

/** 侧栏分组区里的挂载（分区提示）：所属维度与宿主导航能力。 */
export interface PluginUiNavigationContext extends PluginUiHostContext {
  slot: "navigation.groups";
  scope: { dimension: string };
  navigation: NavigationControls;
}

/** 分组对话框表单里的挂载；从某个分组的菜单打开时 scope 带上该分组的取值。 */
export interface PluginUiNavigationFormContext extends PluginUiHostContext {
  slot: "navigation.groups";
  scope: { dimension: string; value?: string };
  form: NavigationFormControls;
}

/** 侧栏分组的一项：会话维度的一个取值，例如一个项目。 */
export interface NavigationGroup {
  id: string;
  label: string;
  /** 悬停时的补充说明，例如固定目录。 */
  detail?: string;
}

/** 分组新建或分组动作的对话框：宿主给外框、标题与按钮，插件在表单区挂自己的内容。 */
export interface NavigationForm {
  title: string;
  /** 提交按钮文字；没有时对话框只读，只有关闭按钮。 */
  submitLabel?: string;
  mount(host: HTMLElement, context: PluginUiNavigationFormContext): void | (() => void);
}

/** 分组区顶部的提示。 */
export interface NavigationNotice {
  mount(host: HTMLElement, context: PluginUiNavigationContext): void | (() => void);
}

/** 对话框表单的宿主能力。 */
export interface NavigationFormControls {
  /**
   * 登记提交：宿主按钮调用，抛错时表单保持打开并显示原因。
   * 新建时 commitFields 让 scope.fields 插件为新取值保存设置，须在创建前调用；返回新分组 id，宿主随即在其中开始新对话。
   */
  onSubmit(submit: (commitFields: (value: string) => Promise<void>) => Promise<string | void>): () => void;
  /** 提交按钮是否可用，例如名称为空时不可用。 */
  setReady(ready: boolean): void;
  /** 正在提交；变化时经 onInvalidate 通知。 */
  busy(): boolean;
  close(): void;
}

/** 分组区的宿主导航能力。 */
export interface NavigationControls {
  /** 重读本插件的分组。 */
  reload(): void;
  /** 在这个分组里开始新对话。 */
  startChat(groupId: string): void;
}

/**
 * 侧栏分组的数据源：会话按 dimension 维度归类，分组 id 即维度取值。
 * 分组行、展开、置顶、搜索、键盘与“新建对话”都由宿主负责，插件只给数据、表单与提示。
 */
export interface NavigationGroupsProvider {
  dimension: string;
  /** 分区标题，例如“项目”。 */
  label: string;
  /** 目录变化、提交表单或插件请求时重读；只经 query 访问本插件。 */
  load(context: Pick<PluginUiHostContext, "query">): Promise<readonly NavigationGroup[]>;
  create?: NavigationForm;
  /** 分组行菜单里的额外动作，每项打开一个对话框。 */
  actions?(group: NavigationGroup): readonly { label: string; form: NavigationForm }[];
  /** 分区顶部的提示，例如未确认的新建请求；没有内容时插件不渲染任何东西。 */
  notice?: NavigationNotice;
}

/** 新建表单里的附加设置：表单确定了新取值、在它被创建之前，宿主依次调用各插件登记的提交。 */
export interface ScopeFieldControls {
  /** 登记本挂载的提交；拒绝时整次新建失败，表单保持打开。返回取消登记。 */
  onCommit(commit: (value: string) => Promise<void>): () => void;
  /** 表单正在提交；变化时经 onInvalidate 通知。 */
  busy(): boolean;
}

/** 插件随下一条消息附带的片段；后端由声明了 `input_part_key(kind)` 的 owner 校验，没有 owner 的 kind 会被拒收。 */
export interface ComposerInputPart {
  kind: string;
  value: unknown;
}

/** 输入框工具位的宿主能力：挂载卸载时宿主自动撤回片段与阻止原因。 */
export interface ComposerControls {
  /** 正在生成回复；变化时经 onInvalidate 通知。 */
  busy(): boolean;
  /** 替换本挂载要附带的片段；传 [] 撤回。 */
  contribute(parts: readonly ComposerInputPart[]): void;
  /** 阻止发送并说明原因（宿主显示在输入框下方并作为发送失败的提示）；null 解除。 */
  block(reason: string | null): void;
  /** 服务端确认收到一条消息后回调。 */
  onSent(callback: () => void): () => void;
}

/**
 * tool.result 插槽的 block：一次调用的名称、参数和已解析的结果值。
 * 宿主只把它交给注册该工具的插件；accepts 返回 false 时退回通用数据展示。
 */
export interface ToolResultFacts {
  name: string;
  arguments: unknown;
  value: unknown;
  error: boolean;
}

/** message.attachment 插槽的 block：附件的不可变描述，以及它所在消息的出处。 */
export interface MessageAttachmentFacts {
  artifact_id: string;
  kind: "file" | "image";
  filename: string | null;
  media_type: string | null;
  size_bytes: number;
  sha256: string;
  origin: {
    kind: "input" | "output" | "tool_result" | "control";
    author: string;
    /** 附件来自工具结果时的工具名。 */
    tool: string | null;
    /** 附件所在消息的时间（ISO 8601）。 */
    at: string;
  };
}

export interface PluginUiRenderer<S extends PluginUiMountSlotName = PluginUiMountSlotName> {
  accepts?(block: PluginUiSlotBlock<S>): boolean;
  prefetch?(context: PluginUiContext<S>): Promise<void>;
  mount(host: HTMLElement, context: PluginUiContext<S>): void | (() => void);
}

/** 输入框建议的一项：选中后写入 `${prefix}${token} `；发出的消息里同样写法显示为提及。 */
export interface ComposerSuggestion {
  token: string;
  label: string;
  description: string;
  /** 有值时照样列出但不能选，显示这个原因。 */
  unavailable?: string;
  /** 同名项并存时用来区分的来源。 */
  badge?: string;
}

/** 输入框建议的数据源：触发、菜单、键盘、IME、无障碍与气泡里的提及显示都由宿主负责。 */
export interface ComposerSuggestProvider {
  /** 菜单分组标题。 */
  label: string;
  /** 行首或空白后输入这些符号时打开菜单。 */
  sigils: readonly ("/" | "@")[];
  /** 写入草稿、识别提及用的前缀；与插件后端识别的写法一致。 */
  prefix: "/" | "@";
  /** 没有任何条目时的说明。 */
  empty: string;
  /** 每次打开菜单重新读取；只经 query 访问本插件。 */
  load(context: Pick<PluginUiHostContext, "query">): Promise<readonly ComposerSuggestion[]>;
}

/** 插件界面模块的默认导出。 */
export interface PluginUiDefinition {
  slots: { [S in PluginUiMountSlotName]?: PluginUiRenderer<S> } & {
    "composer.suggest"?: ComposerSuggestProvider;
    "navigation.groups"?: NavigationGroupsProvider;
  };
}

/** 在插槽元素里挂一个共享 React 的根；返回值可直接作为 mount() 的清理函数。 */
export function render(host: HTMLElement, node: ReactNode): () => void;
/** 输入框上方的一行状态提示。 */
export const Notice: ComponentType<{ children: ReactNode }>;
/**
 * 顶栏右侧的标记：图标加短名，alert 时变色；点开显示 children 面板，点外部或 Esc 收起。
 * 面板里的 code 显示为整块路径，带 data-tone="error" 的元素显示为错误色。
 */
export const HeaderMarker: ComponentType<{
  icon: ReactNode;
  label: string;
  alert?: boolean;
  ariaLabel: string;
  panelLabel: string;
  children: ReactNode;
}>;
/** 标记面板底部的一排文字按钮。 */
export const HeaderMarkerActions: ComponentType<{ children: ReactNode }>;
/** 单选菜单：触发按钮显示当前项，展开后每项带说明，可在底部附一句提示。 */
export const ChoiceMenu: ComponentType<{
  ariaLabel: string;
  value: string;
  choices: readonly { value: string; label: string; description?: string }[];
  note?: string;
  disabled?: boolean;
  onChange(value: string): void;
}>;
/** 过长时滚动显示的单行文字。 */
export const MarqueeText: ComponentType<{ children: ReactNode; still?: boolean }>;
/** 工具结果里的长文本：先给预览（tail 看末尾，否则看开头），可原地展开全文。 */
export const ToolText: ComponentType<{ text: string; tail: boolean; className: string }>;
/** 工具结果的通用数据展示：结构可展开，标量保留原文。 */
export const ToolData: ComponentType<{ value: unknown }>;
export function formatDuration(durationMs: number): string;
export function copyText(text: string): Promise<void>;

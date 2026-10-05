import type { WebEntry, WebMountView, WebUiDisposer } from "@akashic/web-ui-v1";

/**
 * Shell 底栏动作入口。label/iconSvg/order 是可序列化投影，页面可以把它们
 * 转发进自己的 iframe 渲染；onActivate 永远留在贡献者的宿主域，由页面在收到
 * 激活消息后回调，不跨 realm 传递函数。
 */
export interface ShellRailAction extends WebEntry {
  label: string;
  iconSvg: string;
  onActivate: () => void;
}

/** 页面转发给 iframe 的可序列化投影；函数与宿主对象不得出现在这里。 */
export interface ShellRailActionProjection {
  id: string;
  order?: number;
  label: string;
  iconSvg: string;
}

/**
 * 设置工作区顶层分节（shell.settings.v1），如初始配置、模型。
 * 分节可以在自己的 children 里声明子目录，组合层级由分节自己拥有，Shell 不解释。
 */
export interface ShellSettingsSection extends WebEntry {
  label: string;
  route: string;
  iconSvg: string;
}

/**
 * 普通插件配置分节（shell.settings-plugins.v1），在设置工作区“插件”区以 tab 呈现。
 * ≥2 个成员声明同一 family 时折叠为一个组合页；组合标签由成员自带的 familyLabel
 * 给出且同一 family 必须一致，Shell 只拥有折叠机制，不拥有任何 family 词汇。
 */
export interface ShellSettingsPlugin extends WebEntry {
  label: string;
  route: string;
  family?: string;
  familyLabel?: string;
}

/**
 * 按 route 渲染任一设置分节或插件配置分节的表单；route 未知时返回 null。
 * 初始配置等分节用它内嵌其他分节的表单，不需要知道条目属于哪个目录。
 */
export type RenderSettingsRoute = (route: string, host: HTMLElement, props?: unknown) => WebUiDisposer | null;

/**
 * Shell 渲染设置分节时传入的 props。pages 是该分节所属目录的视图，
 * 用于 style 绑定分节自己的 portal 容器；renderRoute 见上。
 */
export interface ShellSettingsRenderProps {
  pages: WebMountView;
  railActions: readonly ShellRailAction[];
  embedded?: boolean;
  renderRoute?: RenderSettingsRoute;
}

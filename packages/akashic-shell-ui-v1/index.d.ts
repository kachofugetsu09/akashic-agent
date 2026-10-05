import type { WebEntry } from "@akashic/web-ui-v1";

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

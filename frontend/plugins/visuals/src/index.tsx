import type { WebHostContextV1 } from "@akashic/web-ui-v1";
import { registerForm, Field } from "../../shared/src/configuration";

// 预检截图用的浏览器；留空即用 Playwright 自带的 Chromium。
export function activate(ctx: WebHostContextV1) {
  return registerForm(ctx, {id: "visuals", title: "可视化", description: "生成交互图表与示意图；发布前会在无界面浏览器里截图自检。",
    switchable: false,
    fields: ({values, change}) =>
      <Field label="预检浏览器路径" name="executable_path" value={values.executable_path} fallback="Playwright 自带 Chromium" change={change} hint="Chromium 或 Chrome 可执行文件的绝对路径；没有安装 Playwright 浏览器时填写" />,
  });
}

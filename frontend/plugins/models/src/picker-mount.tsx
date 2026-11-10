import { createRoot } from "react-dom/client";
import { ModelPickerBody, type ChatModelRuntime, type ModelPickerBodyProps } from "./capsule-picker";

export type { ChatModelRuntime };

// 设置页是原生 DOM：在浮层里挂载与聊天胶囊同一个选择器主体，返回卸载函数。
export function mountModelPicker(host: HTMLElement, props: ModelPickerBodyProps): () => void {
  const root = createRoot(host);
  root.render(<ModelPickerBody {...props} />);
  return () => root.unmount();
}

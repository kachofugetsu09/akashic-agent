import { Undo2 } from "lucide-react";
import type { StagedSteer } from "./use-steers";

// 生成中的插话：还没交给模型前以虚线气泡停在输入框上方，可以撤回到输入框修改。
export function SteerCards({ items, onWithdraw }: { items: StagedSteer[]; onWithdraw: (id: string) => void }) {
  if (!items.length) return null;
  return <ol className="steer-cards" aria-label="待插入的消息">
    {items.map((item) => <li key={item.id} className="steer-card">
      <div>
        <span className="steer-card__label" role="status">下一步交给模型</span>
        <p>{item.text}</p>
      </div>
      <button type="button" onClick={() => onWithdraw(item.id)} aria-label="撤回到输入框" data-tip="撤回到输入框">
        <Undo2 size={16} aria-hidden="true" />
      </button>
    </li>)}
  </ol>;
}

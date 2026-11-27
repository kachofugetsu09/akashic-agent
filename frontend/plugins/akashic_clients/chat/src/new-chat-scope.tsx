import { Check } from "lucide-react";
import type { NavigationGroup } from "@akashic/chat-ui-v1";
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { MarqueeText } from "./marquee-text";
import { FolderMark, type GroupRef, type NavigationGroupsState } from "./navigation-groups";
import { PluginUiSlot } from "./plugin-ui-runtime";
import { useScopeMeta } from "./scope-meta-store";

function ScopeInlineMeta({ dimension, value }: { dimension: string; value: string }) {
  const records = useScopeMeta({ dimension, value });
  const inline = records.filter((record) => record.item.inline);
  if (!inline.length) return null;
  return <>{inline.map(({ key, item }) => <small key={key} className="new-chat-scope__meta-tag">{item.label}</small>)}</>;
}

// 一个分组选项：图标与结构化短标签并排，挑项目时能看清归属与附加策略。
function ScopeOption({ dimension, group }: { dimension: string; group: NavigationGroup }) {
  return <span className="new-chat-scope__option">
    <FolderMark icon={group.icon} color={group.color} size={16} trigger="none" />
    <span>
      <strong><MarqueeText still>{group.label}</MarqueeText></strong>
      <ScopeInlineMeta dimension={dimension} value={group.id} />
      <PluginUiSlot name="scope.badge" scope={{ dimension, value: group.id }} quiet />
      {group.detail ? <small>{group.detail}</small> : null}
    </span>
  </span>;
}

/** 新会话空态输入卡里的分组选择：平时只写维度名（"项目"），选了才显示名称；发出后由标题行显示归属。 */
export function NewChatScope({ groups, onStart, onUngrouped }: {
  groups: NavigationGroupsState;
  onStart: (ref: GroupRef) => void;
  onUngrouped: () => void;
}) {
  const sections = groups.sections.filter((section) => section.groups.length > 0);
  if (!sections.length) return null;
  const active = groups.active;
  const heading = sections.find((section) => section.source.provider.dimension === active?.dimension) ?? sections[0];
  const label = heading.source.provider.label;
  return <DropdownMenu>
    <DropdownMenuTrigger asChild>
      <button type="button" className="new-chat-scope" data-tip={active ? `${label}：${active.label}` : `选择${label}`}
        aria-label={`${label}：${active?.label ?? `不归属${label}`}，点按更换`}>
        {active ? <span className="new-chat-scope__option">
          <FolderMark icon={active.icon} color={active.color} size={16} trigger="none" />
          <strong><MarqueeText still>{active.label}</MarqueeText></strong>
        </span> : <span>{label}</span>}
      </button>
    </DropdownMenuTrigger>
    <DropdownMenuContent className="choice-menu new-chat-scope__menu" align="start" side="top" sideOffset={8} collisionPadding={16}>
      <DropdownMenuItem className="choice-menu__choice" onSelect={onUngrouped}>
        <span className="new-chat-scope__option"><span className="new-chat-scope__blank" aria-hidden="true" /><strong>不归属{label}</strong></span>
        {active ? null : <Check size={16} aria-label="当前" />}
      </DropdownMenuItem>
      {sections.map(({ source, groups: items }) => <div key={source.key} role="group" aria-label={source.provider.label}>
        {sections.length > 1 ? <DropdownMenuLabel>{source.provider.label}</DropdownMenuLabel> : null}
        {items.map((group) => {
          const ref = { dimension: source.provider.dimension, id: group.id };
          const current = active?.dimension === ref.dimension && active.id === ref.id;
          return <DropdownMenuItem key={group.id} className="choice-menu__choice" onSelect={() => { if (!current) onStart(ref); }}>
            <ScopeOption dimension={ref.dimension} group={group} />
            {current ? <Check size={16} aria-label="当前" /> : null}
          </DropdownMenuItem>;
        })}
      </div>)}
    </DropdownMenuContent>
  </DropdownMenu>;
}

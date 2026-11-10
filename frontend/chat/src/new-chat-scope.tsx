import { Check, ChevronDown } from "lucide-react";
import type { NavigationGroup } from "@akashic/chat-ui-v1";
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { MarqueeText } from "./marquee-text";
import { FolderMark, type GroupRef, type NavigationGroupsState } from "./navigation-groups";
import { PluginUiSlot } from "./plugin-ui-runtime";

// 一个分组选项：图标线型与插件短标签并排，第一次挑项目时就能把线型和含义对上号。
function ScopeOption({ dimension, group }: { dimension: string; group: NavigationGroup }) {
  return <span className="new-chat-scope__option">
    <FolderMark />
    <span>
      <strong><MarqueeText still>{group.label}</MarqueeText></strong>
      <PluginUiSlot name="scope.badge" scope={{ dimension, value: group.id }} quiet />
      {group.detail ? <small>{group.detail}</small> : null}
    </span>
  </span>;
}

/** 新会话空态的分组条：首条消息之前可换到另一个分组或不归属任何分组；发出后由标题行显示归属。 */
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
      <button type="button" className="new-chat-scope" aria-label={`${label}：${active?.label ?? `不归属${label}`}，点按更换`}>
        <span className="new-chat-scope__dimension">{label}</span>
        {active ? <span className="new-chat-scope__option">
          <FolderMark />
          <strong><MarqueeText still>{active.label}</MarqueeText></strong>
          <PluginUiSlot name="scope.badge" scope={{ dimension: active.dimension, value: active.id }} quiet />
        </span> : <span>不归属{label}</span>}
        <ChevronDown size={15} aria-hidden="true" />
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

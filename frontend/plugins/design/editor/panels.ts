import { defineComponent, h, ref, type PropType, type Ref, type VNode } from "vue";
import type { Editor } from "@open-pencil/core";
import type { Color, Fill, SceneNode } from "@open-pencil/scene-graph";
import { LayerTreeItem, LayerTreeRoot, PageListRoot, type LayerRow } from "@open-pencil/vue";
import { iconSvg } from "./icons";

const TYPE_LABELS: Record<string, string> = {
  FRAME: "画框", GROUP: "编组", SECTION: "分区", RECTANGLE: "矩形", ELLIPSE: "椭圆", LINE: "直线",
  TEXT: "文字", VECTOR: "矢量", POLYGON: "多边形", STAR: "星形", COMPONENT: "组件", INSTANCE: "实例",
};
const ROUNDED = new Set(["FRAME", "RECTANGLE", "COMPONENT", "INSTANCE"]);

type LayerSlot = {
  node: { id: string; name: string; visible: boolean; type: string }; isSelected: boolean; padLeft: string;
  actions: { select: (additive: boolean) => void; toggleExpand: () => void; toggleVisibility: () => void; rename: (name: string) => void };
};

// 图层名：双击后换成输入框，回车或失焦提交，Esc 放弃。
function layerName(renaming: Ref<string | null>, slot: LayerSlot): VNode {
  const { node, actions } = slot;
  if (renaming.value !== node.id) return h("span", { class: "design-layer-name", title: TYPE_LABELS[node.type] ?? node.type }, node.name);
  return h("input", { class: "design-rename", value: node.name, "aria-label": "图层名称",
    onVnodeMounted: (vnode: VNode) => { const input = vnode.el as HTMLInputElement | null; input?.focus(); input?.select(); },
    onKeydown: (event: KeyboardEvent) => {
      if (event.key === "Enter") (event.target as HTMLInputElement).blur();
      if (event.key === "Escape") renaming.value = null;
    },
    onBlur: (event: FocusEvent) => {
      if (renaming.value === node.id) actions.rename((event.target as HTMLInputElement).value.trim() || node.name);
      renaming.value = null;
    } });
}

function layerRow(renaming: Ref<string | null>, slot: LayerSlot, hasChildren: boolean): VNode {
  const { node, actions } = slot;
  return h("div", {
    class: ["design-layer", slot.isSelected && "is-selected", !node.visible && "is-hidden"],
    style: { paddingInlineStart: `calc(${slot.padLeft} + 8px)` },
    onClick: (event: MouseEvent) => actions.select(event.shiftKey || event.metaKey || event.ctrlKey),
    onDblclick: () => { renaming.value = node.id; },
  }, [
    h("button", { type: "button", class: "design-layer-toggle", "aria-label": "展开或收起", tabindex: -1,
      style: { visibility: hasChildren ? "visible" : "hidden" }, onClick: (event: MouseEvent) => { event.stopPropagation(); actions.toggleExpand(); } }, "▸"),
    layerName(renaming, slot),
    h("button", { type: "button", class: "design-layer-eye", "aria-label": node.visible ? "隐藏图层" : "显示图层",
      innerHTML: iconSvg("eye"), onClick: (event: MouseEvent) => { event.stopPropagation(); actions.toggleVisibility(); } }),
  ]);
}

// 图层面板：页面在上、图层树在下，用 open-pencil 的无样式原件，只补我们自己的行样式。
export const LayersPanel = defineComponent({
  setup() {
    const renaming = ref<string | null>(null);
    return () => h("aside", { class: "design-panel design-panel-start", "aria-label": "图层" }, [
      h(PageListRoot, null, {
        default: ({ pages, currentPageId, actions }: { pages: SceneNode[]; currentPageId: string; actions: { add: () => void; switch: (id: string) => void } }) =>
          h("section", { class: "design-section" }, [
            h("header", { class: "design-section-head" }, [h("h2", "页面"),
              h("button", { type: "button", class: "design-mini", "aria-label": "新建页面", innerHTML: iconSvg("plus"), onClick: actions.add })]),
            ...pages.map((page) => h("button", { type: "button", class: ["design-page", page.id === currentPageId && "is-selected"],
              onClick: () => actions.switch(page.id) }, page.name)),
          ]),
      }),
      h(LayerTreeRoot, null, {
        default: ({ visibleRows }: { visibleRows: LayerRow[] }) =>
          h("section", { class: "design-section design-layers" }, [
            h("header", { class: "design-section-head" }, [h("h2", "图层")]),
            visibleRows.length ? null : h("p", { class: "design-muted" }, "画布上还没有图层"),
            ...visibleRows.map((item) => h(LayerTreeItem, { key: item.node.id, node: item.node, level: item.level, hasChildren: item.hasChildren },
              { default: (slot: LayerSlot) => layerRow(renaming, slot, item.hasChildren) })),
          ]),
      }),
    ]);
  },
});

function hex(color: Color): string {
  const part = (value: number) => Math.round(Math.min(1, Math.max(0, value)) * 255).toString(16).padStart(2, "0");
  return `#${part(color.r)}${part(color.g)}${part(color.b)}`;
}

function parseHex(value: string, alpha: number): Color {
  const channel = (index: number) => parseInt(value.slice(index, index + 2), 16) / 255;
  return { r: channel(1), g: channel(3), b: channel(5), a: alpha };
}

// 属性面板：单选时编辑位置、尺寸、圆角、不透明度、填充与文字大小；每次提交都是一步可撤销的修改。
export const PropertiesPanel = defineComponent({
  props: { editor: { type: Object as PropType<Editor>, required: true } },
  setup(props) {
    const commit = (node: SceneNode, changes: Partial<SceneNode>, label: string) =>
      props.editor.updateNodeWithUndo(node.id, changes, label);
    const number = (node: SceneNode, key: "x" | "y" | "width" | "height" | "rotation" | "cornerRadius" | "fontSize", label: string,
      options: { min?: number; step?: number } = {}) =>
      h("label", { class: "design-field" }, [h("span", label), h("input", {
        type: "number", value: Math.round(Number(node[key] ?? 0) * 100) / 100, min: options.min, step: options.step ?? 1,
        onChange: (event: Event) => {
          const value = Number((event.target as HTMLInputElement).value);
          if (Number.isFinite(value)) commit(node, { [key]: value }, label);
        },
      })]);
    const fill = (node: SceneNode) => {
      const index = node.fills.findIndex((item: Fill) => item.type === "SOLID");
      const current = node.fills[index];
      return h("label", { class: "design-field design-field-wide" }, [h("span", "填充"), current
        ? h("input", { type: "color", value: hex(current.color), onChange: (event: Event) => {
            const fills = node.fills.map((item: Fill, at: number) => at === index
              ? { ...item, color: parseHex((event.target as HTMLInputElement).value, item.color.a) } : item);
            commit(node, { fills }, "填充");
          } })
        : h("span", { class: "design-muted" }, node.fills.length ? "渐变或图片" : "无")]);
    };
    return () => {
      void props.editor.state.sceneVersion;
      const nodes = props.editor.getSelectedNodes();
      const node = nodes.length === 1 ? nodes[0] : null;
      const body = !nodes.length ? [h("p", { class: "design-muted" }, "选中图层后在这里调整属性")]
        : !node ? [h("p", { class: "design-muted" }, `已选 ${nodes.length} 个图层`)]
        : [
            h("p", { class: "design-kind" }, `${TYPE_LABELS[node.type] ?? node.type} · ${node.name}`),
            h("div", { class: "design-grid" }, [number(node, "x", "X"), number(node, "y", "Y"),
              number(node, "width", "宽", { min: 0 }), number(node, "height", "高", { min: 0 }),
              number(node, "rotation", "旋转"), ROUNDED.has(node.type) ? number(node, "cornerRadius", "圆角", { min: 0 }) : null,
              node.type === "TEXT" ? number(node, "fontSize", "字号", { min: 1 }) : null,
              h("label", { class: "design-field" }, [h("span", "不透明"), h("input", {
                type: "number", min: 0, max: 100, value: Math.round(node.opacity * 100),
                onChange: (event: Event) => commit(node, { opacity: Math.min(1, Math.max(0, Number((event.target as HTMLInputElement).value) / 100)) }, "不透明度"),
              })])]),
            fill(node),
          ];
      return h("aside", { class: "design-panel design-panel-end", "aria-label": "属性" },
        [h("section", { class: "design-section" }, [h("header", { class: "design-section-head" }, [h("h2", "属性")]), ...body])]);
    };
  },
});

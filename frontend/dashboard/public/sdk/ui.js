// The shared @akashic/dashboard-ui surface, backed by the host's bundled
// implementations. Plugins import named components from here.
const U = window.__akashicRuntime.UI;
export const {
  ShortcutKey, Label, FieldLabel, Tile, Btn, Chip, Input, SearchInput, Select,
  BrandMark, useTheme, ThemeToggle, cn, Pie, MetricTile, Sparkline, BarChart,
  api, asPageResult, pageCount,
} = U;
export default U;

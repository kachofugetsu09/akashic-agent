// Public Web Host atoms. Product pages and nested visual vocabularies belong
// to ordinary plugins, not to this global runtime surface.
const W = window.__akashicRuntime.WebUi;
export const {
  currentTheme,
  cycleTheme,
  subscribeTheme,
  themes,
  useTheme,
  MaterialButton,
  MaterialFilterChip,
  MaterialIconButton,
} = W;

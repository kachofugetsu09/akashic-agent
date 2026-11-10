// Public Web Host atoms. Product pages and nested visual vocabularies belong
// to ordinary plugins, not to this global runtime surface.
const W = window.__akashicRuntime.WebUi;
export const {
  currentTheme,
  cycleTheme,
  setThemePreference,
  subscribeTheme,
  themeFamilies,
  themes,
  themeSelection,
  useTheme,
  useThemeSelection,
  MaterialButton,
  MaterialFilterChip,
  MaterialIconButton,
} = W;

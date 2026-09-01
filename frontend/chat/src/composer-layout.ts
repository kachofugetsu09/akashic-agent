export function nextComposerExpanded(
  wasExpanded: boolean,
  text: string,
  isOverflowing: () => boolean,
): boolean {
  if (!text) return false;
  if (wasExpanded || text.includes("\n")) return true;
  return isOverflowing();
}

export function clipboardShortcut(
  key: string,
  ctrlKey: boolean,
  metaKey: boolean,
  altKey: boolean,
): "copy" | "paste" | null;
export function pasteKeySequence(
  controlHeld: boolean,
  heldMetaCodes?: string[],
): Array<{ keysym: number; code: string; down: boolean }>;
export function keysymForKey(key: string, code?: string): number | null;

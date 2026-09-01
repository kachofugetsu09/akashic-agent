const NAMED_KEYSYMS = Object.freeze({
  Backspace: 0xff08,
  Tab: 0xff09,
  Enter: 0xff0d,
  Escape: 0xff1b,
  Home: 0xff50,
  ArrowLeft: 0xff51,
  ArrowUp: 0xff52,
  ArrowRight: 0xff53,
  ArrowDown: 0xff54,
  PageUp: 0xff55,
  PageDown: 0xff56,
  End: 0xff57,
  Insert: 0xff63,
  Delete: 0xffff,
  CapsLock: 0xffe5,
  NumLock: 0xff7f,
});

const CODE_KEYSYMS = Object.freeze({
  ShiftLeft: 0xffe1,
  ShiftRight: 0xffe2,
  ControlLeft: 0xffe3,
  ControlRight: 0xffe4,
  AltLeft: 0xffe9,
  AltRight: 0xffea,
  MetaLeft: 0xffeb,
  MetaRight: 0xffec,
});

/** Identify the two clipboard shortcuts that cross the browser boundary. */
export function clipboardShortcut(key, ctrlKey, metaKey, altKey) {
  if (altKey || (!ctrlKey && !metaKey)) return null;
  const value = key.toLowerCase();
  if (value === "c") return "copy";
  if (value === "v") return "paste";
  return null;
}

/** Build a Linux paste chord without leaving a host Meta key pressed remotely. */
export function pasteKeySequence(controlHeld, heldMetaCodes = []) {
  const control = keysymForKey("Control", "ControlLeft");
  const key = keysymForKey("v", "KeyV");
  if (control === null || key === null) throw new Error("paste key mapping is missing");
  const events = heldMetaCodes.map((code) => ({
    keysym: keysymForKey("Meta", code), code, down: false,
  }));
  if (events.some((event) => event.keysym === null)) {
    throw new Error("Meta key mapping is missing");
  }
  if (!controlHeld) events.push({ keysym: control, code: "ControlLeft", down: true });
  events.push(
    { keysym: key, code: "KeyV", down: true },
    { keysym: key, code: "KeyV", down: false },
  );
  if (!controlHeld) events.push({ keysym: control, code: "ControlLeft", down: false });
  for (const code of heldMetaCodes) {
    events.push({ keysym: keysymForKey("Meta", code), code, down: true });
  }
  return events;
}

/** Map one browser keyboard event to the X11 keysym carried by RFB. */
export function keysymForKey(key, code = "") {
  if (Object.hasOwn(CODE_KEYSYMS, code)) return CODE_KEYSYMS[code];
  if (Object.hasOwn(NAMED_KEYSYMS, key)) return NAMED_KEYSYMS[key];
  const functionKey = /^F([1-9]|1[0-2])$/.exec(key);
  if (functionKey) return 0xffbe + Number(functionKey[1]) - 1;
  const points = [...key];
  if (points.length !== 1) return null;
  const value = points[0].codePointAt(0);
  if (value === undefined) return null;
  return value <= 0xff ? value : 0x01000000 | value;
}

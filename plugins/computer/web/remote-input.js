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

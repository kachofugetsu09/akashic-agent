import type { MarkdownIt } from "stream-markdown-parser";

const MARKDOWN_OPERATOR = /[*_^~+=`]/u;
const FACE_SYMBOL = /[・ωΩ∀▽дД﹏꒳｀´＾＿ー…×╥◕≧≦＞＜●○★☆♥♡ﾟ゜ᴗಥಠツシノヽヾ╭╮]/u;
const DECORATED_FACE_SIGNAL = /[・ωΩ∀▽дД﹏꒳｀´＾＿ー…×╥◕≧≦＞＜●○★☆♥♡ﾟ゜ᴗಥಠツシノヽヾ╭╮•✧♬ฅ꒰꒱₍₎٩ᵕᐢ૮ა❛❤ʓԽ໒\[]/u;
const DECORATED_FACE_SIGNAL_GLOBAL = /[・ωΩ∀▽дД﹏꒳｀´＾＿…×╥◕≧≦＞＜●○★☆♥♡ﾟ゜ᴗಥಠヽヾ╭╮•✧♬ฅ꒰꒱₍₎٩ᵕᐢ૮ა❛❤ʓԽ໒\[]/gu;
const ASCII_FACE = /^(?:[=~]?)([xXoOTt^*;:8>])[-_^'.oOqQvVwW]+([xXoOTt^*;:8<])(?:[=~]?)$/u;
const BARE_ASCII_FACE = /^(?:\^[-_.oOqQvVwW]+\^|(?=[-_]*-)(?=[-_]*_)[-_]{3,}|[TtXxOo][-_.][TtXxOo])(?=$|\s|[),）,.!?，。！？])/u;
const MAX_KAOMOJI_LENGTH = 80;
const MAX_DECORATED_KAOMOJI_LENGTH = 160;

interface InlineState {
  src: string;
  pos: number;
  posMax: number;
  push: (type: string, tag: string, nesting: number) => { content: string; markup: string };
}

/** Return a parenthesized kaomoji that Markdown would otherwise style. */
export function readKaomojiLiteral(source: string, offset: number): string | undefined {
  const openingBackticks = /^`+/u.exec(source.slice(offset))?.[0];
  if (openingBackticks) {
    const rest = source.slice(offset + openingBackticks.length);
    const hasMatchingCloser = (rest.match(/`+/gu) ?? []).some((run) => run.length === openingBackticks.length);
    if (hasMatchingCloser) return undefined;
  }

  const decorated = readDecoratedKaomojiLiteral(source, offset);
  if (decorated) return decorated;

  const opener = source[offset];
  if (opener !== "(" && opener !== "（") {
    const bare = BARE_ASCII_FACE.exec(source.slice(offset))?.[0];
    if (bare) return bare;
    if (!MARKDOWN_OPERATOR.test(opener ?? "")) return undefined;

    const nextAsciiOpen = source.indexOf("(", offset + 1);
    const nextWideOpen = source.indexOf("（", offset + 1);
    const forwardOpenings = [nextAsciiOpen, nextWideOpen].filter((value) => value >= 0);
    const nextOpeningOffset = forwardOpenings.length > 0 ? Math.min(...forwardOpenings) : -1;
    const prefix = source.slice(offset, nextOpeningOffset);
    const followsWord = /[\p{L}\p{N}]/u.test(source[offset - 1] ?? "");
    const crossesProse = prefix.includes("`") || /\s[\p{L}\p{N}]/u.test(prefix);
    if (nextOpeningOffset > offset && nextOpeningOffset - offset <= 8 && !followsWord && !crossesProse) {
      const whole = readKaomojiLiteral(source, nextOpeningOffset);
      if (whole) return source.slice(offset, nextOpeningOffset + whole.length);
    }

    const asciiOpen = source.lastIndexOf("(", offset);
    const wideOpen = source.lastIndexOf("（", offset);
    const openingOffset = Math.max(asciiOpen, wideOpen);
    if (openingOffset < 0 || offset - openingOffset >= MAX_KAOMOJI_LENGTH) return undefined;
    const whole = readKaomojiLiteral(source, openingOffset);
    if (!whole || openingOffset + whole.length <= offset) return undefined;
    return source.slice(offset, openingOffset + whole.length);
  }
  const closer = opener === "(" ? ")" : "）";
  const limit = Math.min(source.length, offset + MAX_KAOMOJI_LENGTH);
  const end = source.indexOf(closer, offset + 1);
  if (end < 0 || end >= limit) return undefined;

  const inner = source.slice(offset + 1, end);
  if (!inner || inner.includes("\n") || !MARKDOWN_OPERATOR.test(inner)) return undefined;
  const compact = inner.replace(/\s/gu, "");
  const hasShortNonProseFace = compact.length <= 40
    && /[^\x00-\x7F]/u.test(compact)
    && !/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}]{2}/u.test(compact);
  if (!FACE_SYMBOL.test(compact) && !ASCII_FACE.test(compact) && !hasShortNonProseFace) return undefined;
  return source.slice(offset, end + 1);
}

/** Protect a short standalone decorated face without consuming surrounding prose Markdown. */
function readDecoratedKaomojiLiteral(source: string, offset: number): string | undefined {
  if (offset !== 0 || source.length > MAX_DECORATED_KAOMOJI_LENGTH || source.includes("\n")) return undefined;
  if (!MARKDOWN_OPERATOR.test(source) || !DECORATED_FACE_SIGNAL.test(source)) return undefined;
  if (/!?\[[^\]]+\]\([^\n)]+\)|https?:\/\/|\$[^$\n]+\$/iu.test(source)) return undefined;

  const withoutEntities = source.replace(/&(?:#x?[\da-f]+|[a-z]+);/giu, "");
  const proseCharacters = withoutEntities
    .replace(DECORATED_FACE_SIGNAL_GLOBAL, "")
    .replace(/[xXoOTtqQvVwW]/gu, "");
  if (/[\p{L}\p{N}]/u.test(proseCharacters)) return undefined;
  if (!hasCompleteFaceWrapper(withoutEntities)) return undefined;

  const visible = [...withoutEntities].filter((character) => !/\s/u.test(character));
  const symbolCount = visible.filter((character) => !/[\p{L}\p{N}]/u.test(character)).length;
  if (visible.length === 0 || symbolCount / visible.length < 0.3) return undefined;
  return source;
}

/** Require a closed face boundary before taking over Markdown parsing. */
function hasCompleteFaceWrapper(source: string): boolean {
  let hasPair = false;
  for (const [opening, closing] of [["(", ")"], ["（", "）"], ["꒰", "꒱"], ["₍", "₎"], ["[", "]"]]) {
    const openAt = source.indexOf(opening);
    const closeAt = source.lastIndexOf(closing);
    if ((openAt >= 0) !== (closeAt >= 0)) return false;
    if (openAt >= 0) {
      if (closeAt <= openAt) return false;
      hasPair = true;
    }
  }
  if (hasPair) return true;

  const visible = [...source].filter((character) => !/\s/u.test(character));
  return visible.length >= 3
    && visible[0] === visible.at(-1)
    && /[★☆♥♡ฅ]/u.test(visible[0] ?? "");
}

/** Add one text-token rule without replacing Markstream's stream parser. */
export function configureKaomojiMarkdown(markdown: MarkdownIt): MarkdownIt {
  markdown.inline.ruler.before("text", "akashic_kaomoji_literal", (rawState: unknown, silent?: boolean) => {
    const state = rawState as InlineState;
    const literal = readKaomojiLiteral(state.src, state.pos);
    if (!literal || state.pos + literal.length > state.posMax) return false;
    if (!silent) {
      const token = state.push("kaomoji_literal", "", 0);
      token.content = literal;
      token.markup = literal;
    }
    state.pos += literal.length;
    return true;
  });
  return markdown;
}

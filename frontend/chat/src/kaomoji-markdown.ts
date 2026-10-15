import type { MarkdownIt } from "stream-markdown-parser";

const MARKDOWN_OPERATOR = /[*_^~+=`]/u;
const FACE_SYMBOL = /[・ωΩ∀▽дД﹏꒳｀´＾＿ー…×╥◕≧≦＞＜●○★☆♥♡ﾟ゜ᴗಥಠツシノヽヾ╭╮]/u;
const DECORATED_FACE_SIGNAL = /[・ωΩ∀▽дД﹏꒳｀´＾＿ー…×╥◕≧≦＞＜●○★☆♥♡ﾟ゜ᴗಥಠツシノヽヾ╭╮•✧♬ฅ꒰꒱₍₎٩ᵕᐢ૮ა❛❤ʓԽ໒[]/u;
const DECORATED_FACE_SIGNAL_GLOBAL = /[・ωΩ∀▽дД﹏꒳｀´＾＿…×╥◕≧≦＞＜●○★☆♥♡ﾟ゜ᴗಥಠヽヾ╭╮•✧♬ฅ꒰꒱₍₎٩ᵕᐢ૮ა❛❤ʓԽ໒[]/gu;
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

  const armed = readArmedKaomojiLiteral(source, offset);
  if (armed) return armed;

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
  if (end < 0) return isOpenFaceTail(source.slice(offset)) ? source.slice(offset) : undefined;
  if (end >= limit) return undefined;

  const inner = source.slice(offset + 1, end);
  if (!inner || inner.includes("\n") || !MARKDOWN_OPERATOR.test(inner)) return undefined;
  const compact = inner.replace(/\s/gu, "");
  const hasShortNonProseFace = compact.length <= 40
    && /\P{ASCII}/u.test(compact)
    && !/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}]{2}/u.test(compact);
  if (!FACE_SYMBOL.test(compact) && !ASCII_FACE.test(compact) && !hasShortNonProseFace) return undefined;
  return source.slice(offset, end + 1);
}

// 手臂包着括号脸：_(:з」∠)_、~\(≧▽≦)/~、\(≧▽≦)/、¯\_(ツ)_/¯。手臂里的 _ ~ * \ 是颜文字的一部分，
// 不是强调、下标、删除线或 \( 公式的开头；生成中写到一半（脸还没闭合）同样按字面显示。
const ARMED_FACE = /^(\\_|[~_*]\\|\\|[~_*])[(（]([^()（）\n]{1,30})[)）](\\?_\/|\/[~_*]?|[~_*])?/u;
const ARMED_FACE_OPEN = /^(\\_|[~_*]\\|\\|[~_*])[(（]([^()（）\n]{1,24})$/u;

function readArmedKaomojiLiteral(source: string, offset: number): string | undefined {
  // 前一个字符是反斜杠时，这里是被转义的字面字符，交给转义规则
  if (source[offset - 1] === "\\") return undefined;
  const rest = source.slice(offset);
  const closed = ARMED_FACE.exec(rest);
  if (closed) {
    const [whole, arm, inner, tail] = closed;
    const backslashArm = arm.includes("\\");
    if (!isFaceInner(inner)) return undefined;
    // 单个 \ 只在右边有 / 手臂时才算脸（\(x\) 是公式）；_ * ~ 手臂要两边都有，否则可能是普通强调
    if (arm === "\\" && !tail?.startsWith("/")) return undefined;
    if (!tail && !backslashArm) return undefined;
    // 脸里没有任何符号（_(注意)_）是强调包着括号，除非手臂带反斜杠（¯\_(ツ)_/¯）
    if (!backslashArm && !/[^\p{L}\p{N}\s]/u.test(inner)) return undefined;
    return whole;
  }
  const open = ARMED_FACE_OPEN.exec(rest);
  return open && isFaceInner(open[2]) && /[^\p{L}\p{N}\s]/u.test(open[2]) ? rest : undefined;
}

// 括号里像一张脸：有脸部符号、ASCII 脸，或是不含连续汉字/假名的短符号串。
function isFaceInner(inner: string): boolean {
  const compact = inner.replace(/\s/gu, "");
  if (!compact || compact.endsWith("\\")) return false;
  if (/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}]{2}/u.test(compact)) return false;
  return FACE_SYMBOL.test(compact) || ASCII_FACE.test(compact) || (compact.length <= 40 && /\P{ASCII}/u.test(compact));
}

// 生成中括号还没闭合的脸（"(*´▽"）：括号里已有会被当成 Markdown 的符号和脸部符号时先按字面显示，闭合后不跳。
function isOpenFaceTail(tail: string): boolean {
  const inner = tail.slice(1);
  return tail.length <= 24 && !inner.includes("\n") && MARKDOWN_OPERATOR.test(inner) && isFaceInner(inner);
}

/** 静态路径（micromark）没有可插的行内规则：解析前用同一套判断找出颜文字，把其中的 Markdown 符号转义成字面字符。
 *  围栏代码与行内代码原样跳过。两条渲染路径因此同一结果，生成中正常、落定后不会再变成斜体。 */
export function protectKaomojiLiterals(markdown: string): string {
  let fence: string | null = null;
  return markdown.split("\n").map((line) => {
    const open = /^ {0,3}(`{3,}|~{3,})/u.exec(line);
    if (fence !== null) {
      if (open && open[1][0] === fence[0] && open[1].length >= fence.length && /^ {0,3}[`~]+[ \t]*$/u.test(line)) fence = null;
      return line;
    }
    if (open) {
      fence = open[1];
      return line;
    }
    return protectLine(line);
  }).join("\n");
}

const STATIC_TRIGGER = /[(（\\*_^~+=`-]/u;

function protectLine(line: string): string {
  let out = "";
  let index = 0;
  while (index < line.length) {
    const character = line[index];
    // 1. 行内代码整段照抄
    if (character === "`") {
      const run = /^`+/u.exec(line.slice(index))![0];
      const close = line.indexOf(run, index + run.length);
      if (close > 0) {
        out += line.slice(index, close + run.length);
        index = close + run.length;
        continue;
      }
    }
    // 2. 命中颜文字：转义其中的 Markdown 符号
    const literal = STATIC_TRIGGER.test(character) ? readKaomojiLiteral(line, index) : undefined;
    if (literal) {
      out += literal.replace(/[\\`*_~[\]<>^=+#|-]/gu, "\\$&");
      index += literal.length;
      continue;
    }
    // 3. 原有转义成对照抄，不拆开
    const step = character === "\\" ? 2 : 1;
    out += line.slice(index, index + step);
    index += step;
  }
  return out;
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

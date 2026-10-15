import type { Extension } from "micromark-extension-gfm";
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
  const literal = readLiteral(source, offset);
  // 反引号开头的候选：后面有同长度的反引号串就是行内代码，不算颜文字（只在命中候选时才扫，免得每个反引号都扫到文末）
  const openingBackticks = literal && /^`+/u.exec(literal)?.[0];
  if (openingBackticks) {
    const rest = source.slice(offset + openingBackticks.length);
    const hasMatchingCloser = (rest.match(/`+/gu) ?? []).some((run) => run.length === openingBackticks.length);
    if (hasMatchingCloser) return undefined;
  }
  return literal;
}

// 每次查找都限定在候选的最大长度内（脸 80 字、往前找括号 8 字），长文本里逐字调用也是线性的。
function readLiteral(source: string, offset: number): string | undefined {
  const decorated = readDecoratedKaomojiLiteral(source, offset);
  if (decorated) return decorated;

  const armed = readArmedKaomojiLiteral(source, offset);
  if (armed) return armed;

  const opener = source[offset];
  if (opener !== "(" && opener !== "（") {
    const bare = BARE_ASCII_FACE.exec(source.slice(offset))?.[0];
    if (bare) return bare;
    if (!MARKDOWN_OPERATOR.test(opener ?? "")) return undefined;

    const ahead = source.slice(offset + 1, offset + 9);
    const forwardOpenings = [ahead.indexOf("("), ahead.indexOf("（")].filter((value) => value >= 0);
    const nextOpeningOffset = forwardOpenings.length > 0 ? offset + 1 + Math.min(...forwardOpenings) : -1;
    const prefix = source.slice(offset, nextOpeningOffset);
    const followsWord = /[\p{L}\p{N}]/u.test(source[offset - 1] ?? "");
    const crossesProse = prefix.includes("`") || /\s[\p{L}\p{N}]/u.test(prefix);
    if (nextOpeningOffset > offset && nextOpeningOffset - offset <= 8 && !followsWord && !crossesProse) {
      const whole = readKaomojiLiteral(source, nextOpeningOffset);
      if (whole) return source.slice(offset, nextOpeningOffset + whole.length);
    }

    const behindStart = Math.max(0, offset - MAX_KAOMOJI_LENGTH + 1);
    const behind = source.slice(behindStart, offset + 1);
    const openingIndex = Math.max(behind.lastIndexOf("("), behind.lastIndexOf("（"));
    if (openingIndex < 0) return undefined;
    const openingOffset = behindStart + openingIndex;
    const whole = readKaomojiLiteral(source, openingOffset);
    if (!whole || openingOffset + whole.length <= offset) return undefined;
    return source.slice(offset, openingOffset + whole.length);
  }
  const closer = opener === "(" ? ")" : "）";
  const limit = Math.min(source.length, offset + MAX_KAOMOJI_LENGTH);
  const closerIndex = source.slice(offset + 1, limit).indexOf(closer);
  if (closerIndex < 0) return isOpenFaceTail(source.slice(offset)) ? source.slice(offset) : undefined;
  const end = offset + 1 + closerIndex;

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

type TextConstruct = Exclude<NonNullable<Extension["text"]>[string], unknown[] | undefined>;
type State = Parameters<TextConstruct["tokenize"]>[1];

// 静态路径（micromark）的行内扩展：与富渲染的行内规则同一套判断，命中的颜文字按普通文字输出。
// 扩展只在 micromark 已认出的行内文字里触发：围栏、缩进代码、跨行的行内代码都不会进来；表格先分列再看单元格，
// 逐字核对、遇到单元格边界即放弃，与富渲染路径（分列后才跑颜文字规则）同一结果。
const STATIC_TRIGGERS = "(（\\*_^~+=-";
// 装饰脸（"꒰ *´꒳`* ꒱"、"★*゜ﾟ*★"）只在段首成立，起始字符不在上面的触发字符里：把每行、每个单元格开头的字符也注册上，
// 这些字符只在内容开头才判断，正文中间遇到直接放过。
const CONTENT_START = /(?:^|\n)[ \t]*(?:>[ \t]?)*(?:(?:[-*+]|\d{1,9}[.)])[ \t]+)?(?:\[[ xX]\][ \t]+)?(\S)|\|[ \t]*([^\s|])/gu;

function contentStartCodes(source: string): Set<number> {
  const codes = new Set<number>();
  for (const match of source.matchAll(CONTENT_START)) {
    const code = (match[1] ?? match[2]).charCodeAt(0);
    if (!STATIC_TRIGGERS.includes(String.fromCharCode(code))) codes.add(code);
  }
  return codes;
}

export function kaomojiLiteralExtension(source: string): Extension {
  const construct = literalConstruct(source, false);
  const startOnly = literalConstruct(source, true);
  return { text: Object.fromEntries([
    ...[...STATIC_TRIGGERS].map((character) => [character.charCodeAt(0), construct]),
    ...[...contentStartCodes(source)].map((code) => [code, startOnly]),
  ]) };
}

function literalConstruct(source: string, contentStartOnly: boolean): TextConstruct {
  return { name: "kaomojiLiteral", tokenize(effects, ok, nok) {
    const context = this;
    let literal = "";
    let index = 0;
    const inside: State = (code) => {
      if (index === literal.length) {
        effects.exit("data");
        return ok(code);
      }
      if (code !== literal.charCodeAt(index)) return nok(code);
      index += 1;
      effects.consume(code);
      return inside;
    };
    return (code) => {
      const atContentStart = context.events.length === 0;
      if (contentStartOnly && !atContentStart) return nok(code);
      literal = staticKaomojiLiteral(source, context.now().offset, atContentStart) ?? "";
      if (!literal) return nok(code);
      effects.enter("data");
      return inside(code);
    };
  } };
}

// 只把当前位置前后有界的一段交给判断。富渲染传入的是本段行内源码，"整段只有一个装饰脸"只在段首成立：
// 段首取到行尾；下一行还有字时带上换行（可能是段落续行，按"不止一行"处理）。
function staticKaomojiLiteral(source: string, offset: number, atContentStart: boolean): string | undefined {
  const ahead = source.slice(offset, offset + MAX_DECORATED_KAOMOJI_LENGTH + 1);
  if (atContentStart) {
    const lineEnd = ahead.indexOf("\n");
    const nextLine = source.slice(offset + lineEnd + 1, offset + lineEnd + 1 + MAX_KAOMOJI_LENGTH);
    const single = lineEnd >= 0 && /^[ \t]*(?:\n|$)/u.test(nextLine);
    return readKaomojiLiteral(single ? ahead.slice(0, lineEnd) : ahead, 0);
  }
  const start = Math.max(0, offset - MAX_KAOMOJI_LENGTH);
  return readKaomojiLiteral(source.slice(start, offset) + ahead, offset - start);
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

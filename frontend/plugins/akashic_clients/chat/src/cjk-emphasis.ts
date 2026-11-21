import type { MarkdownIt } from "stream-markdown-parser";

type InlineRule = (state: InlineState, silent: boolean) => boolean;

interface DelimiterScan { can_open: boolean; can_close: boolean; length: number }

interface InlineState {
  src: string;
  posMax: number;
  scanDelims(start: number, canSplitWord: boolean): DelimiterScan;
}

interface InlineRuler {
  at(name: string): { fn: InlineRule } | undefined;
  at(name: string, fn: InlineRule): void;
}

// 中日韩文字与全角标点：按 CommonMark CJK-friendly 提案，紧贴它们的 ** 不再因"标点在内侧"失效。
const CJK = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}\u3000-\u303f\uff00-\uffef\ufe30-\ufe4f]/u;
const WHITESPACE = /\s/u;
const PUNCTUATION = /[\p{P}\p{S}]/u;

// 异体选择符（如 葛 + U+E0100）跟着前一个字算，和静态渲染用的 cjk-friendly 扩展一致。
const VARIATION_SELECTOR = /[\ufe00-\ufe0f\u{e0100}-\u{e01ef}]/u;
const charBefore = (src: string, pos: number): string => {
  if (pos <= 0) return " ";
  const low = src.charCodeAt(pos - 1);
  const start = low >= 0xdc00 && low <= 0xdfff && pos >= 2 ? pos - 2 : pos - 1;
  const char = String.fromCodePoint(src.codePointAt(start) ?? 32);
  return VARIATION_SELECTOR.test(char) && start > 0 ? charBefore(src, start) : char;
};
const charAt = (src: string, pos: number, max: number): string => pos < max ? String.fromCodePoint(src.codePointAt(pos) ?? 32) : " ";

// 与 markdown-it 的 scanDelims 同一判定，只在一侧是中日韩字符时放宽左右侧翼。
function scanCjkDelims(this: InlineState, start: number, canSplitWord: boolean): DelimiterScan {
  const marker = this.src.charCodeAt(start);
  let pos = start;
  while (pos < this.posMax && this.src.charCodeAt(pos) === marker) pos++;
  const length = pos - start;
  const last = charBefore(this.src, start);
  const next = charAt(this.src, pos, this.posMax);
  const lastSpace = WHITESPACE.test(last);
  const nextSpace = WHITESPACE.test(next);
  if (lastSpace || nextSpace) return { can_open: !nextSpace, can_close: !lastSpace, length };
  const lastPunct = PUNCTUATION.test(last);
  const nextPunct = PUNCTUATION.test(next);
  const nearCjk = CJK.test(last) || CJK.test(next);
  const left = lastPunct || (canSplitWord && (nearCjk || !nextPunct));
  const right = nextPunct || (canSplitWord && (nearCjk || !lastPunct));
  return { can_open: left, can_close: right, length };
}

// 只在强调规则运行期间换掉扫描，删除线等其他规则照旧走原判定。
export function configureCjkEmphasis(markdown: MarkdownIt): MarkdownIt {
  const ruler = (markdown as unknown as { inline: { ruler: InlineRuler } }).inline.ruler;
  const original = ruler.at("emphasis")?.fn;
  if (!original) return markdown;
  ruler.at("emphasis", (state, silent) => {
    const scan = state.scanDelims;
    state.scanDelims = scanCjkDelims;
    try {
      return original(state, silent);
    } finally {
      state.scanDelims = scan;
    }
  });
  return markdown;
}

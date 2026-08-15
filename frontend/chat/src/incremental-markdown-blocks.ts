const UNSTABLE_TAIL_BLOCKS = 2;

export type MarkdownBlockParser = (markdown: string) => string[];

export interface IncrementalMarkdownMetrics {
  generation: number;
  parsedCharacters: number;
  maxTailCharacters: number;
  frozenBlocks: number;
}

/** 复用已稳定的顶层 Markdown block，只让解析器处理末尾两个不稳定块。 */
export class IncrementalMarkdownBlocks {
  private readonly parseBlocks: MarkdownBlockParser;
  private source = "";
  private blocks: string[] = [];
  private streaming: boolean | null = null;
  private generation = 0;
  private parsedCharacters = 0;
  private maxTailCharacters = 0;
  private frozenBlocks = 0;

  constructor(parseBlocks: MarkdownBlockParser) {
    this.parseBlocks = parseBlocks;
  }

  parse(source: string, streaming: boolean): string[] {
    if (source === this.source && streaming === this.streaming) return this.blocks;

    // 1. 模式切换或非追加输入建立新 generation，终态始终做一次完整解析。
    if (this.streaming !== null && this.streaming !== streaming) this.resetGeneration();
    if (!streaming || !source.startsWith(this.source)) {
      if (streaming && this.source !== "") this.resetGeneration();
      this.streaming = streaming;
      return this.parseFull(source);
    }

    // 2. 冻结解析器已经给出的稳定 block，只重算原文尾部。
    const firstUnstable = Math.max(0, this.blocks.length - UNSTABLE_TAIL_BLOCKS);
    const frozen = this.blocks.slice(0, firstUnstable);
    const tailStart = frozen.reduce((offset, block) => offset + block.length, 0);
    const tailSource = source.slice(tailStart);
    const tail = this.runParser(tailSource);
    this.source = source;
    this.streaming = true;
    this.blocks = [...frozen, ...tail];
    this.frozenBlocks = frozen.length;
    return this.blocks;
  }

  metrics(): IncrementalMarkdownMetrics {
    return {
      generation: this.generation,
      parsedCharacters: this.parsedCharacters,
      maxTailCharacters: this.maxTailCharacters,
      frozenBlocks: this.frozenBlocks,
    };
  }

  private parseFull(source: string): string[] {
    this.source = source;
    this.blocks = this.runParser(source);
    this.frozenBlocks = 0;
    return this.blocks;
  }

  private runParser(source: string): string[] {
    this.parsedCharacters += source.length;
    this.maxTailCharacters = Math.max(this.maxTailCharacters, source.length);
    return this.parseBlocks(source);
  }

  private resetGeneration(): void {
    this.source = "";
    this.blocks = [];
    this.frozenBlocks = 0;
    this.generation += 1;
  }
}

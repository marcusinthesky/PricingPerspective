const WORDS_PER_MINUTE = 200;
const WORD_PATTERN = /[\p{L}\p{N}]+(?:['’-][\p{L}\p{N}]+)*/gu;

export type ReadingStats = {
  wordCount: number;
  readingTimeMinutes: number;
};

export function countReadableWords(source: string): number {
  const readableSource = source
    .replace(/<!--[\s\S]*?-->/g, " ")
    .replace(/<[^>]*>/g, " ")
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/https?:\/\/\S+/g, " ")
    .replace(/[`*_~>#]/g, " ")
    .replace(/^\s*(?:[-+*]|\d+[.)])\s+/gm, " ");

  return readableSource.match(WORD_PATTERN)?.length ?? 0;
}

export function readingStats(source: string): ReadingStats {
  const wordCount = countReadableWords(source);
  return {
    wordCount,
    readingTimeMinutes: Math.max(1, Math.ceil(wordCount / WORDS_PER_MINUTE)),
  };
}

export function formatReadingTime(minutes: number): string {
  return `${minutes} min read`;
}

/**
 * Eval set loading, language detection, and character utilities.
 */

export interface CharEntry {
  char: string;
  codepoint: number;
  hex: string;
  count: number;
  frequency: number;
  tier: "common" | "uncommon" | "rare" | "unseen" | string;
}

export interface EvalSet {
  description: string;
  char_frequencies: CharEntry[];
}

export function detectLanguage(char: string): string {
  if (!char) return "other";
  const cp = char.codePointAt(0)!;
  if (cp >= 0x3040 && cp <= 0x30ff) return "ja";
  if ((cp >= 0xac00 && cp <= 0xd7af) || (cp >= 0x1100 && cp <= 0x11ff))
    return "ko";
  if (
    (cp >= 0x4e00 && cp <= 0x9fff) ||
    (cp >= 0x3400 && cp <= 0x4dbf) ||
    (cp >= 0x20000 && cp <= 0x2a6df)
  )
    return "zh";
  return "other";
}

export function groupByLanguage(
  chars: CharEntry[],
  languages = ["ja", "ko", "zh"]
): Map<string, CharEntry[]> {
  const groups = new Map<string, CharEntry[]>();
  for (const lang of [...languages, "other"]) {
    groups.set(lang, []);
  }
  for (const entry of chars) {
    const lang = detectLanguage(entry.char);
    const bucket = groups.get(lang) ?? groups.get("other")!;
    bucket.push(entry);
  }
  return groups;
}

export function groupByTier(chars: CharEntry[]): Map<string, CharEntry[]> {
  const groups = new Map<string, CharEntry[]>();
  for (const entry of chars) {
    const tier = entry.tier ?? "unknown";
    if (!groups.has(tier)) groups.set(tier, []);
    groups.get(tier)!.push(entry);
  }
  return groups;
}

export function charToBytes(char: string): Uint8Array {
  return new TextEncoder().encode(char);
}

export function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

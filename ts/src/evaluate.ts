/**
 * Prompt API evaluator for UTF-8 structural validity.
 *
 * Limitations vs the Python byte-prefix protocol:
 * - Input is text, not raw bytes — we cannot feed partial UTF-8 sequences
 * - Output is text — we score the UTF-8 encoding of the generated string
 * - No logit access — no perplexity or log-likelihood metrics
 *
 * What this DOES test:
 * - Given a text prompt ending with a partial character context,
 *   can the model produce valid UTF-8 in its continuation?
 * - DFA scoring (V_partial, V_binary_strict, V_binary_soft) on output bytes
 * - Per-tier and per-language breakdowns matching the paper's structure
 */

import { computeDualScore, type DualScore } from "./dfa.js";
import {
  type CharEntry,
  type EvalSet,
  detectLanguage,
  groupByLanguage,
  charToBytes,
} from "./evalset.js";

export interface EvalConfig {
  /** Max characters to generate per prompt (mapped to Prompt API's response constraint) */
  maxResponseChars?: number;
  /** Limit samples per language (trial mode) */
  trialSamples?: number;
  /** Languages to evaluate */
  languages?: string[];
  /** Called after each character is evaluated */
  onProgress?: (done: number, total: number, latest: CharResult) => void;
}

export interface CharResult {
  char: string;
  codepoint: number;
  tier: string;
  language: string;
  prompt: string;
  generated: string;
  generatedBytes: Uint8Array;
  dfa: DualScore;
  isValidUtf8: boolean;
}

export interface TierStats {
  totalSamples: number;
  validUtf8Rate: number;
  avgDfaPartial: number;
  avgDfaStrict: number;
  avgDfaSoft: number;
}

export interface LangResult {
  language: string;
  totalCharacters: number;
  overallStats: TierStats;
  byTier: Record<string, TierStats>;
  results: CharResult[];
}

export interface EvalResult {
  model: string;
  timestamp: string;
  config: EvalConfig;
  languages: Record<string, LangResult>;
  papersComparable: string[];
  papersNotComparable: string[];
}

function buildPrompt(char: string): string {
  return (
    `Continue this character sequence. Output ONLY the next few characters, nothing else.\n\n` +
    `${char}`
  );
}

function aggregateStats(results: CharResult[]): TierStats {
  if (results.length === 0) {
    return {
      totalSamples: 0,
      validUtf8Rate: 0,
      avgDfaPartial: 0,
      avgDfaStrict: 0,
      avgDfaSoft: 0,
    };
  }
  const n = results.length;
  const valid = results.filter((r) => r.isValidUtf8).length;
  const partial = results.reduce((s, r) => s + r.dfa.partialScore, 0);
  const strict = results.reduce((s, r) => s + r.dfa.binaryStrict, 0);
  const soft = results.reduce((s, r) => s + r.dfa.binarySoft, 0);
  return {
    totalSamples: n,
    validUtf8Rate: valid / n,
    avgDfaPartial: partial / n,
    avgDfaStrict: strict / n,
    avgDfaSoft: soft / n,
  };
}

/**
 * Run a Level 0–style evaluation using the Prompt API.
 *
 * For each OOV character, prompts the model with the character and scores
 * the UTF-8 bytes of the generated continuation.
 */
export async function evaluate(
  evalSet: EvalSet,
  config: EvalConfig = {}
): Promise<EvalResult> {
  const {
    maxResponseChars = 20,
    trialSamples,
    languages = ["ja", "ko", "zh"],
    onProgress,
  } = config;

  // Detect Prompt API — current: window.LanguageModel, legacy: window.ai.languageModel
  const languageModel: any =
    (globalThis as any).LanguageModel ??
    (globalThis as any).ai?.languageModel ??
    (self as any).ai?.languageModel;

  if (!languageModel) {
    throw new Error("Prompt API not available.");
  }

  const session = await languageModel.create();

  const grouped = groupByLanguage(evalSet.char_frequencies, languages);
  const encoder = new TextEncoder();
  const langResults: Record<string, LangResult> = {};
  let done = 0;
  const total = trialSamples
    ? Math.min(trialSamples, evalSet.char_frequencies.length) * languages.length
    : evalSet.char_frequencies.length;

  for (const [lang, chars] of grouped) {
    if (chars.length === 0) continue;
    const subset = trialSamples ? chars.slice(0, trialSamples) : chars;
    const results: CharResult[] = [];

    for (const entry of subset) {
      const prompt = buildPrompt(entry.char);
      let generated: string;

      try {
        generated = await session.prompt(prompt);
      } catch (e) {
        generated = "";
      }

      const generatedBytes = encoder.encode(generated);
      const dfa = computeDualScore(generatedBytes);

      // In browser, strings are always valid UTF-16 → valid UTF-8 on encode.
      // The DFA check here is for completeness and to score structural quality.
      const isValid = dfa.binaryStrict === 1.0;

      const result: CharResult = {
        char: entry.char,
        codepoint: entry.codepoint,
        tier: entry.tier,
        language: lang,
        prompt,
        generated,
        generatedBytes,
        dfa,
        isValidUtf8: isValid,
      };
      results.push(result);

      done++;
      onProgress?.(done, total, result);
    }

    const byTier: Record<string, TierStats> = {};
    const tierGroups = new Map<string, CharResult[]>();
    for (const r of results) {
      if (!tierGroups.has(r.tier)) tierGroups.set(r.tier, []);
      tierGroups.get(r.tier)!.push(r);
    }
    for (const [tier, tResults] of tierGroups) {
      byTier[tier] = aggregateStats(tResults);
    }

    langResults[lang] = {
      language: lang,
      totalCharacters: results.length,
      overallStats: aggregateStats(results),
      byTier,
      results,
    };
  }

  session.destroy();

  return {
    model: "gemini-nano (Prompt API)",
    timestamp: new Date().toISOString(),
    config,
    languages: langResults,
    papersComparable: [
      "DFA V_partial (on generated text bytes)",
      "DFA V_binary_strict",
      "DFA V_binary_soft",
      "Per-tier breakdown (common/uncommon/rare/unseen)",
      "Per-language breakdown (ja/ko/zh)",
    ],
    papersNotComparable: [
      "Byte-prefix protocol (requires raw byte token input)",
      "Incremental validity (steps 1-5 of token generation)",
      "Perplexity / log-likelihood (requires logit access)",
      "Gold vs generated likelihood comparison (Δ_LL)",
      "Pure-byte vs mixed-token breakdown (requires token ID access)",
    ],
  };
}

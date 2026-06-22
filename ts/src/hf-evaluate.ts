/**
 * Full paper evaluation using Transformers.js.
 *
 * Matches the Python bytecanary protocol: byte-prefix generation,
 * DFA scoring, perplexity, likelihood comparison, term matching.
 *
 * Dynamically imported — not part of the main library bundle.
 */

import {
  AutoTokenizer,
  AutoModelForCausalLM,
  Tensor,
} from "@huggingface/transformers";
import { ByteTokenizer } from "./decode.js";
import { computeDualScore, type DualScore } from "./dfa.js";
import { type CharEntry, type EvalSet, detectLanguage, charToBytes } from "./evalset.js";

export interface HFEvalConfig {
  model: string;
  dtype?: string;
  prefixLength?: number;
  maxNewTokens?: number;
  trialSamples?: number;
  languages?: string[];
  level1Data?: Record<string, Level1Sample[]>;
  onProgress?: (msg: string, done: number, total: number) => void;
  onModelProgress?: (progress: { status: string; progress?: number; file?: string }) => void;
}

export interface Level1Sample {
  term: string;
  language: string;
  sentence_prompt: string;
  sentence?: string;
  frequency?: string;
}

export interface HFCharResult {
  char: string;
  codepoint: number;
  tier: string;
  language: string;
  prefixLength: number;
  generatedTokenIds: number[];
  generatedBytes: Uint8Array;
  goldBytes: Uint8Array;
  dfa: DualScore;
  perplexity: number;
  goldLogLikelihood: number;
  genLogLikelihood: number;
  deltaLL: number;
}

export interface HFL1Result {
  term: string;
  language: string;
  prefixLength: number;
  generatedTokenIds: number[];
  generatedBytes: Uint8Array;
  goldBytes: Uint8Array;
  dfa: DualScore;
  termMatch: boolean;
  perplexity: number;
  goldLogLikelihood: number;
  genLogLikelihood: number;
  deltaLL: number;
}

export interface HFTierStats {
  count: number;
  avgVPartial: number;
  avgVBinaryStrict: number;
  avgVBinarySoft: number;
  avgPerplexity: number;
  goldBetterCount: number;
  genBetterCount: number;
  equalCount: number;
  termMatchRate?: number;
}

export interface HFLangResult {
  language: string;
  totalCharacters: number;
  overallStats: HFTierStats;
  byTier: Record<string, HFTierStats>;
  results: HFCharResult[];
  level1?: {
    totalSamples: number;
    stats: HFTierStats;
    results: HFL1Result[];
  };
}

export interface HFEvalResult {
  model: string;
  timestamp: string;
  config: Omit<HFEvalConfig, "onProgress" | "onModelProgress">;
  byteCoverage: number;
  tokenizerType: string;
  languages: Record<string, HFLangResult>;
}

function toTensor(ids: number[]): Tensor {
  return new Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);
}

function tensorToIds(t: Tensor): number[] {
  return Array.from(t.data as BigInt64Array).map(Number);
}

function logSoftmax(logitsData: Float32Array, offset: number, vocabSize: number, targetId: number): number {
  let max = -Infinity;
  for (let i = 0; i < vocabSize; i++) max = Math.max(max, logitsData[offset + i]);
  let sumExp = 0;
  for (let i = 0; i < vocabSize; i++) sumExp += Math.exp(logitsData[offset + i] - max);
  return logitsData[offset + targetId] - max - Math.log(sumExp);
}

async function computeSequenceLL(
  model: any,
  contextIds: number[],
  continuationIds: number[]
): Promise<number> {
  if (continuationIds.length === 0) return 0;
  const fullIds = [...contextIds, ...continuationIds];
  const input = toTensor(fullIds);
  const { logits } = await model({ input_ids: input });
  const data = logits.data as Float32Array;
  const vocabSize = logits.dims[2];
  let ll = 0;
  const startPos = contextIds.length - 1;
  for (let i = 0; i < continuationIds.length; i++) {
    const pos = startPos + i;
    ll += logSoftmax(data, pos * vocabSize, vocabSize, continuationIds[i]);
  }
  return ll;
}

async function computePerplexity(
  model: any,
  allIds: number[]
): Promise<number> {
  if (allIds.length < 2) return Infinity;
  const input = toTensor(allIds);
  const { logits } = await model({ input_ids: input });
  const data = logits.data as Float32Array;
  const vocabSize = logits.dims[2];
  let totalLL = 0;
  const n = allIds.length - 1;
  for (let t = 0; t < n; t++) {
    totalLL += logSoftmax(data, t * vocabSize, vocabSize, allIds[t + 1]);
  }
  return Math.exp(-totalLL / n);
}

function groupByLang(chars: CharEntry[], languages: string[]): Map<string, CharEntry[]> {
  const groups = new Map<string, CharEntry[]>();
  for (const lang of languages) groups.set(lang, []);
  for (const entry of chars) {
    const lang = detectLanguage(entry.char);
    if (groups.has(lang)) groups.get(lang)!.push(entry);
  }
  return groups;
}

function aggregateStats(results: Array<{ dfa: DualScore; perplexity: number; deltaLL: number; termMatch?: boolean }>): HFTierStats {
  const n = results.length;
  if (n === 0) return { count: 0, avgVPartial: 0, avgVBinaryStrict: 0, avgVBinarySoft: 0, avgPerplexity: 0, goldBetterCount: 0, genBetterCount: 0, equalCount: 0 };
  const finitePerps = results.map(r => r.perplexity).filter(p => isFinite(p));
  const termMatches = results.filter(r => r.termMatch !== undefined);
  return {
    count: n,
    avgVPartial: results.reduce((s, r) => s + r.dfa.partialScore, 0) / n,
    avgVBinaryStrict: results.reduce((s, r) => s + r.dfa.binaryStrict, 0) / n,
    avgVBinarySoft: results.reduce((s, r) => s + r.dfa.binarySoft, 0) / n,
    avgPerplexity: finitePerps.length > 0 ? finitePerps.reduce((a, b) => a + b, 0) / finitePerps.length : Infinity,
    goldBetterCount: results.filter(r => r.deltaLL > 0).length,
    genBetterCount: results.filter(r => r.deltaLL < 0).length,
    equalCount: results.filter(r => r.deltaLL === 0).length,
    ...(termMatches.length > 0
      ? { termMatchRate: termMatches.filter(r => r.termMatch).length / termMatches.length }
      : {}),
  };
}

export async function evaluateHF(
  evalSet: EvalSet,
  config: HFEvalConfig
): Promise<HFEvalResult> {
  const {
    model: modelId,
    dtype = "q4",
    prefixLength = 2,
    maxNewTokens = 5,
    trialSamples,
    languages = ["ja", "ko", "zh"],
    level1Data,
    onProgress,
    onModelProgress,
  } = config;

  onProgress?.("Loading tokenizer…", 0, 0);
  const tokenizer = await AutoTokenizer.from_pretrained(modelId);

  onProgress?.("Loading model…", 0, 0);
  const model = await AutoModelForCausalLM.from_pretrained(modelId, {
    dtype,
    progress_callback: onModelProgress,
  } as any);

  const byteTok = new ByteTokenizer(tokenizer);
  onProgress?.(`Tokenizer: ${byteTok.type}, ${byteTok.coverage}/256 byte tokens`, 0, 0);

  const grouped = groupByLang(evalSet.char_frequencies, languages);
  const langResults: Record<string, HFLangResult> = {};

  let done = 0;
  let total = 0;
  for (const [, chars] of grouped) total += Math.min(trialSamples ?? chars.length, chars.length);

  // Level 0
  for (const [lang, chars] of grouped) {
    if (chars.length === 0) continue;
    const subset = trialSamples ? chars.slice(0, trialSamples) : chars;
    const results: HFCharResult[] = [];

    for (const entry of subset) {
      const charBytes = charToBytes(entry.char);
      const pfx = Math.min(prefixLength, charBytes.length);
      const { prefixIds, remainingBytes } = byteTok.getCharPrefixTokens(entry.char, pfx);

      // Generate
      const inputTensor = toTensor(prefixIds);
      const output = await model.generate({
        inputs: inputTensor,
        generation_config: { max_new_tokens: maxNewTokens, do_sample: false },
      } as any);
      const allGenIds = tensorToIds((output as any).sequences ?? output as any);
      const genIds = allGenIds.slice(prefixIds.length);

      // Generated bytes and DFA score
      const genBytes = byteTok.tokenIdsToBytes(genIds);
      const dfa = computeDualScore(genBytes);

      // Gold remaining token IDs
      const goldTokenIds = byteTok.bytesToTokenIds(remainingBytes);

      // Perplexity
      const ppl = await computePerplexity(model, allGenIds);

      // Likelihood comparison
      const goldLL = await computeSequenceLL(model, prefixIds, goldTokenIds);
      const genLL = await computeSequenceLL(model, prefixIds, genIds);
      const deltaLL = goldLL - genLL;

      results.push({
        char: entry.char,
        codepoint: entry.codepoint,
        tier: entry.tier,
        language: lang,
        prefixLength: pfx,
        generatedTokenIds: genIds,
        generatedBytes: genBytes,
        goldBytes: remainingBytes,
        dfa,
        perplexity: ppl,
        goldLogLikelihood: goldLL,
        genLogLikelihood: genLL,
        deltaLL,
      });

      done++;
      onProgress?.(
        `L0 ${lang} ${entry.char} (${entry.tier}) V=${dfa.partialScore.toFixed(3)} ppl=${ppl.toFixed(1)}`,
        done,
        total
      );
    }

    const byTier: Record<string, HFTierStats> = {};
    const tierGroups = new Map<string, HFCharResult[]>();
    for (const r of results) {
      if (!tierGroups.has(r.tier)) tierGroups.set(r.tier, []);
      tierGroups.get(r.tier)!.push(r);
    }
    for (const [tier, tResults] of tierGroups) byTier[tier] = aggregateStats(tResults);

    langResults[lang] = {
      language: lang,
      totalCharacters: results.length,
      overallStats: aggregateStats(results),
      byTier,
      results,
    };
  }

  // Level 1
  if (level1Data) {
    for (const [lang, samples] of Object.entries(level1Data)) {
      if (!langResults[lang]) continue;
      const subset = trialSamples ? samples.slice(0, trialSamples) : samples;
      const l1Results: HFL1Result[] = [];

      for (let si = 0; si < subset.length; si++) {
        const sample = subset[si];
        const term = sample.term;
        const prompt = sample.sentence_prompt;
        if (!term || !prompt) continue;

        const termBytes = new TextEncoder().encode(term);
        const pfx = Math.min(prefixLength, termBytes.length);

        // Split prompt into context (before term) and term prefix
        const context = prompt.endsWith(term) ? prompt.slice(0, -term.length) : prompt;
        const contextEncoded = tokenizer(context, { add_special_tokens: false });
        const contextIds: number[] = tensorToIds(contextEncoded.input_ids);

        const { prefixIds: bytePrefixIds, remainingBytes } = byteTok.getCharPrefixTokens(term, pfx);
        // For SentencePiece with space prefix, strip the leading space token if context already provides it
        const promptIds = [...contextIds, ...bytePrefixIds];

        // Generate
        const inputTensor = toTensor(promptIds);
        const output = await model.generate({
          inputs: inputTensor,
          generation_config: { max_new_tokens: maxNewTokens, do_sample: false },
        } as any);
        const allGenIds = tensorToIds((output as any).sequences ?? output as any);
        const genIds = allGenIds.slice(promptIds.length);

        // Generated bytes and DFA
        const genBytes = byteTok.tokenIdsToBytes(genIds);
        const dfa = computeDualScore(genBytes);

        // Term match: generated bytes start with remaining bytes of target
        const termMatch = remainingBytes.length > 0 &&
          genBytes.length >= remainingBytes.length &&
          remainingBytes.every((b, i) => genBytes[i] === b);

        // Gold token IDs for remaining bytes
        const goldTokenIds = byteTok.bytesToTokenIds(remainingBytes);

        // Perplexity and likelihood
        const ppl = await computePerplexity(model, allGenIds);
        const goldLL = await computeSequenceLL(model, promptIds, goldTokenIds);
        const genLL = await computeSequenceLL(model, promptIds, genIds);

        l1Results.push({
          term,
          language: lang,
          prefixLength: pfx,
          generatedTokenIds: genIds,
          generatedBytes: genBytes,
          goldBytes: remainingBytes,
          dfa,
          termMatch,
          perplexity: ppl,
          goldLogLikelihood: goldLL,
          genLogLikelihood: genLL,
          deltaLL: goldLL - genLL,
        });

        onProgress?.(
          `L1 ${lang} ${term} match=${termMatch} V=${dfa.partialScore.toFixed(3)}`,
          done + si + 1,
          total + subset.length
        );
      }

      langResults[lang].level1 = {
        totalSamples: l1Results.length,
        stats: aggregateStats(l1Results),
        results: l1Results,
      };
    }
  }

  return {
    model: modelId,
    timestamp: new Date().toISOString(),
    config: { model: modelId, dtype, prefixLength, maxNewTokens, trialSamples, languages },
    byteCoverage: byteTok.coverage,
    tokenizerType: byteTok.type,
    languages: langResults,
  };
}

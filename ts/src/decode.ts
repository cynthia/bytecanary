/**
 * Byte-level tokenizer wrapper for Transformers.js tokenizers.
 * Auto-detects SentencePiece (<0xNN>) and GPT-2 BPE byte token formats.
 */

const SPIECE_UNDERLINE = "▁";

function bytesToUnicode(): Map<number, string> {
  const bs: number[] = [];
  for (let i = 0x21; i <= 0x7e; i++) bs.push(i);
  for (let i = 0xa1; i <= 0xac; i++) bs.push(i);
  for (let i = 0xae; i <= 0xff; i++) bs.push(i);
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const result = new Map<number, string>();
  for (let i = 0; i < bs.length; i++) result.set(bs[i], String.fromCodePoint(cs[i]));
  return result;
}

export type TokenizerType = "sentencepiece" | "gpt2";

export class ByteTokenizer {
  type: TokenizerType;
  byteToId: Map<number, number> = new Map();
  idToByte: Map<number, number> = new Map();
  addsSpacePrefix: boolean;
  private unicodeToByte: Map<string, number> | null = null;
  private tokenizer: any;

  constructor(tokenizer: any) {
    this.tokenizer = tokenizer;
    const vocab = this.getVocab();
    this.type = this.detectType(vocab);
    this.buildMaps(vocab);
    if (this.type === "gpt2") {
      this.unicodeToByte = new Map<string, number>();
      for (const [b, u] of bytesToUnicode()) this.unicodeToByte.set(u, b);
    }
    this.addsSpacePrefix = this.detectSpacePrefix();
  }

  private getVocab(): Map<string, number> {
    const model = this.tokenizer.model;
    if (model?.tokens_to_ids instanceof Map) return model.tokens_to_ids;
    if (model?.vocab) {
      if (model.vocab instanceof Map) return model.vocab;
      if (Array.isArray(model.vocab)) {
        const m = new Map<string, number>();
        for (let i = 0; i < model.vocab.length; i++) m.set(model.vocab[i], i);
        return m;
      }
      if (typeof model.vocab === "object") {
        return new Map(
          Object.entries(model.vocab).map(([k, v]) => [k, Number(v)])
        );
      }
    }
    // Fallback: try the tokenizer-level added_tokens_encoder
    if (this.tokenizer.added_tokens_encoder instanceof Map)
      return this.tokenizer.added_tokens_encoder;
    throw new Error("Cannot access tokenizer vocabulary");
  }

  private detectType(vocab: Map<string, number>): TokenizerType {
    if (vocab.has("<0x00>")) return "sentencepiece";
    const b2u = bytesToUnicode();
    if (vocab.has(b2u.get(0)!)) return "gpt2";
    throw new Error(
      "Tokenizer has neither SentencePiece byte tokens (<0xNN>) nor GPT-2 byte-level BPE."
    );
  }

  private buildMaps(vocab: Map<string, number>): void {
    if (this.type === "sentencepiece") {
      for (let b = 0; b < 256; b++) {
        const key = `<0x${b.toString(16).toUpperCase().padStart(2, "0")}>`;
        const id = vocab.get(key);
        if (id !== undefined) {
          this.byteToId.set(b, id);
          this.idToByte.set(id, b);
        }
      }
    } else {
      const b2u = bytesToUnicode();
      for (const [b, u] of b2u) {
        const id = vocab.get(u);
        if (id !== undefined) {
          this.byteToId.set(b, id);
          this.idToByte.set(id, b);
        }
      }
    }
  }

  private detectSpacePrefix(): boolean {
    try {
      const encoded = this.tokenizer.encode("A", { add_special_tokens: false });
      const ids: number[] = Array.isArray(encoded)
        ? encoded
        : Array.from(encoded).map(Number);
      const raw = this.tokenIdsToBytes(ids);
      return raw.length > 0 && raw[0] === 0x20;
    } catch {
      return false;
    }
  }

  bytesToTokenIds(data: Uint8Array): number[] {
    const ids: number[] = [];
    for (const b of data) {
      const id = this.byteToId.get(b);
      if (id === undefined) throw new Error(`No token for byte 0x${b.toString(16)}`);
      ids.push(id);
    }
    return ids;
  }

  tokenIdsToBytes(tokenIds: number[]): Uint8Array {
    if (this.type === "sentencepiece") return this.spieceToBytes(tokenIds);
    return this.gpt2ToBytes(tokenIds);
  }

  private spieceToBytes(tokenIds: number[]): Uint8Array {
    const result: number[] = [];
    const encoder = new TextEncoder();
    for (const id of tokenIds) {
      const b = this.idToByte.get(id);
      if (b !== undefined) {
        result.push(b);
        continue;
      }
      // Non-byte token — decode via tokenizer
      const text: string = this.tokenizer.decode([id], {
        skip_special_tokens: false,
      });
      if (!text) continue;
      if (text === SPIECE_UNDERLINE) {
        result.push(0x20);
      } else if (text.startsWith(SPIECE_UNDERLINE)) {
        result.push(0x20);
        for (const byte of encoder.encode(text.slice(1))) result.push(byte);
      } else {
        for (const byte of encoder.encode(text)) result.push(byte);
      }
    }
    return new Uint8Array(result);
  }

  private gpt2ToBytes(tokenIds: number[]): Uint8Array {
    const result: number[] = [];
    const encoder = new TextEncoder();
    for (const id of tokenIds) {
      const b = this.idToByte.get(id);
      if (b !== undefined) {
        result.push(b);
        continue;
      }
      const text: string = this.tokenizer.decode([id], {
        skip_special_tokens: false,
      });
      if (!text) continue;
      for (const ch of text) {
        const mapped = this.unicodeToByte?.get(ch);
        if (mapped !== undefined) {
          result.push(mapped);
        } else {
          for (const byte of encoder.encode(ch)) result.push(byte);
        }
      }
    }
    return new Uint8Array(result);
  }

  getCharPrefixTokens(
    char: string,
    numBytes: number
  ): { prefixIds: number[]; prefixBytes: Uint8Array; remainingBytes: Uint8Array } {
    const encoder = new TextEncoder();
    const charBytes = encoder.encode(char);
    const prefixBytes = charBytes.slice(0, numBytes);
    const remainingBytes = charBytes.slice(numBytes);

    const inputBytes = this.addsSpacePrefix
      ? new Uint8Array([0x20, ...prefixBytes])
      : prefixBytes;

    return {
      prefixIds: this.bytesToTokenIds(inputBytes),
      prefixBytes,
      remainingBytes,
    };
  }

  get coverage(): number {
    return this.byteToId.size;
  }
}

# ByteCanary

Evaluation code for the paper *Beyond Perplexity: UTF-8 Validity in Byte-aware Language Models*.

Byte-fallback tokenization allows language models to encode arbitrary Unicode, but generation can produce invalid UTF-8 byte sequences. Perplexity does not track this. ByteCanary measures UTF-8 structural validity separately using a DFA-based metric.

## Install

```
pip install -e .
```

Requires Python 3.10+, PyTorch 2.0+, Transformers 4.40+.

## Usage

Works with any HuggingFace model that uses `AutoModelForCausalLM` / `AutoTokenizer`. Auto-detects SentencePiece (`<0xNN>`) and GPT-2 BPE byte token formats.

```sh
# Level 0 with bundled eval set (4000 CJK characters)
bytecanary meta-llama/Llama-3.2-1B

# Trial mode (256 samples per language)
bytecanary meta-llama/Llama-3.2-1B --trial

# Custom eval set
bytecanary google/gemma-2-2b --eval-set data/level0/cjk_unseen3b.json

# Level 0 + Level 1
bytecanary meta-llama/Llama-3.2-1B --level1-data data/level1

# Level 1 only
bytecanary meta-llama/Llama-3.2-1B --level 1 --level1-data data/level1

# Device and dtype
bytecanary mistralai/Mistral-7B-v0.1 --device cuda:1 --dtype bfloat16
```

Results are written to `bytecanary_results/` as JSON.

## Data

**Bundled (included in pip package):**

- `cjk_4000.json` — 4000 CJK characters, 1000 per tier (Common, Uncommon, Rare, Unseen)

**Repo only (not in pip package):**

- `data/level0/cjk_400.json` — 400-character subset
- `data/level0/cjk_unseen3b.json` — 139 unseen 3-byte CJK ideographs
- `data/level1/{ja,ko,zh}/` — 3000 synthetic OOV samples per language (generated with Gemini 3 Pro)

## Reproducing paper results

The paper evaluates a 355M-parameter model with an 8K BPE vocabulary and byte-fallback, trained on 80B tokens (10% EN, 30% each JA/KO/ZH). To reproduce:

```sh
# Level 0 (context-free structural validity)
bytecanary <path-to-checkpoint>

# Level 0 + Level 1 (context-guided byte retrieval)
bytecanary <path-to-checkpoint> --level1-data data/level1
```

The output JSON contains per-tier and per-language breakdowns for V_partial, V_binary_strict, V_binary_soft, perplexity, likelihood comparison, and (for Level 1) Term Match Rate.

## TypeScript / browser

`ts/` contains a TypeScript port of the DFA and scoring, plus an evaluator using the browser Prompt API (`window.LanguageModel`).

```sh
cd ts && npm install && npm run build && npm run serve
```

The browser evaluator scores UTF-8 bytes of generated text through the same DFA. It cannot do the byte-prefix protocol (the Prompt API is text-in/text-out) or compute perplexity. DFA metrics and per-tier breakdowns are comparable to the paper; byte-prefix generation, perplexity, and likelihood comparison are not.

## License

Apache-2.0

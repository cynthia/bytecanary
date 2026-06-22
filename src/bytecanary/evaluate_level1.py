"""Level 1 evaluation: context-guided OOV character completion with term matching."""

import gc
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from .config import EvalConfig
from .decode import ByteTokenizer
from .utils import (
    batch_calculate_log_likelihoods,
    batch_generate,
    calculate_perplexity,
    check_utf8_validity_incremental,
)


class Level1Evaluator:
    def __init__(self, model, tokenizer, byte_tok: ByteTokenizer, config: EvalConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.byte_tok = byte_tok
        self.config = config
        self.device = next(model.parameters()).device
        self.output_dir = Path(config.output_dir) / "level1"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> Dict[str, List[Dict]]:
        if not self.config.level1_data:
            raise ValueError("Level 1 requires --level1-data pointing to the synthetic data directory")
        base = Path(self.config.level1_data)
        result = {}
        for lang in self.config.languages:
            path = base / lang / "synthetic_oov_samples_final.json"
            if not path.exists():
                print(f"Warning: no data for {lang} at {path}")
                result[lang] = []
                continue
            with open(path, "r", encoding="utf-8") as f:
                samples = json.load(f)
            result[lang] = samples
            print(f"Loaded {len(samples)} samples for {lang}")
        return result

    def run(self) -> Dict[str, Any]:
        print("=" * 72)
        print("ByteCanary Level 1: Context-Guided OOV Completion")
        print("=" * 72)

        data = self.load_data()

        if self.config.trial:
            print(f"Trial mode: {self.config.trial_samples} samples/language")
            for lang in data:
                data[lang] = data[lang][: self.config.trial_samples]

        results = {}
        for lang, samples in data.items():
            if not samples:
                continue
            results[lang] = self._evaluate_language(lang, samples)

        self._save(results)
        return results

    def _evaluate_language(self, language: str, samples: List[Dict]) -> Dict[str, Any]:
        print(f"\nEvaluating {language}: {len(samples)} samples")

        eval_by_prefix: Dict[int, list] = {}

        for sample in tqdm(samples, desc=f"Preparing {language}"):
            term = sample.get("term", "")
            prompt = sample.get("sentence_prompt", "")
            if not term or not prompt:
                continue

            term_bytes = term.encode("utf-8")
            byte_length = len(term_bytes)

            if prompt.endswith(term):
                context = prompt[: -len(term)]
            else:
                context = prompt
            context_ids = self.tokenizer.encode(context, add_special_tokens=False)

            for pfx in range(1, byte_length + 1):
                if pfx >= byte_length:
                    prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                    gold_bytes = b""
                else:
                    prefix_ids, _, gold_bytes = self.byte_tok.get_char_prefix_tokens(term, pfx)
                    prompt_ids = context_ids + prefix_ids

                if not prompt_ids:
                    continue

                gold_ids = []
                if gold_bytes:
                    try:
                        gt = gold_bytes.decode("utf-8", errors="ignore")
                        if gt:
                            gold_ids = self.tokenizer.encode(gt, add_special_tokens=False)
                    except Exception:
                        pass

                eval_by_prefix.setdefault(pfx, []).append({
                    "sample": sample,
                    "prompt_ids": prompt_ids,
                    "gold_ids": gold_ids,
                    "gold_bytes": gold_bytes,
                    "total_bytes": byte_length,
                    "actual_prefix_bytes": pfx,
                })

        results_by_prefix = {}
        perplexity_results = {}
        prefix_modes = {}

        for pfx, evals in eval_by_prefix.items():
            print(f"\n  Prefix {pfx}: {len(evals)} samples")

            prompt_list = [e["prompt_ids"] for e in evals]
            gold_list = [e["gold_ids"] for e in evals]

            print(f"    Generating (batch={self.config.batch_size})...")
            gen_list = batch_generate(
                self.model, self.tokenizer, prompt_list, self.device,
                max_new_tokens=self.config.max_new_tokens,
                batch_size=self.config.batch_size,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )

            print("    Log likelihoods...")
            gold_ll = batch_calculate_log_likelihoods(
                self.model, self.tokenizer, prompt_list, gold_list, self.device,
                batch_size=self.config.batch_size,
            )
            gen_ll = batch_calculate_log_likelihoods(
                self.model, self.tokenizer, prompt_list, gen_list, self.device,
                batch_size=self.config.batch_size,
            )

            pfx_results = []
            for e, gids, gll, ell in zip(evals, gen_list, gold_ll, gen_ll):
                incr = check_utf8_validity_incremental(
                    self.byte_tok, e["prompt_ids"], gids,
                )

                match = False
                gb = e["gold_bytes"]
                if incr and gb:
                    for ir in incr:
                        gen_tokens = ir["token_ids"][len(e["prompt_ids"]):]
                        gen_bytes = self.byte_tok.token_ids_to_bytes(gen_tokens)
                        if gen_bytes.startswith(gb):
                            match = True
                            break

                pfx_results.append({
                    "sample_term": e["sample"].get("term", ""),
                    "sample_language": e["sample"].get("language", language),
                    "prefix_bytes": e["actual_prefix_bytes"],
                    "total_bytes": e["total_bytes"],
                    "gold_bytes": gb.hex() if gb else "",
                    "gold_token_ids": e["gold_ids"],
                    "gold_log_likelihood": gll,
                    "generated_log_likelihood": ell,
                    "prompt_length": len(e["prompt_ids"]),
                    "initial_token_ids": e["prompt_ids"],
                    "generated_token_ids": gids,
                    "incremental_results": incr,
                    "term_match": match,
                    "error": None,
                })

            results_by_prefix[pfx] = pfx_results
            del evals, prompt_list, gold_list, gen_list, gold_ll, gen_ll
            gc.collect()

        for pfx, results in results_by_prefix.items():
            stats = _aggregate_l1(results, self.model, self.tokenizer, self.device, self.config.batch_size)
            perplexity_results[pfx] = stats["perplexity"]
            prefix_modes[pfx] = stats["mode"]

        return {
            "language": language,
            "total_samples": len(samples),
            "results_by_prefix": results_by_prefix,
            "perplexity_by_prefix": perplexity_results,
            "prefix_modes": prefix_modes,
        }

    def _save(self, results: Dict[str, Any]):
        summary = {
            "level": "level1",
            "config": {
                "model": self.config.model,
                "trial": self.config.trial,
                "max_new_tokens": self.config.max_new_tokens,
                "batch_size": self.config.batch_size,
            },
            "languages": {},
        }
        for lang, lr in results.items():
            summary["languages"][lang] = {
                "total_samples": lr["total_samples"],
                "perplexity_by_prefix": lr["perplexity_by_prefix"],
                "prefix_modes": lr["prefix_modes"],
            }

        path = self.output_dir / "summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSummary: {path}")

        if self.config.save_detailed:
            stripped = {}
            for lang, lr in results.items():
                lc = dict(lr)
                new_rbp = {}
                for pfx, pr in lc.get("results_by_prefix", {}).items():
                    new_rbp[pfx] = [{k: v for k, v in r.items() if k != "incremental_results"} for r in pr]
                lc["results_by_prefix"] = new_rbp
                stripped[lang] = lc
            dp = self.output_dir / "detailed_results.json"
            with open(dp, "w", encoding="utf-8") as f:
                json.dump(stripped, f, indent=2, ensure_ascii=False)
            print(f"Details: {dp}")

        _print_l1_summary(summary)


def _aggregate_l1(results, model, tokenizer, device, batch_size):
    token_seqs = []
    utf8_by_step = {i: [] for i in range(1, 6)}
    utf8_by_step_pure = {i: [] for i in range(1, 6)}
    utf8_by_step_mixed = {i: [] for i in range(1, 6)}
    dfa_partial_by_step = {i: [] for i in range(1, 6)}
    dfa_strict_by_step = {i: [] for i in range(1, 6)}
    dfa_soft_by_step = {i: [] for i in range(1, 6)}
    ll_diffs = []
    term_matches = []
    cum_partial = []
    cum_binary = []
    ema_vals = []
    total = 0
    errors = 0

    for r in results:
        total += 1
        if r.get("error"):
            errors += 1
            continue

        incr = r.get("incremental_results", [])
        if not incr:
            continue

        token_seqs.append(incr[-1]["token_ids"])

        gll = r.get("gold_log_likelihood")
        ell = r.get("generated_log_likelihood")
        if isinstance(gll, dict) and isinstance(ell, dict):
            gv = gll.get("total_log_likelihood")
            ev = ell.get("total_log_likelihood")
            if gv is not None and ev is not None:
                ll_diffs.append(gv - ev)

        generated_ids = r.get("generated_token_ids", [])
        step_ps = []
        step_bs = []
        ema = None
        for sr in incr:
            sn = sr["num_tokens"]
            if sn not in utf8_by_step:
                continue
            v = 1 if sr["is_valid_utf8"] else 0
            utf8_by_step[sn].append(v)

            tokens_so_far = generated_ids[:sn] if generated_ids else []
            is_pure = all(tid < 256 for tid in tokens_so_far) if tokens_so_far else False
            if is_pure:
                utf8_by_step_pure[sn].append(v)
            else:
                utf8_by_step_mixed[sn].append(v)

            ps = sr.get("dfa_partial_score", 0.0)
            bs = sr.get("dfa_binary_strict", 0.0)
            bsoft = sr.get("dfa_binary_soft", 0.0)
            dfa_partial_by_step[sn].append(ps)
            dfa_strict_by_step[sn].append(bs)
            dfa_soft_by_step[sn].append(bsoft)

            step_ps.append(ps)
            step_bs.append(bs)
            ema = ps if ema is None else 0.1 * ps + 0.9 * ema

        if step_ps:
            cum_partial.append(sum(step_ps) / len(step_ps))
            cum_binary.append(sum(step_bs) / len(step_bs))
            ema_vals.append(ema or 0.0)

        term_matches.append(1 if r.get("term_match", False) else 0)

    def _m(lst):
        return sum(lst) / len(lst) if lst else 0.0

    ppl = calculate_perplexity(model, tokenizer, token_seqs, device, batch_size) if token_seqs else {
        "perplexity": float("inf"), "avg_loss": float("inf"), "num_sequences": 0, "total_tokens": 0
    }

    if not ll_diffs:
        ll_stats = {
            "gold_better_count": 0, "generated_better_count": 0, "equal_count": 0,
            "total_comparisons": 0, "gold_win_rate": 0.0,
            "avg_log_likelihood_diff": 0.0, "median_log_likelihood_diff": 0.0, "stdev_log_likelihood_diff": 0.0,
        }
    else:
        gb = sum(1 for d in ll_diffs if d > 0)
        gn = sum(1 for d in ll_diffs if d < 0)
        eq = sum(1 for d in ll_diffs if d == 0)
        ll_stats = {
            "gold_better_count": gb, "generated_better_count": gn, "equal_count": eq,
            "total_comparisons": len(ll_diffs),
            "gold_win_rate": gb / len(ll_diffs),
            "avg_log_likelihood_diff": statistics.mean(ll_diffs),
            "median_log_likelihood_diff": statistics.median(ll_diffs),
            "stdev_log_likelihood_diff": statistics.stdev(ll_diffs) if len(ll_diffs) > 1 else 0.0,
        }

    mode = {
        "total_samples": total,
        "errors": errors,
        "utf8_validity_rates": {f"step_{s}": _m(utf8_by_step[s]) for s in range(1, 6)},
        "utf8_validity_rates_pure": {f"step_{s}": _m(utf8_by_step_pure[s]) for s in range(1, 6)},
        "utf8_validity_rates_mixed": {f"step_{s}": _m(utf8_by_step_mixed[s]) for s in range(1, 6)},
        "term_match_rate": _m(term_matches),
        "dfa_partial_rates": {f"step_{s}": _m(dfa_partial_by_step[s]) for s in range(1, 6)},
        "dfa_binary_strict_rates": {f"step_{s}": _m(dfa_strict_by_step[s]) for s in range(1, 6)},
        "dfa_binary_soft_rates": {f"step_{s}": _m(dfa_soft_by_step[s]) for s in range(1, 6)},
        "dfa_cumulative_partial": _m(cum_partial),
        "dfa_cumulative_binary": _m(cum_binary),
        "dfa_ema_partial": _m(ema_vals),
        "likelihood_comparison": ll_stats,
    }

    return {"perplexity": ppl, "mode": mode}


def _print_l1_summary(summary):
    print("\n" + "=" * 72)
    print("LEVEL 1 SUMMARY")
    print("=" * 72)
    for lang, ls in summary["languages"].items():
        print(f"\n{lang.upper()}: {ls['total_samples']} samples")
        for pfx, ms in ls["prefix_modes"].items():
            print(f"  Prefix {pfx}: {ms['total_samples']} samples, {ms['errors']} errors")
            ppl = ls["perplexity_by_prefix"].get(pfx, {})
            print(f"    Perplexity: {ppl.get('perplexity', float('inf')):.2f}")
            print(f"    Term Match: {ms['term_match_rate']:.2%}")
            for step in ["step_1", "step_2", "step_3", "step_4", "step_5"]:
                v = ms["utf8_validity_rates"].get(step, 0)
                d = ms["dfa_partial_rates"].get(step, 0)
                print(f"    {step}: validity={v:.2%}  DFA_partial={d:.4f}")

"""Level 0 evaluation: single OOV character completion with DFA scoring."""

import gc
import json
import statistics
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .config import EvalConfig
from .decode import ByteTokenizer
from .utils import (
    batch_calculate_log_likelihoods,
    batch_generate,
    calculate_perplexity,
    check_utf8_validity_incremental,
)


def _bundled_eval_set() -> str:
    ref = resources.files("bytecanary") / "data" / "cjk_4000.json"
    with resources.as_file(ref) as p:
        return str(p)


def detect_language(char: str) -> str:
    if not char:
        return "other"
    cp = ord(char[0])
    if 0x3040 <= cp <= 0x30FF:
        return "ja"
    if 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
        return "ko"
    if (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF):
        return "zh"
    return "other"


class Level0Evaluator:
    def __init__(self, model, tokenizer, byte_tok: ByteTokenizer, config: EvalConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.byte_tok = byte_tok
        self.config = config
        self.device = next(model.parameters()).device
        self.output_dir = Path(config.output_dir) / "level0"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_eval_set(self) -> List[Dict[str, Any]]:
        path = self.config.eval_set or _bundled_eval_set()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        chars = data.get("char_frequencies", [])
        tiers = {}
        for c in chars:
            t = c.get("tier", "unknown")
            tiers[t] = tiers.get(t, 0) + 1
        print(f"Loaded {len(chars)} characters from {Path(path).name}")
        print(f"  Tiers: {tiers}")
        return chars

    def run(self) -> Dict[str, Any]:
        print("=" * 72)
        print("ByteCanary Level 0: Single OOV Character Completion")
        print("=" * 72)

        chars = self.load_eval_set()
        groups = self._group_by_language(chars)

        if self.config.trial:
            print(f"Trial mode: {self.config.trial_samples} samples/language")
            for lang in groups:
                groups[lang] = groups[lang][: self.config.trial_samples]

        results = {}
        for lang, char_list in groups.items():
            if not char_list:
                continue
            results[lang] = self._evaluate_language(lang, char_list)

        self._save(results)
        return results

    def _group_by_language(self, chars):
        groups = {lang: [] for lang in self.config.languages}
        groups["other"] = []
        for ci in chars:
            lang = detect_language(ci["char"])
            if lang in groups:
                groups[lang].append(ci)
            else:
                groups["other"].append(ci)
        return groups

    def _evaluate_language(self, language: str, char_list: List[Dict]) -> Dict[str, Any]:
        print(f"\nEvaluating {language}: {len(char_list)} characters")

        samples_by_prefix: Dict[int, list] = {}

        for ci in tqdm(char_list, desc=f"Preparing {language}"):
            char = ci["char"]
            char_bytes = char.encode("utf-8")
            byte_length = len(char_bytes)

            for pfx in range(1, byte_length + 1):
                prefix_ids, prefix_bytes, remaining = self.byte_tok.get_char_prefix_tokens(char, pfx)
                if not prefix_ids:
                    continue

                gold_ids = []
                if remaining:
                    try:
                        gold_text = remaining.decode("utf-8", errors="ignore")
                        if gold_text:
                            gold_ids = self.tokenizer.encode(gold_text, add_special_tokens=False)
                    except Exception:
                        pass

                samples_by_prefix.setdefault(pfx, []).append({
                    "char_info": ci,
                    "prefix_token_ids": prefix_ids,
                    "gold_token_ids": gold_ids,
                    "gold_bytes": remaining,
                    "total_bytes": byte_length,
                    "actual_prefix_bytes": pfx,
                })

        results_by_prefix = {}
        perplexity_results = {}
        prefix_modes = {}
        perplexity_by_tier = {}
        prefix_modes_by_tier = {}

        for pfx, samples in samples_by_prefix.items():
            print(f"\n  Prefix {pfx}: {len(samples)} samples")

            prompt_list = [s["prefix_token_ids"] for s in samples]
            gold_list = [s["gold_token_ids"] for s in samples]

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
            for s, gids, gll, ell in zip(samples, gen_list, gold_ll, gen_ll):
                incr = check_utf8_validity_incremental(
                    self.byte_tok, s["prefix_token_ids"], gids,
                )
                pfx_results.append({
                    "char": s["char_info"]["char"],
                    "codepoint": s["char_info"].get("codepoint", ord(s["char_info"]["char"])),
                    "frequency": s["char_info"].get("count", 0),
                    "tier": s["char_info"].get("tier", "unknown"),
                    "prefix_bytes": s["actual_prefix_bytes"],
                    "total_bytes": s["total_bytes"],
                    "gold_bytes": s["gold_bytes"].hex() if s["gold_bytes"] else "",
                    "gold_token_ids": s["gold_token_ids"],
                    "gold_log_likelihood": gll,
                    "generated_log_likelihood": ell,
                    "initial_token_ids": s["prefix_token_ids"],
                    "generated_token_ids": gids,
                    "incremental_results": incr,
                    "error": None,
                })

            results_by_prefix[pfx] = pfx_results
            del samples, prompt_list, gold_list, gen_list, gold_ll, gen_ll
            gc.collect()

        for pfx, results in results_by_prefix.items():
            stats = _aggregate_stats(results, self.model, self.tokenizer, self.device, self.config.batch_size)
            perplexity_results[pfx] = stats["perplexity"]
            prefix_modes[pfx] = stats["mode"]
            perplexity_by_tier[pfx] = stats["tier_perplexity"]
            prefix_modes_by_tier[pfx] = stats["tier_modes"]

        return {
            "language": language,
            "total_characters": len(char_list),
            "results_by_prefix": results_by_prefix,
            "perplexity_by_prefix": perplexity_results,
            "prefix_modes": prefix_modes,
            "perplexity_by_tier": perplexity_by_tier,
            "prefix_modes_by_tier": prefix_modes_by_tier,
        }

    def _save(self, results: Dict[str, Any]):
        summary = {
            "level": "level0",
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
                "total_characters": lr["total_characters"],
                "perplexity_by_prefix": lr["perplexity_by_prefix"],
                "prefix_modes": lr["prefix_modes"],
                "perplexity_by_tier": lr.get("perplexity_by_tier", {}),
                "prefix_modes_by_tier": lr.get("prefix_modes_by_tier", {}),
            }

        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSummary: {summary_path}")

        if self.config.save_detailed:
            stripped = {}
            for lang, lr in results.items():
                lang_copy = dict(lr)
                new_rbp = {}
                for pfx, pfx_results in lang_copy.get("results_by_prefix", {}).items():
                    new_rbp[pfx] = [{k: v for k, v in r.items() if k != "incremental_results"} for r in pfx_results]
                lang_copy["results_by_prefix"] = new_rbp
                stripped[lang] = lang_copy

            detail_path = self.output_dir / "detailed_results.json"
            with open(detail_path, "w", encoding="utf-8") as f:
                json.dump(stripped, f, indent=2, ensure_ascii=False)
            print(f"Details: {detail_path}")

        _print_summary(summary)


def _aggregate_stats(results, model, tokenizer, device, batch_size):
    token_seqs = []
    utf8_by_step = {i: [] for i in range(1, 6)}
    utf8_by_step_pure = {i: [] for i in range(1, 6)}
    utf8_by_step_mixed = {i: [] for i in range(1, 6)}
    dfa_partial_by_step = {i: [] for i in range(1, 6)}
    dfa_strict_by_step = {i: [] for i in range(1, 6)}
    dfa_soft_by_step = {i: [] for i in range(1, 6)}
    ll_diffs = []
    cum_partial = []
    cum_binary = []
    ema_vals = []
    total = 0
    errors = 0

    tier_data: Dict[str, dict] = {}

    for r in results:
        total += 1
        tier = r.get("tier", "unknown")
        if tier not in tier_data:
            tier_data[tier] = _empty_tier()
        tier_data[tier]["total"] += 1

        if r.get("error"):
            errors += 1
            tier_data[tier]["errors"] += 1
            continue

        incr = r.get("incremental_results", [])
        if not incr:
            continue

        token_seqs.append(incr[-1]["token_ids"])
        tier_data[tier]["token_seqs"].append(incr[-1]["token_ids"])

        gll = r.get("gold_log_likelihood")
        ell = r.get("generated_log_likelihood")
        if isinstance(gll, dict) and isinstance(ell, dict):
            gv = gll.get("total_log_likelihood")
            ev = ell.get("total_log_likelihood")
            if gv is not None and ev is not None:
                ll_diffs.append(gv - ev)
                tier_data[tier]["ll_diffs"].append(gv - ev)

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
            tier_data[tier]["utf8_by_step"][sn].append(v)

            tokens_so_far = generated_ids[:sn] if generated_ids else []
            is_pure = all(tid < 256 for tid in tokens_so_far) if tokens_so_far else False
            if is_pure:
                utf8_by_step_pure[sn].append(v)
                tier_data[tier]["utf8_by_step_pure"][sn].append(v)
            else:
                utf8_by_step_mixed[sn].append(v)
                tier_data[tier]["utf8_by_step_mixed"][sn].append(v)

            ps = sr.get("dfa_partial_score", 0.0)
            bs = sr.get("dfa_binary_strict", 0.0)
            bsoft = sr.get("dfa_binary_soft", 0.0)
            dfa_partial_by_step[sn].append(ps)
            dfa_strict_by_step[sn].append(bs)
            dfa_soft_by_step[sn].append(bsoft)
            tier_data[tier]["dfa_partial"][sn].append(ps)
            tier_data[tier]["dfa_strict"][sn].append(bs)
            tier_data[tier]["dfa_soft"][sn].append(bsoft)

            step_ps.append(ps)
            step_bs.append(bs)
            ema = ps if ema is None else 0.1 * ps + 0.9 * ema

        if step_ps:
            cum_partial.append(sum(step_ps) / len(step_ps))
            cum_binary.append(sum(step_bs) / len(step_bs))
            ema_vals.append(ema or 0.0)
            tier_data[tier]["cum_partial"].append(sum(step_ps) / len(step_ps))
            tier_data[tier]["cum_binary"].append(sum(step_bs) / len(step_bs))
            tier_data[tier]["ema_vals"].append(ema or 0.0)

    ppl = calculate_perplexity(model, tokenizer, token_seqs, device, batch_size) if token_seqs else {
        "perplexity": float("inf"), "avg_loss": float("inf"), "num_sequences": 0, "total_tokens": 0
    }

    mode = {
        "total_samples": total,
        "errors": errors,
        "utf8_validity_rates": {f"step_{s}": _mean(utf8_by_step[s]) for s in range(1, 6)},
        "utf8_validity_rates_pure": {f"step_{s}": _mean(utf8_by_step_pure[s]) for s in range(1, 6)},
        "utf8_validity_rates_mixed": {f"step_{s}": _mean(utf8_by_step_mixed[s]) for s in range(1, 6)},
        "dfa_partial_rates": {f"step_{s}": _mean(dfa_partial_by_step[s]) for s in range(1, 6)},
        "dfa_binary_strict_rates": {f"step_{s}": _mean(dfa_strict_by_step[s]) for s in range(1, 6)},
        "dfa_binary_soft_rates": {f"step_{s}": _mean(dfa_soft_by_step[s]) for s in range(1, 6)},
        "dfa_cumulative_partial": _mean(cum_partial),
        "dfa_cumulative_binary": _mean(cum_binary),
        "dfa_ema_partial": _mean(ema_vals),
        "likelihood_comparison": _ll_stats(ll_diffs),
    }

    tier_ppl = {}
    tier_modes = {}
    for tier, td in tier_data.items():
        tier_ppl[tier] = (
            calculate_perplexity(model, tokenizer, td["token_seqs"], device, batch_size)
            if td["token_seqs"]
            else {"perplexity": float("inf"), "avg_loss": float("inf"), "num_sequences": 0, "total_tokens": 0}
        )
        tier_modes[tier] = {
            "total_samples": td["total"],
            "errors": td["errors"],
            "utf8_validity_rates": {f"step_{s}": _mean(td["utf8_by_step"][s]) for s in range(1, 6)},
            "utf8_validity_rates_pure": {f"step_{s}": _mean(td["utf8_by_step_pure"][s]) for s in range(1, 6)},
            "utf8_validity_rates_mixed": {f"step_{s}": _mean(td["utf8_by_step_mixed"][s]) for s in range(1, 6)},
            "dfa_partial_rates": {f"step_{s}": _mean(td["dfa_partial"][s]) for s in range(1, 6)},
            "dfa_binary_strict_rates": {f"step_{s}": _mean(td["dfa_strict"][s]) for s in range(1, 6)},
            "dfa_binary_soft_rates": {f"step_{s}": _mean(td["dfa_soft"][s]) for s in range(1, 6)},
            "dfa_cumulative_partial": _mean(td["cum_partial"]),
            "dfa_cumulative_binary": _mean(td["cum_binary"]),
            "dfa_ema_partial": _mean(td["ema_vals"]),
            "likelihood_comparison": _ll_stats(td["ll_diffs"]),
        }

    return {"perplexity": ppl, "mode": mode, "tier_perplexity": tier_ppl, "tier_modes": tier_modes}


def _empty_tier():
    return {
        "token_seqs": [],
        "utf8_by_step": {i: [] for i in range(1, 6)},
        "utf8_by_step_pure": {i: [] for i in range(1, 6)},
        "utf8_by_step_mixed": {i: [] for i in range(1, 6)},
        "dfa_partial": {i: [] for i in range(1, 6)},
        "dfa_strict": {i: [] for i in range(1, 6)},
        "dfa_soft": {i: [] for i in range(1, 6)},
        "ll_diffs": [],
        "cum_partial": [],
        "cum_binary": [],
        "ema_vals": [],
        "total": 0,
        "errors": 0,
    }


def _mean(lst):
    return sum(lst) / len(lst) if lst else 0.0


def _ll_stats(diffs):
    if not diffs:
        return {
            "gold_better_count": 0,
            "generated_better_count": 0,
            "equal_count": 0,
            "total_comparisons": 0,
            "gold_win_rate": 0.0,
            "avg_log_likelihood_diff": 0.0,
            "median_log_likelihood_diff": 0.0,
            "stdev_log_likelihood_diff": 0.0,
        }
    gold_better = sum(1 for d in diffs if d > 0)
    gen_better = sum(1 for d in diffs if d < 0)
    equal = sum(1 for d in diffs if d == 0)
    return {
        "gold_better_count": gold_better,
        "generated_better_count": gen_better,
        "equal_count": equal,
        "total_comparisons": len(diffs),
        "gold_win_rate": gold_better / len(diffs),
        "avg_log_likelihood_diff": statistics.mean(diffs),
        "median_log_likelihood_diff": statistics.median(diffs),
        "stdev_log_likelihood_diff": statistics.stdev(diffs) if len(diffs) > 1 else 0.0,
    }


def _print_summary(summary):
    print("\n" + "=" * 72)
    print("LEVEL 0 SUMMARY")
    print("=" * 72)
    for lang, ls in summary["languages"].items():
        print(f"\n{lang.upper()}: {ls['total_characters']} characters")
        for pfx, ms in ls["prefix_modes"].items():
            print(f"  Prefix {pfx}: {ms['total_samples']} samples, {ms['errors']} errors")
            ppl = ls["perplexity_by_prefix"].get(pfx, {})
            print(f"    Perplexity: {ppl.get('perplexity', float('inf')):.2f}")
            for step in ["step_1", "step_2", "step_3", "step_4", "step_5"]:
                v = ms["utf8_validity_rates"].get(step, 0)
                d = ms["dfa_partial_rates"].get(step, 0)
                print(f"    {step}: validity={v:.2%}  DFA_partial={d:.4f}")

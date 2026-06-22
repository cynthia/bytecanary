"""Batched generation, perplexity, log likelihood, and incremental validity checking."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any

from .dfa import compute_dual_score
from .decode import ByteTokenizer, analyze_utf8_validity


def check_utf8_validity_incremental(
    byte_tok: ByteTokenizer,
    initial_token_ids: List[int],
    generated_token_ids: List[int],
) -> List[Dict[str, Any]]:
    results = []
    for i in range(1, len(generated_token_ids) + 1):
        combined = initial_token_ids + generated_token_ids[:i]
        byte_data = byte_tok.token_ids_to_bytes(combined)
        validity = analyze_utf8_validity(byte_data)
        dfa = compute_dual_score(byte_data)
        try:
            decoded = byte_data.decode("utf-8", errors="ignore")
        except Exception:
            decoded = ""
        results.append({
            "num_tokens": i,
            "token_ids": combined[:],
            "is_valid_utf8": validity["valid"],
            "decoded_text": decoded,
            "error_info": validity,
            "byte_length": len(byte_data),
            "dfa_partial_score": dfa.partial_score,
            "dfa_binary_strict": dfa.binary_strict,
            "dfa_binary_soft": dfa.binary_soft,
        })
    return results


def batch_generate(
    model,
    tokenizer,
    prompt_token_ids_list: List[List[int]],
    device: torch.device,
    max_new_tokens: int = 5,
    batch_size: int = 32,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
) -> List[List[int]]:
    model.eval()
    all_generated = []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    with torch.no_grad():
        for start in range(0, len(prompt_token_ids_list), batch_size):
            batch = prompt_token_ids_list[start : start + batch_size]
            max_len = max(len(p) for p in batch)

            input_ids_list = []
            attn_list = []
            for prompt in batch:
                pad_len = max_len - len(prompt)
                input_ids_list.append([pad_id] * pad_len + prompt)
                attn_list.append([0] * pad_len + [1] * len(prompt))

            input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
            attention_mask = torch.tensor(attn_list, dtype=torch.long, device=device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            for i in range(len(batch)):
                generated = outputs[i][max_len:].cpu().tolist()
                all_generated.append(generated)

            del outputs, input_ids, attention_mask
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()

    return all_generated


def batch_calculate_log_likelihoods(
    model,
    tokenizer,
    prompt_list: List[List[int]],
    continuation_list: List[List[int]],
    device: torch.device,
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    model.eval()
    results = []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    empty_result = {
        "total_log_likelihood": 0.0,
        "avg_log_likelihood": 0.0,
        "perplexity": float("inf"),
        "token_log_likelihoods": [],
    }

    with torch.no_grad():
        for start in range(0, len(prompt_list), batch_size):
            end = min(start + batch_size, len(prompt_list))
            bp = prompt_list[start:end]
            bc = continuation_list[start:end]

            seqs = []
            prompt_lens = []
            cont_lens = []

            for prompt, cont in zip(bp, bc):
                if not cont:
                    results.append(dict(empty_result))
                    continue
                seqs.append(prompt + cont)
                prompt_lens.append(len(prompt))
                cont_lens.append(len(cont))

            if not seqs:
                continue

            max_len = max(len(s) for s in seqs)
            ids_batch = []
            attn_batch = []
            seq_lens = []

            for seq in seqs:
                sl = len(seq)
                seq_lens.append(sl)
                pad_len = max_len - sl
                ids_batch.append([pad_id] * pad_len + seq)
                attn_batch.append([0] * pad_len + [1] * sl)

            input_ids = torch.tensor(ids_batch, dtype=torch.long, device=device)
            attention_mask = torch.tensor(attn_batch, dtype=torch.long, device=device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            for i, (sl, pl, cl) in enumerate(zip(seq_lens, prompt_lens, cont_lens)):
                lls = []
                pad_len = max_len - sl
                seq_start = pad_len

                for j in range(cl):
                    pos = seq_start + pl + j - 1
                    if pos < seq_start:
                        continue
                    token_logits = logits[i, pos, :]
                    log_probs = F.log_softmax(token_logits, dim=-1)
                    target = input_ids[i, pos + 1].item()
                    lls.append(log_probs[target].item())

                if lls:
                    total = sum(lls)
                    avg = total / len(lls)
                    results.append({
                        "total_log_likelihood": total,
                        "avg_log_likelihood": avg,
                        "perplexity": float(np.exp(-avg)),
                        "token_log_likelihoods": lls,
                    })
                else:
                    results.append(dict(empty_result))

            del outputs, logits, input_ids, attention_mask
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()

    return results


def calculate_perplexity(
    model,
    tokenizer,
    token_ids_list: List[List[int]],
    device: torch.device,
    batch_size: int = 32,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    valid = [s for s in token_ids_list if len(s) >= 2]

    if not valid:
        return {"perplexity": float("inf"), "avg_loss": float("inf"), "num_sequences": len(token_ids_list), "total_tokens": 0}

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    with torch.no_grad():
        for start in range(0, len(valid), batch_size):
            batch = valid[start : start + batch_size]
            max_len = max(len(s) for s in batch)

            ids_batch = []
            attn_batch = []
            lengths = []

            for seq in batch:
                sl = len(seq)
                lengths.append(sl)
                pad_len = max_len - sl
                ids_batch.append([pad_id] * pad_len + seq)
                attn_batch.append([0] * pad_len + [1] * sl)

            input_ids = torch.tensor(ids_batch, dtype=torch.long, device=device)
            attention_mask = torch.tensor(attn_batch, dtype=torch.long, device=device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            logits = outputs.logits

            for i, sl in enumerate(lengths):
                pad_len = max_len - sl
                seq_start = pad_len
                seq_logits = logits[i, seq_start : seq_start + sl - 1, :]
                seq_labels = input_ids[i, seq_start + 1 : seq_start + sl]
                loss = F.cross_entropy(
                    seq_logits.reshape(-1, seq_logits.size(-1)),
                    seq_labels.reshape(-1),
                    reduction="sum",
                )
                if not torch.isnan(loss):
                    total_loss += loss.item()
                    total_tokens += sl - 1

            del outputs, logits, input_ids, attention_mask
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()

    if total_tokens == 0:
        return {"perplexity": float("inf"), "avg_loss": float("inf"), "num_sequences": len(token_ids_list), "total_tokens": 0}

    avg_loss = total_loss / total_tokens
    return {
        "perplexity": float(np.exp(avg_loss)),
        "avg_loss": float(avg_loss),
        "num_sequences": len(token_ids_list),
        "total_tokens": total_tokens,
    }

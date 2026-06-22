#!/usr/bin/env python3
"""
Example: evaluate any HuggingFace model with ByteCanary.

Usage:
    # Level 0 only (bundled CJK 4000 eval set)
    bytecanary meta-llama/Llama-3.2-1B --trial

    # Level 0 with a custom eval set
    bytecanary google/gemma-2-2b --eval-set data/level0/cjk_unseen3b.json

    # Level 0 + Level 1 (needs the synthetic data directory)
    bytecanary meta-llama/Llama-3.2-1B --level1-data data/level1

    # Level 1 only
    bytecanary meta-llama/Llama-3.2-1B --level 1 --level1-data data/level1

    # Specify device and dtype
    bytecanary mistralai/Mistral-7B-v0.1 --device cuda:1 --dtype bfloat16

Bundled data (included in pip install):
    data/level0/cjk_4000.json      4000 CJK chars (common/uncommon/rare/unseen)

Repo-only data (in bytecanary/data/, not in pip package):
    data/level0/cjk_400.json       400-char subset
    data/level0/cjk_unseen3b.json  139 unseen 3-byte CJK ideographs (control set)
    data/level1/ja/                3000 Japanese synthetic OOV samples
    data/level1/ko/                3000 Korean synthetic OOV samples
    data/level1/zh/                3000 Chinese synthetic OOV samples

Results are saved to bytecanary_results/{level0,level1}/:
    - summary.json: aggregated metrics (validity rates, DFA scores, perplexity)
    - detailed_results.json: per-character / per-sample results
"""

import sys
from bytecanary.cli import main

if __name__ == "__main__":
    sys.exit(main())

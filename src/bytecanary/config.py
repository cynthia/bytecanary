from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class EvalConfig:
    model: str
    device: Optional[str] = None
    eval_set: Optional[str] = None
    output_dir: str = "bytecanary_results"
    batch_size: int = 64
    max_new_tokens: int = 5
    trial: bool = False
    trial_samples: int = 256
    languages: List[str] = field(default_factory=lambda: ["ja", "ko", "zh"])
    temperature: float = 1.0
    do_sample: bool = False
    save_detailed: bool = True
    dtype: Optional[str] = None
    trust_remote_code: bool = False
    level1_data: Optional[str] = None
    top_p: float = 1.0
    top_k: int = 50

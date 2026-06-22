"""ByteCanary: UTF-8 structural validity evaluation for byte-level language models."""

from .config import EvalConfig
from .decode import ByteTokenizer
from .dfa import DualScore, UTF8Analysis, UTF8State, UTF8StateMachine, compute_dual_score
from .evaluate import Level0Evaluator
from .evaluate_level1 import Level1Evaluator

__all__ = [
    "EvalConfig",
    "ByteTokenizer",
    "Level0Evaluator",
    "Level1Evaluator",
    "UTF8StateMachine",
    "UTF8State",
    "UTF8Analysis",
    "DualScore",
    "compute_dual_score",
]

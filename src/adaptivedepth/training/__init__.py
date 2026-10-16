"""Route-aware teacher/student training for Phase 5."""

from adaptivedepth.training.data import TrainingExample, load_training_examples
from adaptivedepth.training.trainer import find_latest_checkpoint, train_phase5

__all__ = [
    "TrainingExample",
    "find_latest_checkpoint",
    "load_training_examples",
    "train_phase5",
]

"""
Model Service - Singleton pattern for loading PyTorch models.
Uses lazy initialization + fallback to randomly-initialized models
when pre-trained weights are not available (for demo/dev).
"""
import os
import torch
from pathlib import Path
from functools import lru_cache
import logging

from models.ml_models import (
    LearningPathRecommender,
    StruggleDetector,
    CodeGrader,
    load_model,
)

logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model instances (lazy loaded)
_learning_path_model = None
_struggle_detector_model = None
_code_grader_model = None


def _load_or_init(model_instance, weight_path: str, model_name: str):
    """Load weights if available, else return initialized model (demo mode)."""
    path = Path(weight_path)
    if path.exists():
        try:
            model = load_model(model_instance, str(path), DEVICE)
            logger.info(f"Loaded {model_name} from {path}")
            return model
        except Exception as e:
            logger.warning(f"Failed to load {model_name} weights: {e}. Using random init.")
    else:
        logger.info(f"{model_name} weights not found at {path}. Using random init (demo mode).")

    model_instance.eval()
    return model_instance.to(DEVICE)


def get_learning_path_model() -> LearningPathRecommender:
    global _learning_path_model
    if _learning_path_model is None:
        model = LearningPathRecommender(
            num_topics=20, embedding_dim=64, hidden_dim=128, skill_levels=5
        )
        _learning_path_model = _load_or_init(
            model, "models/learning_path_model.pt", "LearningPathRecommender"
        )
    return _learning_path_model


def get_struggle_detector_model() -> StruggleDetector:
    global _struggle_detector_model
    if _struggle_detector_model is None:
        model = StruggleDetector(
            vocab_size=8000, embedding_dim=128, hidden_dim=256,
            num_heads=4, num_layers=3, max_seq_len=128
        )
        _struggle_detector_model = _load_or_init(
            model, "models/struggle_detector_model.pt", "StruggleDetector"
        )
    return _struggle_detector_model


def get_code_grader_model() -> CodeGrader:
    global _code_grader_model
    if _code_grader_model is None:
        model = CodeGrader(
            vocab_size=2000, embedding_dim=64,
            hidden_dim=128, num_ast_features=16
        )
        _code_grader_model = _load_or_init(
            model, "models/code_grader_model.pt", "CodeGrader"
        )
    return _code_grader_model
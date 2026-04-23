"""
Training Script for EduAI BD ML Models

Trains all three models with synthetic data.
In production: replace with real student data from the BD platform.

Usage:
    python train.py --model all          # train all models
    python train.py --model learning_path
    python train.py --model struggle
    python train.py --model grader
    python train.py --epochs 50 --batch_size 64
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
import logging

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from models.ml_models import (
    LearningPathRecommender,
    StruggleDetector,
    CodeGrader,
    save_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATASETS
# ─────────────────────────────────────────────────────────────────────────────

class LearningPathDataset(Dataset):
    """Synthetic dataset for learning path recommender."""

    def __init__(self, size: int = 5000, num_topics: int = 20, skill_levels: int = 5):
        np.random.seed(42)
        self.size = size
        self.num_topics = num_topics

        # Simulate Bangladeshi student profiles
        self.topic_skills = torch.randint(0, skill_levels + 1, (size, num_topics))
        self.extra = torch.rand(size, 4)  # hours, exp, age, language

        # Simulate completed topics (cumulative)
        self.completed = (torch.rand(size, num_topics) < 0.3).long()

        # Targets: next best topic + recommended difficulty + hours
        self.target_topic = torch.randint(0, num_topics, (size,))
        self.target_difficulty = torch.randint(0, skill_levels, (size,))
        self.target_hours = torch.rand(size) * 15 + 1  # 1-16 hours

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return (
            self.topic_skills[idx],
            self.extra[idx],
            self.completed[idx],
            self.target_topic[idx],
            self.target_difficulty[idx],
            self.target_hours[idx],
        )


class StruggleDataset(Dataset):
    """Synthetic dataset for struggle detector."""

    NUM_TYPES = 8
    VOCAB_SIZE = 8000

    def __init__(self, size: int = 8000, max_len: int = 128):
        np.random.seed(42)
        self.size = size
        self.max_len = max_len

        # Random tokenized questions
        self.input_ids = torch.randint(1, self.VOCAB_SIZE, (size, max_len))
        # Add padding
        for i in range(size):
            pad_len = np.random.randint(0, max_len // 2)
            if pad_len > 0:
                self.input_ids[i, -pad_len:] = 0

        self.attention_mask = (self.input_ids != 0).long()
        self.struggle_labels = torch.randint(0, self.NUM_TYPES, (size,))
        self.severity = torch.rand(size)
        self.intervention_labels = torch.randint(0, 15, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.struggle_labels[idx],
            self.severity[idx],
            self.intervention_labels[idx],
        )


class CodeGraderDataset(Dataset):
    """Synthetic dataset for code grader."""

    CODE_VOCAB = 2000
    NUM_AST = 16

    def __init__(self, size: int = 6000, code_len: int = 256):
        np.random.seed(42)
        self.size = size
        self.student_tokens = torch.randint(0, self.CODE_VOCAB, (size, code_len))
        self.ref_tokens = torch.randint(0, self.CODE_VOCAB, (size, code_len))
        self.ast_features = torch.rand(size, self.NUM_AST)
        self.scores = torch.rand(size) * 100
        self.feedback_labels = torch.randint(0, 5, (size,))
        self.concept_mastery = torch.rand(size, 12)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return (
            self.student_tokens[idx],
            self.ref_tokens[idx],
            self.ast_features[idx],
            self.scores[idx],
            self.feedback_labels[idx],
            self.concept_mastery[idx],
        )


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOPS
# ─────────────────────────────────────────────────────────────────────────────

def train_learning_path(epochs: int = 30, batch_size: int = 64, lr: float = 1e-3):
    logger.info("=" * 50)
    logger.info("Training LearningPathRecommender")
    logger.info("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = LearningPathDataset(size=5000)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = LearningPathRecommender(num_topics=20).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            topic_skills, extra, completed, t_topic, t_diff, t_hours = [b.to(device) for b in batch]
            optimizer.zero_grad()

            out = model(topic_skills, extra, completed)
            loss_topic = ce_loss(out["topic_scores"], t_topic)
            loss_diff = ce_loss(out["difficulty_logits"], t_diff)
            loss_hours = mse_loss(out["estimated_hours"], t_hours / 16.0)

            loss = loss_topic + 0.5 * loss_diff + 0.3 * loss_hours
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_train = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                topic_skills, extra, completed, t_topic, t_diff, t_hours = [b.to(device) for b in batch]
                out = model(topic_skills, extra, completed)
                loss = (ce_loss(out["topic_scores"], t_topic) +
                        0.5 * ce_loss(out["difficulty_logits"], t_diff))
                val_loss += loss.item()
        avg_val = val_loss / len(val_loader)

        if epoch % 5 == 0 or epoch == epochs:
            logger.info(f"Epoch {epoch:3d}/{epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_model(model, str(MODELS_DIR / "learning_path_model.pt"))

    logger.info(f"✅ LearningPathRecommender saved | Best Val Loss: {best_val_loss:.4f}")
    return model


def train_struggle_detector(epochs: int = 25, batch_size: int = 64, lr: float = 2e-4):
    logger.info("=" * 50)
    logger.info("Training StruggleDetector")
    logger.info("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = StruggleDataset(size=8000)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = StruggleDetector(vocab_size=8000, num_layers=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs
    )

    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss()

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            ids, mask, s_label, severity, intervention = [b.to(device) for b in batch]
            optimizer.zero_grad()

            out = model(ids, mask)
            loss = (ce_loss(out["struggle_logits"], s_label) +
                    bce_loss(out["severity"], severity) +
                    0.3 * ce_loss(out["intervention_logits"], intervention))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = out["struggle_logits"].argmax(dim=-1)
            correct += (preds == s_label).sum().item()
            total += len(s_label)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                ids, mask, s_label, severity, intervention = [b.to(device) for b in batch]
                out = model(ids, mask)
                preds = out["struggle_logits"].argmax(dim=-1)
                val_correct += (preds == s_label).sum().item()
                val_total += len(s_label)

        val_acc = val_correct / val_total
        if epoch % 5 == 0 or epoch == epochs:
            logger.info(f"Epoch {epoch:3d}/{epochs} | "
                       f"Loss: {total_loss/len(train_loader):.4f} | "
                       f"Train Acc: {correct/total:.3f} | Val Acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, str(MODELS_DIR / "struggle_detector_model.pt"))

    logger.info(f"✅ StruggleDetector saved | Best Val Acc: {best_val_acc:.3f}")
    return model


def train_code_grader(epochs: int = 30, batch_size: int = 64, lr: float = 1e-3):
    logger.info("=" * 50)
    logger.info("Training CodeGrader")
    logger.info("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = CodeGraderDataset(size=6000)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = CodeGrader(vocab_size=2000, num_ast_features=16).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss()

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for batch in train_loader:
            s_tok, r_tok, ast_feat, scores, fb_labels, concepts = [b.to(device) for b in batch]
            optimizer.zero_grad()

            out = model(s_tok, r_tok, ast_feat)
            loss_score = mse_loss(out["score"], scores)
            loss_fb = ce_loss(out["feedback_logits"], fb_labels)
            loss_concept = bce_loss(out["concept_mastery"], concepts)

            loss = loss_score / 100 + loss_fb + loss_concept
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs:
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    s_tok, r_tok, ast_feat, scores, fb_labels, concepts = [b.to(device) for b in batch]
                    out = model(s_tok, r_tok, ast_feat)
                    val_loss += mse_loss(out["score"], scores).item() / 100
            avg_val = val_loss / len(val_loader)
            logger.info(f"Epoch {epoch:3d}/{epochs} | "
                       f"Train: {total_loss/len(train_loader):.4f} | Val MSE: {avg_val:.4f}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                save_model(model, str(MODELS_DIR / "code_grader_model.pt"))

    logger.info(f"✅ CodeGrader saved | Best Val MSE: {best_val_loss:.4f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train EduAI BD Models")
    parser.add_argument("--model", default="all",
                        choices=["all", "learning_path", "struggle", "grader"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"Training: {args.model} | Epochs: {args.epochs} | BS: {args.batch_size}")

    if args.model in ("all", "learning_path"):
        train_learning_path(args.epochs, args.batch_size, args.lr)

    if args.model in ("all", "struggle"):
        train_struggle_detector(args.epochs, args.batch_size, lr=2e-4)

    if args.model in ("all", "grader"):
        train_code_grader(args.epochs, args.batch_size, args.lr)

    logger.info("\n🎉 All models trained and saved to models/")


if __name__ == "__main__":
    main()
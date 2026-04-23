"""
PyTorch ML Models for EduAI BD Platform

1. LearningPathRecommender - Collaborative filtering + content-based hybrid
2. StruggleDetector       - BiLSTM + Attention NLP model
3. CodeQualityScorer      - AST-based + learned features model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. LEARNING PATH RECOMMENDER
# ─────────────────────────────────────────────────────────────────────────────

class StudentEncoder(nn.Module):
    """Encodes student profile into latent representation."""

    def __init__(self, num_topics: int = 20, skill_levels: int = 5,
                 embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.skill_embedding = nn.Embedding(skill_levels + 1, embedding_dim)
        self.topic_embedding = nn.Embedding(num_topics + 1, embedding_dim)

        input_dim = num_topics * embedding_dim + 4  # 4 = extra features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.output_dim = hidden_dim // 2

    def forward(self, topic_skills: torch.Tensor, extra_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            topic_skills: (batch, num_topics) skill levels per topic
            extra_features: (batch, 4) [hours_per_week, prior_exp, age_group, language_pref]
        """
        skill_embs = self.skill_embedding(topic_skills)          # (B, T, E)
        flat = skill_embs.view(skill_embs.size(0), -1)          # (B, T*E)
        combined = torch.cat([flat, extra_features.float()], dim=-1)
        return self.encoder(combined)


class LearningPathRecommender(nn.Module):
    """
    Recommends next topics and difficulty levels for each student.
    Uses a hybrid approach: collaborative filtering + content embeddings.
    """

    def __init__(self, num_topics: int = 20, num_students: int = 10000,
                 embedding_dim: int = 64, hidden_dim: int = 128, skill_levels: int = 5):
        super().__init__()
        self.num_topics = num_topics
        self.num_students = num_students

        # Student encoder
        self.student_encoder = StudentEncoder(num_topics, skill_levels, embedding_dim, hidden_dim)

        # Topic content encoder
        self.topic_content = nn.Embedding(num_topics, embedding_dim)

        # Attention over topics
        latent_dim = self.student_encoder.output_dim
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=4, batch_first=True)
        self.attn_proj = nn.Linear(embedding_dim, latent_dim)

        # Path predictor head
        combined_dim = latent_dim * 2
        self.path_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_topics),   # score each topic
        )

        # Difficulty predictor head
        self.difficulty_head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, skill_levels),         # recommended difficulty
        )

        # Estimated completion time head
        self.time_head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),                       # positive time output
        )

    def forward(self, topic_skills: torch.Tensor, extra_features: torch.Tensor,
                completed_topics: torch.Tensor) -> Dict[str, torch.Tensor]:
        student_repr = self.student_encoder(topic_skills, extra_features)  # (B, latent)

        # Attend over topic content using student as query
        all_topic_ids = torch.arange(self.num_topics, device=topic_skills.device)
        topic_embs = self.topic_content(all_topic_ids).unsqueeze(0).expand(
            topic_skills.size(0), -1, -1)                                  # (B, T, E)

        q = student_repr.unsqueeze(1)                                      # (B, 1, latent)
        # Project topic embeddings to latent dim for query
        q_proj = self.attn_proj(
            student_repr.unsqueeze(1).expand(-1, self.num_topics, -1)
        )  # Simplified attention via dot product
        attn_out = (topic_embs * q_proj).mean(dim=1)                       # (B, E)
        attn_repr = self.attn_proj(attn_out)                               # (B, latent)

        combined = torch.cat([student_repr, attn_repr], dim=-1)            # (B, 2*latent)

        topic_scores = self.path_head(combined)                            # (B, num_topics)
        # Mask already-completed topics
        topic_scores = topic_scores - completed_topics.float() * 1e9

        difficulty_logits = self.difficulty_head(combined)                 # (B, skill_levels)
        est_time = self.time_head(combined).squeeze(-1)                    # (B,)

        return {
            "topic_scores": topic_scores,
            "difficulty_logits": difficulty_logits,
            "estimated_hours": est_time,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. STRUGGLE DETECTOR (NLP)
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class StruggleDetector(nn.Module):
    """
    Detects student struggles from natural language questions.

    Outputs:
      - struggle_type: [confusion, syntax_error, logic_error, concept_gap,
                        motivation_low, time_pressure, language_barrier]
      - severity: float [0, 1]
      - suggested_intervention: topic recommendation
    """

    STRUGGLE_TYPES = [
        "confusion", "syntax_error", "logic_error", "concept_gap",
        "motivation_low", "time_pressure", "language_barrier", "no_struggle"
    ]
    NUM_INTERVENTIONS = 15

    def __init__(self, vocab_size: int = 8000, embedding_dim: int = 128,
                 hidden_dim: int = 256, num_heads: int = 4, num_layers: int = 3,
                 max_seq_len: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(embedding_dim, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads,
            dim_feedforward=hidden_dim, dropout=0.1,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pooling attention
        self.pool_attn = nn.Linear(embedding_dim, 1)

        # Output heads
        self.struggle_classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, len(self.STRUGGLE_TYPES)),
        )
        self.severity_regressor = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.intervention_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.NUM_INTERVENTIONS),
        )

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x = self.embedding(input_ids)               # (B, L, E)
        x = self.pos_enc(x)

        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)  # (B, L, E)

        # Attention pooling
        attn_w = self.pool_attn(x).squeeze(-1)      # (B, L)
        if attention_mask is not None:
            attn_w = attn_w.masked_fill(attention_mask == 0, float('-inf'))
        attn_w = F.softmax(attn_w, dim=-1)
        pooled = (x * attn_w.unsqueeze(-1)).sum(dim=1)  # (B, E)

        return {
            "struggle_logits": self.struggle_classifier(pooled),
            "severity": self.severity_regressor(pooled).squeeze(-1),
            "intervention_logits": self.intervention_head(pooled),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. CODE QUALITY SCORER
# ─────────────────────────────────────────────────────────────────────────────

class CodeFeatureExtractor(nn.Module):
    """Extracts learned features from tokenized code."""

    def __init__(self, vocab_size: int = 2000, embedding_dim: int = 64,
                 hidden_dim: int = 128, max_len: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(embedding_dim, max_len)

        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embedding_dim, hidden_dim, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
            )
            for k in [3, 5, 7]
        ])
        self.output_dim = hidden_dim * 3

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)    # (B, L, E)
        x = self.pos_enc(x)
        x = x.transpose(1, 2)           # (B, E, L) for Conv1d
        features = [F.adaptive_max_pool1d(conv(x), 1).squeeze(-1)
                    for conv in self.conv_blocks]
        return torch.cat(features, dim=-1)   # (B, 3*H)


class CodeGrader(nn.Module):
    """
    Auto-grades Python code assignments.

    Inputs:
      - student_code tokens
      - reference_code tokens (optional)
      - hand-crafted AST features (line_count, complexity, etc.)

    Outputs:
      - score: float [0, 100]
      - feedback_category: [correct, minor_error, logic_error, incomplete, wrong_approach]
      - concept_mastery: per-concept score vector
    """

    FEEDBACK_CATS = ["correct", "minor_error", "logic_error", "incomplete", "wrong_approach"]
    NUM_CONCEPTS = 12

    def __init__(self, vocab_size: int = 2000, embedding_dim: int = 64,
                 hidden_dim: int = 128, num_ast_features: int = 16):
        super().__init__()
        self.code_encoder = CodeFeatureExtractor(vocab_size, embedding_dim, hidden_dim)
        feat_dim = self.code_encoder.output_dim

        # Cross-code similarity (student vs reference)
        self.similarity_proj = nn.Linear(feat_dim * 2, 64)

        # Combined head input
        combined_dim = feat_dim + 64 + num_ast_features
        self.backbone = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        latent = hidden_dim // 2

        # Output heads
        self.score_head = nn.Sequential(
            nn.Linear(latent, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        self.feedback_head = nn.Linear(latent, len(self.FEEDBACK_CATS))
        self.concept_head = nn.Sequential(
            nn.Linear(latent, self.NUM_CONCEPTS),
            nn.Sigmoid()
        )

    def forward(self, student_tokens: torch.Tensor,
                ref_tokens: torch.Tensor,
                ast_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        student_feats = self.code_encoder(student_tokens)
        ref_feats = self.code_encoder(ref_tokens)

        sim_input = torch.cat([student_feats, ref_feats], dim=-1)
        sim_repr = F.relu(self.similarity_proj(sim_input))

        combined = torch.cat([student_feats, sim_repr, ast_features.float()], dim=-1)
        latent = self.backbone(combined)

        return {
            "score": self.score_head(latent).squeeze(-1) * 100,
            "feedback_logits": self.feedback_head(latent),
            "concept_mastery": self.concept_head(latent),
        }


# ─────────────────────────────────────────────────────────────────────────────
# MODEL FACTORY & CHECKPOINT UTILS
# ─────────────────────────────────────────────────────────────────────────────

def create_learning_path_model(**kwargs) -> LearningPathRecommender:
    return LearningPathRecommender(**kwargs)

def create_struggle_detector(**kwargs) -> StruggleDetector:
    return StruggleDetector(**kwargs)

def create_code_grader(**kwargs) -> CodeGrader:
    return CodeGrader(**kwargs)


def save_model(model: nn.Module, path: str):
    torch.save({
        "state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
    }, path)
    print(f"Model saved to {path}")


def load_model(model: nn.Module, path: str, device: str = "cpu") -> nn.Module:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
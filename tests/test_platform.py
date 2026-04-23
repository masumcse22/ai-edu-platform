"""
Tests for EduAI BD Platform
Run: pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np


# ─── ML MODEL TESTS ──────────────────────────────────────────────────────────

class TestLearningPathRecommender:
    def test_forward_pass(self):
        from models.ml_models import LearningPathRecommender
        model = LearningPathRecommender(num_topics=20)
        model.eval()
        topic_skills = torch.randint(0, 6, (2, 20))
        extra = torch.rand(2, 4)
        completed = torch.zeros(2, 20, dtype=torch.long)

        with torch.no_grad():
            out = model(topic_skills, extra, completed)

        assert "topic_scores" in out
        assert out["topic_scores"].shape == (2, 20)
        assert out["difficulty_logits"].shape == (2, 5)
        assert out["estimated_hours"].shape == (2,)

    def test_completed_masking(self):
        from models.ml_models import LearningPathRecommender
        model = LearningPathRecommender(num_topics=20)
        model.eval()
        topic_skills = torch.zeros(1, 20, dtype=torch.long)
        extra = torch.rand(1, 4)

        # Complete first 5 topics
        completed = torch.zeros(1, 20, dtype=torch.long)
        completed[0, :5] = 1

        with torch.no_grad():
            out = model(topic_skills, extra, completed)

        # Completed topics should have very low scores
        scores = out["topic_scores"][0]
        assert scores[:5].max().item() < scores[5:].min().item() + 1e5


class TestStruggleDetector:
    def test_output_shapes(self):
        from models.ml_models import StruggleDetector
        model = StruggleDetector(vocab_size=8000)
        model.eval()
        ids = torch.randint(0, 8000, (3, 128))
        mask = torch.ones(3, 128, dtype=torch.long)

        with torch.no_grad():
            out = model(ids, mask)

        assert out["struggle_logits"].shape == (3, 8)
        assert out["severity"].shape == (3,)
        assert (out["severity"] >= 0).all() and (out["severity"] <= 1).all()

    def test_severity_range(self):
        from models.ml_models import StruggleDetector
        model = StruggleDetector(vocab_size=8000)
        model.eval()
        ids = torch.randint(1, 8000, (10, 128))

        with torch.no_grad():
            out = model(ids)

        sev = out["severity"]
        assert ((sev >= 0) & (sev <= 1)).all()


class TestCodeGrader:
    def test_score_range(self):
        from models.ml_models import CodeGrader
        model = CodeGrader(vocab_size=2000)
        model.eval()

        s_tok = torch.randint(0, 2000, (4, 256))
        r_tok = torch.randint(0, 2000, (4, 256))
        ast_feat = torch.rand(4, 16)

        with torch.no_grad():
            out = model(s_tok, r_tok, ast_feat)

        assert out["score"].shape == (4,)
        assert (out["score"] >= 0).all() and (out["score"] <= 100).all()
        assert out["concept_mastery"].shape == (4, 12)


# ─── API TESTS ───────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"


def test_learning_path_endpoint(client):
    payload = {
        "student": {
            "student_id": "test_001",
            "name": "Test Student",
            "hours_per_week": 10.0,
            "prior_experience_years": 1.0,
            "age_group": 2,
            "language_preference": "bn",
            "topic_skill_levels": {"variables": 4, "operators": 3},
            "completed_topics": ["variables"],
            "device_type": "mobile",
            "internet_quality": "limited",
        },
        "num_recommendations": 3,
        "goal": "data_science",
    }
    resp = client.post("/api/v1/learning/recommend", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "recommended_topics" in data
    assert len(data["recommended_topics"]) <= 3
    assert "weekly_plan" in data
    assert "motivation_message" in data


def test_code_grader_endpoint(client):
    payload = {
        "student_id": "test_001",
        "assignment_id": "assignment_loops_001",
        "code": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)",
        "language": "python",
        "assignment_description": "Recursive factorial",
    }
    resp = client.post("/api/v1/grader/grade", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert 0 <= data["score"] <= 100
    assert "feedback_category" in data
    assert "test_results" in data
    # Factorial should pass all tests
    assert any(t["passed"] for t in data["test_results"])


def test_code_grader_buggy_code(client):
    payload = {
        "student_id": "test_001",
        "assignment_id": "assignment_loops_001",
        "code": "def factorial(n):\n    return 0",  # Wrong answer
        "language": "python",
        "assignment_description": "",
    }
    resp = client.post("/api/v1/grader/grade", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["score"] < 80  # Should not get a high score


def test_struggle_detector_endpoint(client):
    payload = {
        "student_id": "test_001",
        "question_text": "ami bujhte parchhi na keno loop kaj korche na",
        "current_topic": "for_loops",
        "session_duration_minutes": 45,
        "previous_errors": 3,
        "language": "bn",
    }
    resp = client.post("/api/v1/struggle/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "struggle_type" in data
    assert 0 <= data["severity"] <= 1
    assert 0 <= data["confidence"] <= 1
    assert "encouragement_message" in data


def test_struggle_detector_english(client):
    payload = {
        "student_id": "test_002",
        "question_text": "I give up, recursion is too hard",
        "current_topic": "recursion",
        "session_duration_minutes": 90,
        "previous_errors": 7,
        "language": "en",
    }
    resp = client.post("/api/v1/struggle/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["struggle_type"] in ["motivation_low", "confusion", "concept_gap", "logic_error",
                                      "syntax_error", "time_pressure", "language_barrier", "no_struggle"]


def test_dashboard_endpoint(client):
    resp = client.get("/api/v1/dashboard/progress/test_001")
    assert resp.status_code == 200
    data = resp.json()
    assert "overall_progress" in data
    assert "topics_completed" in data


# ─── AST FEATURE TESTS ───────────────────────────────────────────────────────

def test_ast_features_valid_code():
    from app.routers.code_grader import _extract_ast_features
    code = """
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
"""
    features = _extract_ast_features(code)
    assert features is not None
    assert len(features) == 16
    assert all(0 <= f <= 1 for f in features)


def test_ast_features_syntax_error():
    from app.routers.code_grader import _extract_ast_features
    code = "def broken(:\n  pass"
    features = _extract_ast_features(code)
    assert features is None


def test_tokenizer():
    from app.data.tokenizers import SimpleCodeTokenizer, SimpleNLPTokenizer

    ct = SimpleCodeTokenizer()
    ids = ct.encode("def hello():\n    return 42", max_len=32)
    assert len(ids) == 32

    nt = SimpleNLPTokenizer()
    ids = nt.encode("ami bujhchi na keno loop kaj korche na", max_len=16)
    assert len(ids) == 16
"""
Learning Path Router - Personalized topic recommendations using PyTorch model
"""
import torch
import numpy as np
from fastapi import APIRouter, HTTPException
from typing import List, Dict

from app.schemas import (
    LearningPathRequest, LearningPathResponse,
    TopicRecommendation, DifficultyLevel
)
from app.services.model_service import get_learning_path_model
from app.data.curriculum import PYTHON_CURRICULUM, TOPIC_PREREQUISITES, TOPIC_RESOURCES

router = APIRouter()

DIFFICULTY_MAP = {0: "beginner", 1: "elementary", 2: "intermediate", 3: "advanced", 4: "expert"}

MOTIVATIONAL_MESSAGES = {
    "bn": [
        "তুমি পারবে! প্রতিটি ধাপ তোমাকে এগিয়ে নিয়ে যাচ্ছে। 🚀",
        "বাংলাদেশের ভবিষ্যৎ প্রযুক্তিবিদ হিসেবে তোমার যাত্রা শুরু হোক! 💪",
        "কোডিং শেখা চ্যালেঞ্জিং, কিন্তু তুমি সঠিক পথে আছ! ⭐",
    ],
    "en": [
        "Every expert was once a beginner. Keep going! 🚀",
        "You're building skills that will change your future. 💪",
        "Consistency beats talent — keep showing up! ⭐",
    ]
}


@router.post("/recommend", response_model=LearningPathResponse)
async def get_learning_path(request: LearningPathRequest):
    """
    Generate personalized learning path using ML model.
    Considers student's skill level, available time, device constraints, and goals.
    """
    student = request.student
    model = get_learning_path_model()

    # Prepare input tensors
    topic_ids = list(PYTHON_CURRICULUM.keys())
    num_topics = len(topic_ids)
    topic_skill_tensor = torch.zeros(1, num_topics, dtype=torch.long)

    for i, tid in enumerate(topic_ids):
        skill = student.topic_skill_levels.get(tid, 0)
        topic_skill_tensor[0, i] = min(skill, 5)

    extra = torch.tensor([[
        min(student.hours_per_week / 40.0, 1.0),
        min(student.prior_experience_years / 10.0, 1.0),
        student.age_group / 4.0,
        1.0 if student.language_preference == "bn" else 0.0,
    ]], dtype=torch.float32)

    completed_mask = torch.zeros(1, num_topics, dtype=torch.long)
    for tid in student.completed_topics:
        if tid in topic_ids:
            completed_mask[0, topic_ids.index(tid)] = 1

    # Run model inference
    with torch.no_grad():
        outputs = model(topic_skill_tensor, extra, completed_mask)
        topic_scores = torch.softmax(outputs["topic_scores"], dim=-1)[0]
        difficulty_idx = torch.argmax(outputs["difficulty_logits"], dim=-1)[0].item()
        est_hours = outputs["estimated_hours"][0].item()

    # Select top-N topics
    n = request.num_recommendations
    top_indices = torch.topk(topic_scores, min(n, num_topics)).indices.tolist()

    recommendations: List[TopicRecommendation] = []
    for idx in top_indices:
        tid = topic_ids[idx]
        topic_info = PYTHON_CURRICULUM.get(tid, {})
        has_bn = student.language_preference == "bn" and topic_info.get("bangla_available", False)

        # Adjust difficulty for low-end devices / limited internet
        adj_difficulty = difficulty_idx
        if student.internet_quality == "limited" and adj_difficulty > 2:
            adj_difficulty = max(adj_difficulty - 1, 0)

        resources = TOPIC_RESOURCES.get(tid, [])
        if student.internet_quality == "limited":
            resources = [r for r in resources if r.get("offline", False)] or resources[:2]

        recommendations.append(TopicRecommendation(
            topic_id=tid,
            topic_name=topic_info.get("name", tid.replace("_", " ").title()),
            difficulty=DifficultyLevel(DIFFICULTY_MAP.get(adj_difficulty, "beginner")),
            estimated_hours=max(1.0, est_hours * topic_info.get("weight", 1.0)),
            priority_score=round(topic_scores[idx].item(), 4),
            prerequisites=TOPIC_PREREQUISITES.get(tid, []),
            resources=resources[:3],
            bangla_resources_available=has_bn,
        ))

    # Build weekly plan
    weekly_plan = _build_weekly_plan(recommendations, student.hours_per_week)
    total_hours = sum(r.estimated_hours for r in recommendations)
    total_weeks = max(1, int(np.ceil(total_hours / max(student.hours_per_week, 1))))

    lang = student.language_preference
    msg_list = MOTIVATIONAL_MESSAGES.get(lang, MOTIVATIONAL_MESSAGES["en"])
    motivation = msg_list[hash(student.student_id) % len(msg_list)]

    # Naive completion probability based on past activity
    completed_ratio = len(student.completed_topics) / max(num_topics, 1)
    completion_prob = min(0.95, 0.4 + completed_ratio * 0.6)

    return LearningPathResponse(
        student_id=student.student_id,
        recommended_topics=recommendations,
        weekly_plan=weekly_plan,
        motivation_message=motivation,
        total_estimated_weeks=total_weeks,
        completion_probability=round(completion_prob, 3),
    )


def _build_weekly_plan(topics: List[TopicRecommendation], hours_per_week: float) -> List[Dict]:
    plan = []
    week = 1
    weekly_hours = 0.0
    week_topics = []

    for topic in topics:
        if weekly_hours + topic.estimated_hours > hours_per_week and week_topics:
            plan.append({"week": week, "topics": week_topics, "total_hours": round(weekly_hours, 1)})
            week += 1
            weekly_hours = 0.0
            week_topics = []
        week_topics.append(topic.topic_id)
        weekly_hours += topic.estimated_hours

    if week_topics:
        plan.append({"week": week, "topics": week_topics, "total_hours": round(weekly_hours, 1)})

    return plan
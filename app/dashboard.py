"""
Dashboard Router - Student progress and analytics
"""
from fastapi import APIRouter
from app.schemas import StudentProgress

router = APIRouter()


@router.get("/progress/{student_id}", response_model=StudentProgress)
async def get_progress(student_id: str):
    """Get student progress dashboard data."""
    # In production: pull from database
    return StudentProgress(
        student_id=student_id,
        overall_progress=0.45,
        topics_completed=9,
        topics_total=20,
        average_score=72.5,
        struggle_frequency=0.2,
        streak_days=5,
        last_active="2024-01-15",
        weak_areas=["recursion", "oop", "file_io"],
        strong_areas=["variables", "loops", "conditionals"],
    )


@router.get("/leaderboard")
async def get_leaderboard():
    """Class leaderboard (privacy-preserving: anonymized)."""
    return {
        "leaderboard": [
            {"rank": 1, "student_id": "Student_A", "score": 94.5, "streak": 14},
            {"rank": 2, "student_id": "Student_B", "score": 91.2, "streak": 10},
            {"rank": 3, "student_id": "Student_C", "score": 88.7, "streak": 8},
        ]
    }
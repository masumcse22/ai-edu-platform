"""
Pydantic schemas for EduAI BD API
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum


class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    ELEMENTARY = "elementary"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class StruggleType(str, Enum):
    CONFUSION = "confusion"
    SYNTAX_ERROR = "syntax_error"
    LOGIC_ERROR = "logic_error"
    CONCEPT_GAP = "concept_gap"
    MOTIVATION_LOW = "motivation_low"
    TIME_PRESSURE = "time_pressure"
    LANGUAGE_BARRIER = "language_barrier"
    NO_STRUGGLE = "no_struggle"


class FeedbackCategory(str, Enum):
    CORRECT = "correct"
    MINOR_ERROR = "minor_error"
    LOGIC_ERROR = "logic_error"
    INCOMPLETE = "incomplete"
    WRONG_APPROACH = "wrong_approach"


# ─── Student Profile ──────────────────────────────────────────────────────────

class StudentProfile(BaseModel):
    student_id: str = Field(..., description="Unique student identifier")
    name: str = Field(..., description="Student name")
    hours_per_week: float = Field(5.0, ge=0.5, le=40.0, description="Available study hours")
    prior_experience_years: float = Field(0.0, ge=0.0, le=30.0)
    age_group: int = Field(2, ge=0, le=4, description="0=<15, 1=15-18, 2=19-25, 3=26-35, 4=35+")
    language_preference: str = Field("en", description="bn=Bangla, en=English")
    topic_skill_levels: Dict[str, int] = Field(
        default_factory=dict,
        description="Topic → skill level (0-5)"
    )
    completed_topics: List[str] = Field(default_factory=list)
    device_type: str = Field("mobile", description="mobile|desktop|low_end")
    internet_quality: str = Field("limited", description="limited|moderate|good")

    class Config:
        json_schema_extra = {
            "example": {
                "student_id": "BD_2024_001",
                "name": "Rahim Ahmed",
                "hours_per_week": 8.0,
                "prior_experience_years": 0.5,
                "age_group": 2,
                "language_preference": "bn",
                "topic_skill_levels": {"variables": 4, "loops": 2, "functions": 1},
                "completed_topics": ["variables", "data_types"],
                "device_type": "mobile",
                "internet_quality": "limited"
            }
        }


# ─── Learning Path ────────────────────────────────────────────────────────────

class LearningPathRequest(BaseModel):
    student: StudentProfile
    num_recommendations: int = Field(5, ge=1, le=10)
    goal: str = Field("general", description="general|data_science|web_dev|automation|job_ready")


class TopicRecommendation(BaseModel):
    topic_id: str
    topic_name: str
    difficulty: DifficultyLevel
    estimated_hours: float
    priority_score: float
    prerequisites: List[str]
    resources: List[Dict[str, str]]
    bangla_resources_available: bool


class LearningPathResponse(BaseModel):
    student_id: str
    recommended_topics: List[TopicRecommendation]
    weekly_plan: List[Dict[str, Any]]
    motivation_message: str
    total_estimated_weeks: int
    completion_probability: float


# ─── Code Grader ─────────────────────────────────────────────────────────────

class CodeSubmission(BaseModel):
    student_id: str
    assignment_id: str
    code: str = Field(..., max_length=5000)
    language: str = Field("python", description="Only python supported")
    assignment_description: str = ""

    @validator("code")
    def validate_code(cls, v):
        if not v.strip():
            raise ValueError("Code cannot be empty")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "student_id": "BD_2024_001",
                "assignment_id": "assignment_loops_001",
                "code": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)",
                "language": "python",
                "assignment_description": "Write a recursive factorial function"
            }
        }


class TestResult(BaseModel):
    test_name: str
    passed: bool
    expected: Any
    actual: Any
    error: Optional[str]


class GradingResult(BaseModel):
    student_id: str
    assignment_id: str
    score: float = Field(..., ge=0, le=100)
    feedback_category: FeedbackCategory
    detailed_feedback: str
    test_results: List[TestResult]
    concept_mastery: Dict[str, float]
    suggestions: List[str]
    execution_time_ms: float
    passed: bool


# ─── Struggle Detector ───────────────────────────────────────────────────────

class StudentQuestion(BaseModel):
    student_id: str
    question_text: str = Field(..., min_length=3, max_length=1000)
    current_topic: Optional[str] = None
    session_duration_minutes: int = Field(0, ge=0)
    previous_errors: int = Field(0, ge=0)
    language: str = Field("en", description="en|bn|mixed")

    class Config:
        json_schema_extra = {
            "example": {
                "student_id": "BD_2024_001",
                "question_text": "ami bujhte parchhi na keno loop kaj korche na",
                "current_topic": "for_loops",
                "session_duration_minutes": 45,
                "previous_errors": 3,
                "language": "bn"
            }
        }


class StruggleAnalysis(BaseModel):
    student_id: str
    struggle_type: StruggleType
    severity: float = Field(..., ge=0.0, le=1.0)
    confidence: float
    detected_issues: List[str]
    recommended_intervention: str
    suggested_resources: List[Dict[str, str]]
    encouragement_message: str
    alert_instructor: bool


# ─── Dashboard ───────────────────────────────────────────────────────────────

class StudentProgress(BaseModel):
    student_id: str
    overall_progress: float
    topics_completed: int
    topics_total: int
    average_score: float
    struggle_frequency: float
    streak_days: int
    last_active: str
    weak_areas: List[str]
    strong_areas: List[str]
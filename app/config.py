"""
Configuration settings for EduAI BD Platform
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    APP_NAME: str = "EduAI BD"
    VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Model settings
    LEARNING_PATH_MODEL: str = "models/learning_path_model.pt"
    STRUGGLE_DETECTOR_MODEL: str = "models/struggle_detector_model.pt"
    CODE_GRADER_MODEL: str = "models/code_grader_model.pt"

    # NLP settings
    MAX_SEQUENCE_LENGTH: int = 512
    EMBEDDING_DIM: int = 128
    HIDDEN_DIM: int = 256

    # Grader settings
    CODE_TIMEOUT_SECONDS: int = 10
    MAX_CODE_LENGTH: int = 5000

    # Learning path settings
    NUM_TOPICS: int = 20
    DIFFICULTY_LEVELS: int = 5

    class Config:
        env_file = ".env"


settings = Settings()
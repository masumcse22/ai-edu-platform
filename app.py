"""
HuggingFace Spaces entry point for EduAI BD.
HF Spaces runs on port 7860 by default.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from app.main import app  # noqa: F401

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=7860,
        log_level="info",
    )
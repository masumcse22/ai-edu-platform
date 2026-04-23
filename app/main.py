"""
AI-Powered Education Platform
FastAPI backend with ML-driven personalization
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from pathlib import Path

from app.routers import learning_path, code_grader, struggle_detector, dashboard
from app.core.config import settings

app = FastAPI(
    title="EduAI BD - Personalized Python Learning Platform",
    description="ML-powered education platform for Bangladeshi students",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent.parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Include routers
app.include_router(learning_path.router, prefix="/api/v1/learning", tags=["Learning Path"])
app.include_router(code_grader.router, prefix="/api/v1/grader", tags=["Code Grader"])
app.include_router(struggle_detector.router, prefix="/api/v1/struggle", tags=["Struggle Detector"])
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["Dashboard"])


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent.parent / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<h1>EduAI BD Platform Running</h1>")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "platform": "EduAI BD", "version": "1.0.0"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, reload=True)
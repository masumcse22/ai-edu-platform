# EduAI BD — AI-Powered Python Education Platform

> Personalized ML-driven learning for Bangladeshi students

## Features

- **📚 Personalized Learning Paths** — PyTorch model recommends next topics based on student skill level, available time, device constraints, and goals
- **⚙️ Auto Code Grader** — Sandboxed Python execution + ML scoring + AST feature analysis
- **🧠 Struggle Detector** — BiLSTM + Transformer NLP model detects confusion, syntax errors, motivation issues from student questions in Bangla/English

## Architecture

```
FastAPI Backend
├── /api/v1/learning/recommend    → LearningPathRecommender (PyTorch)
├── /api/v1/grader/grade          → CodeGrader (PyTorch + sandboxed exec)
├── /api/v1/struggle/analyze      → StruggleDetector (Transformer NLP)
└── /api/v1/dashboard/progress    → Student analytics
```

## Bangladeshi Context

- Bangla language support in UI and interventions
- Optimized for mobile and limited-bandwidth environments
- Bangla resource tagging in curriculum
- Mixed Bangla-English (code-switching) NLP support
- Intervention messages in Bangla

## Running Locally

```bash
pip install -r requirements.txt
python train.py --model all --epochs 30   # train models
python app.py                              # start server → localhost:7860
```

## API Docs

Visit `/api/docs` for interactive Swagger documentation.
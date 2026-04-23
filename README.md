<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=30&pause=1000&color=00C896&center=true&vCenter=true&width=600&lines=EduAI+BD+%F0%9F%87%A7%F0%9F%87%A9;ML-Powered+Python+Education;Built+for+Bangladesh" alt="Typing SVG" />

<br/>

<p align="center">
  <strong>🇧🇩 Bangladesh's First AI-Driven Python Learning Platform</strong><br/>
  <em>Personalized · Bangla-Aware · Mobile-First · ML-Powered</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/Language-Bangla%20%2B%20English-green?style=for-the-badge"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-active%20development-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/platform-mobile%20%7C%20web-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/NLP-code--switching%20support-purple?style=flat-square"/>
</p>

</div>

---

## 📖 Overview

**EduAI BD** is an intelligent, ML-driven Python education platform purpose-built for Bangladeshi students. It combines deep learning recommendation systems, sandboxed code execution, and multilingual NLP to deliver a truly personalized learning experience — even on low-bandwidth mobile devices.

> _"The right lesson, at the right time, in the language you think in."_

---

## ✨ Core Features

<table>
<tr>
<td width="33%" valign="top">

### 📚 Personalized Learning Paths
A **PyTorch-based recommender model** dynamically adapts to each student's:
- Current skill level
- Available study time
- Device & bandwidth constraints
- Learning goals & pace

No two students see the same path.

</td>
<td width="33%" valign="top">

### ⚙️ Auto Code Grader
A multi-layer grading engine combining:
- **Sandboxed Python execution** (secure, isolated)
- **ML scoring model** (PyTorch)
- **AST feature analysis** for code quality

Gives structured, meaningful feedback — not just pass/fail.

</td>
<td width="33%" valign="top">

### 🧠 Struggle Detector
**BiLSTM + Transformer NLP** pipeline that detects:
- Conceptual confusion
- Syntax frustration
- Motivational dips

Supports **Bangla, English, and code-switching** (mixed-language input).

</td>
</tr>
</table>

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        EduAI BD — FastAPI Backend                │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  POST /api/v1/learning/recommend                          │    │
│  │  └──► LearningPathRecommender  [PyTorch]                  │    │
│  │       Input : skill_level, time, device, goals            │    │
│  │       Output: ordered topic list + resource tags          │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  POST /api/v1/grader/grade                                │    │
│  │  └──► CodeGrader  [PyTorch + Sandboxed Exec + AST]        │    │
│  │       Input : student_code, problem_id                    │    │
│  │       Output: score, feedback, quality_metrics            │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  POST /api/v1/struggle/analyze                            │    │
│  │  └──► StruggleDetector  [BiLSTM + Transformer NLP]        │    │
│  │       Input : student_message (Bangla/English/mixed)      │    │
│  │       Output: struggle_type, confidence, intervention     │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  GET  /api/v1/dashboard/progress                          │    │
│  │  └──► Student Analytics Engine                            │    │
│  │       Output: progress charts, streak data, weaknesses   │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
│  Interactive Docs →  /api/docs  (Swagger UI)                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🇧🇩 Built for Bangladesh

EduAI BD is not a generic platform with a Bengali translation slapped on. It's engineered from the ground up for the Bangladeshi context:

| Feature | Details |
|---|---|
| 🗣️ **Bangla UI** | Full Bangla interface and intervention messages |
| 🔀 **Code-switching NLP** | Understands mixed Bangla-English student queries |
| 📱 **Mobile-first** | Optimized for Android devices and limited data |
| 📶 **Low-bandwidth mode** | Lightweight responses for 2G/3G environments |
| 🏷️ **Bangla curriculum tags** | Topics tagged with localized Bangla resources |
| 💬 **Local interventions** | Motivational nudges written in Bangla |

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10+**
- `pip` or `conda`
- (Optional) CUDA-enabled GPU for faster training

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/eduai-bd.git
cd eduai-bd

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train all ML models (recommended: GPU)
python train.py --model all --epochs 30

#    Or train individual models:
python train.py --model recommender --epochs 20
python train.py --model grader --epochs 15
python train.py --model struggle --epochs 30

# 4. Start the FastAPI server
python app.py
# → Server running at http://localhost:7860
# → API docs at  http://localhost:7860/api/docs
```

---

## 🧩 API Reference

### `POST /api/v1/learning/recommend`
Returns a personalized learning path for a student.

```json
{
  "student_id": "s_001",
  "skill_level": "beginner",
  "available_minutes": 30,
  "device": "mobile",
  "goals": ["web scraping", "data analysis"]
}
```

### `POST /api/v1/grader/grade`
Grades submitted Python code with ML + AST analysis.

```json
{
  "problem_id": "p_042",
  "student_code": "def fizzbuzz(n):\n    ..."
}
```

### `POST /api/v1/struggle/analyze`
Detects struggle type from a student's message (Bangla, English, or mixed).

```json
{
  "student_id": "s_001",
  "message": "ami bujhte parchi na keno loop ta kaj korche na"
}
```

> 📖 Full interactive documentation available at `/api/docs`

---

## 🧠 ML Model Details

| Model | Architecture | Input | Output |
|---|---|---|---|
| **LearningPathRecommender** | PyTorch MLP + Embedding | Student profile features | Ranked topic list |
| **CodeGrader** | PyTorch + rule-based AST | Code string + AST features | Score + feedback dict |
| **StruggleDetector** | BiLSTM + Transformer (BERT-based) | Raw text (multilingual) | Struggle label + confidence |

---

## 📁 Project Structure

```
eduai-bd/
├── app.py                    # FastAPI entrypoint
├── train.py                  # Model training CLI
├── requirements.txt
├── models/
│   ├── recommender.py        # LearningPathRecommender (PyTorch)
│   ├── grader.py             # CodeGrader (PyTorch + AST)
│   └── struggle_detector.py  # BiLSTM + Transformer NLP
├── api/
│   └── v1/
│       ├── learning.py
│       ├── grader.py
│       ├── struggle.py
│       └── dashboard.py
├── curriculum/
│   └── topics.json           # Bangla-tagged topic graph
├── sandbox/
│   └── executor.py           # Sandboxed Python runner
└── utils/
    ├── bangla_nlp.py         # Bangla/code-switching preprocessing
    └── analytics.py          # Progress tracking
```

---

## 🤝 Contributing

Contributions are very welcome — especially from Bangladeshi developers and educators!

```bash
# Fork → clone → create branch
git checkout -b feature/your-feature-name

# Make changes, then
git commit -m "feat: describe your change"
git push origin feature/your-feature-name
# Open a Pull Request
```

Please read `CONTRIBUTING.md` before submitting.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made with ❤️ for Bangladeshi learners**

_EduAI BD — শিখি, বাড়ি, এগিয়ে যাই_

</div>
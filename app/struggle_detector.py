"""
Struggle Detector Router - NLP-based student struggle detection
Supports Bangla/English mixed input (code-switching common in BD)
"""
import re
import torch
import numpy as np
from fastapi import APIRouter
from typing import List, Dict

from app.schemas import StudentQuestion, StruggleAnalysis, StruggleType
from app.services.model_service import get_struggle_detector_model
from app.data.tokenizers import SimpleNLPTokenizer
from app.data.interventions import INTERVENTION_MAP, STRUGGLE_RESOURCES

router = APIRouter()

STRUGGLE_TYPES = [
    "confusion", "syntax_error", "logic_error", "concept_gap",
    "motivation_low", "time_pressure", "language_barrier", "no_struggle"
]

# Keyword patterns for heuristic augmentation (Bangla + English mixed)
STRUGGLE_KEYWORDS = {
    "confusion": ["bujhchi na", "bujhi na", "don't understand", "confused", "ki kore",
                  "keno", "why", "unclear", "what does", "বুঝতে পারছি না"],
    "syntax_error": ["error", "syntax", "SyntaxError", "IndentationError", "invalid",
                     "kaj korche na", "কাজ করছে না"],
    "logic_error": ["wrong output", "expected", "but got", "not working", "incorrect",
                    "ঠিক হচ্ছে না", "thik hochhe na"],
    "concept_gap": ["what is", "explain", "ki", "কি", "কেন", "difference between",
                    "never learned", "don't know what"],
    "motivation_low": ["give up", "too hard", "hate", "boring", "why do we", "kosto",
                       "কষ্ট", "difficult"],
    "time_pressure": ["deadline", "fast", "quick", "rush", "hurry", "taratari",
                      "তাড়াতাড়ি"],
    "language_barrier": ["english mane", "translate", "bangla te", "বাংলায়", "i don't know english"],
}

ENCOURAGEMENT_MESSAGES = {
    "confusion": {
        "bn": "কনফিউশন মানেই তুমি কিছু নতুন শিখছ! এটা স্বাভাবিক। আস্তে আস্তে বুঝবে। 💙",
        "en": "Confusion means you're learning something new! That's totally normal. 💙"
    },
    "language_barrier": {
        "bn": "বাংলায় প্রোগ্রামিং শেখা সম্পূর্ণ সম্ভব! আমরা তোমাকে সাহায্য করব। 🇧🇩",
        "en": "Learning in your native language is completely valid! We have Bangla resources. 🇧🇩"
    },
    "motivation_low": {
        "bn": "হাল ছেড়ো না! বাংলাদেশের প্রতিটি সফল developer একসময় তোমার মতোই ছিল। 🔥",
        "en": "Don't give up! Every expert started exactly where you are. 🔥"
    },
    "no_struggle": {
        "bn": "দারুণ প্রশ্ন! তুমি সঠিক পথে আছ। 🌟",
        "en": "Great question! You're on the right track. 🌟"
    },
    "_default": {
        "bn": "তুমি ভালো করছ! সমস্যা হওয়াটা শেখার অংশ। 💪",
        "en": "You're doing well! Problems are part of learning. 💪"
    }
}


@router.post("/analyze", response_model=StruggleAnalysis)
async def analyze_struggle(question: StudentQuestion):
    """
    Analyze student question to detect struggle patterns.
    Supports Bangla, English, and code-switched text.
    """
    model = get_struggle_detector_model()
    tokenizer = SimpleNLPTokenizer()

    # Preprocess: normalize mixed script
    cleaned_text = _preprocess_text(question.question_text, question.language)

    # Tokenize
    token_ids = tokenizer.encode(cleaned_text, max_len=128)
    input_tensor = torch.tensor([token_ids], dtype=torch.long)
    attn_mask = torch.tensor([[1 if t != 0 else 0 for t in token_ids]], dtype=torch.long)

    # ML inference
    with torch.no_grad():
        outputs = model(input_tensor, attn_mask)
        struggle_probs = torch.softmax(outputs["struggle_logits"], dim=-1)[0]
        severity = outputs["severity"][0].item()
        intervention_idx = torch.argmax(outputs["intervention_logits"], dim=-1)[0].item()

    ml_struggle_idx = torch.argmax(struggle_probs).item()
    ml_confidence = struggle_probs[ml_struggle_idx].item()

    # Heuristic augmentation (boosts accuracy for code-switched text)
    heuristic_scores = _heuristic_detection(cleaned_text)
    blended = _blend_scores(struggle_probs.numpy(), heuristic_scores)

    final_idx = int(np.argmax(blended))
    final_type = StruggleType(STRUGGLE_TYPES[final_idx])
    final_confidence = float(blended[final_idx])

    # Adjust severity with contextual signals
    if question.previous_errors > 3:
        severity = min(1.0, severity + 0.2)
    if question.session_duration_minutes > 60:
        severity = min(1.0, severity + 0.1)

    # Detected issues extraction
    detected_issues = _extract_issues(cleaned_text, final_type)

    # Get intervention
    intervention = INTERVENTION_MAP.get(final_type.value, INTERVENTION_MAP["_default"])
    resources = STRUGGLE_RESOURCES.get(final_type.value, [])

    # Bangla resources for language-limited students
    if question.language == "bn":
        resources = [r for r in resources if r.get("lang") in ("bn", "both")] or resources

    # Encouragement message
    lang = question.language if question.language in ("bn", "en") else "en"
    msg_key = final_type.value if final_type.value in ENCOURAGEMENT_MESSAGES else "_default"
    encouragement = ENCOURAGEMENT_MESSAGES[msg_key].get(lang, ENCOURAGEMENT_MESSAGES[msg_key]["en"])

    # Alert instructor for high severity
    alert = severity > 0.7 and final_type != StruggleType.NO_STRUGGLE

    return StruggleAnalysis(
        student_id=question.student_id,
        struggle_type=final_type,
        severity=round(severity, 3),
        confidence=round(final_confidence, 3),
        detected_issues=detected_issues,
        recommended_intervention=intervention,
        suggested_resources=resources[:3],
        encouragement_message=encouragement,
        alert_instructor=alert,
    )


def _preprocess_text(text: str, lang: str) -> str:
    """Normalize mixed Bangla-English text."""
    text = text.lower().strip()
    # Normalize common Bangla romanizations
    bn_norm = {
        "ami": "i", "kore": "how", "na": "not", "ki": "what",
        "keno": "why", "tai": "so", "kintu": "but", "ar": "and",
    }
    if lang in ("bn", "mixed"):
        words = text.split()
        text = " ".join(bn_norm.get(w, w) for w in words)
    return text


def _heuristic_detection(text: str) -> np.ndarray:
    """Keyword-based struggle scoring."""
    scores = np.zeros(len(STRUGGLE_TYPES))
    for i, stype in enumerate(STRUGGLE_TYPES[:-1]):  # exclude no_struggle
        keywords = STRUGGLE_KEYWORDS.get(stype, [])
        hits = sum(1 for kw in keywords if kw.lower() in text)
        scores[i] = min(hits * 0.3, 1.0)
    if scores.max() == 0:
        scores[-1] = 0.6  # no_struggle
    return scores / (scores.sum() + 1e-8)


def _blend_scores(ml: np.ndarray, heuristic: np.ndarray) -> np.ndarray:
    """Blend ML and heuristic scores."""
    blended = 0.65 * ml + 0.35 * heuristic
    return blended / (blended.sum() + 1e-8)


def _extract_issues(text: str, struggle_type: StruggleType) -> List[str]:
    issues = []
    if "error" in text:
        error_match = re.search(r'(\w+error)', text, re.IGNORECASE)
        if error_match:
            issues.append(f"Error mentioned: {error_match.group(1)}")
    if struggle_type == StruggleType.CONCEPT_GAP:
        what_match = re.search(r'what (?:is|are) (.+?)[\?\.]+', text)
        if what_match:
            issues.append(f"Concept unclear: {what_match.group(1)[:50]}")
    if not issues:
        issues.append(f"General {struggle_type.value.replace('_', ' ')} detected")
    return issues
"""
Code Grader Router - Auto-grade Python assignments with ML + sandboxed execution
"""
import ast
import sys
import io
import time
import traceback
import signal
import torch
import numpy as np
from fastapi import APIRouter, HTTPException
from contextlib import redirect_stdout, redirect_stderr
from typing import List, Dict, Any, Optional

from app.schemas import CodeSubmission, GradingResult, TestResult, FeedbackCategory
from app.services.model_service import get_code_grader_model
from app.data.assignments import ASSIGNMENT_REGISTRY
from app.data.tokenizers import SimpleCodeTokenizer

router = APIRouter()

FEEDBACK_MAP = {0: "correct", 1: "minor_error", 2: "logic_error", 3: "incomplete", 4: "wrong_approach"}
CONCEPTS = [
    "variables", "conditionals", "loops", "functions", "recursion",
    "lists", "dicts", "string_ops", "file_io", "error_handling",
    "oop", "algorithms"
]

FEEDBACK_MESSAGES = {
    "correct": "🎉 Excellent work! Your solution is correct and well-structured.",
    "minor_error": "🔧 Almost there! Minor issues found. Check edge cases.",
    "logic_error": "🧠 Logic needs revision. Trace through your code step by step.",
    "incomplete": "📝 Solution is incomplete. Make sure to handle all requirements.",
    "wrong_approach": "💡 Different approach needed. Review the concept and try again.",
}


@router.post("/grade", response_model=GradingResult)
async def grade_code(submission: CodeSubmission):
    """
    Grade a Python code submission using:
    1. Sandboxed test execution
    2. AST feature extraction
    3. ML model scoring
    """
    start_time = time.time()

    # Get assignment config
    assignment = ASSIGNMENT_REGISTRY.get(submission.assignment_id)
    if not assignment:
        # Dynamic grading without predefined tests
        assignment = _create_dynamic_assignment(submission)

    # Extract AST features
    ast_features = _extract_ast_features(submission.code)
    if ast_features is None:
        return GradingResult(
            student_id=submission.student_id,
            assignment_id=submission.assignment_id,
            score=0.0,
            feedback_category=FeedbackCategory.MINOR_ERROR,
            detailed_feedback="❌ Syntax Error: Could not parse your code. Check for syntax mistakes.",
            test_results=[],
            concept_mastery={c: 0.0 for c in CONCEPTS},
            suggestions=["Review Python syntax basics", "Use an IDE for syntax highlighting"],
            execution_time_ms=(time.time() - start_time) * 1000,
            passed=False,
        )

    # Run test cases in sandbox
    test_results = _run_tests_sandboxed(submission.code, assignment.get("tests", []))

    # ML model inference
    model = get_code_grader_model()
    tokenizer = SimpleCodeTokenizer()

    student_tokens = torch.tensor(
        tokenizer.encode(submission.code, max_len=256), dtype=torch.long
    ).unsqueeze(0)
    ref_tokens = torch.tensor(
        tokenizer.encode(assignment.get("reference_solution", "pass"), max_len=256), dtype=torch.long
    ).unsqueeze(0)
    ast_tensor = torch.tensor([ast_features], dtype=torch.float32)

    with torch.no_grad():
        outputs = model(student_tokens, ref_tokens, ast_tensor)
        ml_score = outputs["score"][0].item()
        feedback_idx = torch.argmax(outputs["feedback_logits"], dim=-1)[0].item()
        concept_scores = outputs["concept_mastery"][0].tolist()

    # Combine test score + ML score
    test_score = (sum(1 for t in test_results if t.passed) / max(len(test_results), 1)) * 100
    final_score = 0.6 * test_score + 0.4 * ml_score
    final_score = round(min(100.0, max(0.0, final_score)), 2)

    # Determine feedback
    feedback_cat = FeedbackCategory(FEEDBACK_MAP.get(int(feedback_idx), "minor_error"))
    if test_score == 100 and final_score >= 85:
        feedback_cat = FeedbackCategory.CORRECT
    elif test_score == 0:
        feedback_cat = FeedbackCategory.LOGIC_ERROR

    concept_mastery = {CONCEPTS[i]: round(concept_scores[i], 3) for i in range(len(CONCEPTS))}
    suggestions = _generate_suggestions(feedback_cat, concept_mastery, ast_features)
    exec_ms = (time.time() - start_time) * 1000

    return GradingResult(
        student_id=submission.student_id,
        assignment_id=submission.assignment_id,
        score=final_score,
        feedback_category=feedback_cat,
        detailed_feedback=FEEDBACK_MESSAGES.get(feedback_cat.value, ""),
        test_results=test_results,
        concept_mastery=concept_mastery,
        suggestions=suggestions,
        execution_time_ms=round(exec_ms, 2),
        passed=final_score >= 60.0,
    )


def _extract_ast_features(code: str) -> Optional[List[float]]:
    """Extract 16 AST-based features from code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    features = {
        "num_functions": 0, "num_classes": 0, "num_loops": 0,
        "num_conditionals": 0, "num_try_except": 0, "has_recursion": 0,
        "max_depth": 0, "num_imports": 0, "num_list_comps": 0,
        "num_lambda": 0, "num_assert": 0, "cyclomatic_complexity": 1,
        "loc": len(code.splitlines()), "avg_line_len": 0,
        "has_docstring": 0, "num_return": 0,
    }

    lines = [l for l in code.splitlines() if l.strip()]
    features["avg_line_len"] = np.mean([len(l) for l in lines]) / 100 if lines else 0

    func_names = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            features["num_functions"] += 1
            func_names.add(node.name)
            if (node.body and isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, ast.Constant)):
                features["has_docstring"] = 1
        elif isinstance(node, ast.ClassDef):
            features["num_classes"] += 1
        elif isinstance(node, (ast.For, ast.While)):
            features["num_loops"] += 1
        elif isinstance(node, ast.If):
            features["num_conditionals"] += 1
            features["cyclomatic_complexity"] += 1
        elif isinstance(node, ast.Try):
            features["num_try_except"] += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            features["num_imports"] += 1
        elif isinstance(node, ast.ListComp):
            features["num_list_comps"] += 1
        elif isinstance(node, ast.Lambda):
            features["num_lambda"] += 1
        elif isinstance(node, ast.Assert):
            features["num_assert"] += 1
        elif isinstance(node, ast.Return):
            features["num_return"] += 1
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in func_names:
                features["has_recursion"] = 1

    # Normalize
    return [
        min(features["num_functions"] / 5, 1),
        min(features["num_classes"] / 3, 1),
        min(features["num_loops"] / 5, 1),
        min(features["num_conditionals"] / 5, 1),
        min(features["num_try_except"] / 3, 1),
        float(features["has_recursion"]),
        min(features["max_depth"] / 5, 1),
        min(features["num_imports"] / 5, 1),
        min(features["num_list_comps"] / 3, 1),
        min(features["num_lambda"] / 3, 1),
        float(features["has_docstring"]),
        min(features["cyclomatic_complexity"] / 10, 1),
        min(features["loc"] / 50, 1),
        min(features["avg_line_len"], 1),
        min(features["num_return"] / 5, 1),
        min(features["num_assert"] / 3, 1),
    ]


def _run_tests_sandboxed(code: str, test_cases: List[Dict]) -> List[TestResult]:
    """Run test cases in a restricted environment."""
    results = []

    # Safe globals - restrict dangerous modules
    safe_globals = {
        "__builtins__": {
            k: v for k, v in __builtins__.items()
            if k not in {"open", "exec", "eval", "__import__", "compile", "input"}
        } if isinstance(__builtins__, dict) else {
            k: getattr(__builtins__, k) for k in dir(__builtins__)
            if k not in {"open", "exec", "eval", "__import__", "compile", "input"}
        }
    }

    # Execute student code
    student_ns = dict(safe_globals)
    stdout_capture = io.StringIO()
    try:
        with redirect_stdout(stdout_capture):
            exec(compile(code, "<student>", "exec"), student_ns)
    except Exception as e:
        for tc in test_cases:
            results.append(TestResult(
                test_name=tc.get("name", "test"),
                passed=False,
                expected=tc.get("expected"),
                actual=None,
                error=f"Execution error: {str(e)}"
            ))
        return results

    for tc in test_cases:
        test_code = tc.get("test_code", "")
        expected = tc.get("expected")
        try:
            test_ns = dict(student_ns)
            out = io.StringIO()
            with redirect_stdout(out):
                exec(compile(test_code, "<test>", "exec"), test_ns)
            actual = test_ns.get("_result", out.getvalue().strip())
            passed = actual == expected
            results.append(TestResult(
                test_name=tc.get("name", "test"),
                passed=passed,
                expected=expected,
                actual=actual,
                error=None
            ))
        except Exception as e:
            results.append(TestResult(
                test_name=tc.get("name", "test"),
                passed=False,
                expected=expected,
                actual=None,
                error=str(e)
            ))

    return results


def _create_dynamic_assignment(submission: CodeSubmission) -> Dict:
    return {
        "id": submission.assignment_id,
        "description": submission.assignment_description,
        "reference_solution": "pass",
        "tests": [],
    }


def _generate_suggestions(cat: FeedbackCategory, concepts: Dict, ast_features: List) -> List[str]:
    suggestions = []
    weak = [c for c, s in concepts.items() if s < 0.4]
    if weak:
        suggestions.append(f"Review these concepts: {', '.join(weak[:3])}")
    if cat == FeedbackCategory.LOGIC_ERROR:
        suggestions.extend(["Use print() to trace variable values", "Draw a flowchart first"])
    if cat == FeedbackCategory.INCOMPLETE:
        suggestions.append("Re-read the problem statement carefully")
    if ast_features and ast_features[11] > 0.8:  # high cyclomatic complexity
        suggestions.append("Consider breaking your function into smaller parts")
    if not suggestions:
        suggestions.append("Great work! Try adding edge case handling")
    return suggestions[:4]
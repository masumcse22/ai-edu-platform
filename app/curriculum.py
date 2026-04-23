"""
Python curriculum for EduAI BD.
20 topics from beginner to advanced, with Bangladeshi context.
"""

PYTHON_CURRICULUM = {
    "variables": {
        "name": "Variables & Data Types",
        "weight": 0.8,
        "bangla_available": True,
        "difficulty_base": 0,
        "category": "fundamentals",
    },
    "operators": {
        "name": "Operators & Expressions",
        "weight": 0.8,
        "bangla_available": True,
        "difficulty_base": 0,
        "category": "fundamentals",
    },
    "input_output": {
        "name": "Input & Output",
        "weight": 0.7,
        "bangla_available": True,
        "difficulty_base": 0,
        "category": "fundamentals",
    },
    "conditionals": {
        "name": "Conditionals (if/else)",
        "weight": 1.0,
        "bangla_available": True,
        "difficulty_base": 1,
        "category": "control_flow",
    },
    "for_loops": {
        "name": "For Loops",
        "weight": 1.0,
        "bangla_available": True,
        "difficulty_base": 1,
        "category": "control_flow",
    },
    "while_loops": {
        "name": "While Loops",
        "weight": 0.9,
        "bangla_available": True,
        "difficulty_base": 1,
        "category": "control_flow",
    },
    "functions": {
        "name": "Functions",
        "weight": 1.2,
        "bangla_available": True,
        "difficulty_base": 2,
        "category": "functions",
    },
    "lists": {
        "name": "Lists & List Operations",
        "weight": 1.1,
        "bangla_available": True,
        "difficulty_base": 1,
        "category": "data_structures",
    },
    "dicts": {
        "name": "Dictionaries",
        "weight": 1.1,
        "bangla_available": False,
        "difficulty_base": 2,
        "category": "data_structures",
    },
    "strings": {
        "name": "String Manipulation",
        "weight": 1.0,
        "bangla_available": True,
        "difficulty_base": 1,
        "category": "data_structures",
    },
    "file_io": {
        "name": "File I/O",
        "weight": 1.0,
        "bangla_available": False,
        "difficulty_base": 2,
        "category": "io",
    },
    "error_handling": {
        "name": "Error Handling (try/except)",
        "weight": 1.0,
        "bangla_available": False,
        "difficulty_base": 2,
        "category": "robustness",
    },
    "oop_basics": {
        "name": "OOP Basics (Classes & Objects)",
        "weight": 1.4,
        "bangla_available": False,
        "difficulty_base": 3,
        "category": "oop",
    },
    "recursion": {
        "name": "Recursion",
        "weight": 1.3,
        "bangla_available": False,
        "difficulty_base": 3,
        "category": "algorithms",
    },
    "sorting_searching": {
        "name": "Sorting & Searching",
        "weight": 1.2,
        "bangla_available": False,
        "difficulty_base": 3,
        "category": "algorithms",
    },
    "modules": {
        "name": "Modules & Packages",
        "weight": 0.9,
        "bangla_available": False,
        "difficulty_base": 2,
        "category": "ecosystem",
    },
    "numpy_basics": {
        "name": "NumPy Basics",
        "weight": 1.3,
        "bangla_available": False,
        "difficulty_base": 3,
        "category": "data_science",
    },
    "pandas_basics": {
        "name": "Pandas for Data Analysis",
        "weight": 1.4,
        "bangla_available": False,
        "difficulty_base": 3,
        "category": "data_science",
    },
    "ml_intro": {
        "name": "Intro to Machine Learning (sklearn)",
        "weight": 1.6,
        "bangla_available": False,
        "difficulty_base": 4,
        "category": "ml",
    },
    "project_capstone": {
        "name": "Capstone Project",
        "weight": 2.0,
        "bangla_available": False,
        "difficulty_base": 4,
        "category": "project",
    },
}

TOPIC_PREREQUISITES = {
    "operators": ["variables"],
    "input_output": ["variables"],
    "conditionals": ["operators"],
    "for_loops": ["conditionals"],
    "while_loops": ["conditionals"],
    "functions": ["for_loops", "while_loops"],
    "lists": ["for_loops"],
    "dicts": ["lists"],
    "strings": ["variables"],
    "file_io": ["functions"],
    "error_handling": ["functions"],
    "oop_basics": ["functions", "dicts"],
    "recursion": ["functions"],
    "sorting_searching": ["lists", "functions"],
    "modules": ["functions"],
    "numpy_basics": ["lists", "functions"],
    "pandas_basics": ["numpy_basics"],
    "ml_intro": ["pandas_basics", "numpy_basics"],
    "project_capstone": ["ml_intro", "oop_basics", "error_handling"],
}

TOPIC_RESOURCES = {
    "variables": [
        {"title": "Python Variables (Bangla)", "url": "https://www.w3schools.com/python/python_variables.asp",
         "type": "tutorial", "offline": True, "lang": "both"},
        {"title": "Variables Video (BD Creator)", "url": "https://youtube.com/watch?v=example1",
         "type": "video", "offline": False, "lang": "bn"},
        {"title": "Practice Problems", "url": "https://www.hackerrank.com/domains/python",
         "type": "practice", "offline": False, "lang": "en"},
    ],
    "for_loops": [
        {"title": "For Loops Tutorial", "url": "https://www.w3schools.com/python/python_for_loops.asp",
         "type": "tutorial", "offline": True, "lang": "en"},
        {"title": "Loop Exercises", "url": "https://pynative.com/python-for-loop-exercise-with-solution/",
         "type": "practice", "offline": False, "lang": "en"},
    ],
    "functions": [
        {"title": "Python Functions", "url": "https://realpython.com/defining-your-own-python-function/",
         "type": "article", "offline": False, "lang": "en"},
        {"title": "Functions Practice", "url": "https://www.practicepython.org/",
         "type": "practice", "offline": False, "lang": "en"},
    ],
    "ml_intro": [
        {"title": "Scikit-learn Quickstart", "url": "https://scikit-learn.org/stable/getting_started.html",
         "type": "docs", "offline": False, "lang": "en"},
        {"title": "ML for Beginners (Free)", "url": "https://www.coursera.org/learn/machine-learning",
         "type": "course", "offline": False, "lang": "en"},
    ],
}
# Default empty resources for topics not explicitly listed
for topic in PYTHON_CURRICULUM:
    if topic not in TOPIC_RESOURCES:
        TOPIC_RESOURCES[topic] = [
            {"title": f"Learn {PYTHON_CURRICULUM[topic]['name']}",
             "url": f"https://docs.python.org/3/tutorial/",
             "type": "docs", "offline": True, "lang": "en"}
        ]
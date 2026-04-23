"""
Intervention strategies and resources for each struggle type.
"""

INTERVENTION_MAP = {
    "confusion": (
        "Break the concept into smaller steps. Try working through an example "
        "with pen and paper before coding. Use the 'rubber duck' technique: "
        "explain your code line by line out loud."
    ),
    "syntax_error": (
        "Read the error message carefully — it tells you exactly which line. "
        "Check: colons after if/for/def, matching parentheses/quotes, "
        "and correct indentation (4 spaces)."
    ),
    "logic_error": (
        "Add print() statements to check variable values at each step. "
        "Trace through your algorithm manually with a simple example. "
        "Compare your output with the expected output step by step."
    ),
    "concept_gap": (
        "Review the concept with a concrete real-world example. "
        "Watch a short video tutorial, then try a simple exercise before "
        "returning to the harder problem."
    ),
    "motivation_low": (
        "Take a 10-minute break! Come back and try a simpler problem first "
        "to rebuild confidence. Remember why you started learning to code. "
        "Your progress is real — look at what you've already built!"
    ),
    "time_pressure": (
        "Focus on the core requirement first — get a working solution, "
        "then optimize. Break the problem into the smallest possible tasks. "
        "It's okay to submit a partial solution."
    ),
    "language_barrier": (
        "আপনার ভাষায় শেখা সম্পূর্ণ ঠিক আছে! (Learning in your language is fine!) "
        "Check our Bangla resources. Many concepts have direct translations. "
        "Code keywords (def, for, if) are the same in all languages."
    ),
    "no_struggle": (
        "Great question! You're engaging deeply with the material. "
        "Keep exploring — try extending the solution or solving a harder variant."
    ),
    "_default": (
        "Review the relevant section of the tutorial. "
        "Try breaking the problem into smaller pieces and solving each separately."
    ),
}

STRUGGLE_RESOURCES = {
    "confusion": [
        {"title": "Python Visualizer", "url": "https://pythontutor.com", "lang": "both",
         "description": "Visualize your code execution step by step"},
        {"title": "W3Schools Python", "url": "https://www.w3schools.com/python/", "lang": "en",
         "description": "Simple, clear explanations with examples"},
    ],
    "syntax_error": [
        {"title": "Common Python Errors", "url": "https://realpython.com/python-syntax/", "lang": "en",
         "description": "Guide to understanding Python syntax errors"},
        {"title": "Python IDLE", "url": "https://docs.python.org/3/library/idle.html", "lang": "en",
         "description": "Built-in IDE with error highlighting"},
    ],
    "language_barrier": [
        {"title": "Python Bangla Tutorial", "url": "https://www.shikho.com/python-bangla", "lang": "bn",
         "description": "সম্পূর্ণ বাংলায় Python শিখুন"},
        {"title": "Bangla Coding Community", "url": "https://discord.gg/bdpython", "lang": "bn",
         "description": "বাংলাদেশি Python শিক্ষার্থীদের কমিউনিটি"},
    ],
    "motivation_low": [
        {"title": "100 Days of Code Challenge", "url": "https://www.100daysofcode.com/", "lang": "en",
         "description": "Structured daily coding habit builder"},
    ],
    "concept_gap": [
        {"title": "Real Python Tutorials", "url": "https://realpython.com/", "lang": "en",
         "description": "In-depth Python tutorials"},
        {"title": "CS Dojo YouTube", "url": "https://youtube.com/@csdojo", "lang": "en",
         "description": "Visual explanations for CS concepts"},
    ],
}
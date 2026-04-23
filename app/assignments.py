"""
Assignment registry with test cases for auto-grading.
"""

ASSIGNMENT_REGISTRY = {
    "assignment_loops_001": {
        "title": "Factorial with Recursion",
        "description": "Write a recursive factorial function",
        "reference_solution": """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
""",
        "tests": [
            {"name": "test_factorial_0", "test_code": "_result = factorial(0)", "expected": 1},
            {"name": "test_factorial_1", "test_code": "_result = factorial(1)", "expected": 1},
            {"name": "test_factorial_5", "test_code": "_result = factorial(5)", "expected": 120},
            {"name": "test_factorial_10", "test_code": "_result = factorial(10)", "expected": 3628800},
        ],
    },
    "assignment_list_001": {
        "title": "List Sum",
        "description": "Write a function that returns the sum of a list without using sum()",
        "reference_solution": """
def list_sum(lst):
    total = 0
    for item in lst:
        total += item
    return total
""",
        "tests": [
            {"name": "test_empty", "test_code": "_result = list_sum([])", "expected": 0},
            {"name": "test_single", "test_code": "_result = list_sum([5])", "expected": 5},
            {"name": "test_multiple", "test_code": "_result = list_sum([1, 2, 3, 4, 5])", "expected": 15},
        ],
    },
    "assignment_string_001": {
        "title": "Palindrome Check",
        "description": "Check if a string is a palindrome",
        "reference_solution": """
def is_palindrome(s):
    s = s.lower().replace(' ', '')
    return s == s[::-1]
""",
        "tests": [
            {"name": "test_palindrome", "test_code": "_result = is_palindrome('racecar')", "expected": True},
            {"name": "test_not_palindrome", "test_code": "_result = is_palindrome('hello')", "expected": False},
            {"name": "test_phrase", "test_code": "_result = is_palindrome('A man a plan a canal Panama')", "expected": True},
        ],
    },
    "assignment_fibonacci_001": {
        "title": "Fibonacci Sequence",
        "description": "Generate first N fibonacci numbers",
        "reference_solution": """
def fibonacci(n):
    if n <= 0:
        return []
    if n == 1:
        return [0]
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib
""",
        "tests": [
            {"name": "test_fib_5", "test_code": "_result = fibonacci(5)", "expected": [0, 1, 1, 2, 3]},
            {"name": "test_fib_1", "test_code": "_result = fibonacci(1)", "expected": [0]},
            {"name": "test_fib_0", "test_code": "_result = fibonacci(0)", "expected": []},
        ],
    },
}
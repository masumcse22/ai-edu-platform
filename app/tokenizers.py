"""
Simple tokenizers for EduAI BD.
In production, replace with HuggingFace tokenizer fine-tuned on BD student data.
"""
import re
from typing import List


PYTHON_KEYWORDS = [
    "def", "class", "if", "else", "elif", "for", "while", "return", "import",
    "from", "try", "except", "finally", "with", "as", "pass", "break",
    "continue", "lambda", "yield", "assert", "raise", "in", "not", "and",
    "or", "True", "False", "None", "print", "len", "range", "list", "dict",
    "tuple", "set", "str", "int", "float", "bool", "input", "open", "type",
    "self", "super", "__init__", "append", "extend", "pop", "get",
]

NLP_COMMON = [
    "what", "how", "why", "when", "where", "error", "wrong", "help",
    "understand", "confused", "not", "work", "code", "function", "loop",
    "variable", "syntax", "output", "expected", "got", "please", "explain",
    # Bangla romanized
    "ami", "ki", "keno", "kore", "bujhchi", "na", "thik", "hocche",
    "kaj", "korche", "ektu", "help", "chai", "pari",
    # Bangla unicode common words
    "না", "কি", "কেন", "কীভাবে", "বুঝি", "ভুল", "সমস্যা",
]


class SimpleCodeTokenizer:
    """Character + keyword level tokenizer for Python code."""

    def __init__(self):
        self.vocab = {tok: i + 1 for i, tok in enumerate(PYTHON_KEYWORDS)}
        self.vocab["<PAD>"] = 0
        self.vocab["<UNK>"] = len(self.vocab)
        self.vocab_size = len(self.vocab) + 500  # room for char tokens

        # Add character-level vocab
        chars = list("abcdefghijklmnopqrstuvwxyz0123456789_()[]{}:,. \n\t=+-*/")
        for c in chars:
            if c not in self.vocab:
                self.vocab[c] = len(self.vocab)

    def encode(self, code: str, max_len: int = 256) -> List[int]:
        tokens = re.findall(r'\b\w+\b|[^\w\s]', code)
        ids = [self.vocab.get(tok, self.vocab.get("<UNK>", 1)) for tok in tokens]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return ids[:max_len]


class SimpleNLPTokenizer:
    """Word-level tokenizer for student questions (EN + BN)."""

    def __init__(self):
        self.vocab = {tok: i + 1 for i, tok in enumerate(NLP_COMMON)}
        self.vocab["<PAD>"] = 0
        self.vocab["<UNK>"] = len(self.vocab)

    def encode(self, text: str, max_len: int = 128) -> List[int]:
        text = text.lower().strip()
        tokens = text.split()
        ids = [self.vocab.get(tok, self.vocab.get("<UNK>", 1)) for tok in tokens]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return ids[:max_len]
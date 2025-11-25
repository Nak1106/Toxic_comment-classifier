import re
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import DATA_DIR


def load_lexicon(path: Optional[Path] = None) -> List[str]:
    """
    Load a list of offensive terms.

    If a file is provided, each line is treated as one term.
    Otherwise, use a small built in list that you can extend later.
    """
    if path is None:
        # You can extend this list or replace with a file in data/offensive_lexicon.txt
        base_terms = [
            "idiot",
            "stupid",
            "dumb",
            "bastard",
            "bitch",
            "asshole",
            "moron",
            "fuck",
            "fucking",
            "shit",
            "slut",
            "whore",
        ]
        return base_terms

    if not path.exists():
        raise FileNotFoundError(f"Lexicon file not found at {path}")

    terms = []
    with path.open("r", encoding="utf8") as f:
        for line in f:
            t = line.strip().lower()
            if t and not t.startswith("#"):
                terms.append(t)
    return terms


def compile_lexicon_regex(terms: List[str]) -> re.Pattern:
    """
    Compile a regex that matches any of the lexicon terms as whole words.
    """
    escaped = [re.escape(t) for t in terms]
    pattern = r"\b(" + "|".join(escaped) + r")\b"
    return re.compile(pattern, flags=re.IGNORECASE)


def extract_lexicon_features(
    texts: List[str],
    lexicon_terms: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Turn each text into a small lexicon based feature vector.

    For now we use:
      [0] total number of matches
      [1] binary flag: at least one match
      [2] matches per token (length normalized), zero if length is zero

    You can add more features later if needed.
    """
    if lexicon_terms is None:
        lexicon_terms = load_lexicon()

    regex = compile_lexicon_regex(lexicon_terms)
    feats = []

    for txt in texts:
        if not isinstance(txt, str):
            txt = ""
        tokens = txt.split()
        length = max(len(tokens), 1)
        matches = regex.findall(txt)
        count = len(matches)
        has_match = 1.0 if count > 0 else 0.0
        ratio = float(count) / float(length)
        feats.append([float(count), has_match, ratio])

    return np.array(feats, dtype="float32")


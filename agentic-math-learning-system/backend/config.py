# config.py
# ─────────────────────────────────────────────────────────────────────
# LLMConfig centralises all LLM parameters so students can see at a
# glance how temperature and max_tokens differ per task type.
#
# Rule of thumb used throughout this system:
#   temperature ≤ 0.2  →  deterministic, accurate outputs (math, formulas)
#   temperature ≥ 0.7  →  creative, varied outputs (synthetic examples)
# ─────────────────────────────────────────────────────────────────────
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    """
    Immutable configuration for a single LLM task type.

    Attributes:
        model:       Ollama model tag to invoke (e.g. "llama3:latest").
        temperature: Sampling temperature — controls output randomness.
        max_tokens:  Maximum number of tokens the model may generate.
    """

    model: str
    temperature: float
    max_tokens: int


# ── Per-task config constants ─────────────────────────────────────────

# Low temperature (≤ 0.2) for deterministic, accurate math reasoning.
# High temperature would introduce randomness that corrupts formula steps
# and produces inconsistent chain-of-thought reasoning.
TUTOR_CONFIG = LLMConfig(model="llama3:latest", temperature=0.1, max_tokens=1024)

# Low temperature for MCQ generation ensures well-formed, unambiguous
# questions with exactly one correct answer and three plausible distractors.
PRACTICE_CONFIG = LLMConfig(model="llama3:latest", temperature=0.2, max_tokens=512)

# Low temperature for formula extraction — we want exact LaTeX/Unicode
# representations, not creative paraphrases of mathematical notation.
FORMULA_EXTRACT_CONFIG = LLMConfig(model="llama3:latest", temperature=0.1, max_tokens=512)

# High temperature (≥ 0.7) for synthetic example generation encourages
# variety so students see multiple distinct worked examples rather than
# near-identical repetitions of the same numerical substitution.
EXAMPLE_GEN_CONFIG = LLMConfig(model="llama3:latest", temperature=0.8, max_tokens=1024)

# Low temperature for chunk classification — we want consistent labels
# ("formula" | "explanation" | "example") without sampling noise causing
# the same chunk to be classified differently on repeated runs.
CLASSIFY_CONFIG = LLMConfig(model="llama3:latest", temperature=0.1, max_tokens=64)

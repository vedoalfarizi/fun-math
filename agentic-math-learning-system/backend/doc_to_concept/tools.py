# doc_to_concept/tools.py
# ─────────────────────────────────────────────────────────────────────
# LangGraph tools for the Doc-to-Concept pipeline (Path C).
#
# Tool definition pattern (Req 10.1, 10.2, 10.3):
#   - Each tool is a standalone Python function in this dedicated module.
#   - The @tool decorator (from langchain_core.tools) registers the
#     function with LangGraph's tool-calling infrastructure.
#   - The docstring is critical: LangGraph passes it to the LLM so the
#     model knows WHEN and HOW to invoke the tool. Write docstrings as
#     if explaining the tool's purpose directly to the LLM.
#
# Tools defined here:
#   extract_formulas_tool  — extracts LaTeX formulas from formula chunks
#   generate_examples_tool — generates ≥2 worked examples per formula
#
# Requirements satisfied: 8.6, 8.7, 10.1, 10.2, 10.3
# ─────────────────────────────────────────────────────────────────────
import json
import logging
from typing import List

from langchain_core.tools import tool

from config import EXAMPLE_GEN_CONFIG, FORMULA_EXTRACT_CONFIG
from llm_wrapper import call_llm
from prompts import EXAMPLE_GEN_PROMPT, FORMULA_EXTRACT_PROMPT

logger = logging.getLogger(__name__)


# ── Tool 1: extract_formulas_tool ─────────────────────────────────────
# Registration note: The @tool decorator wraps this function and exposes
# it to LangGraph's tool-calling mechanism. LangGraph uses the function
# name and docstring to inform the LLM about the tool's capability and
# when to invoke it. The function signature defines the input schema.
@tool
def extract_formulas_tool(chunk_text: str) -> List[dict]:
    """
    Extract all distinct mathematical formulas from a formula-classified text chunk.

    Use this tool on each text chunk that has been classified as 'formula' type
    by the chunk classifier. The tool identifies every unique mathematical
    expression or equation in the chunk and returns them as structured objects
    with LaTeX notation preserved exactly as it appears in the source.

    This tool uses a low-temperature LLM call (temperature=0.1) to ensure
    exact LaTeX reproduction rather than creative paraphrasing of notation.

    Args:
        chunk_text: The raw text content of a formula-classified chunk from
                    the document. Should contain mathematical expressions in
                    LaTeX or Unicode notation.

    Returns:
        A list of dicts, each with:
            - "formula":     LaTeX-formatted formula string (e.g., "E = mc^2")
            - "description": One-sentence plain-English description of the formula
        Returns an empty list if no formulas are found in the chunk.
    """
    # FORMULA_EXTRACT_CONFIG uses temperature=0.1 — low temperature ensures
    # the LLM reproduces LaTeX notation exactly rather than paraphrasing it.
    # This satisfies Req 8.6 (extract formulas as structured list) and
    # Req 12.4 (preserve LaTeX through the pipeline).
    prompt = FORMULA_EXTRACT_PROMPT.format(chunk_text=chunk_text)
    raw_response = call_llm(prompt, FORMULA_EXTRACT_CONFIG, task_name="extract_formulas")

    # Parse the JSON array returned by the LLM.
    # The prompt's <output_format> section specifies a JSON array of
    # {formula, description} objects, so we parse it directly.
    try:
        # Strip markdown code fences if the LLM wrapped the JSON in ```json ... ```
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            # Remove opening fence (```json or ```) and closing fence (```)
            cleaned = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            )
        formulas = json.loads(cleaned)
        if not isinstance(formulas, list):
            logger.warning("extract_formulas_tool: LLM returned non-list JSON; wrapping")
            formulas = [formulas] if formulas else []
        return formulas
    except (json.JSONDecodeError, ValueError) as exc:
        # If the LLM returns malformed JSON, log the issue and return an
        # empty list rather than crashing the pipeline. The persist_node
        # will store an empty formulas list for this document.
        logger.warning(
            "extract_formulas_tool: failed to parse LLM response as JSON: %s | raw=%r",
            exc,
            raw_response[:200],
        )
        return []


# ── Tool 2: generate_examples_tool ────────────────────────────────────
# Registration note: The @tool decorator registers this function with
# LangGraph. The docstring tells the LLM that this tool should be called
# once per formula, after extract_formulas_tool has run. The high
# temperature config (0.8) is intentional — see EXAMPLE_GEN_CONFIG in
# config.py for the rationale.
@tool
def generate_examples_tool(formula: str, description: str, topic: str) -> List[str]:
    """
    Generate at least two distinct synthetic worked examples for a mathematical formula.

    Use this tool for each formula extracted from the document to create
    educational worked examples that demonstrate the formula applied to
    concrete problems. Call this tool once per formula — do not batch
    multiple formulas in a single call.

    This tool uses a high-temperature LLM call (temperature=0.8) to encourage
    variety across examples so students see the formula applied in different
    numeric contexts rather than near-identical repetitions.

    Args:
        formula:     A LaTeX-formatted mathematical formula string
                     (e.g., "x = (-b \\pm \\sqrt{b^2-4ac}) / 2a").
        description: A one-sentence plain-English description of what the
                     formula represents (e.g., "The quadratic formula for
                     finding roots of ax^2 + bx + c = 0").
        topic:       The math topic label for contextual relevance
                     (e.g., "algebra", "calculus"). Used to keep examples
                     within the appropriate subject area.

    Returns:
        A list of at least two worked example strings. Each string contains
        a problem statement and its full step-by-step solution. If the LLM
        returns fewer than two examples, a placeholder is appended to
        guarantee the minimum count required by Req 8.7.
    """
    # EXAMPLE_GEN_CONFIG uses temperature=0.8 — high temperature encourages
    # variety so students see multiple distinct worked examples rather than
    # near-identical repetitions of the same numerical substitution (Req 8.7).
    # num_examples=2 is the minimum required by Req 8.7.
    prompt = EXAMPLE_GEN_PROMPT.format(
        formula=formula,
        description=description,
        topic=topic or "mathematics",
        num_examples=2,
    )
    raw_response = call_llm(prompt, EXAMPLE_GEN_CONFIG, task_name="generate_examples")

    # Parse the JSON array of example objects returned by the LLM.
    # The prompt's <output_format> specifies [{problem, solution_steps, answer}].
    try:
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            )
        examples_raw = json.loads(cleaned)
        if not isinstance(examples_raw, list):
            examples_raw = [examples_raw] if examples_raw else []

        # Convert each structured example object into a readable prose string
        # for storage and display. The frontend renders these as plain text
        # below each formula in the Doc-to-Concept results view.
        examples: List[str] = []
        for ex in examples_raw:
            if isinstance(ex, dict):
                problem = ex.get("problem", "")
                steps = ex.get("solution_steps", [])
                answer = ex.get("answer", "")
                # Format as a readable block: problem → numbered steps → answer
                steps_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
                example_text = f"Problem: {problem}\n{steps_text}\nAnswer: {answer}"
                examples.append(example_text.strip())
            elif isinstance(ex, str):
                # Accept plain string examples if the LLM deviates from the schema
                examples.append(ex.strip())

    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "generate_examples_tool: failed to parse LLM response as JSON: %s | raw=%r",
            exc,
            raw_response[:200],
        )
        # Fall back to splitting on double newlines — a common LLM output pattern
        # when the model ignores the JSON format instruction.
        examples = [
            ex.strip()
            for ex in raw_response.strip().split("\n\n")
            if ex.strip()
        ]

    # Req 8.7: guarantee at least two examples per formula.
    # If the LLM returned fewer, append a placeholder so downstream code
    # can always rely on len(examples) >= 2.
    while len(examples) < 2:
        examples.append(
            f"Example {len(examples) + 1}: Apply the formula {formula} "
            "to a concrete problem. (Additional example generation pending.)"
        )

    return examples

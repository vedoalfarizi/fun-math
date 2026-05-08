# practice_agent/tools.py
# ─────────────────────────────────────────────────────────────────────
# LangGraph tools for Path B — Practice Agent.
#
# Tools in LangGraph are plain Python functions decorated with @tool
# from langchain_core.tools.  The decorator does two things:
#   1. Wraps the function so LangGraph can invoke it as a node action.
#   2. Reads the function's docstring and type annotations to build a
#      JSON schema that is passed to the LLM.  The LLM uses this schema
#      to decide *when* to call the tool and *what arguments* to supply.
#
# This means the docstring is not just documentation for human readers —
# it is the LLM's instruction manual for the tool.  Write it as if you
# are explaining the tool's purpose and usage to the language model.
#
# Satisfies Req 7.4 (Deep_Dive_Tool invoked on incorrect answer),
#           Req 7.5 (tool defined as explicit Python function with docstring),
#           Req 10.1 (tools in dedicated tools.py per agentic path),
#           Req 10.2 (complete docstrings for LLM tool schema),
#           Req 10.3 (registration-site comment explaining LangGraph exposure).
# ─────────────────────────────────────────────────────────────────────
from langchain_core.tools import tool

from config import TUTOR_CONFIG
from llm_wrapper import call_llm
from prompts import DEEP_DIVE_PROMPT


# ── Tool: deep_dive_tool ──────────────────────────────────────────────
#
# Registration note (Req 10.3):
#   The @tool decorator from langchain_core.tools registers this
#   function as a LangGraph-compatible tool.  LangGraph reads the
#   function's docstring and type annotations to build the tool schema
#   that is passed to the LLM.  The LLM uses the docstring to decide
#   *when* to call the tool (only on incorrect answers) and the type
#   annotations to know *what arguments* to supply (topic, question,
#   incorrect_answer).  Writing a clear, precise docstring is therefore
#   critical for correct tool invocation — the LLM will not call the
#   tool appropriately if the docstring is vague or missing.
@tool
def deep_dive_tool(topic: str, question: str, incorrect_answer: str) -> str:
    """
    Provide a detailed, step-by-step explanation of the concept tested by the question.

    Use this tool ONLY when a student has answered a multiple-choice question
    incorrectly.  Do NOT call this tool for correct answers.

    The explanation is structured in four parts:
      1. Misconception   — identifies the specific misunderstanding implied by
                           the student's wrong answer choice.
      2. Concept Review  — re-explains the underlying math concept from first
                           principles using clear, accessible language.
      3. Correct Solution — walks through the correct solution step-by-step,
                            showing every intermediate calculation.
      4. Worked Example  — provides a similar but distinct problem and solves
                           it completely to reinforce understanding.

    The output is formatted with Markdown headings (## Misconception, etc.)
    so the frontend can render it in a visually distinct panel (Req 11.3).

    Args:
        topic:            The math topic of the question (e.g., "quadratic equations",
                          "integration by parts", "matrix multiplication").
                          Used to frame the concept review at the right level.
        question:         The full text of the MCQ question stem that the student
                          answered incorrectly.  Provides context for identifying
                          the specific misconception.
        incorrect_answer: The text of the answer option the student selected
                          (e.g., "A) x = 3").  Used to diagnose the misconception
                          and tailor the explanation to the student's error.

    Returns:
        A multi-section Markdown string containing the misconception analysis,
        concept review, correct solution, and a worked example.  The string
        is stored in PracticeState["explanation"] and returned to the client
        in PracticeAnswerResponse.explanation.
    """
    # Render the DEEP_DIVE_PROMPT with the three context variables.
    # TUTOR_CONFIG uses temperature=0.1 for accurate, deterministic
    # step-by-step reasoning — we want precise math, not creative variation.
    prompt = DEEP_DIVE_PROMPT.format(
        topic=topic,
        question=question,
        incorrect_answer=incorrect_answer,
    )

    explanation = call_llm(prompt, TUTOR_CONFIG, task_name="deep_dive")
    return explanation

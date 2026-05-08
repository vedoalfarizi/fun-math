# practice_agent/nodes.py
# ─────────────────────────────────────────────────────────────────────
# Node functions for the Practice Agent LangGraph (Path B).
#
# Each function is a pure transformation: it receives the current
# PracticeState dict, performs one focused task, and returns an updated
# copy of the state.  Keeping nodes as standalone functions makes the
# graph topology in graph.py easy to read.
#
# The Practice Agent spans two HTTP requests:
#
#   POST /api/practice/start
#     Entry point: generate_question_node → END
#     Generates an MCQ and returns it to the client.
#
#   POST /api/practice/answer
#     Entry point: evaluate_answer_node
#     Correct:   evaluate_answer_node → persist_node → END
#     Incorrect: evaluate_answer_node → deep_dive_node → persist_node → END
#
# Satisfies Req 7.1 (generate MCQ), 7.2 (evaluate answer),
#           Req 7.3 (correct → next_question, incorrect → deep_dive),
#           Req 7.4 (deep_dive_tool invoked on incorrect),
#           Req 7.6 (MCQ generation with PRACTICE_CONFIG temp≤0.2),
#           Req 7.7 (persist full state machine history).
# ─────────────────────────────────────────────────────────────────────
import json
import logging
import re
import uuid

from practice_agent.state import PracticeState
from practice_agent.tools import deep_dive_tool
from config import PRACTICE_CONFIG
from database import get_session
from llm_wrapper import call_llm
from models.db_models import PracticeSession
from prompts import MCQ_GENERATION_PROMPT

logger = logging.getLogger(__name__)


# ── Node 1: generate_question_node ────────────────────────────────────
def generate_question_node(state: PracticeState) -> PracticeState:
    """
    Generate a multiple-choice question using the Ollama LLM.

    Calls the LLM with MCQ_GENERATION_PROMPT and PRACTICE_CONFIG
    (temperature=0.2) to produce a deterministic, well-formed MCQ
    (Req 7.6).  Parses the JSON response to extract question_text,
    options, and correct_option.

    A new session_id and question_id are generated here if not already
    present in the state (i.e., on the first /start call).

    Args:
        state: PracticeState with topic and difficulty_level set.

    Returns:
        Updated PracticeState with session_id, question_id,
        question_text, options, correct_option, and graph_status set.
    """
    # Generate UUIDs for the session and question.
    # session_id may already be set if the router pre-populated it;
    # we use setdefault-style logic to avoid overwriting an existing value.
    session_id = state.get("session_id") or str(uuid.uuid4())
    question_id = str(uuid.uuid4())

    # Render the MCQ generation prompt with topic and difficulty.
    prompt = MCQ_GENERATION_PROMPT.format(
        topic=state["topic"],
        difficulty_level=state["difficulty_level"],
    )

    # PRACTICE_CONFIG uses temperature=0.2 for deterministic, unambiguous
    # questions with exactly one correct answer (Req 7.6).
    raw_response = call_llm(prompt, PRACTICE_CONFIG, task_name="mcq_generation")

    # Parse the LLM's JSON response.
    # The prompt instructs the LLM to output valid JSON only, but we apply
    # a defensive extraction step to handle any stray prose the LLM may
    # prepend or append despite the instruction.
    question_text, options, correct_option = _parse_mcq_response(raw_response)

    logger.info(
        "generate_question_node | session=%s question_id=%s topic=%r difficulty=%s",
        session_id,
        question_id,
        state["topic"],
        state["difficulty_level"],
    )

    return {
        **state,
        "session_id": session_id,
        "question_id": question_id,
        "question_text": question_text,
        "options": options,
        "correct_option": correct_option,
        # graph_status signals to the client that the session is now
        # waiting for an answer via POST /api/practice/answer.
        "graph_status": "evaluate_answer",
        # Initialise history as an empty list; persist_node will populate it.
        "history": state.get("history") or [],
    }


def _parse_mcq_response(raw: str) -> tuple[str, list[str], str]:
    """
    Extract question_text, options list, and correct_option from the LLM response.

    The LLM is instructed to return valid JSON matching the MCQ_GENERATION_PROMPT
    output_format schema.  This helper extracts the JSON block defensively,
    handling cases where the LLM wraps the JSON in a markdown code fence or
    adds a brief preamble.

    Args:
        raw: Raw string response from the LLM.

    Returns:
        Tuple of (question_text, options, correct_option) where:
            question_text — the MCQ question stem string
            options       — list of four strings in "A) ..." format
            correct_option — single letter "A", "B", "C", or "D"

    Raises:
        ValueError: If the JSON cannot be parsed or required keys are missing.
    """
    # Strip markdown code fences if the LLM wrapped the JSON in ```json ... ```.
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    # Find the first '{' and last '}' to isolate the JSON object.
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in LLM response: {raw[:200]!r}")

    json_str = cleaned[start:end]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse MCQ JSON: {exc}\nRaw: {raw[:200]!r}") from exc

    # Validate required keys.
    for key in ("question_text", "options", "correct_option"):
        if key not in data:
            raise ValueError(f"MCQ JSON missing required key '{key}': {data}")

    question_text: str = data["question_text"]
    raw_options: dict = data["options"]
    correct_option: str = data["correct_option"].strip().upper()

    # Normalise options dict {"A": "...", "B": "...", ...} into a list
    # ["A) ...", "B) ...", "C) ...", "D) ..."] matching PracticeStartResponse.options.
    options = [f"{letter}) {text}" for letter, text in sorted(raw_options.items())]

    return question_text, options, correct_option


# ── Node 2: evaluate_answer_node ──────────────────────────────────────
def evaluate_answer_node(state: PracticeState) -> PracticeState:
    """
    Compare the student's selected option to the correct option and set outcome.

    This node performs a simple string comparison — no LLM call is needed
    because the correct answer was determined deterministically during MCQ
    generation and stored server-side in state["correct_option"].

    The LangGraph conditional edge in graph.py reads state["outcome"] to
    route to either persist_node (correct) or deep_dive_node (incorrect).

    Args:
        state: PracticeState with selected_option and correct_option set.

    Returns:
        Updated PracticeState with 'outcome' set to "correct" or "incorrect".
    """
    selected = state["selected_option"].strip().upper()
    correct = state["correct_option"].strip().upper()

    outcome = "correct" if selected == correct else "incorrect"

    logger.info(
        "evaluate_answer_node | session=%s selected=%s correct=%s outcome=%s",
        state.get("session_id"),
        selected,
        correct,
        outcome,
    )

    return {**state, "outcome": outcome}


# ── Node 3: deep_dive_node ────────────────────────────────────────────
def deep_dive_node(state: PracticeState) -> PracticeState:
    """
    Invoke the deep_dive_tool to generate a detailed concept explanation.

    This node is only reached when outcome == "incorrect" (via the
    conditional edge in graph.py).  It calls deep_dive_tool with the
    topic, question text, and the student's incorrect answer so the
    explanation is targeted at the specific misconception (Req 7.4).

    The explanation is stored in state["explanation"] and later returned
    to the client in PracticeAnswerResponse.explanation, where the
    frontend renders it in a visually distinct yellow panel (Req 11.3).

    Args:
        state: PracticeState with topic, question_text, selected_option,
               and outcome == "incorrect".

    Returns:
        Updated PracticeState with 'explanation' populated and
        'graph_status' set to "retry".
    """
    # Retrieve the text of the option the student selected so the
    # deep_dive_tool can reference the specific wrong answer in its explanation.
    selected_letter = state["selected_option"].strip().upper()
    # Find the matching option text from the options list (e.g., "A) ...")
    selected_text = next(
        (opt for opt in state.get("options", []) if opt.startswith(selected_letter)),
        selected_letter,  # fallback to just the letter if options list is unavailable
    )

    # Invoke the deep_dive_tool.
    # deep_dive_tool is a @tool-decorated function; calling .invoke() passes
    # the arguments through LangGraph's tool invocation layer, which handles
    # argument validation and logging.
    explanation = deep_dive_tool.invoke({
        "topic": state["topic"],
        "question": state["question_text"],
        "incorrect_answer": selected_text,
    })

    logger.info(
        "deep_dive_node | session=%s explanation_length=%d",
        state.get("session_id"),
        len(explanation),
    )

    return {
        **state,
        "explanation": explanation,
        # graph_status "retry" signals to the client that the student
        # should review the explanation and attempt the question again.
        "graph_status": "retry",
    }


# ── Node 4: persist_node ──────────────────────────────────────────────
def persist_node(state: PracticeState) -> PracticeState:
    """
    Persist the completed turn to the PostgreSQL practice_sessions table.

    Implements an upsert pattern (Req 7.7):
      - If a PracticeSession row with the given session_id already exists,
        the new turn is appended to its 'history' JSON list.
      - If no row exists, a new PracticeSession is created first.

    Each history entry stored in the JSON column contains:
        question_text    — the MCQ question stem
        options          — the four answer options
        correct_option   — the correct answer letter (for review purposes)
        selected_option  — the student's chosen answer
        outcome          — "correct" or "incorrect"
        explanation      — deep dive text (None if outcome was "correct")

    Uses get_session() from database.py so the connection is properly
    returned to the pool after the write completes.

    Args:
        state: Fully populated PracticeState after evaluate_answer_node
               (and optionally deep_dive_node) has run.

    Returns:
        Updated PracticeState with 'graph_status' set to "next_question"
        if outcome was "correct", or unchanged "retry" if incorrect.
    """
    with get_session() as db:
        # Attempt to load an existing session row.
        record = (
            db.query(PracticeSession)
            .filter_by(session_id=state["session_id"])
            .first()
        )

        if not record:
            # First answer for this session — create a new row.
            logger.info(
                "persist_node | creating new practice session session_id=%s",
                state["session_id"],
            )
            record = PracticeSession(
                session_id=state["session_id"],
                topic=state.get("topic"),
                difficulty_level=state.get("difficulty_level"),
                history=[],
            )
            db.add(record)

        # Build the turn dict to append to the history JSON column.
        turn = {
            "question_id": state.get("question_id"),
            "question_text": state.get("question_text"),
            "options": state.get("options", []),
            "correct_option": state.get("correct_option"),
            "selected_option": state.get("selected_option"),
            "outcome": state.get("outcome"),
            # explanation is None when the student answered correctly.
            "explanation": state.get("explanation"),
        }

        # SQLAlchemy does not detect in-place mutations to JSON columns by default.
        # Reassigning the attribute ensures the ORM marks the column as dirty
        # and includes it in the UPDATE statement.
        new_history = (record.history or []) + [turn]
        record.history = new_history

        db.commit()

    # Log outside the session using the local variable — accessing record.history
    # here would raise DetachedInstanceError because the session is already closed.
    logger.info(
        "persist_node | session=%s outcome=%s turns_total=%d",
        state["session_id"],
        state.get("outcome"),
        len(new_history),
    )

    # Set graph_status to "next_question" when the student answered correctly,
    # signalling the frontend to offer a new question.  For incorrect answers,
    # graph_status was already set to "retry" by deep_dive_node.
    updated_status = (
        "next_question" if state.get("outcome") == "correct" else state.get("graph_status", "retry")
    )

    return {**state, "graph_status": updated_status}

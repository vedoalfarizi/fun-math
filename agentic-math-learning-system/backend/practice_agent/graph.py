# practice_agent/graph.py
# ─────────────────────────────────────────────────────────────────────
# LangGraph graph definition for the Practice Agent (Path B).
#
# The graph topology is defined in a single function so students can
# read the full execution flow in one place (Req 10.4).  Every node,
# edge, and conditional branch is visible here without needing to trace
# through multiple files.
#
# Key LangGraph concepts demonstrated (contrasted with Path A):
#
#   Path A (Adaptive Tutor) — single-request pipeline:
#     The entire graph runs within one HTTP request.  The entry point
#     is always "router" and the graph runs to END in one shot.
#
#   Path B (Practice Agent) — two-request state machine:
#     The graph is invoked TWICE across two separate HTTP requests.
#     /start  → graph runs from "generate_question" entry → END
#     /answer → graph runs from "evaluate_answer" entry → END
#     This demonstrates how LangGraph supports resumable, stateful
#     workflows where the client holds state between invocations.
#
# Satisfies Req 7.3 (correct → next_question, incorrect → deep_dive),
#           Req 7.4 (deep_dive_node reached on incorrect answer),
#           Req 10.4 (graph topology in a single, clearly commented function).
# ─────────────────────────────────────────────────────────────────────
from langgraph.graph import END, StateGraph

from practice_agent.nodes import (
    deep_dive_node,
    evaluate_answer_node,
    generate_question_node,
    persist_node,
)
from practice_agent.state import PracticeState


def build_practice_graph() -> StateGraph:
    """
    Construct and compile the Practice Agent LangGraph.

    Graph topology
    ──────────────
    Entry point: generate_question  (used by POST /api/practice/start)

    /start path:
        generate_question → END
        (returns the MCQ to the client; graph halts here)

    /answer path (graph re-entered at evaluate_answer):
        evaluate_answer ──▶ persist ──▶ END          (outcome == "correct")
        evaluate_answer ──▶ deep_dive ──▶ persist ──▶ END  (outcome == "incorrect")

    The conditional edge after 'evaluate_answer' reads state["outcome"]
    to choose between the two answer paths.  This is the key pattern
    students should study: a single evaluation step gates whether the
    Deep_Dive_Tool is invoked.

    Node responsibilities:
        generate_question — calls LLM with MCQ_GENERATION_PROMPT; parses
                            question_text, options, correct_option from JSON
        evaluate_answer   — compares selected_option to correct_option;
                            sets state["outcome"] to "correct" or "incorrect"
        deep_dive         — invokes deep_dive_tool; populates state["explanation"]
                            with a targeted concept explanation
        persist           — upserts the completed turn to PostgreSQL
                            practice_sessions table

    Two-request invocation pattern:
        The compiled graph is invoked with different entry points depending
        on the HTTP endpoint:
          /start  → _graph.invoke(state)  (default entry = generate_question)
          /answer → _graph.invoke(state, config={"entry_point": "evaluate_answer"})
        See router.py for the exact invocation pattern.

    Returns:
        A compiled LangGraph StateGraph ready to be invoked with an
        initial PracticeState dict.
    """
    # Initialise the graph with the PracticeState TypedDict as the state schema.
    # LangGraph uses the TypedDict to validate state keys at each node boundary.
    graph = StateGraph(PracticeState)

    # ── Register nodes ────────────────────────────────────────────────
    # Each node is a pure function defined in nodes.py.
    # The string name appears in LangGraph's execution trace and in the
    # conditional edge mapping below.
    graph.add_node("generate_question", generate_question_node)
    graph.add_node("evaluate_answer", evaluate_answer_node)
    graph.add_node("deep_dive", deep_dive_node)
    graph.add_node("persist", persist_node)

    # ── Entry point ───────────────────────────────────────────────────
    # The default entry point is "generate_question", used by /start.
    # The /answer route overrides this by passing entry_point="evaluate_answer"
    # in the graph invocation config (see router.py).
    graph.set_entry_point("generate_question")

    # ── /start path: generate_question → END ─────────────────────────
    # After generating the MCQ, the graph halts immediately.
    # The question is returned to the client; the graph does NOT proceed
    # to evaluate_answer here — that happens in the next HTTP request.
    # This edge demonstrates how LangGraph supports partial graph execution.
    graph.add_edge("generate_question", END)

    # ── /answer path: conditional edge after evaluate_answer ──────────
    # This is the core branching decision of Path B.  LangGraph evaluates
    # the lambda after evaluate_answer_node completes and uses the returned
    # string to look up the next node in the mapping dict.
    #
    # The lambda reads state["outcome"] which was set by evaluate_answer_node.
    # The mapping keys must exactly match the values evaluate_answer_node writes:
    #   "correct"   → persist (no explanation needed; advance to next question)
    #   "incorrect" → deep_dive (invoke Deep_Dive_Tool before persisting)
    graph.add_conditional_edges(
        "evaluate_answer",                  # source node
        lambda state: state["outcome"],     # function that returns the routing key
        {
            "correct": "persist",           # correct answer → skip deep dive
            "incorrect": "deep_dive",       # wrong answer → explain the concept
        },
    )

    # ── Incorrect answer tail: deep_dive → persist → END ─────────────
    # After the deep dive explanation is generated, the turn is persisted
    # (including the explanation) and the graph halts.
    graph.add_edge("deep_dive", "persist")

    # ── Shared tail: persist → END ────────────────────────────────────
    # Both the correct and incorrect paths converge at persist → END.
    graph.add_edge("persist", END)

    # ── Compile ───────────────────────────────────────────────────────
    # compile() validates the graph structure (checks for unreachable
    # nodes, missing edges, etc.) and returns an executable object.
    # The compiled graph is stored as a module-level singleton in
    # router.py so it is built only once at application startup.
    return graph.compile()

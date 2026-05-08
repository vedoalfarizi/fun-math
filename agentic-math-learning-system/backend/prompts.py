# prompts.py
# ─────────────────────────────────────────────────────────────────────
# All prompt templates are defined here as named string constants so
# that business logic modules never contain inline prompt text.
# Centralising prompts in one file satisfies Req 9.1 and makes it easy
# for students to study prompt engineering patterns in isolation.
#
# Structural conventions used throughout this file (Req 9.2, 9.3):
#   <context>      — retrieved knowledge chunks injected at inference time
#   <question>     — the student's raw query
#   <instructions> — behavioural rules the LLM must follow
#   <examples>     — in-context learning (ICL) demonstrations (Req 9.4)
#   <output_format>— exact schema the LLM must produce
#
# Every delimiter section carries a Python comment explaining its role.
# ─────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════
# PATH A — ADAPTIVE TUTOR
# ══════════════════════════════════════════════════════════════════════

# ── TUTOR_ROUTER_PROMPT ───────────────────────────────────────────────
# Purpose: Classify an incoming student query so the LangGraph router
# node can decide whether to run the full RAG pipeline or answer
# directly.  A low-temperature LLM call is used here (TUTOR_CONFIG,
# temp=0.1) because we need a deterministic, binary classification.
#
# <question>     — the raw student query to classify
# <instructions> — the two valid output labels and decision criteria
# <output_format>— forces a single-word response to simplify parsing
TUTOR_ROUTER_PROMPT = """\
<question>
{query}
</question>

<instructions>
You are a query classifier for a math tutoring system.
Decide whether answering the question above requires retrieving
knowledge from a document store, or whether it can be answered
directly from general mathematical knowledge.

Rules:
- Reply "knowledge_required" if the question asks about a specific
  theorem, formula, proof, or concept that may be in the knowledge base.
- Reply "direct_answer" if the question is a straightforward arithmetic
  calculation or a general math fact that any LLM can answer reliably
  without external context.
- Do NOT explain your reasoning. Output only one of the two labels.
</instructions>

<output_format>
knowledge_required | direct_answer
</output_format>
"""

# ── TUTOR_RAG_PROMPT ──────────────────────────────────────────────────
# Purpose: Generate a Chain-of-Thought (CoT) answer grounded in
# retrieved context chunks.  Used by generate_node when route ==
# "knowledge_required".  Satisfies Req 6.5 (XML delimiters), 6.6 (CoT
# instruction), and 9.4 (in-context learning example).
#
# <context>      — TOP_N reranked chunks injected at inference time;
#                  grounding the answer in retrieved text reduces
#                  hallucination risk
# <examples>     — one ICL demonstration showing the expected CoT
#                  output format so the LLM mimics the structure
#                  (few-shot prompting, Req 9.4)
# <question>     — the student's question
# <instructions> — CoT reasoning rules; the LLM must list steps before
#                  stating the final answer (Req 6.6)
# <output_format>— two labelled sections the frontend renders separately
TUTOR_RAG_PROMPT = """\
<context>
{context}
</context>

<examples>
The following example demonstrates the required Chain-of-Thought output
format.  Reproduce this structure exactly for every response.

Q: What is the quadratic formula and how is it derived?

## Reasoning
Step 1 — Start from the standard form of a quadratic equation:
         ax² + bx + c = 0
Step 2 — Divide every term by a (assuming a ≠ 0):
         x² + (b/a)x + (c/a) = 0
Step 3 — Complete the square by adding and subtracting (b/2a)²:
         (x + b/2a)² = (b² - 4ac) / 4a²
Step 4 — Take the square root of both sides:
         x + b/2a = ± √(b² - 4ac) / 2a
Step 5 — Isolate x:
         x = (-b ± √(b² - 4ac)) / 2a

## Answer
The quadratic formula is x = (-b ± √(b² - 4ac)) / 2a.
It is derived by completing the square on the general form ax² + bx + c = 0.
</examples>

<question>
{question}
</question>

<instructions>
You are an expert math tutor.  Use ONLY the information in <context> to
answer the question.  If the context does not contain enough information,
say so explicitly rather than guessing.

Follow the Chain-of-Thought format shown in <examples>:
1. Under the heading "## Reasoning", list every logical step you take,
   referencing the context where relevant.
2. Under the heading "## Answer", state the final answer concisely.

Do not skip steps.  Show all intermediate calculations.
</instructions>

<output_format>
## Reasoning
<step-by-step reasoning here>

## Answer
<concise final answer here>
</output_format>
"""

# ── TUTOR_DIRECT_PROMPT ───────────────────────────────────────────────
# Purpose: Answer a query that the router classified as "direct_answer"
# — no retrieved context is available, so the LLM relies on its
# parametric knowledge.  CoT is still required (Req 6.6) to keep the
# output format consistent with the RAG path.
#
# <question>     — the student's question
# <instructions> — CoT rules identical to the RAG path for consistency
# <output_format>— same two-section schema as TUTOR_RAG_PROMPT so the
#                  frontend can render both paths identically
TUTOR_DIRECT_PROMPT = """\
<question>
{question}
</question>

<instructions>
You are an expert math tutor answering from your own knowledge.
No external context has been retrieved for this question.

Follow the Chain-of-Thought format:
1. Under the heading "## Reasoning", list every logical step you take.
2. Under the heading "## Answer", state the final answer concisely.

Show all intermediate calculations.  Do not skip steps.
</instructions>

<output_format>
## Reasoning
<step-by-step reasoning here>

## Answer
<concise final answer here>
</output_format>
"""


# ══════════════════════════════════════════════════════════════════════
# PATH B — PRACTICE AGENT
# ══════════════════════════════════════════════════════════════════════

# ── MCQ_GENERATION_PROMPT ─────────────────────────────────────────────
# Purpose: Generate a well-formed multiple-choice question for the
# Practice Agent.  Used by generate_question_node with PRACTICE_CONFIG
# (temp=0.2) to ensure deterministic, unambiguous questions (Req 7.6).
#
# <instructions> — topic, difficulty, and structural requirements for
#                  the MCQ; low temperature keeps the question precise
# <output_format>— strict JSON schema that nodes.py parses to extract
#                  question_text, options, and correct_option
MCQ_GENERATION_PROMPT = """\
<instructions>
Generate a single multiple-choice math question on the topic below.

Topic: {topic}
Difficulty: {difficulty_level}

Requirements:
- The question stem must be unambiguous and self-contained.
- Provide exactly four answer options labelled A, B, C, and D.
- Exactly one option must be correct; the other three must be plausible
  but clearly wrong to a student who understands the concept.
- Do not include the answer explanation in the question or options.
- Output valid JSON only — no prose before or after the JSON block.
</instructions>

<output_format>
{{
  "question_text": "<question stem>",
  "options": {{
    "A": "<option A text>",
    "B": "<option B text>",
    "C": "<option C text>",
    "D": "<option D text>"
  }},
  "correct_option": "<A | B | C | D>"
}}
</output_format>
"""

# ── DEEP_DIVE_PROMPT ──────────────────────────────────────────────────
# Purpose: Produce a detailed concept explanation when a student answers
# an MCQ incorrectly.  Invoked by deep_dive_tool (Req 7.4, 7.5).
# Uses TUTOR_CONFIG (temp=0.1) for accurate, step-by-step reasoning.
#
# <context>      — the original question and the student's wrong answer,
#                  giving the LLM the misconception to address
# <instructions> — four-part explanation structure that targets the
#                  specific misconception rather than restating the topic
# <output_format>— labelled sections the frontend renders in a
#                  visually distinct "deep dive" panel (Req 11.3)
DEEP_DIVE_PROMPT = """\
<context>
Topic: {topic}
Question: {question}
Student's incorrect answer: {incorrect_answer}
</context>

<instructions>
You are an expert math tutor helping a student who answered incorrectly.
Provide a thorough explanation structured in four parts:

1. Misconception — Identify the specific misunderstanding implied by
   the student's wrong answer.
2. Concept Review — Re-explain the underlying concept from first
   principles, using clear language suitable for the difficulty level.
3. Correct Solution — Walk through the correct solution step-by-step,
   showing every calculation.
4. Worked Example — Provide a similar but distinct example problem and
   solve it completely to reinforce understanding.

Be encouraging and precise.  Do not simply state the correct answer
without explanation.
</instructions>

<output_format>
## Misconception
<identify the specific misunderstanding>

## Concept Review
<first-principles explanation>

## Correct Solution
<step-by-step solution to the original question>

## Worked Example
<similar problem and full solution>
</output_format>
"""


# ══════════════════════════════════════════════════════════════════════
# PATH C — DOC-TO-CONCEPT AGENT
# ══════════════════════════════════════════════════════════════════════

# ── FORMULA_EXTRACT_PROMPT ────────────────────────────────────────────
# Purpose: Extract all distinct mathematical formulas from a chunk that
# has been classified as "formula" type.  Used by extract_formulas_tool
# with FORMULA_EXTRACT_CONFIG (temp=0.1) for exact LaTeX output
# (Req 8.6).
#
# <context>      — the raw text of the formula-classified chunk;
#                  providing the source text prevents the LLM from
#                  hallucinating formulas not present in the document
# <instructions> — extraction rules that preserve LaTeX syntax and
#                  require a structured list output
# <output_format>— JSON array so nodes.py can iterate over formulas
#                  and pass each one to EXAMPLE_GEN_PROMPT
FORMULA_EXTRACT_PROMPT = """\
<context>
{chunk_text}
</context>

<instructions>
You are a mathematical formula extractor.  Read the text in <context>
and identify every distinct mathematical formula or equation present.

Rules:
- Preserve the original LaTeX or Unicode notation exactly as it appears.
- Do not paraphrase, simplify, or rewrite any formula.
- Each formula must be a self-contained expression (e.g., "E = mc²",
  "x = (-b ± √(b²-4ac)) / 2a").
- Exclude prose descriptions; include only the symbolic expressions.
- If no formulas are present, return an empty array.
- Output valid JSON only — no prose before or after the JSON block.
</instructions>

<output_format>
[
  {{
    "formula": "<LaTeX or Unicode formula string>",
    "description": "<one-sentence plain-English description of what the formula represents>"
  }}
]
</output_format>
"""

# ── EXAMPLE_GEN_PROMPT ────────────────────────────────────────────────
# Purpose: Generate synthetic worked examples for a single extracted
# formula.  Used by generate_examples_tool with EXAMPLE_GEN_CONFIG
# (temp=0.8) to encourage variety across examples (Req 8.7).
# High temperature is intentional here — see EXAMPLE_GEN_CONFIG comment
# in config.py.
#
# <context>      — the formula and its description so the LLM has
#                  precise scope for the examples it generates
# <instructions> — minimum count (≥ 2), numeric variety requirement,
#                  and step-by-step solution format
# <output_format>— JSON array consumed by generate_examples_node to
#                  build the DocConceptsResponse
EXAMPLE_GEN_PROMPT = """\
<context>
Formula: {formula}
Description: {description}
Topic: {topic}
</context>

<instructions>
You are a math educator creating worked examples for students.
Generate at least {num_examples} distinct worked examples that
demonstrate the formula in <context>.

Requirements:
- Each example must use different numeric values so students see
  variety rather than near-identical repetitions.
- Show every calculation step; do not skip intermediate results.
- State the final answer clearly at the end of each example.
- Examples should progress from simpler to more complex where possible.
- Output valid JSON only — no prose before or after the JSON block.
</instructions>

<output_format>
[
  {{
    "problem": "<problem statement using the formula>",
    "solution_steps": [
      "<step 1>",
      "<step 2>",
      "<step N>"
    ],
    "answer": "<final numeric or symbolic answer>"
  }}
]
</output_format>
"""

# ── CHUNK_CLASSIFY_PROMPT ─────────────────────────────────────────────
# Purpose: Classify a single text chunk as "formula", "explanation", or
# "example".  Used by classify_node with CLASSIFY_CONFIG (temp=0.1) for
# deterministic classification (Req 8.4).  The classification is stored
# as chunk metadata in ChromaDB (Req 4.4) and drives downstream
# formula-extraction logic.
#
# <context>      — the raw chunk text to classify; providing the full
#                  text (not a summary) ensures accurate classification
# <instructions> — precise definitions of each label so the LLM applies
#                  them consistently across all chunks in a document
# <output_format>— single-word response to simplify parsing in
#                  classify_node; no JSON overhead needed here
CHUNK_CLASSIFY_PROMPT = """\
<context>
{chunk_text}
</context>

<instructions>
Classify the text in <context> into exactly one of three categories:

- "formula"      — The chunk consists primarily of mathematical
                   equations, symbolic expressions, or LaTeX notation.
                   Example: a block containing "F = ma" or the
                   quadratic formula derivation.

- "explanation"  — The chunk consists primarily of prose that explains
                   a mathematical concept, theorem, or procedure in
                   natural language without being a worked example.

- "example"      — The chunk consists primarily of a worked problem:
                   a specific numeric scenario is set up and solved
                   step-by-step.

Rules:
- Output only one of the three labels: formula | explanation | example
- Do not output any other text, punctuation, or explanation.
- If the chunk contains a mix, choose the dominant category.
</instructions>

<output_format>
formula | explanation | example
</output_format>
"""

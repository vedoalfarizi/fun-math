// src/pages/AdaptiveTutor.tsx
// ─────────────────────────────────────────────────────────────────────
// Path A — Adaptive Tutor chat interface (Req 11.1, 11.2, 11.4, 11.5).
//
// Features:
//  • Scrollable chat history showing student questions and tutor answers.
//  • Topic filter input to narrow ChromaDB retrieval.
//  • Question input + Send button; submits on Enter key too.
//  • Renders LLM response with "## Reasoning" in muted text and
//    "## Answer" in bold (Req 11.2).
//  • Displays a route_taken badge ("RAG" or "Direct") per response.
//  • Shows LoadingSpinner while the API call is in flight (Req 11.4).
//  • Shows ErrorBanner on non-2xx API errors (Req 11.5).
//  • Generates a stable UUID session_id on component mount.
// ─────────────────────────────────────────────────────────────────────

import { useEffect, useRef, useState } from 'react'
import { tutorChat, type TutorChatResponse } from '../api/client'
import ErrorBanner from '../components/ErrorBanner'
import LoadingSpinner from '../components/LoadingSpinner'

// ── Types ─────────────────────────────────────────────────────────────

/** One entry in the visible chat history. */
interface ChatEntry {
  id: string
  role: 'student' | 'tutor'
  /** Raw question text (student) or full LLM response string (tutor). */
  text: string
  /** Only present on tutor entries. */
  routeTaken?: string
}

// ── Helpers ───────────────────────────────────────────────────────────

/**
 * Generate a RFC-4122 v4 UUID.
 * Uses crypto.randomUUID() when available (all modern browsers),
 * falling back to a manual implementation for older environments.
 */
function generateUUID(): string {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID()
  }
  // Fallback: manual v4 UUID construction
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0
    const v = c === 'x' ? r : (r & 0x3) | 0x8
    return v.toString(16)
  })
}

/**
 * Parse the LLM response string into separate reasoning and answer sections.
 *
 * The backend CoT prompt instructs the LLM to structure its output with
 * "## Reasoning" and "## Answer" headings. This function splits on those
 * headings so the UI can style each section differently (Req 11.2).
 *
 * If neither heading is present the entire text is treated as the answer.
 */
function parseResponse(text: string): { reasoning: string; answer: string } {
  // Normalise line endings and trim surrounding whitespace.
  const normalised = text.replace(/\r\n/g, '\n').trim()

  // Match case-insensitively to handle minor LLM formatting variations.
  const reasoningMatch = normalised.match(/##\s*reasoning\s*\n([\s\S]*?)(?=##\s*answer|$)/i)
  const answerMatch = normalised.match(/##\s*answer\s*\n([\s\S]*)/i)

  const reasoning = reasoningMatch ? reasoningMatch[1].trim() : ''
  const answer = answerMatch ? answerMatch[1].trim() : normalised

  return { reasoning, answer }
}

/**
 * Map the raw route_taken value from the API to a short display label.
 * The backend returns "knowledge_required" or "direct_answer".
 */
function routeLabel(routeTaken: string): 'RAG' | 'Direct' {
  return routeTaken === 'knowledge_required' ? 'RAG' : 'Direct'
}

// ── Component ─────────────────────────────────────────────────────────

export default function AdaptiveTutor() {
  // Stable session ID generated once on mount — persists across questions
  // in the same browser session so the backend can track conversation history.
  const [sessionId] = useState<string>(() => generateUUID())

  // Chat history displayed in the scrollable message list.
  const [history, setHistory] = useState<ChatEntry[]>([])

  // Controlled inputs
  const [topic, setTopic] = useState('')
  const [query, setQuery] = useState('')

  // API call state
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Ref to the bottom of the chat list — used to auto-scroll after each reply.
  const bottomRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to the latest message whenever history changes.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [history])

  // ── Submit handler ──────────────────────────────────────────────────

  async function handleSend() {
    const trimmedQuery = query.trim()
    if (!trimmedQuery || isLoading) return

    // Append the student's question to the chat history immediately so the
    // UI feels responsive before the API call completes.
    const studentEntry: ChatEntry = {
      id: generateUUID(),
      role: 'student',
      text: trimmedQuery,
    }
    setHistory((prev) => [...prev, studentEntry])
    setQuery('')
    setError(null)
    setIsLoading(true)

    try {
      const response: TutorChatResponse = await tutorChat({
        session_id: sessionId,
        query: trimmedQuery,
        topic: topic.trim() || undefined,
      })

      const tutorEntry: ChatEntry = {
        id: generateUUID(),
        role: 'tutor',
        text: response.response,
        routeTaken: response.route_taken,
      }
      setHistory((prev) => [...prev, tutorEntry])
    } catch (err) {
      // apiFetch throws an Error with a descriptive message on non-2xx (Req 11.5).
      setError(err instanceof Error ? err.message : 'An unexpected error occurred.')
    } finally {
      setIsLoading(false)
    }
  }

  // Allow submitting with the Enter key (Shift+Enter inserts a newline).
  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  // ── Render ──────────────────────────────────────────────────────────

  return (
    <div className="flex h-full flex-col gap-4">
      {/* ── Page header ─────────────────────────────────────────────── */}
      <div>
        <h2 className="text-xl font-semibold text-gray-800">Adaptive Tutor</h2>
        <p className="mt-1 text-sm text-gray-500">
          Ask any math question. The tutor retrieves relevant knowledge (RAG) or
          answers directly, then reasons step-by-step before giving a final answer.
        </p>
      </div>

      {/* ── Error banner ─────────────────────────────────────────────── */}
      <ErrorBanner error={error} onDismiss={() => setError(null)} />

      {/* ── Topic filter ─────────────────────────────────────────────── */}
      <div className="flex items-center gap-2">
        <label htmlFor="topic-filter" className="shrink-0 text-sm font-medium text-gray-700">
          Topic filter
        </label>
        <input
          id="topic-filter"
          type="text"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          placeholder="e.g. quadratic equations (optional)"
          className="flex-1 rounded-md border border-gray-300 px-3 py-1.5 text-sm shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
      </div>

      {/* ── Chat history ─────────────────────────────────────────────── */}
      {/*
        The chat area is scrollable. flex-1 + min-h-0 lets it grow to fill
        available vertical space while keeping the input bar pinned at the bottom.
      */}
      <div className="flex-1 overflow-y-auto rounded-lg border border-gray-200 bg-gray-50 p-4">
        {history.length === 0 && !isLoading && (
          <p className="text-center text-sm text-gray-400">
            No messages yet — ask a question below to get started.
          </p>
        )}

        <div className="flex flex-col gap-4">
          {history.map((entry) =>
            entry.role === 'student' ? (
              // ── Student bubble ──────────────────────────────────────
              <div key={entry.id} className="flex justify-end">
                <div className="max-w-[75%] rounded-2xl rounded-br-sm bg-blue-600 px-4 py-2.5 text-sm text-white shadow-sm">
                  {entry.text}
                </div>
              </div>
            ) : (
              // ── Tutor bubble ────────────────────────────────────────
              <TutorMessage key={entry.id} entry={entry} />
            ),
          )}

          {/* Loading spinner appears inside the chat area while awaiting reply */}
          {isLoading && (
            <div className="flex justify-start">
              <div className="rounded-2xl rounded-bl-sm bg-white px-4 py-3 shadow-sm">
                <LoadingSpinner isLoading={isLoading} />
              </div>
            </div>
          )}

          {/* Invisible anchor element — scrolled into view after each update */}
          <div ref={bottomRef} />
        </div>
      </div>

      {/* ── Input bar ────────────────────────────────────────────────── */}
      <div className="flex items-end gap-2">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a math question… (Enter to send, Shift+Enter for newline)"
          rows={2}
          disabled={isLoading}
          className="flex-1 resize-none rounded-lg border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:bg-gray-100 disabled:text-gray-400"
          aria-label="Question input"
        />
        <button
          onClick={handleSend}
          disabled={isLoading || !query.trim()}
          className="shrink-0 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 disabled:cursor-not-allowed disabled:opacity-50"
          aria-label="Send question"
        >
          Send
        </button>
      </div>
    </div>
  )
}

// ── TutorMessage sub-component ────────────────────────────────────────

/**
 * Renders a single tutor response bubble.
 *
 * Layout:
 *  ┌─────────────────────────────────────────────────────┐
 *  │ [RAG] or [Direct] badge                             │
 *  │                                                     │
 *  │ ## Reasoning  (muted grey text)                     │
 *  │ <reasoning steps>                                   │
 *  │                                                     │
 *  │ ## Answer  (bold label)                             │
 *  │ <final answer>                                      │
 *  └─────────────────────────────────────────────────────┘
 */
function TutorMessage({ entry }: { entry: ChatEntry }) {
  const { reasoning, answer } = parseResponse(entry.text)
  const badge = entry.routeTaken ? routeLabel(entry.routeTaken) : null

  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] rounded-2xl rounded-bl-sm bg-white px-4 py-3 shadow-sm border border-gray-100">
        {/* Route badge — "RAG" (blue) or "Direct" (green) */}
        {badge && (
          <span
            className={`mb-2 inline-block rounded-full px-2 py-0.5 text-xs font-semibold ${
              badge === 'RAG'
                ? 'bg-blue-100 text-blue-700'
                : 'bg-green-100 text-green-700'
            }`}
            title={
              badge === 'RAG'
                ? 'Answer was generated using retrieved knowledge (RAG pipeline)'
                : 'Answer was generated directly without retrieval'
            }
          >
            {badge}
          </span>
        )}

        {/* Reasoning section — rendered in muted text (Req 11.2) */}
        {reasoning && (
          <div className="mb-3">
            <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-gray-400">
              Reasoning
            </p>
            {/* Preserve whitespace and line breaks from the LLM output */}
            <p className="whitespace-pre-wrap text-sm text-gray-500">{reasoning}</p>
          </div>
        )}

        {/* Answer section — rendered in bold (Req 11.2) */}
        <div>
          <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-gray-700">
            Answer
          </p>
          <p className="whitespace-pre-wrap text-sm font-semibold text-gray-900">{answer}</p>
        </div>
      </div>
    </div>
  )
}

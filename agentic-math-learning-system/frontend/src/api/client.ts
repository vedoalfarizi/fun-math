// src/api/client.ts
// ─────────────────────────────────────────────────────────────────────
// Typed API client for the Agentic Math Learning System backend.
// All requests go through apiFetch<T>() so error handling is consistent.
// VITE_API_URL defaults to http://localhost:8080 for local development.
// ─────────────────────────────────────────────────────────────────────

// ── Base URL ─────────────────────────────────────────────────────────
// Vite exposes env variables prefixed with VITE_ via import.meta.env.
const API_BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8080'

// ── Request / Response types (mirror Pydantic models exactly) ────────

// Path A — Adaptive Tutor
export interface TutorChatRequest {
  session_id: string
  query: string
  topic?: string
}

export interface TutorChatResponse {
  session_id: string
  response: string
  route_taken: string
  context_chunks: string[]
}

// Path B — Practice Agent
export interface PracticeStartRequest {
  topic: string
  difficulty_level: 'easy' | 'medium' | 'hard'
}

export interface PracticeStartResponse {
  session_id: string
  question_id: string
  question_text: string
  options: string[]
}

export interface PracticeAnswerRequest {
  session_id: string
  question_id: string
  selected_option: string
}

export interface PracticeAnswerResponse {
  session_id: string
  outcome: 'correct' | 'incorrect'
  explanation?: string
}

// Path C — Doc-to-Concept Agent
export interface DocConceptsResponse {
  document_id: string
  filename: string
  topic?: string
  formulas: Array<{
    formula: string
    examples: string[]
  }>
}

// ── Base fetch helper ─────────────────────────────────────────────────

/**
 * Generic fetch wrapper that:
 * 1. Prepends API_BASE_URL to the path.
 * 2. Sets Content-Type: application/json for non-FormData bodies.
 * 3. Throws a descriptive Error on non-2xx responses (Req 11.5).
 * 4. Returns the parsed JSON body typed as T.
 */
export async function apiFetch<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const url = `${API_BASE_URL}${path}`

  // Only set Content-Type for JSON bodies; FormData sets its own boundary.
  const headers: HeadersInit =
    options.body instanceof FormData
      ? {}
      : { 'Content-Type': 'application/json', ...options.headers }

  const response = await fetch(url, { ...options, headers })

  if (!response.ok) {
    // Extract field-level detail from FastAPI 422 responses when available.
    let detail = response.statusText
    try {
      const errorBody = await response.json()
      detail = errorBody.detail ?? detail
    } catch {
      // ignore JSON parse errors on error responses
    }
    throw new Error(`API error ${response.status}: ${detail}`)
  }

  return response.json() as Promise<T>
}

// ── Typed API wrappers ────────────────────────────────────────────────

/** POST /api/tutor/chat — send a question to the Adaptive Tutor. */
export async function tutorChat(req: TutorChatRequest): Promise<TutorChatResponse> {
  return apiFetch<TutorChatResponse>('/api/tutor/chat', {
    method: 'POST',
    body: JSON.stringify(req),
  })
}

/** POST /api/practice/start — generate a new MCQ for the given topic. */
export async function practiceStart(req: PracticeStartRequest): Promise<PracticeStartResponse> {
  return apiFetch<PracticeStartResponse>('/api/practice/start', {
    method: 'POST',
    body: JSON.stringify(req),
  })
}

/** POST /api/practice/answer — submit a student's MCQ answer. */
export async function practiceAnswer(req: PracticeAnswerRequest): Promise<PracticeAnswerResponse> {
  return apiFetch<PracticeAnswerResponse>('/api/practice/answer', {
    method: 'POST',
    body: JSON.stringify(req),
  })
}

/**
 * POST /api/docs/upload — upload a PDF for concept extraction.
 * Uses FormData so the backend receives a multipart/form-data request.
 */
export async function uploadDoc(file: File, topic?: string): Promise<{ document_id: string }> {
  const formData = new FormData()
  formData.append('file', file)
  if (topic) formData.append('topic', topic)
  return apiFetch<{ document_id: string }>('/api/docs/upload', {
    method: 'POST',
    body: formData,
  })
}

/** GET /api/docs/{document_id}/concepts — retrieve extracted formulas and examples. */
export async function getDocConcepts(documentId: string): Promise<DocConceptsResponse> {
  return apiFetch<DocConceptsResponse>(`/api/docs/${documentId}/concepts`)
}

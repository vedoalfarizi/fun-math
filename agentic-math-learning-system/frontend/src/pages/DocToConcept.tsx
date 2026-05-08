// src/pages/DocToConcept.tsx
// ─────────────────────────────────────────────────────────────────────
// Path C — Doc-to-Concept Agent page (Req 11.1, 11.4, 11.5, 11.6).
//
// Features:
//  • PDF file input (accept=".pdf") — validates client-side before upload.
//  • Optional topic text input to label the document in ChromaDB.
//  • "Upload & Analyse" button — posts a FormData payload via uploadDoc().
//  • After upload, polls getDocConcepts() until the document record is
//    ready (formulas array is non-empty), then renders results.
//  • Extracted formulas rendered in monospace <code> blocks.
//  • Worked examples rendered in prose below each formula.
//  • LoadingSpinner shown during upload and polling (Req 11.4).
//  • ErrorBanner shown on 422 or other API errors (Req 11.5).
// ─────────────────────────────────────────────────────────────────────

import { useRef, useState } from 'react'
import {
  uploadDoc,
  getDocConcepts,
  type DocConceptsResponse,
} from '../api/client'
import ErrorBanner from '../components/ErrorBanner'
import LoadingSpinner from '../components/LoadingSpinner'

// ── Constants ─────────────────────────────────────────────────────────

/** How long to wait between polling attempts (ms). */
const POLL_INTERVAL_MS = 2_000

/** Maximum number of polling attempts before giving up. */
const MAX_POLL_ATTEMPTS = 30

// ── Types ─────────────────────────────────────────────────────────────

/** Describes the current phase of the upload + processing pipeline. */
type Phase =
  | 'idle'        // Nothing happening — waiting for user input.
  | 'uploading'   // POST /api/docs/upload in flight.
  | 'processing'  // Polling GET /api/docs/{id}/concepts.
  | 'done'        // Results are ready to display.
  | 'error'       // A non-recoverable error occurred.

// ── Component ─────────────────────────────────────────────────────────

export default function DocToConcept() {
  // ── State ─────────────────────────────────────────────────────────

  /** The PDF file selected by the user. */
  const [file, setFile] = useState<File | null>(null)

  /** Optional topic label for the document. */
  const [topic, setTopic] = useState('')

  /** Current pipeline phase — drives UI visibility. */
  const [phase, setPhase] = useState<Phase>('idle')

  /** Human-readable status message shown below the spinner. */
  const [statusMessage, setStatusMessage] = useState('')

  /** Extracted concepts returned by the backend. */
  const [concepts, setConcepts] = useState<DocConceptsResponse | null>(null)

  /** Error message shown in the ErrorBanner. */
  const [error, setError] = useState<string | null>(null)

  /** Hidden file input ref — triggered by the styled button. */
  const fileInputRef = useRef<HTMLInputElement>(null)

  // ── Derived flags ──────────────────────────────────────────────────

  const isLoading = phase === 'uploading' || phase === 'processing'
  const canSubmit = file !== null && !isLoading

  // ── Handlers ──────────────────────────────────────────────────────

  /** Handle file selection from the native file picker. */
  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const selected = e.target.files?.[0] ?? null

    // Client-side validation: only accept PDF files.
    if (selected && !selected.name.toLowerCase().endsWith('.pdf')) {
      setError('Only PDF files are supported. Please select a .pdf file.')
      setFile(null)
      // Reset the input so the same file can be re-selected after correction.
      if (fileInputRef.current) fileInputRef.current.value = ''
      return
    }

    setFile(selected)
    setError(null)
    // Reset any previous results when a new file is chosen.
    setConcepts(null)
    setPhase('idle')
  }

  /**
   * Poll GET /api/docs/{documentId}/concepts until the backend has finished
   * processing (formulas array is non-empty) or the attempt limit is reached.
   *
   * The backend pipeline (parse → chunk → classify → embed → extract → generate)
   * runs asynchronously after the upload returns, so we need to poll.
   */
  async function pollForConcepts(documentId: string): Promise<void> {
    setPhase('processing')
    setStatusMessage('Processing document — extracting formulas and generating examples…')

    for (let attempt = 1; attempt <= MAX_POLL_ATTEMPTS; attempt++) {
      await sleep(POLL_INTERVAL_MS)

      try {
        const result = await getDocConcepts(documentId)

        // The document is ready when the backend has populated the formulas array.
        if (result.formulas && result.formulas.length > 0) {
          setConcepts(result)
          setPhase('done')
          setStatusMessage('')
          return
        }

        // Update the status message so the user knows we are still working.
        setStatusMessage(
          `Processing… (attempt ${attempt}/${MAX_POLL_ATTEMPTS})`,
        )
      } catch (err) {
        // A 404 means the record isn't written yet — keep polling.
        // Any other error is surfaced to the user.
        const message = err instanceof Error ? err.message : String(err)
        if (!message.includes('404')) {
          throw err
        }
      }
    }

    // Exceeded the maximum number of attempts.
    throw new Error(
      'Document processing is taking longer than expected. ' +
        'Please try again or check the backend logs.',
    )
  }

  /** Handle the "Upload & Analyse" button click. */
  async function handleUpload() {
    if (!canSubmit) return

    setError(null)
    setConcepts(null)
    setPhase('uploading')
    setStatusMessage('Uploading PDF…')

    try {
      // Step 1 — POST the PDF to the backend.
      const { document_id } = await uploadDoc(file!, topic.trim() || undefined)

      // Step 2 — Poll until the agentic pipeline finishes.
      await pollForConcepts(document_id)
    } catch (err) {
      // Surface the error in the ErrorBanner (Req 11.5).
      setError(err instanceof Error ? err.message : 'An unexpected error occurred.')
      setPhase('error')
      setStatusMessage('')
    }
  }

  /** Reset the form so the user can upload another document. */
  function handleReset() {
    setFile(null)
    setTopic('')
    setConcepts(null)
    setError(null)
    setPhase('idle')
    setStatusMessage('')
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  // ── Render ─────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col gap-6">
      {/* ── Page header ───────────────────────────────────────────── */}
      <div>
        <h2 className="text-xl font-semibold text-gray-800">Doc-to-Concept Agent</h2>
        <p className="mt-1 text-sm text-gray-500">
          Upload a math PDF to extract formulas and generate worked examples.
          The backend parses the document with LlamaParse, classifies chunks,
          embeds them in ChromaDB, then uses the LLM to extract and illustrate
          every formula.
        </p>
      </div>

      {/* ── Error banner ──────────────────────────────────────────── */}
      <ErrorBanner error={error} onDismiss={() => setError(null)} />

      {/* ── Upload form ───────────────────────────────────────────── */}
      {phase !== 'done' && (
        <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm">
          <div className="flex flex-col gap-4">
            {/* PDF file picker */}
            <div>
              <label
                htmlFor="pdf-upload"
                className="mb-1 block text-sm font-medium text-gray-700"
              >
                PDF file <span className="text-red-500">*</span>
              </label>

              {/*
                The native <input type="file"> is hidden; a styled button
                triggers it. This gives full control over the visual design
                while keeping the input accessible.
              */}
              <input
                ref={fileInputRef}
                id="pdf-upload"
                type="file"
                accept=".pdf"
                onChange={handleFileChange}
                disabled={isLoading}
                className="sr-only"
                aria-describedby="pdf-upload-hint"
              />

              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isLoading}
                  className="rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 disabled:cursor-not-allowed disabled:opacity-50"
                  aria-controls="pdf-upload"
                >
                  Choose PDF…
                </button>

                {/* Show selected filename or placeholder */}
                <span
                  id="pdf-upload-hint"
                  className={`text-sm ${file ? 'text-gray-800' : 'text-gray-400'}`}
                >
                  {file ? file.name : 'No file selected'}
                </span>
              </div>
            </div>

            {/* Optional topic input */}
            <div>
              <label
                htmlFor="doc-topic"
                className="mb-1 block text-sm font-medium text-gray-700"
              >
                Topic <span className="text-gray-400 font-normal">(optional)</span>
              </label>
              <input
                id="doc-topic"
                type="text"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="e.g. calculus, linear algebra…"
                disabled={isLoading}
                className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:bg-gray-100 disabled:text-gray-400"
              />
            </div>

            {/* Upload button */}
            <div>
              <button
                type="button"
                onClick={handleUpload}
                disabled={!canSubmit}
                className="rounded-lg bg-blue-600 px-5 py-2 text-sm font-medium text-white shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 disabled:cursor-not-allowed disabled:opacity-50"
              >
                Upload &amp; Analyse
              </button>
            </div>
          </div>

          {/* Loading spinner + status message */}
          {isLoading && (
            <div className="mt-4 flex flex-col items-center gap-2">
              <LoadingSpinner isLoading={isLoading} />
              {statusMessage && (
                <p className="text-sm text-gray-500">{statusMessage}</p>
              )}
            </div>
          )}
        </div>
      )}

      {/* ── Results ───────────────────────────────────────────────── */}
      {phase === 'done' && concepts && (
        <ConceptResults concepts={concepts} onReset={handleReset} />
      )}
    </div>
  )
}

// ── ConceptResults sub-component ──────────────────────────────────────

/**
 * Renders the extracted formulas and their worked examples.
 *
 * Layout per formula:
 *  ┌──────────────────────────────────────────────────────────────┐
 *  │  Formula N                                                   │
 *  │  ┌────────────────────────────────────────────────────────┐  │
 *  │  │  <formula in monospace code block>                     │  │
 *  │  └────────────────────────────────────────────────────────┘  │
 *  │  Worked Examples                                             │
 *  │  1. <example prose>                                          │
 *  │  2. <example prose>                                          │
 *  └──────────────────────────────────────────────────────────────┘
 */
function ConceptResults({
  concepts,
  onReset,
}: {
  concepts: DocConceptsResponse
  onReset: () => void
}) {
  return (
    <div className="flex flex-col gap-6">
      {/* Results header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-800">
            Extracted Concepts
          </h3>
          <p className="mt-0.5 text-sm text-gray-500">
            <span className="font-medium">{concepts.filename}</span>
            {concepts.topic && (
              <>
                {' '}
                &mdash; topic:{' '}
                <span className="font-medium">{concepts.topic}</span>
              </>
            )}
          </p>
          <p className="mt-1 text-sm text-gray-400">
            {concepts.formulas.length} formula
            {concepts.formulas.length !== 1 ? 's' : ''} found
          </p>
        </div>

        {/* Allow the user to upload another document */}
        <button
          type="button"
          onClick={onReset}
          className="shrink-0 rounded-md border border-gray-300 bg-white px-3 py-1.5 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1"
        >
          Upload another
        </button>
      </div>

      {/* Formula cards */}
      {concepts.formulas.length === 0 ? (
        <p className="rounded-lg border border-yellow-200 bg-yellow-50 px-4 py-3 text-sm text-yellow-800">
          No formulas were extracted from this document. Try uploading a PDF
          with explicit mathematical notation.
        </p>
      ) : (
        <ol className="flex flex-col gap-6" aria-label="Extracted formulas">
          {concepts.formulas.map((item, index) => (
            <li
              key={index}
              className="rounded-lg border border-gray-200 bg-white p-5 shadow-sm"
            >
              {/* Formula label */}
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-400">
                Formula {index + 1}
              </p>

              {/*
                Formula rendered in a monospace code block (Req 11.6).
                The <pre> preserves whitespace and line breaks so that
                multi-line LaTeX or aligned equations display correctly.
              */}
              <pre className="overflow-x-auto rounded-md bg-gray-900 px-4 py-3 text-sm text-green-300">
                <code>{item.formula}</code>
              </pre>

              {/* Worked examples */}
              {item.examples && item.examples.length > 0 && (
                <div className="mt-4">
                  <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-400">
                    Worked Examples
                  </p>
                  <ol className="flex flex-col gap-3" aria-label={`Examples for formula ${index + 1}`}>
                    {item.examples.map((example, exIdx) => (
                      <li key={exIdx} className="flex gap-3">
                        {/* Example number badge */}
                        <span
                          className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-blue-100 text-xs font-semibold text-blue-700"
                          aria-hidden="true"
                        >
                          {exIdx + 1}
                        </span>
                        {/* Example prose — whitespace-pre-wrap preserves step-by-step formatting */}
                        <p className="whitespace-pre-wrap text-sm text-gray-700 leading-relaxed">
                          {example}
                        </p>
                      </li>
                    ))}
                  </ol>
                </div>
              )}
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}

// ── Utility ───────────────────────────────────────────────────────────

/** Promise-based sleep helper used by the polling loop. */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

// src/components/ErrorBanner.tsx
// ─────────────────────────────────────────────────────────────────────
// Shared error display shown when a backend API call returns a non-2xx
// status (Req 11.5). Renders nothing when error is null/undefined.
// ─────────────────────────────────────────────────────────────────────

interface ErrorBannerProps {
  /** The error message to display. Pass null or undefined to hide the banner. */
  error: string | null | undefined
  /** Called when the user dismisses the banner. */
  onDismiss: () => void
}

export default function ErrorBanner({ error, onDismiss }: ErrorBannerProps) {
  // Render nothing when there is no error — keeps the DOM clean.
  if (!error) return null

  return (
    <div
      className="flex items-start gap-3 rounded-md bg-red-50 border border-red-300 px-4 py-3 text-red-800"
      role="alert"
      aria-live="assertive"
    >
      {/* Error icon */}
      <span className="mt-0.5 text-red-500" aria-hidden="true">⚠</span>

      {/* Error message */}
      <p className="flex-1 text-sm">{error}</p>

      {/* Dismissible × button */}
      <button
        onClick={onDismiss}
        className="ml-auto text-red-500 hover:text-red-700 focus:outline-none focus:ring-2 focus:ring-red-400 rounded"
        aria-label="Dismiss error"
      >
        ×
      </button>
    </div>
  )
}

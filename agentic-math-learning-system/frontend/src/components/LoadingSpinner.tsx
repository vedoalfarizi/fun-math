// src/components/LoadingSpinner.tsx
// ─────────────────────────────────────────────────────────────────────
// Shared loading indicator shown while any backend API call is in
// progress (Req 11.4). Uses Tailwind's animate-spin utility.
// ─────────────────────────────────────────────────────────────────────

interface LoadingSpinnerProps {
  /** When false the spinner is not rendered, keeping the DOM clean. */
  isLoading: boolean
}

export default function LoadingSpinner({ isLoading }: LoadingSpinnerProps) {
  if (!isLoading) return null

  return (
    <div className="flex items-center justify-center py-4" role="status" aria-label="Loading">
      {/* animate-spin rotates the element continuously via a CSS animation */}
      <div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-500 border-t-transparent" />
      <span className="sr-only">Loading…</span>
    </div>
  )
}

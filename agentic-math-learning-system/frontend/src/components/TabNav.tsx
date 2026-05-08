// src/components/TabNav.tsx
// ─────────────────────────────────────────────────────────────────────
// Three-tab navigation bar for the three agentic paths (Req 11.1).
// The active tab is highlighted with a blue underline and bold text.
// onSelect is called with the 0-based tab index when a tab is clicked.
// ─────────────────────────────────────────────────────────────────────

interface TabNavProps {
  /** Index of the currently active tab (0, 1, or 2). */
  activeTab: number
  /** Callback invoked with the new tab index when a tab is clicked. */
  onSelect: (index: number) => void
}

// Tab labels correspond to the three agentic paths.
const TABS = ['Adaptive Tutor', 'Practice Agent', 'Doc-to-Concept'] as const

export default function TabNav({ activeTab, onSelect }: TabNavProps) {
  return (
    <nav
      className="flex border-b border-gray-200 bg-white"
      role="tablist"
      aria-label="Agentic paths"
    >
      {TABS.map((label, index) => {
        const isActive = index === activeTab
        return (
          <button
            key={label}
            role="tab"
            aria-selected={isActive}
            aria-controls={`tabpanel-${index}`}
            onClick={() => onSelect(index)}
            className={[
              'px-6 py-3 text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-inset focus:ring-blue-500',
              isActive
                ? 'border-b-2 border-blue-600 text-blue-600'
                : 'text-gray-500 hover:text-gray-700 hover:border-b-2 hover:border-gray-300',
            ].join(' ')}
          >
            {label}
          </button>
        )
      })}
    </nav>
  )
}

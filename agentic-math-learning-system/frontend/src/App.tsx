// src/App.tsx
// ─────────────────────────────────────────────────────────────────────
// Root application component. Renders the TabNav and conditionally
// mounts the active page component based on the selected tab (Req 11.1).
//
// Page components are lazy-loaded to keep the initial bundle small —
// each agentic path is only loaded when the student navigates to it.
// ─────────────────────────────────────────────────────────────────────
import { lazy, Suspense, useState } from 'react'
import TabNav from './components/TabNav'
import LoadingSpinner from './components/LoadingSpinner'

// Lazy-load page components — each path is a separate chunk.
const AdaptiveTutor = lazy(() => import('./pages/AdaptiveTutor'))
const PracticeAgent = lazy(() => import('./pages/PracticeAgent'))
const DocToConcept = lazy(() => import('./pages/DocToConcept'))

// Map tab index → page component for clean conditional rendering.
const PAGES = [AdaptiveTutor, PracticeAgent, DocToConcept] as const

export default function App() {
  // activeTab tracks which of the three agentic paths is displayed.
  const [activeTab, setActiveTab] = useState(0)

  const ActivePage = PAGES[activeTab]

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Application header */}
      <header className="bg-white shadow-sm">
        <div className="mx-auto max-w-5xl px-4 py-4">
          <h1 className="text-xl font-bold text-gray-900">
            🧮 Agentic Math Learning System
          </h1>
          <p className="text-sm text-gray-500 mt-0.5">
            Explore Agentic RAG, stateful workflows, and LangGraph in action
          </p>
        </div>
        {/* Tab navigation — onSelect updates activeTab state */}
        <div className="mx-auto max-w-5xl px-4">
          <TabNav activeTab={activeTab} onSelect={setActiveTab} />
        </div>
      </header>

      {/* Main content area — Suspense shows a spinner while lazy chunks load */}
      <main
        className="mx-auto max-w-5xl px-4 py-6"
        role="tabpanel"
        id={`tabpanel-${activeTab}`}
        aria-label={['Adaptive Tutor', 'Practice Agent', 'Doc-to-Concept'][activeTab]}
      >
        <Suspense fallback={<LoadingSpinner isLoading={true} />}>
          <ActivePage />
        </Suspense>
      </main>
    </div>
  )
}
